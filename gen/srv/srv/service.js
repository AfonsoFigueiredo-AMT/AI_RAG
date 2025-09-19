import cds from '@sap/cds';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StructuredOutputParser } from '@langchain/core/output_parsers';
import { OutputFixingParser } from 'langchain/output_parsers';
import { z } from 'zod';
import { OpenAI } from 'openai';
import fetch from 'node-fetch';

class AIService extends cds.ApplicationService {
  async init() {
    // Initialize OpenAI
    let apiKey;
    if (process.env.VCAP_SERVICES) {
      const vcap = JSON.parse(process.env.VCAP_SERVICES);
      if (vcap['user-provided']) {
        const openaiService = vcap['user-provided'].find(s => s.name === 'openai-service');
        if (openaiService?.credentials?.OPENAI_API_KEY) {
          apiKey = openaiService.credentials.OPENAI_API_KEY;
        }
      }
    }
    if (!apiKey) throw new Error("OPENAI_API_KEY not found in VCAP_SERVICES");

    this.openai = new OpenAI({ apiKey });
    process.env.OPENAI_API_KEY = apiKey;

    console.log('OpenAI API Key set successfully');

    this.db = await cds.connect.to('db');
    this.on('queryLLM', this.queryLLM);
    this.on('populateCoordinates', this.populateCoordinates);

    await super.init();
  }

  async runVectorSearch(query) {
    const sql = `
      SELECT TOP 10
        COSINE_SIMILARITY("EMBEDDING", VECTOR_EMBEDDING(?, 'QUERY', 'SAP_NEB.20240715')) AS Similarity,
        VECTOR_EMBEDDING(?, 'QUERY', 'SAP_NEB.20240715') AS QueryEmbedding,
        *
      FROM "922E760E5EEB45D786D4B63D18D570D1"."CLIENTS"
      ORDER BY COSINE_SIMILARITY(
      VECTOR_EMBEDDING(?, 'QUERY', 'SAP_NEB.20240715'),"EMBEDDING") DESC;
      `;

    const result = await this.db.run(sql, [query, query, query]);
    return result;
  }

  // Geocoding function using Nominatim (OpenStreetMap)
  async geocodeAddress(address) {
    const query = encodeURIComponent(address);
    const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&limit=1`;
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'MyCAPApp/1.0' } // Simplified User-Agent
      });
      const data = await response.json();
      if (data.length > 0) {
        return {
          lat: parseFloat(data[0].lat),
          lon: parseFloat(data[0].lon)
        };
      }
      return null;
    } catch (err) {
      console.error('Geocoding error for address', address, ':', err);
      return null;
    }
  }

  // Fetch route geometry from OSRM
  async getRouteGeometry(coordinates) {
    if (coordinates.length < 2) return null;
    const coordsStr = coordinates.map(c => `${c.lon},${c.lat}`).join(';');
    const url = `http://router.project-osrm.org/route/v1/driving/${coordsStr}?overview=full&geometries=geojson`;
    try {
      const response = await fetch(url);
      const data = await response.json();
      if (data.routes && data.routes.length > 0) {
        return data.routes[0].geometry;
      }
      return null;
    } catch (err) {
      console.error('OSRM route error:', err);
      return null;
    }
  }

  // Endpoint to populate Latitude and Longitude
  async populateCoordinates(req) {
    try {
      // Fetch all clients without coordinates, including PostalCode
      const clients = await this.db.run(
        'SELECT ID, Address, PostalCode, City, Country FROM "922E760E5EEB45D786D4B63D18D570D1"."CLIENTS" WHERE Latitude IS NULL OR Longitude IS NULL'
      );

      if (clients.length === 0) {
        return { message: 'All clients already have coordinates or no clients found.' };
      }

      // Process each client with a 1-second delay to respect Nominatim rate limits
      const results = {
        updated: [],
        failed: []
      };
      for (const client of clients) {
        const fullAddress = `${client.Address}, ${client.PostalCode} ${client.City}, ${client.Country}`;
        console.log(`Geocoding: ${fullAddress}`);
        const coords = await this.geocodeAddress(fullAddress);

        if (coords) {
          // Update the client record with coordinates
          await this.db.run(
            'UPDATE "922E760E5EEB45D786D4B63D18D570D1"."CLIENTS" SET Latitude = ?, Longitude = ? WHERE ID = ?',
            [coords.lat, coords.lon, client.ID]
          );
          results.updated.push({ id: client.ID, address: fullAddress, lat: coords.lat, lon: coords.lon });
          console.log(`Updated ID ${client.ID} with lat: ${coords.lat}, lon: ${coords.lon}`);
        } else {
          results.failed.push({ id: client.ID, address: fullAddress });
          console.warn(`Failed to geocode: ${fullAddress}`);
        }

        // Delay to respect Nominatim rate limit (1 second)
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      return {
        message: `Processed ${clients.length} clients.`,
        updated: results.updated,
        failed: results.failed
      };
    } catch (err) {
      console.error('Error in populateCoordinates:', err);
      return req.reject(500, `Failed to populate coordinates: ${err.message}`);
    }
  }

  async queryLLM(req) {
    const query = req.data?.prompt || req;
    if (!query) return req.reject(400, 'query is required');

    try {
      // Get context from vector search
      const contextRecords = await this.runVectorSearch(query);
      const context = contextRecords
        .map(record => {
          const columns = Object.keys(record);
          const formattedFields = columns
            .map(col => {
              const value = record[col] !== null && record[col] !== undefined ? record[col] : 'N/A';
              return `${col}: ${value}`;
            })
            .join(', ');
          return `Address Record: {${formattedFields}}`;
        })
        .join('\n');

      // Define the prompt template
      const promptTemplate = ChatPromptTemplate.fromMessages([
        [
          'system',
          `You are a Routing Expert and Travel Agent Specialist. Your job is to analyze the provided client addresses and generate the most efficient, quickest route to visit all specified clients in a logical order.

          Key Guidelines:
          - Use ONLY the provided context (address records) to identify and select relevant client addresses. Ignore any external knowledge or assumptions about locations.
          - Optimize the route for efficiency: Prioritize based on geographical proximity (e.g., group by city/region, then street order), estimated travel time (infer from address details like city, region, country), and logical flow (e.g., start from a central point if not specified, avoid backtracking).
          - If the query specifies multiple clients (e.g., "route to clients X, Y, and Z"), map them to the most relevant addresses from the context based on names, IDs, or details. If exact matches aren't found, select the top semantically similar ones.
          - Consider practical factors: Suggest modes of travel (e.g., driving, walking) based on urban vs. rural addresses; note potential delays (e.g., traffic in cities); ensure safety (e.g., avoid unsafe areas if inferable from data).
          - Structure your response:
            - Start with a brief overview of the route (total estimated steps/distance if inferable).
            - List the sequence of addresses with directions between them.
            - End with tips for execution (e.g., "Use GPS for real-time updates").
          - Provide a structured JSON response only.

          Additional Information: Assume standard road networks; if addresses lack coordinates, use descriptive sequencing (e.g., "from City A to City B via main highway").

          Context:
          {context}
          Question:
          {query}
          {formatInstructions}`,
        ],
      ]);

      // Define the output schema
      const outputSchema = z.object({
        answer: z.string().describe('The detailed route description, including sequence, directions, and tips based on the context.'),
        confidence: z.number().min(0).max(1).describe('Confidence score for the route accuracy (0 to 1, based on address match quality and optimization feasibility).'),
        relevantAddresses: z
          .array(
            z.object({
              addressId: z.string().describe('The ID of the relevant Address in the route sequence.'),
              relevance: z.number().min(0).max(1).describe('Relevance score of the Address to the query and its position in the route.'),
            })
          )
          .describe('Ordered list of relevant Addresses used in the route, with relevance reflecting match and sequencing fit.'),
      });

      // Set up the parser
      const parser = StructuredOutputParser.fromZodSchema(outputSchema);
      const formatInstructions = parser.getFormatInstructions();
      const parserWithFix = OutputFixingParser.fromLLM(getChatModel(), parser);

      // Create the LLM chain
      const llmChain = promptTemplate.pipe(getChatModel()).pipe(parserWithFix);

      // Invoke the LLM
      let response = await llmChain.invoke({
        context,
        query,
        formatInstructions,
      });

      // Fetch addresses for relevantAddresses, including PostalCode
      const addressIds = response.relevantAddresses.map(addr => addr.addressId);
      const addresses = await this.db.run(
        `SELECT ID, CompanyName, Address, PostalCode, City, Country, Latitude, Longitude FROM "922E760E5EEB45D786D4B63D18D570D1"."CLIENTS" WHERE ID IN (${addressIds.map(() => '?').join(',')})`,
        addressIds
      );

      // Sort addresses to match the order in relevantAddresses and geocode if necessary
      const orderedAddresses = await Promise.all(response.relevantAddresses.map(async relAddr => {
        const addr = addresses.find(a => a.ID.toString() === relAddr.addressId);
        if (!addr) return null;

        let lat = addr.Latitude;
        let lon = addr.Longitude;
        if (!lat || !lon) {
          const fullAddress = `${addr.Address}, ${addr.PostalCode} ${addr.City}, ${addr.Country}`;
          const coords = await this.geocodeAddress(fullAddress);
          if (coords) {
            // Update database with coordinates
            await this.db.run(
              'UPDATE "922E760E5EEB45D786D4B63D18D570D1"."CLIENTS" SET Latitude = ?, Longitude = ? WHERE ID = ?',
              [coords.lat, coords.lon, addr.ID]
            );
            lat = coords.lat;
            lon = coords.lon;
          }
        }

        return {
          id: addr.ID,
          companyName: addr.CompanyName,
          address: `${addr.Address}, ${addr.PostalCode} ${addr.City}, ${addr.Country}`,
          lat,
          lon
        };
      }));

      // Filter out null entries and addresses without coordinates
      const validAddresses = orderedAddresses.filter(addr => addr !== null && addr.lat && addr.lon);

      // Get route geometry from OSRM
      const coordinates = validAddresses.map(addr => ({ lat: addr.lat, lon: addr.lon }));
      const routeGeometry = await this.getRouteGeometry(coordinates);

      // Generate Google Maps URL as fallback
      let mapsUrl = '';
      if (validAddresses.length > 0) {
        const origin = encodeURIComponent(validAddresses[0].address);
        const destination = encodeURIComponent(validAddresses[validAddresses.length - 1].address);
        const waypoints = validAddresses.slice(1, -1).map(addr => encodeURIComponent(addr.address)).join('|');
        mapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${origin}&destination=${destination}`;
        if (waypoints) {
          mapsUrl += `&waypoints=${waypoints}`;
        }
        mapsUrl += '&travelmode=driving';
      }

      // Add map data to response
      response = {
        ...response,
        mapsUrl: mapsUrl || 'No valid addresses found for mapping',
        leafletMapData: {
          waypoints: validAddresses.map(addr => ({
            id: addr.id,
            companyName: addr.companyName,
            address: addr.address,
            coordinates: [addr.lat, addr.lon]
          })),
          routeGeometry: routeGeometry || null
        }
      };

      return response;
    } catch (err) {
      console.error('Error in queryLLM:', err);
      return req.reject(500, `Failed to process query: ${err.message}`);
    }
  }

  // Normalize vector L2
  normalizeL2(vec) {
    const norm = Math.sqrt(vec.reduce((acc, val) => acc + val * val, 0));
    return norm === 0 ? vec : vec.map(v => v / norm);
  }
}

const getChatModel = () => {
  return new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
    temperature: 0.1,
  });
};

const getEmbedding = async function(prompt) {
  const embResp = await this.openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: prompt,
    encoding_format: 'float'
  });

  return embResp.data[0].embedding;
};

export default AIService;