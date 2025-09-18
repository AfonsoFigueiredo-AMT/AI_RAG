import cds from '@sap/cds';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StructuredOutputParser } from '@langchain/core/output_parsers';
import { OutputFixingParser } from 'langchain/output_parsers';
import { z } from 'zod';
import { OpenAI } from 'openai';

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
    this.on('findClosest', this.onFindClosest);
    this.on('queryLLM', this.queryLLM);

    await super.init();
  }

  async onFindClosest(req) {
    const { prompt } = req.data;
    if (!prompt) return req.reject(400, 'prompt is required');

    try {
      // Generate embedding
      const embedding = await getEmbedding(prompt);

      // Slice to 768 dimensions and normalize
      let userEmbedding = embedding.slice(0, 768);
      userEmbedding = this.normalizeL2(userEmbedding);

      // Convert embedding to JSON string for REAL_VECTOR
      const userEmbeddingStr = JSON.stringify(userEmbedding);

      // Query using COSINE_SIMILARITY
      const query = `
        SELECT
          "AddressID",
          "AddresseeFullName",
          COSINE_SIMILARITY("Embedding", TO_REAL_VECTOR(?)) as similarity
        FROM "922E760E5EEB45D786D4B63D18D570D1"."HDI_ADDRESS_2"
        WHERE "Embedding" IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 1
      `;

      // Execute the query
      const result = await this.db.run(query, [userEmbeddingStr]);

      if (!result || result.length === 0) {
        return req.reject(404, 'No matching address found');
      }

      return result[0];
    } catch (err) {
      console.error('Error in findClosest handler:', err);
      return req.reject(500, `Failed to find closest address: ${err.message}`);
    }
  }

  async runVectorSearch(query) {
    const sql = `
      SELECT TOP 10
        COSINE_SIMILARITY("Embedding", VECTOR_EMBEDDING(?, 'QUERY', 'SAP_NEB.20240715')) AS Similarity,
        VECTOR_EMBEDDING(?, 'QUERY', 'SAP_NEB.20240715') AS QueryEmbedding,
        *
      FROM "922E760E5EEB45D786D4B63D18D570D1"."HDI_ADDRESS_2"
      ORDER BY COSINE_SIMILARITY(
      VECTOR_EMBEDDING(?, 'QUERY', 'SAP_NEB.20240715'), "Embedding") DESC;
      `;

    const result = await this.db.run(sql, [query, query, query]);
    return result;
  }

  async queryLLM(req) {
    const query = req;
    if (!query) return req.reject(400, 'query is required');

    try {
      // Get context from vector search
      const contextRecords = await this.runVectorSearch(query);
      const context = contextRecords
        .map(record => {
          // Get all column names from the record (including 'similarity' and table columns)
          const columns = Object.keys(record);
          // Format each column as "ColumnName: value"
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

    // Define the output schema (unchanged, as it fits routing: answer describes the route, relevantAddresses lists the sequence)
    const outputSchema = z.object({
      answer: z.string().describe('The detailed route description, including sequence, directions, and tips based on the context.'),
      confidence: z.number().min(0).max(1).describe('Confidence score for the route accuracy (0 to 1, based on address match quality and optimization feasibility).'),
      relevantAddresses: z
        .array(
          z.object({
            addressId: z.string().describe('The AddressID of the relevant Address in the route sequence.'),
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
      const response = await llmChain.invoke({
        context,
        query,
        formatInstructions,
      });

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

const getEmbedding = async (prompt) => {
  embResp = await this.openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: prompt,
    encoding_format: 'float'
  });

  return embResp.data[0].embedding
};

export default AIService;