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
    const queryVector = await getEmbedding(query);
    const queryVectorStr = JSON.stringify(queryVector.slice(0, 768));
    const sql = `
      SELECT TOP 5
        "SupplierInvoiceIDByInvcgParty",
        "InvoiceGrossAmount",
        "SupplierInvoice",
        COSINE_SIMILARITY("Embedding", TO_REAL_VECTOR(?)) as "COSINE_SIMILARITY"
      FROM "922E760E5EEB45D786D4B63D18D570D1"."HDI_SUPPLIERINVOICE"
      WHERE "Embedding" IS NOT NULL
      ORDER BY COSINE_SIMILARITY("Embedding", TO_REAL_VECTOR(?)) DESC
    `;

    const result = await this.db.run(sql, [queryVectorStr, queryVectorStr]);
    return result;
  }

  async queryLLM(req) {
    const query = req;
    if (!query) return req.reject(400, 'query is required');

    try {
      // Get context from vector search
      const contextRecords = await this.runVectorSearch(query);
      const context = contextRecords
        .map(record => `Invoice ID: ${record.SupplierInvoiceIDByInvcgParty}, Amount: ${record.InvoiceGrossAmount}, Details: ${record.SupplierInvoice}`)
        .join('\n');

      // Define the prompt template
      const promptTemplate = await ChatPromptTemplate.fromMessages([
        [
          'system',
          `You are an accountant. Use the following context to answer the question. Provide a structured JSON response.
          Context:
          {context}
          Question:
          {query}
          {formatInstructions}`,
        ],
      ]);

      // Define the output schema
      const outputSchema = z.object({
        answer: z.string().describe('The answer to the query based on the context.'),
        confidence: z.number().min(0).max(1).describe('Confidence score for the answer (0 to 1).'),
        relevantInvoices: z
          .array(
            z.object({
              invoiceId: z.string().describe('The SupplierInvoiceIDByInvcgParty of the relevant invoice.'),
              relevance: z.number().min(0).max(1).describe('Relevance score of the invoice to the query.'),
            })
          )
          .describe('List of relevant invoices from the context.'),
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