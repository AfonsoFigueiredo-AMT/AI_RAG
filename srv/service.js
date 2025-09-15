const cds = require('@sap/cds');
const { OpenAI } = require('openai'); // Certifique-se que o SDK está instalado

class AddressService extends cds.ApplicationService {
  async init() {
    // Carrega API Key do OpenAI via VCAP_SERVICES
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

    // Eventos
    this.on('READ', 'Addresses', this.readAddress);
    this.on('findClosest', this.onFindClosest);

    await super.init();
  }

  // Handler READ via sinônimo
  async readAddress(req) {
    try {
      const db = await cds.connect.to('db');
      const data = await db.run(`SELECT * FROM "ADDRESSSERVICE_ADDRESSES"`);
      console.log('Fetched Addresses data:', JSON.stringify(data, null, 2));
      return data;
    } catch (err) {
      console.error('Error in readAddress:', err);
      return req.reject(500, `Failed to read addresses: ${err.message}`);
    }
  }

  async onFindClosest(req) {
    const { prompt } = req.data;
    if (!prompt) return req.reject(400, 'prompt is required');

    try {
      // Cria embedding do prompt
      const embResp = await this.openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: prompt,
        encoding_format: "float"
      });

      let userEmbedding = embResp.data[0].embedding.slice(0, 768);
      userEmbedding = this.normalizeL2(userEmbedding);

      // Busca todas as addresses
      const db = await cds.connect.to('db');
      const addrs = await db.run(`SELECT * FROM "ADDRESSSERVICE_ADDRESSES"`);

      // Calcula similaridade
      let best = null;
      let bestScore = -Infinity;

      for (const a of addrs) {
        const dbEmb = this.parseEmbedding(a.EMBEDDING);
        if (!dbEmb || dbEmb.length !== userEmbedding.length) continue;

        const similarity = this.cosineSim(userEmbedding, dbEmb);
        if (similarity > bestScore) {
          bestScore = similarity;
          best = a;
        }
      }

      return best;

    } catch (err) {
      console.error('Error in findClosest handler:', err);
      return req.reject(500, `Failed to find closest address: ${err.message}`);
    }
  }

  // Normaliza vetor L2
  normalizeL2(vec) {
    const norm = Math.sqrt(vec.reduce((acc, val) => acc + val * val, 0));
    return norm === 0 ? vec : vec.map(v => v / norm);
  }

  // Converte string de embedding em array de números
  parseEmbedding(raw) {
    if (!raw) return null;
    let str = raw.toString().trim();
    if (str.startsWith('"') && str.endsWith('"')) str = str.slice(1, -1);

    try {
      const arr = JSON.parse(str);
      if (Array.isArray(arr)) return arr;
    } catch {
      const matches = str.match(/-?\d+(\.\d+)?(e-?\d+)?/g);
      if (matches) return matches.map(Number);
    }
    return null;
  }

  // Calcula similaridade cosseno
  cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
}

module.exports = AddressService;
