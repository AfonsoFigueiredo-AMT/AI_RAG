const cds = require('@sap/cds');
const OpenAI = require('openai');

class AddressService extends cds.ApplicationService {
  async init() {
    const { Addresses } = this.entities;
    
    let apiKey;
    if (process.env.VCAP_SERVICES) {
      const vcap = JSON.parse(process.env.VCAP_SERVICES);

      if (vcap['user-provided']) {
        const openaiService = vcap['user-provided'].find(s => s.name === 'openai-service');
        if (openaiService && openaiService.credentials && openaiService.credentials.OPENAI_API_KEY) {
          apiKey = openaiService.credentials.OPENAI_API_KEY;
        }
      }
    }

    if (!apiKey) {
      throw new Error("OPENAI_API_KEY not found in VCAP_SERVICES");
    }

    this.openai = new OpenAI({ apiKey });

    this.on('findClosest', this.onFindClosest);

    return await super.init();
  }

  async onFindClosest(req) {
    const { prompt } = req.data;

    if (!prompt) {
      return req.reject(400, 'prompt is required');
    }

    try {
      const embResp = await this.openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: prompt,
        encoding_format: "float"
      });
      let userEmbedding = embResp.data[0].embedding;

      userEmbedding = userEmbedding.slice(0, 768);

      userEmbedding = this.normalizeL2(userEmbedding);

      const { Addresses } = this.entities;
      const addrs = await SELECT.from(Addresses);

      let best = null;
      let bestScore = -Infinity;

      for (const a of addrs) {
        let dbEmb = this.parseEmbedding(a.EMBEDDING);

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

  normalizeL2(vec) {
    const norm = Math.sqrt(vec.reduce((acc, val) => acc + val * val, 0));
    if (norm === 0) return vec;
    return vec.map(v => v / norm);
  }

  parseEmbedding(raw) {
    if (!raw) return null;

    let str = raw.toString().trim();
    if (str.startsWith('"') && str.endsWith('"')) {
      str = str.slice(1, -1);
    }
    try {
      const arr = JSON.parse(str);
      if (Array.isArray(arr)) return arr;
    } catch (e) {
      const matches = str.match(/-?\d+(\.\d+)?(e-?\d+)?/g);
      if (matches) return matches.map(Number);
    }

    return null;
  }

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

