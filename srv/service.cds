service AIService {
  action queryLLM(query: String, k: Integer) returns {
    answer: String;
    confidence: Double;
    relevantInvoices: array of {
      invoiceId: String;
      relevance: Double;
    };
  };
  action findClosest(prompt : String);
}
