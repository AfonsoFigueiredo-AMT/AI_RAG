service AIService {
  action queryLLM(prompt: String) returns {
    answer: String;
    confidence: Double;
    relevantAddresses: array of {
      addressId: String;
      relevance: Double;
    };
  };
  // action findClosest(prompt : String);
}
