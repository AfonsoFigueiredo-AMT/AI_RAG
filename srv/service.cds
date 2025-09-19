service AIService {
  action queryLLM(prompt: String) returns {};
  action populateCoordinates() returns {
    message: String;
    updated: array of {
      id: Integer;
      address: String;
      lat: Double;
      lon: Double;
    };
    failed: array of {
      id: Integer;
      address: String;
    };
  };
}