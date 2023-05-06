namespace ggml_api.DTOs;
public record ContinuationInput
(
    int maxTokens,
    string[] reversePrompts,
    string input,
    string model = "",
    bool ignore_eos = false,
    int top_k = 40,
    float top_p = 0.8f,
    float temperature = 0.1f,
    int mirostat = 0,
    float entropy = 0.0f,
    float learningRate = 0.0f,
    float tailFreeSamplingRate = 0.0f,
    float typical_p = 0.0f,
    bool penalizeNewLines = false,
    bool penalizeSpaces = false,
    float repetition_penalty = 0.2f,
    bool includeIngest = false,
    bool includeStatistics = false
);
