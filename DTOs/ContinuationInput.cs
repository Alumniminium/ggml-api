namespace ggml_api.DTOs;
public record ContinuationInput
(
    int maxTokens,
    string input,
    string[] reversePrompts =null,
    string model = "ggml-wizardlm-7b-q5_1.bin",
    bool ignore_eos = false,
    int top_k = 40,
    float top_p = 0.8f,
    float temperature = 0.1f,
    int mirostat = 2,
    float entropy = 3.0f,
    float learningRate = 0.01f,
    float tailFreeSamplingRate = 1.0f,
    float typical_p = 1.0f,
    bool penalizeNewLines = false,
    bool penalizeSpaces = false,
    float repetition_penalty = 1.2f,
    bool includeIngest = true,
    bool includeStatistics = true
);
