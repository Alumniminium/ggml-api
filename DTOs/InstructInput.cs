namespace ggml_api.DTOs;

public record InstructInput
(
    int maxTokens,
    string[] reversePrompts,
    string instruction = "Tell me a joke",
    string input = "",
    string output = "",
    string model = "",
    bool ignore_eos = false,
    int top_k = 40,
    float top_p = 0.8f,
    float temperature = 0.85f,
    float repetition_penalty = 0.2f,
    bool includeIngest = false,
    bool includeStatistics = false
);
