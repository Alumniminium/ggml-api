namespace ggml_api.DTOs;

public record InstructInput(int maxTokens, string instruction, string input="", string output = "", string model = "", int top_k = 40, float top_p = 0.8f, float temperature = 0.85f, float repetition_penalty = 0.2f);
public record ContinuationInput(int maxTokens, string input, string model = "", int top_k = 40, float top_p = 0.8f, float temperature = 0.1f, float repetition_penalty = 0.2f);
