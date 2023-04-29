using Microsoft.AspNetCore.Mvc;

namespace ggml_api.Controllers;

public record InstructInput(int maxTokens, string instruction, string input="", string output = "", string model = "", int top_k = 40, float top_p = 0.8f, float temperature = 0.85f, float repetition_penalty = 0.2f);
public record ContinuationInput(int maxTokens, string input, string model = "", int top_k = 40, float top_p = 0.8f, float temperature = 0.1f, float repetition_penalty = 0.2f);

[ApiController]
[Route("[controller]")]
public class LLaMAController : ControllerBase
{
    private readonly ILogger<LLaMAController> _logger;
    public LLaMAController(ILogger<LLaMAController> logger) => _logger = logger;

    [HttpGet]
    [Route("/v1/models")]
    [Produces("text/plain")]
    public async IAsyncEnumerable<string> GetModels()
    {
        var models = Directory.GetFiles("/models");
        foreach (var model in models)
        {
            _logger.LogInformation("{model}", model);
            yield return Path.GetFileName(model) + Environment.NewLine;
        }
    }

    [HttpPost]
    [Route("/instruction")]
    [Produces("text/plain")]
    public async IAsyncEnumerable<string> GetInstructionResponse([FromBody]InstructInput dto)
    {
        while(AI.Runner.Busy)
            await Task.Delay(100);

        AI.Runner.Busy = true;
        
        SetupModel(dto.model);

        AI.Runner.Busy = true;

        _logger.LogInformation(""""Instruction: "{Instruction}",  Input: "{Input}"", Response: "{Response}"""", dto.instruction, dto.input, dto.output);
        foreach (var token in AI.Runner.Instruction(dto.instruction, dto.input, dto.output, HttpContext.RequestAborted))
        {
            _logger.LogInformation("{token}", token);
            yield return token;
        }
        foreach (var segment in AI.Runner.InferenceStream(dto.maxTokens, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, HttpContext.RequestAborted))
        {
            _logger.LogInformation("Generated token: {segment}", segment);
            yield return segment;
        }
        AI.Runner.Clear();
        AI.Runner.Busy = false;
        yield return Environment.NewLine;
    }

    [HttpPost]
    [Route("/continuation")]
    [Produces("text/plain")]
    public async IAsyncEnumerable<string> GetContinuation([FromBody]ContinuationInput dto)
    {
        while (AI.Runner.Busy)
            await Task.Delay(100);
        AI.Runner.Busy = true;

        SetupModel(dto.model);
        
        AI.Runner.Busy = true;

        _logger.LogInformation(""""Continuation: Input: "{Input}"""", dto.input);
        foreach (var token in AI.Runner.IngestPrompt(dto.input, HttpContext.RequestAborted))
        {
            _logger.LogInformation("{token}", token);
            yield return token;
        }
        foreach (var token in AI.Runner.InferenceStream(dto.maxTokens, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, HttpContext.RequestAborted))
        {
            _logger.LogInformation("{token}", token);
            yield return token;
        }
        AI.Runner.Clear();
        AI.Runner.Busy = false;
        yield return Environment.NewLine;
    }

    private static void SetupModel(string model)
    {
        if (!string.IsNullOrWhiteSpace(model) && AI.Model.ModelName != model)
        {
            AI.Model.Dispose();
            AI.Model = LLaMA.NET.LLaMAModel.FromPath($"/models/{model}");
            AI.Runner = AI.Model.CreateRunner(Program.THREAD_COUNT);
        }
    }
}
