using ggml_api.DTOs;
using Microsoft.AspNetCore.Mvc;

namespace ggml_api.Controllers;

[ApiController]
[Route("[controller]")]
public class LLaMAController : ControllerBase
{
    private readonly ILogger<LLaMAController> _logger;
    private readonly ILLMService _llm;
    public LLaMAController(ILogger<LLaMAController> logger, ILLMService llm)
    {
        _logger = logger;
        _llm = llm;
    }

    [HttpGet]
    [Route("/models")]
    [Produces("text/plain")]
    public IEnumerable<string> GetModels()
    {
        var models = Directory.GetFiles(Program.MODEL_DIR);
        foreach (var model in models)
        {
            _logger.LogInformation("{model}", model);
            yield return Path.GetFileName(model) + Environment.NewLine;
        }
    }

    [HttpPost]
    [Route("/instruction")]
    [Produces("text/plain")]
    public async IAsyncEnumerable<string> GetInstructionResponse([FromBody] InstructInput dto)
    {
        await foreach (var token in _llm.Instruct(dto, HttpContext.RequestAborted))
            yield return token;
    }

    [HttpPost]
    [Route("/completion")]
    [Produces("text/plain")]
    public async IAsyncEnumerable<string> GetContinuation([FromBody] ContinuationInput dto)
    {
        await foreach (var token in _llm.Complete(dto, HttpContext.RequestAborted))
            yield return token;
    }
}
