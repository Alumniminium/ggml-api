using System.Diagnostics;
using System.Runtime.CompilerServices;
using ggml_api.DTOs;
using LLaMA.NET;

namespace ggml_api;

public interface ILLMService
{
    public IAsyncEnumerable<string> Instruct(InstructInput instruct, CancellationToken txt);
    public IAsyncEnumerable<string> Complete(ContinuationInput continuation, CancellationToken txt);
}

public class LLMService : ILLMService
{
    public LLaMAModel Model = LLaMAModel.FromPath("/models/ggml-wizardlm-7b-q5_1.bin");
    public LLaMARunner Runner;

    public LLMService() => Runner = Model.CreateRunner(Program.THREAD_COUNT);

    public async IAsyncEnumerable<string> Complete(ContinuationInput dto, [EnumeratorCancellation]CancellationToken ct)
    {
        var waitTime = 0d;
        var modelLoadTime = 0d;
        var ingestTime = 0d;
        var inferTime = 0d;

        var start = Stopwatch.GetTimestamp();
        while (Runner.Busy)
            await Task.Delay(100);
        waitTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        start = Stopwatch.GetTimestamp();

        Runner.Busy = true;

        SetupModel(dto.model);
        modelLoadTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        start = Stopwatch.GetTimestamp();

        foreach (var token in Runner.IngestPrompt(dto.input, ct))
            yield return token;
        ingestTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        start = Stopwatch.GetTimestamp();

        foreach (var token in Runner.InferenceStream(dto.maxTokens, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, ct))
            yield return token;
        inferTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        Runner.Clear();
        Runner.Busy = false;
        
        yield return $"{Environment.NewLine}[Statistics]{Environment.NewLine}Wait: {waitTime:0.00}s{Environment.NewLine}Model Load: {modelLoadTime:0.00}s{Environment.NewLine}Ingest: {ingestTime:0.00}s{Environment.NewLine}Infer: {inferTime:0.00}s"; 
    }

    public async IAsyncEnumerable<string> Instruct(InstructInput dto, [EnumeratorCancellation]CancellationToken ct)
    {
        var waitTime = 0d;
        var modelLoadTime = 0d;
        var ingestTime = 0d;
        var inferTime = 0d;

        var start = Stopwatch.GetTimestamp();

         while(Runner.Busy)
            await Task.Delay(100);
        waitTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        start = Stopwatch.GetTimestamp();

        Runner.Busy = true;

        SetupModel(dto.model);
        modelLoadTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        start = Stopwatch.GetTimestamp();

        foreach (var token in Runner.Instruction(dto.instruction, dto.input, dto.output, ct))
            yield return token;
        ingestTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        start = Stopwatch.GetTimestamp();

        foreach (var token in Runner.InferenceStream(dto.maxTokens, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, ct))
            yield return token;
        inferTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        Runner.Clear();
        Runner.Busy = false;
        yield return $"{Environment.NewLine}[Statistics]{Environment.NewLine}Wait: {waitTime:0.00}s{Environment.NewLine}Model Load: {modelLoadTime:0.00}s{Environment.NewLine}Ingest: {ingestTime:0.00}s{Environment.NewLine}Infer: {inferTime:0.00}s";
    }

    private void SetupModel(string model)
    {
        if (!string.IsNullOrWhiteSpace(model) && Model.ModelName != model)
        {
            Model.Dispose();
            Model = LLaMAModel.FromPath($"/models/{model}");
            Runner = Model.CreateRunner(Program.THREAD_COUNT);
            Runner.Busy = true;
        }
    }
}