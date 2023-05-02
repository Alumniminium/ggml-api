using System.Diagnostics;
using System.Runtime.CompilerServices;
using ggml_api.DTOs;
using LLaMA.NET;

namespace ggml_api;

public interface ILLMService
{
    public IAsyncEnumerable<string> Instruct(InstructInput instruct, CancellationToken txt);
    public IAsyncEnumerable<string> Complete(ContinuationInput continuation, CancellationToken txt);
    public string GetStatistics();
}

public class LLMService : ILLMService
{
    public LLaMAModel Model;
    public LLaMARunner Runner;

    private double _WaitTime = 0d;
    private double _ModelLoadTime = 0d;
    private double _IngestTime = 0d;
    private double _InferTime = 0d;
    private int _Tokens;

    private double TotalTime => _WaitTime + _ModelLoadTime + _IngestTime + _InferTime;
    private double TimePerToken => TotalTime / _Tokens;
    public List<string> ChatHistory = new();

    public LLMService()
    {
        Model = new LLaMAModel(Path.Combine(Program.MODEL_DIR, Program.DEFAULT_MODEL));
        Runner = Model.CreateRunner(Program.THREAD_COUNT);
    }


    public async IAsyncEnumerable<string> Complete(ContinuationInput dto, [EnumeratorCancellation] CancellationToken ct)
    {
        ResetStatistics();
        await BusyWait();
        SetupModel(dto.model);

        var start = Stopwatch.GetTimestamp();

        if (dto.input == "forget everything")
        {
            Runner.Clear();
            yield return "\n\n[New Conversation]\n\n";
            yield break;
        }

        if (dto.input != "continue")
        {
            foreach (var token in Runner.IngestPrompt(dto.input, ct))
            {
                _Tokens++;
                if (dto.includeIngest)
                    yield return token;
            }
        }
        _IngestTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        if (dto.includeIngest && !string.IsNullOrWhiteSpace(dto.input))
            yield return dto.input;

        start = Stopwatch.GetTimestamp();
        foreach (var token in Runner.InferenceStream(dto.maxTokens, dto.reversePrompts, dto.ignore_eos, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, ct))
        {
            _Tokens++;
            yield return token;
        }
        _InferTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        if (dto.includeStatistics)
            yield return GetStatistics();

        Runner.Busy = false;
    }

    public async IAsyncEnumerable<string> Instruct(InstructInput dto, [EnumeratorCancellation] CancellationToken ct)
    {
        ResetStatistics();
        await BusyWait();
        SetupModel(dto.model);

        var start = Stopwatch.GetTimestamp();
        foreach (var token in Runner.Instruction(dto.instruction, dto.input, dto.output, ct))
        {
            _Tokens++;
            if (dto.includeIngest)
                yield return token;
        }
        _IngestTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        if (!dto.includeIngest && !string.IsNullOrWhiteSpace(dto.output))
            yield return dto.output;

        start = Stopwatch.GetTimestamp();
        foreach (var token in Runner.InferenceStream(dto.maxTokens, dto.reversePrompts, dto.ignore_eos, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, ct))
        {
            _Tokens++;
            yield return token;
        }
        _InferTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        if (dto.includeStatistics)
            yield return GetStatistics();

        Runner.Busy = false;
    }

    private async Task BusyWait()
    {
        var start = Stopwatch.GetTimestamp();

        while (Runner.Busy)
            await Task.Delay(100);
        Runner.Busy = true;
        _WaitTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
    }

    private void SetupModel(string model)
    {
        var start = Stopwatch.GetTimestamp();
        if (!string.IsNullOrWhiteSpace(model) && Model.ModelName != model)
        {
            Model.Dispose();
            Model = new LLaMAModel(Path.Combine(Program.MODEL_DIR, model));
            Runner = Model.CreateRunner(Program.THREAD_COUNT);
            Runner.Busy = true;
        }
        _ModelLoadTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
    }

    public string GetStatistics()
    {
        var stats = new Dictionary<string, double>
        {
            { "Wait Time ", _WaitTime },
            { "Model Load", _ModelLoadTime },
            { "Ingestion ", _IngestTime },
            { "Inference ", _InferTime },
            { "Total Time", TotalTime },
            { "Token Time", TimePerToken },
            { "Num Tokens", _Tokens },
        };

        return $"{Environment.NewLine}{{END}}{Environment.NewLine}" + string.Join(Environment.NewLine, stats.Select(kv => $"{kv.Key}: {kv.Value:0.00} sec")) + Environment.NewLine;
    }

    private void ResetStatistics()
    {
        _WaitTime = 0d;
        _ModelLoadTime = 0d;
        _IngestTime = 0d;
        _InferTime = 0d;
        _Tokens = 0;
    }
}