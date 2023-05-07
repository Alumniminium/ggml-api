using System.Diagnostics;
using System.Runtime.CompilerServices;
using ggml_api.DTOs;
using LLaMA.NET;

namespace ggml_api;

public interface ILLMService
{
    public IAsyncEnumerable<string> Complete(ContinuationInput dto, CancellationToken txt);
    public string GetStatistics();
}

public class LLMService : ILLMService
{
    public LLM LLM;
    private double _WaitTime = 0d;
    private double _ModelLoadTime = 0d;
    private double _IngestTime = 0d;
    private double _InferTime = 0d;
    private int _Tokens;

    private double TotalTime => _WaitTime + _ModelLoadTime + _IngestTime + _InferTime;
    private double TimePerToken => TotalTime / _Tokens;
    public List<string> ChatHistory = new();

    public LLMService() => LLM = new LLM(Path.Combine(Program.MODEL_DIR, Program.DEFAULT_MODEL), Program.THREAD_COUNT);


    public async IAsyncEnumerable<string> Complete(ContinuationInput dto, [EnumeratorCancellation] CancellationToken ct)
    {
        ResetStatistics();
        await BusyWait();
        var result = SetupModel(dto.model);
        
        if(result != LLM.ModelName)
        {
            yield return result;
            yield break;
        }

        var start = Stopwatch.GetTimestamp();

        if (string.Equals(dto.input, "forget everything", StringComparison.InvariantCultureIgnoreCase))
        {
            LLM.ClearContext();
            yield return "\n\n[New Conversation]\n\n";
            yield break;
        }

        if (!string.Equals(dto.input, "continue", StringComparison.InvariantCultureIgnoreCase))
        {
            foreach (var token in LLM.IngestPrompt(dto.input))
            {
                _Tokens++;
                if (dto.includeIngest)
                    yield return token;
            }
        }
        _IngestTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        start = Stopwatch.GetTimestamp();
        foreach (var token in LLM.InferenceStream(dto.maxTokens, dto.reversePrompts, dto.ignore_eos, dto.top_k, dto.top_p, dto.temperature, dto.repetition_penalty, dto.mirostat, dto.entropy, dto.learningRate, dto.tailFreeSamplingRate, dto.typical_p, dto.penalizeNewLines, dto.penalizeSpaces, ct))
        {
            _Tokens++;
            yield return token;
        }
        _InferTime = Stopwatch.GetElapsedTime(start).TotalSeconds;

        if (dto.includeStatistics)
            yield return GetStatistics();

        LLM.Busy = false;
    }

    private async Task BusyWait()
    {
        var start = Stopwatch.GetTimestamp();

        while (LLM.Busy)
            await Task.Delay(100);
        LLM.Busy = true;
        _WaitTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
    }

    private string SetupModel(string model)
    {
        var start = Stopwatch.GetTimestamp();
        try
        {
            if (!string.IsNullOrWhiteSpace(model) && LLM.ModelName != model)
            {
                LLM.Dispose();
                LLM = new LLM(Path.Combine(Program.MODEL_DIR, model), Program.THREAD_COUNT)
                {
                    Busy = true
                };
            }
        }
        catch
        {
            return "Model not found";
        }
        _ModelLoadTime = Stopwatch.GetElapsedTime(start).TotalSeconds;
        return LLM.ModelName;
    }

    public string GetStatistics()
    {
        var stats = new Dictionary<string, double>
        {
            { "Wait Time  (sec)", _WaitTime },
            { "Model Load (sec)", _ModelLoadTime },
            { "Ingestion  (sec)", _IngestTime },
            { "Inference  (sec)", _InferTime },
            { "Total Time (sec)", TotalTime },
            { "Token Time (sec)", TimePerToken },
            { "Num Tokens (int)", _Tokens },
        };

        return $"{Environment.NewLine}{{END}}{Environment.NewLine}" + string.Join(Environment.NewLine, stats.Select(kv => $"{kv.Key}: {kv.Value:0.00}")) + Environment.NewLine;
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