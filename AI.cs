namespace ggml_api;

public static class AI
{
    public static LLaMA.NET.LLaMAModel Model = LLaMA.NET.LLaMAModel.FromPath("/models/ggml-wizardlm-7b-q5_1.bin");
    public static LLaMA.NET.LLaMARunner Runner = Model.CreateRunner(Program.THREAD_COUNT);
}
