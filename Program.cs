using Microsoft.AspNetCore.Http.Features;

namespace ggml_api;
public static class Program
{
    public static int THREAD_COUNT => Environment.GetEnvironmentVariable("THREAD_COUNT") is null ? Environment.ProcessorCount / 2 : int.Parse(Environment.GetEnvironmentVariable("THREAD_COUNT"));
    public static string MODEL_DIR => Environment.GetEnvironmentVariable("MODEL_DIR") ?? "/models/llm/";
    public static string DEFAULT_MODEL => Environment.GetEnvironmentVariable("DEFAULT_MODEL") ?? "ggml-wizardlm-7b-q5_1.bin";

    static void Main()
    {
        var builder = WebApplication.CreateBuilder();        
        
        builder.Services.ConfigureHttpJsonOptions(options => 
        {
            options.SerializerOptions.TypeInfoResolver = new AppJsonSerializerContext();
        });
        builder.Services.AddControllers(options => options.OutputFormatters.Add(new StringOutputFormatter()));
        builder.Services.AddControllers(options => options.OutputFormatters.Add(new AsyncStringOutputFormatter()));
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        builder.WebHost.UseKestrel(options =>
        {
            options.Limits.MaxRequestBodySize = null;
            options.Limits.MaxRequestBufferSize = null;
            options.Listen(System.Net.IPAddress.Any, 5165);
        });
        builder.Services.Configure<FormOptions>(x =>
        {
            x.ValueLengthLimit = int.MaxValue;
            x.MultipartBodyLengthLimit = int.MaxValue;
        });
        
        builder.Services.AddSingleton<ILLMService, LLMService>();

        var app = builder.Build();
        app.UseSwagger();
        app.UseSwaggerUI(c =>
        {
            c.SwaggerEndpoint("/swagger/v1/swagger.json", "Her.st LLaMA API");
            c.RoutePrefix = string.Empty;
        });

        app.MapControllers();
        app.Run();
    }
}