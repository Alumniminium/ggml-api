using Microsoft.AspNetCore.Http.Features;

namespace ggml_api;
public static class Program
{
    public static int THREAD_COUNT => Environment.GetEnvironmentVariable("THREAD_COUNT") is null ? Environment.ProcessorCount / 2 : int.Parse(Environment.GetEnvironmentVariable("THREAD_COUNT"));

    static void Main()
    {
        var builder = WebApplication.CreateBuilder();
        builder.Services.AddControllers(options => options.OutputFormatters.Add(new AsyncStringOutputFormatter()));
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        builder.WebHost.UseKestrel(options =>
        {
            options.Limits.MaxRequestBodySize = null;
            options.Limits.MaxRequestBufferSize = null;
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