using System;
using System.Text.Json.Serialization;
using ggml_api.DTOs;

namespace ggml_api;

[JsonSerializable(typeof(ContinuationInput))]
public partial class AppJsonSerializerContext : JsonSerializerContext
{

}