- API for transforming and optimizing LLM/VLM models before passing to QAIRT Converter.
- It accepts both floating-point and quantized ONNX models as input (including ONNX+encodings and QDQ ONNX, the latter in beta) and always produces an ONNX model as output.

## When to use
- The optimizer is usually required for LLM/LVM models.
- It is also recommended to run on other models when the situations below apply
      - The model contains attention layers (MHA) → apply ** MHA→SHA conversion ** via convert_mha_to_sha().
      - Highly recommended for all attention-based models.
      - The model was quantized at a specific sequence/context length and you need to deploy at different lengths
        (for example, longer context window) → apply AR/CL rewriting via change_seq_length(), change_context_length(), or change_seq_and_context_length
      - The model is an ** MoE (Mixture-of-Experts) architecture ** → apply MoE adaptation via adapt_moe().
      - The LLM produces a ** compiled binary too large ** for the device’s execution partitions → apply model splitting via split_llm(). Each split is compiled into its own binary and all splits are executed in pipeline on the same device.
  
