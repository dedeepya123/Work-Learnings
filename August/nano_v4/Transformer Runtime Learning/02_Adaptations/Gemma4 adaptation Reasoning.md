## Gemma4Text Model and Gemma4DecoderLayer Reasoning

"The core Transformer computation remains unchanged. The main adaptation happens at the model orchestration level. In the HF implementation, the text model dynamically derives cache state and attention masks from the DynamicCache and runtime sequence state. For the compiled HTP graph, we make that state explicit: the cache is represented using fixed buffers and explicit positions, while the global and sliding attention masks are prepared as fixed-shape tensors outside the model. The model then selects the appropriate tensors for each layer based on the static layer configuration. RoPE can similarly receive precomputed position embeddings rather than always constructing them internally.

The decoder layer itself requires almost no structural adaptation because it doesn't create the dynamic state. It mainly passes the explicit tensors into attention and performs the same normalization, residual, attention and MLP computation as HF. The notable interface change is that K/V states are returned upward because another part of the system needs them."

<img width="1192" height="622" alt="image" src="https://github.com/user-attachments/assets/b6bb70e5-cffd-46c2-adf9-0136631e4bb3" />

Two are directly about dynamic/static shape/state:
1. KV cache
2. attention mask
RoPE is different:
3. RoPE
   → graph/operator representation

And then we have additional adaptations:

4. cache memory layout
5. output interface
6. GeLU numerical/backend matching

This matters because when you encounter another model, you shouldn't search only for:

"Where are the three static-shape problems?"

Instead ask:

"What aspects of this HF implementation are incompatible or suboptimal for the target runtime?"

That is the broader question.


``` text
                 HF MODEL
                    │
                    ▼
          inspect the forward path
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
    STATE          SHAPE         OPS
       │            │             │
       ▼            ▼             ▼
 dynamic?        dynamic?      unsupported?
 object?         growing?      inefficient?
       │            │             │
       └────────────┼─────────────┘
                    ▼
              REPRESENTATION
                 CHANGE
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
   fixed buffer  fixed shape   explicit ops
        │           │            │
        └───────────┼────────────┘
                    ▼
             compiled graph
                    │
                    ▼
               accelerator
```

### Gemma4TextAttention

"This is where the main adapted runtime representations are consumed. The attention mathematics remains the same, but several representations change for the target runtime. RoPE is expressed as explicit real/imaginary complex multiplication rather than HF's rotate_half representation, which gives a cleaner graph for quantization and backend lowering. K/V are written into fixed cache buffers using explicit cache positions rather than dynamically extending the cache, so the cache update needs metadata describing the write location and layout. The attention mask is no longer constructed here; it arrives as a fixed-shape tensor prepared outside the model. The rest of the attention computation — Q/K/V projections, normalization, GQA, softmax, V aggregation and output projection — remains fundamentally the same."

### Gemma4TextMLP

"The MLP doesn't have the dynamic state problems of attention, so its structure doesn't need adaptation for static execution. The main change is numerical/operator matching: the activation is switched from PyTorch's tanh-based GeLU approximation to erf-based GeLU when that better matches the backend's implementation. The gated MLP structure, projections and tensor shapes remain unchanged."
