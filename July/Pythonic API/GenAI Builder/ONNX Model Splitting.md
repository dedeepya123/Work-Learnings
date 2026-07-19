## ONNX Model Splitting 

### Why Do We Split at All?
The Core Problem — HTP Memory Constraints
LLaMA 3.2 3B full ONNX graph:
  All 28 transformer layers in ONE graph
  → ~6-8 GB of weights
  → HTP on-device VTCM: only 8-16 MB! 
  → Graph finalize FAILS if too large!

"Getting Graph finalize failure if we attempt to use whole model. We need to split the model into 4 or 8 logical splits to accelerate with QNN-HTP" — AISW-153846

## What is the Split Logic?

qairt.optimizer.onnx: Found 32 post-FFN residual adds (valid split points)

Split 1: 8 layers, boundary at "/transformer_h_7_Add_1/Add_output_0"
Split 2: 8 layers, boundary at "/transformer_h_15_Add_1/Add_output_0"
Split 3: 8 layers, boundary at "/transformer_h_23_Add_1/Add_output_0"

The splitter uses post-FFN residual adds as split boundaries!
Why residual adds?
  Each transformer layer ends with a residual add:
  output = LayerNorm(x + FFN(Attention(x)))
                         ↑
                 This Add node = natural split point!
  
  It's a single tensor output → clean cut with minimal
  data transfer between splits 

## LLaMA 3.2 3B Split Anatomy
With num_splits=3 + split_embedding=True:

``` text
Full LLaMA 3.2 3B (28 transformer layers)
        │
        ▼ LLMSplitter
        │
        ├── Split 1: embedding_table   ← Gather layer only
        │           → embedding_table.bin 
        │
        ├── Split 2: layers 0-13       ← 14 transformer layers
        │   (ar1_cl4096_2_of_4)        → split_0/model.bin 
        │
        ├── Split 3: layers 14-27      ← 14 transformer layers
        │   (ar1_cl4096_3_of_4)        → split_1/model.bin 
        │
        └── Split 4: lm_head           ← logits projection
            (ar1_cl4096_4_of_4)        → split_2/model.bin 
This is exactly what your cache shows — ar1_cl4096_2_of_4, 3_of_4 compiled
```
## Which File to Refer — LLMSplitter API

qairt/optimizer/onnx/passes/splitters/llm_splitter.py
Usage (from confirmed internal test_script.py):

``` text
from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.splitters import LLMSplitter

graph_context = GraphContext.from_files(
    model_path="ar1_cl4096.onnx",
    encodings_path="ar1_cl4096.encodings"
)

config = LLMSplitter.Config(
    num_splits=3,           # ← N transformer layer splits
    split_embedding=True,   # ← extract embedding as separate split
    split_lm_head=True,     # ← extract lm_head as separate split
)

split_models = LLMSplitter(config).split(graph_context)

```
## SplitModelConfig — How to Configure in Builder
From confirmed internal AISW-172893 example:


from qairt.api.transforms.model_transformer_config import (
    SplitModelConfig,
    ModelTransformerConfig,
)

#  Configure splitting in builder
builder._transformation_config = ModelTransformerConfig(
    split_model_config=SplitModelConfig(
        num_splits=3,           # number of transformer splits
        split_embedding=True,   #  extract embedding separately
        split_lm_head=True,     #  extract lm_head separately
    )
)
Or via transform options string (CLI pattern):


--transform-options 'split.num_splits=3;
                     split.split_embedding=True;
                     split.split_lm_head=True'
                     Split Configuration Reference

<img width="620" height="138" alt="image" src="https://github.com/user-attachments/assets/caef1303-7a3d-40f5-b5b2-1ff29caf3fa5" />

<img width="653" height="175" alt="image" src="https://github.com/user-attachments/assets/2e78cf86-7191-4145-9d0e-722e80a034d3" />

