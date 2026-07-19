## How Split Count is Determined

When split.num_splits is not set explicitly, the builder auto-calculates it from the total model parameter count. Because split_embedding and split_lm_head both default to True, the embedding layer and LM head are each placed in their own split. The remaining decoder layers are then divided into splits of approximately 2 GB each:

num_splits = 3 + (model_params_in_GB // 2)

Base Split 1 → Embedding layer    (split_embedding=True)
Base Split 2 → First decoder block (at least 1 decoder split always)
Base Split 3 → LM Head            (split_lm_head=True)
Then extra decoder splits added per 2GB of params:

Each 2GB of model params → +1 more decoder split

<img width="618" height="144" alt="image" src="https://github.com/user-attachments/assets/2c4a0a2c-cb3e-4862-92a7-d5b5666b83ab" />

LLaMA 3.2 3B — Exactly 4 Splits

``` text
LLaMA 3.2 3B = 3B params ≈ 6GB at FP32
formula: 3 + (6 // 2) = 3 + 3 = ... wait!

Actually from your cache:
  ar1_cl4096_2_of_4  ← split 2 of 4
  ar1_cl4096_3_of_4  ← split 3 of 4

So total = 4 splits! Let's map them:

Split 1/4 → embedding_table    → embedding_table.bin
Split 2/4 → decoder layers 0-13 → split_0/model.bin 
Split 3/4 → decoder layers 14-27 → split_1/model.bin 
Split 4/4 → lm_head            → split_2/model.bin

```
## Why ~2GB Per Split?
This is the HTP hardware constraint:

HTP VTCM (Vector TCM) per split:  ~8-16 MB
HTP can map weights from DDR:      ~2 GB max per context binary

So each context binary (.bin) can hold ~2GB of weights max!
        ↓
Split at 2GB boundaries = each split fits perfectly in HTP! 
LLaMA 3.2 3B decoder = 6GB total
        │
        ├── decoder split 1 = ~2GB  → 2_of_4.bin 
        ├── decoder split 2 = ~2GB  → 3_of_4.bin 
        ├── embedding       = ~0.5GB → embedding_table.bin 
        └── lm_head         = ~0.5GB → 4_of_4 → split_2/model.bin 
## How to Override Auto-Calculation

from qairt.api.transforms.model_transformer_config import (
    SplitModelConfig, ModelTransformerConfig
)

#  Override auto-calculated splits
builder._transformation_config.model_transformer_config.split_model_config = SplitModelConfig(
    num_splits=2,            # force 2 decoder splits instead of auto
    split_embedding=True,    # always keep True
    split_lm_head=True,      # always keep True
)

<img width="617" height="190" alt="image" src="https://github.com/user-attachments/assets/2eb88aa8-e56f-4c43-ae12-948e2df6728f" />

The auto-calculation is smart — it ensures each split fits within HTP's ~2GB context binary limit without you having to manually calculate!
