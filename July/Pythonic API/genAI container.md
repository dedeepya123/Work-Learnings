 ## What is the Container Object?

container = builder.build()
``` text
# container is an LLMContainer object

# It holds references to ALL compiled artifacts:
# ├── .bin files (from cache)
# ├── embedding table
# ├── tokenizer
# └── gen_ai_config
```
The Full Story After .build()
``` text
builder.build()
        │
        ▼
  LLMContainer  ← in-memory references to .bin files in cache
        │
        │  Two things you can do:
        │
        ├── 1️ container.save()    ← serialize to disk
        │                              (what you already did!)
        │
        └── 2️ container.export()  ← export to dialog format
                                       (for genie-t2t-run CLI)
``` 
1️ container.save() — What It Does
``` text
container.save("./serialized_output", exist_ok=True)
Takes ALL .bin files from cache:
  compile_7f070638.../ar1_cl4096_3_of_4.bin
  compile_31ca75df.../ar1_cl4096_2_of_4.bin
  compile_91f91a61.../ar1_cl4096_4_of_4.bin
        │
        ▼ Reorganizes into clean deployment structure
        │
serialized_output/
├── embedding_table.bin              ← split 1/4
├── embedding_table_quant_param.json
├── models/
│   ├── split_0/model.bin            ← split 2/4 (layers 0-13)
│   ├── split_1/model.bin            ← split 3/4 (layers 14-27)
│   ├── split_2/model.bin            ← split 4/4 (lm_head)
│   └── use_cases.json               ← AR/CL metadata
├── gen_ai_config.json               ← Genie runtime config
├── backend_extensions.json          ← HTP backend settings
├── metadata.json                    ← model metadata
└── tokenizer.json                   ← tokenizer
```
2️ container.export() — What It Does
``` text
container.export("./dialog_output")
Takes serialized_output structure
        │
        ▼ Reorganizes into Genie dialog format
        │
dialog_output/
├── genie_config.json         ← Genie dialog config (NOT gen_ai_config!)
└── artifacts/
    ├── embedding_table.bin   ← same bins, different structure
    ├── split_model_1.bin
    ├── split_model_2.bin
    ├── split_model_3.bin
    └── tokenizer.json
```
