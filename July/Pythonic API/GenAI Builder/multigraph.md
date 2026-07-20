## What is multi_graph?

GenAIBuilder.HTP: DEBUG | multi_graph | gen_ai_builder_htp.py:254
- Multi-graph set to: False   ← DEFAULT

GenAIBuilder.HTP: DEBUG | multi_graph | gen_ai_builder_htp.py:267
- Context length set to single value: [4096]   ← only ONE CL!
  
**multi_graph controls whether a SINGLE context binary .bin contains graphs for MULTIPLE context lengths (CL values)**

 Default Behaviour — multi_graph=False
auto_regression_number = [1, 128]   ← multiple ARs   (weight sharing)
context_length         = [4096]     ← ONE CL only

## What you get per split .bin:
``` text
┌─────────────────────────────────┐
│  model.bin                      │
│  ├── Graph: AR=1,   CL=4096     │
│  ├── Graph: AR=128, CL=4096     │
│  └── Shared weights ~1.5GB      │
└─────────────────────────────────┘
```
ONE context length = ONE set of KV cache shapes compiled in!
When multi_graph=True — Multiple CLs in ONE .bin
 
``` text
"MultiGraph means creating a context binary that has graphs inside, each for a different context length 
In addition to creating different graphs for prefill/decode. 
With CL_LIST of 256, 512, 1K, 2K, 4K — you end up with 10 graphs inside each context binary: AR128_CL256, AR128_CL512... AR1_CL256, AR1_CL512..."
```
# Enable multi_graph:
builder.multi_graph = True
builder._transformation_config.model_transformer_config.arn_cl_options.context_length = [256, 512, 1024, 2048, 4096]
builder._transformation_config.model_transformer_config.arn_cl_options.auto_regression_number = [1, 128]
``` text
What you get per split .bin:
┌─────────────────────────────────────────────────────┐
│  model.bin  (multi_graph=True)                      │
│                                                     │
│  ├── Graph: AR=1,   CL=256                          │
│  ├── Graph: AR=128, CL=256                          │
│  ├── Graph: AR=1,   CL=512                          │
│  ├── Graph: AR=128, CL=512                          │
│  ├── Graph: AR=1,   CL=1024                         │
│  ├── Graph: AR=128, CL=1024                         │
│  ├── Graph: AR=1,   CL=2048                         │
│  ├── Graph: AR=128, CL=2048                         │
│  ├── Graph: AR=1,   CL=4096                         │
│  ├── Graph: AR=128, CL=4096                         │
│                                                     │
│  └── Shared weights ~1.5GB  ← ALL graphs share!    │
└─────────────────────────────────────────────────────┘
```

5 CL values × 2 AR values = 10 graphs in ONE .bin
 
## Why Would You Want Multiple CLs?
Without multi_graph (default):
  CL=4096 always allocated
  Short prompt "Hi" = 5 tokens
  → KV cache = [1, 32, 4096, 128]  ← 4096 slots allocated!
  → Wasteful for short prompts! 

## With multi_graph=True:
  Genie picks the RIGHT graph at runtime:
  Prompt = 5 tokens  → use CL=256  graph (small, fast!) 
  Prompt = 2000 tokens→ use CL=2048 graph               
  Prompt = 4000 tokens→ use CL=4096 graph               

## Dynamic CL selection = memory efficient! 
How It Relates to AR/CL Conversion
``` text 
AR/CL Conversion Stage:

multi_graph=False:                multi_graph=True:
  context_length=[4096]             context_length=[256,512,1024,2048,4096]
         │                                  │
         ▼                                  ▼
  2 ONNX files:                    10 ONNX files:
  ├── ar1_cl4096.onnx               ├── ar1_cl256.onnx
  └── ar128_cl4096.onnx             ├── ar128_cl256.onnx
                                    ├── ar1_cl512.onnx
                                    ├── ar128_cl512.onnx
                                    │   ...
                                    └── ar128_cl4096.onnx
         │                                  │
         ▼                                  ▼
  1 .bin per split                  1 .bin per split
  (2 graphs inside)                 (10 graphs inside)
  weight shared                  ALL weight shared 
 ```

## AVERAGE PERFORMANCE CHANGES (Multi-CL vs Non-Multi-CL):
  Init Time:              +58.0%  ← longer to load (more graphs)
  Prompt Processing:      +29.5%  ← longer TTFT (more complex)
  Time to First Token:    -22.6%  ← FASTER! 
  Token Generation:       -15.3%  ← FASTER! 
  Prompt Processing Rate: +29.4%
  Token Generation Rate:  +16.6%  ← 16% faster decode! 
 
 Summary — multi_graph vs Default

## Setting	CL Graphs	ARN Graphs	Total per split	Use case
multi_graph=False (default)	1 CL	2 AR	2 graphs	Fixed context length 
multi_graph=True	N CLs	2 AR	N×2 graphs	Dynamic CL selection 
Simple rule: multi_graph=False = ONE context length baked in → always allocates max KV cache. multi_graph=True = MULTIPLE context lengths → Genie picks smallest fitting graph at runtime → faster + memory efficient for variable-length prompts



## 1️. Why Does Prompt Processing Time INCREASE with multi_graph?
``` text
multi_graph=False (CL=4096 always):
  Prompt = "What is Snapdragon?" (7 tokens)
  → padded to AR=128 fixed shape
  → runs on CL=4096 graph
  → attention mask tells model:
    "only attend to these 7 positions"
  → FIXED graph, FIXED KV shape → FAST 

multi_graph=True (CL selected dynamically):
  Prompt = "What is Snapdragon?" (7 tokens)
  → Genie must first DECIDE which CL to use!
  → check use_cases.json
  → select CL=256 graph
  → switch context binary
  → load that graph variant
  → THEN run inference
```
Decision overhead + graph switching = SLOWER! 
The overhead is from runtime graph selection + context switching — Genie must decide which CL graph to load before even starting inference!

## 2️. multi_graph=False 
multi_graph=False:
  CL = 4096  fixed, always

  KV cache pre-allocated = [1, 32, 4096, 128]
  → 4096 slots always in DDR
  → REGARDLESS of prompt length!

  Max tokens one conversation = 4096
  "one person in one session"  exactly right!

  prompt tokens + generated tokens <= 4096
  if you exceed 4096 → KV overflow → context window full

## 3️. How Does Dynamic CL Selection Work?
multi_graph=True with CL_LIST = [256, 512, 1024, 2048, 4096]

 **"based on seq_len we decide which CL?"**

ANSWER: YES — based on CURRENT conversation length!
``` text
How Genie decides at RUNTIME:
        │
        ▼
  count current tokens:
  prompt_tokens + past_kv_tokens = total_used
        │
        ├── total_used <= 256   → use CL=256  graph 
        ├── total_used <= 512   → use CL=512  graph 
        ├── total_used <= 1024  → use CL=1024 graph 
        ├── total_used <= 2048  → use CL=2048 graph 
        └── total_used <= 4096  → use CL=4096 graph 
```   
 How KV Cache Grows Over Conversation
Turn 1: "Hi"  (2 tokens)
  → CL=256 selected
  → KV allocated: [1, 32, 256, 128]  ← small! 
  → fast, memory efficient

Turn 2: "Tell me everything about Snapdragon..." (50 tokens)
  total = 2 + 50 = 52 tokens
  → still CL=256 

Turn 3: long response generated... 
  total = 200 tokens
  → still CL=256 

Turn 4: more conversation...
  total = 280 tokens  ← exceeds 256!
  → UPGRADE to CL=512 graph! 
  → new KV allocated: [1, 32, 512, 128]

...and so on until CL=4096 max!

## "seq_len is fixed, right?"

seq_len (input_tokens_per_inference) = FIXED 
  → how many tokens processed per forward pass
  → always = AR value (128 for prefill, 1 for decode)
  → NEVER changes!

context_length (CL) = DYNAMIC with multi_graph
  → total conversation window size
  → GROWS as conversation gets longer
  → Genie selects bigger CL graph when needed
``` text
┌─────────────────────────────────────────────┐
│  seq_len = AR = how many tokens IN per pass │
│  context_length = total conversation window │
│                                             │
│  seq_len is ALWAYS fixed                  │
│  context_length is dynamic (multi_graph)     │
└─────────────────────────────────────────────┘
``` 
 Summary Table
<img width="596" height="209" alt="image" src="https://github.com/user-attachments/assets/a48c24fb-a013-444f-a0bc-51292a8f4be2" />

 One-Line Summary
seq_len is always fixed (it's your AR value — how many tokens per pass). What changes with multi_graph=True is the conversation window size — Genie dynamically picks the smallest CL graph that fits your current conversation length, saving DDR memory for short conversations while still supporting up to max CL=4096 for long ones! (from memory: multi_graph CL_LIST investigation, 2026-07-20)
