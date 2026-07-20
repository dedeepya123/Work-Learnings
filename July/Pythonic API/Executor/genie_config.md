genie_config.json — Structure & Every Field Explained!

 Your dialog_output/genie_config.json — Full Structure

{
  "model_config": {

    "model": {
      "bos_token_id":   128000,
      "eos_token_id":   128001,
      "context_length": 4096,
      "vocab_size":     128256
    },

    "search": {
      "method":           "greedy",
      "temperature":      1.0,
      "top_k":            1,
      "top_p":            1.0,
      "stop_on_eos":      true,
      "max_length":       256,
      "max_total_length": 4096
    }

  },

  "backend_config": {
    "type":            "htp",
    "vtcm_mb":         8,
    "device_id":       0,
    "num_activations": 1
  },

  "tokenizer": {
    "type": "llm_tokenizer",
    "path": "artifacts/tokenizer.json"
  },

  "engine": {
    "type":         "SnapdragonLLM",
    "context_size": 4096,
    "model_config": {
      "model_path":    "artifacts/",
      "use_cases_path":"artifacts/use_cases.json"
    }
  }
}

## Every Field — Explained!
model_config.model — Model Identity Fields
bos_token_id:   128000   ← Beginning Of Sentence token ID
                          Genie prepends this to every prompt!
                          LLaMA 3: 128000 = <|begin_of_text|>

eos_token_id:   128001   ← End Of Sentence token ID
                          Genie STOPS generation when it sees this!
                          LLaMA 3: 128001 = <|end_of_text|>

context_length: 4096     ← Max KV cache size
                          = your cl value from build!
                          Genie pre-allocates KV for this many tokens

vocab_size:     128256   ← number of tokens in vocabulary
                          Genie uses this for logits sampling
                          LLaMA 3.2: 128256 tokens
model_config.search — Token Sampling Strategy
method:           "greedy"  ← sampling method
                              "greedy" = always pick highest prob token
                              "top_k"  = sample from top K tokens
                              "top_p"  = nucleus sampling

temperature:      1.0       ← controls randomness
                              < 1.0 = more deterministic
                              > 1.0 = more creative/random
                              1.0   = neutral (your case)

top_k:            1         ← greedy = top_k=1
                              top_k=50 = sample from 50 highest tokens

top_p:            1.0       ← nucleus sampling threshold
                              top_p=0.9 = sample from tokens
                              covering 90% of probability mass

stop_on_eos:      true      ← stop generation on eos_token 
                              if False → generates until max_length

max_length:       256       ← max NEW tokens to generate
                              (NOT including prompt tokens!)

max_total_length: 4096      ← absolute max = context_length
                              prompt tokens + generated tokens <= 4096
backend_config — HTP Hardware Settings
type:            "htp"   ← backend type
                           "htp" = Hexagon DSP 
                           "cpu" = CPU backend

vtcm_mb:         8       ← VTCM allocation in MB for this session
                           SM8850 has 8-16MB VTCM
                           8MB = conservative, safe setting

device_id:       0       ← which DSP device to use
                           0 = primary DSP 
                           1 = secondary (multi-DSP systems)

num_activations: 1       ← number of concurrent activations
                           1 = one model active at a time 
tokenizer — Tokenizer Config
type: "llm_tokenizer"       ← Genie's built-in LLM tokenizer
                               reads HF tokenizer.json format

## engine — Genie Inference Engine Settings
type: "SnapdragonLLM"     ← Genie engine type
                             "SnapdragonLLM" = HTP LLM engine 
                             "SnapdragonLLMCPU" = CPU engine

context_size: 4096        ← must match context_length above!

model_config:
  model_path: "artifacts/"   ← folder with .bin files
                                relative path! CWD sensitive!

  use_cases_path: "artifacts/use_cases.json"
                              ← tells Genie which AR/CL graphs
                                 are available at runtime

## use_cases.json — How Genie Picks AR/CL at Runtime

``` text
{
  "use_cases": [
    {
      "name": "ar1_cl4096",
      "ar":   1,
      "cl":   4096,
      "bin_files": [
        "split_model_1.bin",
        "split_model_2.bin",
        "split_model_3.bin"
      ]
    },
    {
      "name": "ar128_cl4096",
      "ar":   128,
      "cl":   4096,
      "bin_files": [
        "split_model_1.bin",
        "split_model_2.bin",
        "split_model_3.bin"
      ]
    }
  ]
}
```
At runtime Genie reads use_cases.json:
  Prefill → pick use_case where ar=128 
  Decode  → pick use_case where ar=1  

Both share SAME .bin files (weight sharing!) 

## How Genie Reads genie_config.json — Full Flow
``` text
genie-t2t-run -c genie_config.json
        │
        ├── formConfigString(genie_config.json)
        │     reads: context_length, vocab_size,
        │            bos_id, eos_id, sampler info,
        │            engine info, tokenizer info
        │     → builds internal configString JSON
        │
        ├── GenieDialogConfig_createFromJson(configString)
        │     → fills GenieDialogConfig_Handle
        │
        ├── GenieDialog_create(configHandle, &DialogHandle)
        │     → loads .bin files from model_path
        │     → initializes HTP contexts
        │     → pre-allocates KV cache (context_length)
        │
        └── GenieDialog_query(DialogHandle, prompt)
              → tokenize using tokenizer.json
              → prefill: ar=128 graph
              → decode:  ar=1   graph (loop)
              → stop: when eos_token_id seen
              → return generated text
```
Two Types of Fields in genie_config.json

<img width="609" height="100" alt="image" src="https://github.com/user-attachments/assets/228c3eb5-f799-4ae6-8fcc-a0472f01f212" />

<img width="637" height="160" alt="image" src="https://github.com/user-attachments/assets/20e3426d-8e62-42cf-874a-ff38bb249fc6" />
                          
