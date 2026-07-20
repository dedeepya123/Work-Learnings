## What is T2TExecutor?
 The T2TExecutor handles text-to-text generation on target via Genie. It supports two modes of execution:

    - Native: This is the default mode of execution on platforms with native python support if no device is
      specified. Execution is performed via native python bindings.
    - Device: This is the mode of execution when a device is specified. Execution is performed via subprocess.
      See :class:`qairt.api.configs.device.Device` for supported device types.

"The executor object contains: genie-config, .bin files, tokenizer, sdk-dependencies. 
It can push artifacts to device and return a target object for execution."

``` text
T2TExecutor = Text-to-Text Executor
             ↑
    The BRIDGE between your Python code
    on Linux host and actual inference
    on the device!
```

## Three Layers of T2TExecutor
``` text
Your Python Script (Linux x86 host)
        │
        │  Layer 1: LLMContainer
        │  (holds .bin paths + gen_ai_config)
        │
        ▼
  container.get_executor(device)
        │
        │  Layer 2: T2TExecutor
        │  (prepares environment + manages device)
        │
        ▼
  executor.generate(prompt)
        │
        │  Layer 3: GenieT2TRunModule / GenieNativeT2TRunner
        │  (actual Genie execution engine)
        │
        ▼
  GenerationExecutionResult
  ├── generated_text 
  └── metrics 
```
## What Happens Inside container.get_executor(device)

"When a user issues get_executor(), we implicitly invoke prepare_environment() — which pushes artifacts and prepares device"
``` text
container.get_executor(device)
        │
        ├── STEP 1: prepare_environment()  ← auto-invoked!
        │           │
        │           ├── Creates temp dir on device:
        │           │   /data/local/tmp/<uuid>/
        │           │
        │           ├── Pushes ALL artifacts via ADB:
        │           │   adb push libQnnHtp.so          → /data/local/tmp/<uuid>/lib/
        │           │   adb push libQnnHtpV81Skel.so   → /data/local/tmp/<uuid>/lib/
        │           │   adb push libQnnHtpNetRunExtensions.so
        │           │   adb push libQnnSystem.so
        │           │   adb push libGenie.so
        │           │   adb push genie-t2t-run          → /data/local/tmp/<uuid>/bin/
        │           │   adb push model.bin files        → /data/local/tmp/<uuid>/
        │           │   adb push genie_config.json      → /data/local/tmp/<uuid>/
        │           │   adb push tokenizer.json         → /data/local/tmp/<uuid>/
        │           │
        │           └── Sets env vars on device:
        │               export LD_LIBRARY_PATH=/data/local/tmp/<uuid>/lib
        │               export ADSP_LIBRARY_PATH=/data/local/tmp/<uuid>/lib
        │
        └── Returns T2TExecutor object 


DEBUG - Host command: adb ['-H', 'localhost', '-s', '232dcfdd',
        'push', '.../lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so',
        '/data/local/tmp/01165311-9c19-47a6-ac40-dc77af1f7467/lib/libQnnHtpV81Skel.so']


DEBUG - Pushed genie-t2t-run to /data/local/tmp/.../bin/genie-t2t-run 
DEBUG - Pushing genie_config.json to /data/local/tmp/.../genie_config.json 
```
## What Happens Inside executor.generate(prompt)

Confirmed from t2t_executor.py internal source:

## Inside T2TExecutor.generate():
``` text
def generate(self, prompt: str) -> GenerationExecutionResult:

    # Calls _generate_non_native for device execution
    return self._generate_non_native(prompt)

def _generate_non_native(self, prompt):
    # Uses GenieT2TRunModule to run on device
    t2t_result = self._runner.run(
        GenieT2TRunExecutionConfig(prompt=prompt)
    )
    return t2t_result

Internally it runs this on the device:

```
# What gets executed on the Android device:
cd /data/local/tmp/<uuid>/
export LD_LIBRARY_PATH=$PWD/lib
export ADSP_LIBRARY_PATH=$PWD/lib
./bin/genie-t2t-run \
    -c genie_config.json \
    -p "What is Qualcomm Snapdragon?" \
    --profile profile.json

## What T2TExecutor Contains Internally

T2TExecutor object
├── _runner          ← GenieT2TRunModule (device runner)
├── _device          ← Device(type=ANDROID, serial_id=...)
├── _container       ← LLMContainer (has .bin paths)
├── _gen_ai_config   ← GenieConfig (genie_config.json)
└── _sdk_path        ← QAIRT SDK path (for pushing libs)



# OLD (deprecated but works):
executor: T2TExecutor = container.get_executor(device)
result = executor.generate(prompt)

# NEW (recommended — WorkflowBuilder):
from qairt.gen_ai_api.workflow_builder import WorkflowBuilder

executor = WorkflowBuilder.from_builders(
    {'genai': builder}          # ← your GenAIBuilderHTP
).build().get_executor(device)

result = executor.generate(prompt)

## Why new way?

WorkflowBuilder knows the builder context
→ better export decisions (HTP params, device type)
→ more extensible (supports multimodal in future)
→ export is executor's responsibility, not container's

## GenerationExecutionResult — What You Get Back


result: GenerationExecutionResult = executor.generate(prompt)

# What's inside:
result.generated_text        ← the model's output text ✅
result.metrics               ← dict of performance metrics

# Metrics keys:
{
  'init_time':               11438794,  # us — model load time
  'prompt_processing_time':  7363621,   # us — prefill time
  'time_to_first_token':     7363630,   # us — TTFT
  'token_generation_time':   173043480, # us — decode time
  'prompt_processing_rate':  1.9,       # toks/sec — prefill speed
  'token_generation_rate':   1.06,      # toks/sec — decode speed
  'adapter_switch_time':     ...,       # us — LoRA switch time
  'token_acceptance_rate':   ...,       # toks/inference — speculative
}

## Summary — T2TExecutor in One Picture

``` text
LLMContainer.load(serialized_output/)
        ↓
container.get_executor(device)
        ↓ [prepare_environment() auto-called]
        ↓ pushes .bin + libs + genie-t2t-run + config via ADB
        ↓
T2TExecutor (ready to run!)
        ↓
executor.generate("What is Snapdragon?")
        ↓ runs genie-t2t-run on device
        ↓ streams back tokens
        ↓
GenerationExecutionResult
  ├── generated_text 
  └── metrics (TTFT, TPS...) 
```

## Flow
### ACT 1 — container.get_executor(device) → "Setup Phase"
Think of this like setting up a remote workstation:
``` text
Your Linux machine has:
  ├── split_0/model.bin    ← compiled for SM8850
  ├── split_1/model.bin
  ├── split_2/model.bin
  ├── embedding_table.bin
  ├── genie_config.json
  └── tokenizer.json
```
get_executor() does THREE things automatically:

``` text
  1️ CREATES a workspace on device:
     /data/local/tmp/<uuid>/       ← fresh isolated workspace
                                     new UUID per session

  2️ SHIPS everything to device:
     ┌─────────────────────────────────────────────────┐
     │  Linux Host          ADB           Android       │
     │                                                  │
     │  model.bin    ──────push──────→  /data/tmp/bin/ │
     │  libQnnHtp.so ──────push──────→  /data/tmp/lib/ │
     │  libQnnHtpV81 ──────push──────→  /data/tmp/lib/ │
     │  libGenie.so  ──────push──────→  /data/tmp/lib/ │
     │  genie-t2t-run──────push──────→  /data/tmp/bin/ │
     │  genie_config ──────push──────→  /data/tmp/     │
     │  tokenizer    ──────push──────→  /data/tmp/     │
     └─────────────────────────────────────────────────┘

  3️ SETS environment on device:
     export LD_LIBRARY_PATH=/data/tmp/<uuid>/lib
     export ADSP_LIBRARY_PATH=/data/tmp/<uuid>/lib
     ← tells Android where to find Genie + QNN libs!

  Returns → T2TExecutor object 
  (a handle to the prepared device environment)
```
### ACT 2 — executor.generate(prompt) → "Inference Phase"

Now device is ready — inference happens in 5 steps:

``` text
Your prompt: "What is Qualcomm Snapdragon?"
        │
        ▼

STEP 1️ — Tokenization (on host or device)
  "What is Qualcomm Snapdragon?"
  → [128000, 3923, 374, 56835, 6054, 55845, 30]
     ↑ token IDs using tokenizer.json

STEP 2️ — Genie picks execution graph
  Uses use_cases.json to decide:
  7 tokens → prefill mode → use AR=128 graph
  (pad to 128 tokens with padding mask)
  or
  1 token  → decode mode  → use AR=1  graph

STEP 3️ — Genie orchestrates splits sequentially
  token_ids
      ↓
  embedding_table.bin   ← lookup embedding vector
      ↓  [1, 128, 3072] hidden state
  split_0/model.bin     ← layers 0-13 run on HTP
      ↓  [1, 128, 3072] hidden state updated
  split_1/model.bin     ← layers 14-27 run on HTP
      ↓  [1, 128, 3072] final hidden state
  split_2/model.bin     ← lm_head projection
      ↓  [1, 128, 128256] logits over vocabulary

STEP 4️ — Sampling next token
  logits [128256] → argmax or temperature sampling
  → next_token_id = 56835 ("Qualcomm")
  → append to KV cache
  → feed back as next AR=1 decode step

STEP 5️ — Loop until EOS or max_tokens
  [56835] → split_0 → split_1 → split_2 → next token
  [next]  → split_0 → split_1 → split_2 → next token
  ...
  [128001] = <eos> → STOP! 
        │
        ▼
  result.generated_text = "Qualcomm Snapdragon is a line of..."
  result.metrics = {TTFT, TPS, ...}

```
## Doubt Clarified
``` text
T2TExecutor
    │
    ├── NON-NATIVE path  (Android device via ADB)
    │        │
    │        └──→ GenieT2TRunModule
    │                    │
    │                    └──→ genie-t2t-run CLI on device
    │                              │
    │                              └──→ Genie C++ APIs 
    │
    └── NATIVE path  (x86 Linux host)
             │
             └──→ GenieNativeT2TRunner
                          │
                          └──→ libPyGenie pybind
                                    │
                                    └──→ Genie C++ APIs
```

Both paths wrap the SAME underlying Genie C++ APIs — just via different mechanisms! 
## Key Difference How they reach Genie

<img width="644" height="169" alt="image" src="https://github.com/user-attachments/assets/78a9da91-b190-4982-bb52-78a44eb961b4" />

## One More Layer — What Genie Does Inside Both
Both paths ultimately reach the same Genie Dialog API:
``` text
Genie C++ API (same for both paths!)
        │
        ├── genie::Dialog::create(config)   ← loads .bin files
        │                                      sets up KV cache
        │                                      initializes splits
        │
        ├── genie::Dialog::query(prompt)    ← runs inference
        │        │
        │        ├── tokenize prompt
        │        ├── prefill  → AR=128 graph
        │        ├── decode   → AR=1   graph (loop)
        │        └── return generated tokens
        │
        └── genie::Dialog::reset()          ← clears KV cache
```
### One-Line Summary
"T2TExecutor is a smart wrapper that picks the right path to Genie — if you're targeting an Android device it uses GenieT2TRunModule which shells out to genie-t2t-run via ADB, and if you're running locally on x86 it uses GenieNativeT2TRunner which calls Genie directly via pybind — both ultimately call the same Genie Dialog C++ APIs underneath!"



