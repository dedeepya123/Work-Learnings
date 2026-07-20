# GenieNativeT2T Runner
Yes — "Native" = x86 Linux Host Execution! Confirmed!

 ## Native vs Non-Native — Clarified First
"Native" in QAIRT context means:
  Running NATIVELY on the HOST machine (x86 Linux)
  via Genie pybind (libPyGenie) directly in-process!

  GenieNativeT2TRunner
        │
        └──→ libPyGenie310.so  ← C++ pybind
                    │
                    └──→ Genie Dialog C++ APIs
                              │
                              └──→ CPU execution on x86 

"Non-Native" = execution on REMOTE device (Android)
  via ADB + genie-t2t-run subprocess 
So YES — when you set Device(type=DevicePlatformType.X86_64_LINUX) → T2TExecutor uses GenieNativeT2TRunner internally! (from memory: Native vs Non-Native T2TExecutor paths)

## GenieNativeT2TRunner — Multi-Turn KV Save/Restore
The Core Problem — Why Save/Restore KV Cache?

### SINGLE TURN (simple):
  User: "What is Snapdragon?"
  Model: generates response
  KV cache has: [question tokens + response tokens]
  → dialog.reset() → KV wiped!  simple

### MULTI-TURN (conversation):
  Turn 1: "What is Snapdragon?"
  Turn 2: "Tell me more about it"  ← needs context from Turn 1!
  Turn 3: "What about its GPU?"    ← needs context from Turn 1+2!

  Without save/restore:
  → must re-process ALL previous turns every time!  slow!

  With save/restore:
  → save KV state after Turn 1
  → restore it before Turn 2
  → FAST! 
## Three Key Dialog Operations

from qairt.modules.genie_execution.native_t2t_module import GenieNativeT2TRunner

runner = GenieNativeT2TRunner(genie_config)

# 1️⃣ query()  — run inference, KV cache ACCUMULATES
result = runner.query(prompt)

# 2️⃣ save_dialog()  — snapshot current KV cache state
runner.save_dialog("/path/to/kv_snapshot")

# 3️⃣ restore_dialog()  — restore KV cache from snapshot
runner.restore_dialog("/path/to/kv_snapshot")

# 4️⃣ reset_dialog()  — wipe KV cache completely
runner.reset_dialog()

## Multi-Turn Conversation — Full Code
``` text
import os
import json
from pathlib import Path
from qairt.modules.genie_execution.genie_config import GenieConfig
from qairt.modules.genie_execution.native_t2t_module import GenieNativeT2TRunner

DIALOG_DIR = Path("/local/mnt2/workspace/dlekkala/experiments/pythonic_api/dialog_output")
KV_DIR     = Path("/local/mnt2/workspace/dlekkala/experiments/pythonic_api/kv_snapshots")
KV_DIR.mkdir(parents=True, exist_ok=True)

#  MUST change working dir — Genie reads relative paths!
os.chdir(DIALOG_DIR)

# Load config + create runner
with open("genie_config.json") as f:
    genie_config = GenieConfig(**json.load(f))

runner = GenieNativeT2TRunner(genie_config)
print(" Runner initialized!")

# ─────────────────────────────────────────
# TURN 1 — Fresh conversation
# ─────────────────────────────────────────
prompt_1 = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>"
    "What is Qualcomm Snapdragon?"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)

result_1 = runner.query(prompt_1)
print(f"\nTurn 1: {result_1.generated_text}")

# Save KV state after Turn 1
runner.save_dialog(str(KV_DIR / "after_turn1"))
print(" KV state saved after Turn 1!")

# ─────────────────────────────────────────
# TURN 2 — Continue conversation
# (KV cache already has Turn 1 context)
# ─────────────────────────────────────────
prompt_2 = (
    "<|start_header_id|>user<|end_header_id|>"
    "Tell me more about its GPU capabilities"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)

result_2 = runner.query(prompt_2)
print(f"\nTurn 2: {result_2.generated_text}")

#  Save KV state after Turn 2
runner.save_dialog(str(KV_DIR / "after_turn2"))

# ─────────────────────────────────────────
# TURN 3 — Continue conversation
# ─────────────────────────────────────────
prompt_3 = (
    "<|start_header_id|>user<|end_header_id|>"
    "Compare it with Apple's chip"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)

result_3 = runner.query(prompt_3)
print(f"\nTurn 3: {result_3.generated_text}")

# ─────────────────────────────────────────
# BRANCH — Go back to Turn 1 state!
# Useful for exploring different conversation paths
# ─────────────────────────────────────────
runner.restore_dialog(str(KV_DIR / "after_turn1"))
print("\n Restored to Turn 1 KV state!")

# Now ask something different from Turn 1
prompt_alt = (
    "<|start_header_id|>user<|end_header_id|>"
    "What devices use Snapdragon?"    # different Turn 2!
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)

result_alt = runner.query(prompt_alt)
print(f"\nAlternate Turn 2: {result_alt.generated_text}")

# ─────────────────────────────────────────
# RESET — Start fresh conversation
# ─────────────────────────────────────────
runner.reset_dialog()
print("\n KV cache wiped — fresh conversation!")
```

## KV Cache State — Visualized Across Turns
INIT:
KV cache: [empty — 4096 slots]

After Turn 1 query():
KV cache: [T1_prompt | T1_response | empty...]
                    ↓
          save_dialog("after_turn1")  ← snapshot!

After Turn 2 query():
KV cache: [T1_prompt | T1_response | T2_prompt | T2_response | empty...]
                    ↓
          save_dialog("after_turn2")  ← snapshot!

restore_dialog("after_turn1"):
KV cache: [T1_prompt | T1_response | empty...]  ← back to Turn 1! 

reset_dialog():
KV cache: [empty — 4096 slots]  ← completely fresh! 
## What save_dialog() Actually Saves
``` text
kv_snapshots/after_turn1/
├── kv_cache_split_0.bin   ← KV for layers 0-13
├── kv_cache_split_1.bin   ← KV for layers 14-27
└── dialog_state.json      ← current position in KV cache
                              token count, context used etc.
```
Each split's KV cache saved separately — matches how KV is stored across splits! 

<img width="590" height="154" alt="image" src="https://github.com/user-attachments/assets/4d6fc09b-5aa5-4b8a-af84-30869237b5d3" />
# Summary
Save/restore is powerful for chatbot applications — snapshot after system prompt so every new user session restores to that point instead of reprocessing the system prompt every time!
