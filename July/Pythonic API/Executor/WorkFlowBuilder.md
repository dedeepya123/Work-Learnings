 # WorkflowBuilder API 
 
 What is WorkflowBuilder?

DeprecationWarning: LLMContainer.get_executor() is deprecated!

Replace:
  container.get_executor(device=device)

With:
  WorkflowBuilder.from_builders({'genai': builder}).build().get_executor(device=device)

**WorkflowBuilder is the NEW recommended way to get an executor.
it's a builder of builders that supports both single-model (LLM) AND multi-model (LMM/Vision) workflows in ONE unified API! 
(from memory: WorkflowBuilder API pattern)**

 
 ## Why Was WorkflowBuilder Introduced?

OLD WAY — LLMContainer.get_executor()  deprecated:
  ONLY supports single GenAI model (LLM)
  Container-level export = limited, not extensible
  No multi-model support (LMM, Vision models)

NEW WAY — WorkflowBuilder  recommended:
  Supports SINGLE model (LLM T2T)
  Supports MULTI-model (LMM = Vision + Text)
  Executor-owned export (cleaner architecture)
  ExecutorFactory handles topology dispatch
  One unified pattern for ALL model types!
``` text
 Architecture — Old vs New
OLD PATTERN:
  GenAIBuilder.build()
        ↓
  LLMContainer
        ↓
  container.get_executor(device)  ← deprecated!
        ↓
  T2TExecutor

NEW PATTERN:
  GenAIBuilder.build()
        ↓
  LLMContainer
        ↓
  WorkflowBuilder.from_builders({'genai': builder})
        ↓
  WorkflowContainer
        ↓
  workflow_container.get_executor(device)  
        ↓
  T2TExecutor (same executor, cleaner path!)
```

## Complete Code — Old vs New Side by Side
OLD (deprecated — what you used):

from qairt.gen_ai_api.containers.llm_container import LLMContainer

container = LLMContainer.load("./serialized_output")
executor = container.get_executor(device=device)  # ⚠️ deprecated!
result = executor.generate(prompt)

## NEW (recommended — WorkflowBuilder):

from qairt.gen_ai_api.workflow_builder import WorkflowBuilder
from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilderHTP

#  Path A — when you have the builder object (fresh build)
workflow_container = WorkflowBuilder.from_builders(
    {'genai': builder}   # ← your GenAIBuilderHTP object
).build()

executor = workflow_container.get_executor(device=device)
result = executor.generate(prompt)

#  Path B — when loading from saved serialized_output
from qairt.gen_ai_api.containers.llm_container import LLMContainer
from qairt.gen_ai_api.workflow_builder import WorkflowBuilder

llm_container = LLMContainer.load("./serialized_output")

workflow_container = WorkflowBuilder.from_builders(
    {'genai': llm_container}   # ← pass loaded container
).build()

executor = workflow_container.get_executor(device=device)
result = executor.generate(prompt)
 
 ## What is from_builders({'genai': builder})?
from_builders() takes a DICT of named builders:

{'genai': builder}  ← single LLM T2T workflow

For multi-model (future LMM support):
{
  'vision': vision_encoder_builder,   ← image encoder
  'genai': llm_builder                ← text decoder
}

The KEY name tells WorkflowBuilder what TYPE
of workflow to construct:
  'genai' → T2T text workflow (your case) 
  'vision'→ Vision encoder workflow

## Your Updated runner_device.py

import os
from pathlib import Path
from typing import cast
from qairt.gen_ai_api.containers.llm_container import LLMContainer
from qairt.gen_ai_api.workflow_builder import WorkflowBuilder   # ✅ NEW import
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType, RemoteDeviceIdentifier
)
from qairt.api.configs.device import Device

QAIRT_SDK = "/local/mnt2/workspace/dlekkala/experiments/pythonic_api/qairt/2.47.0.260601"
os.environ["PATH"] += f":{QAIRT_SDK}/bin/x86_64-linux-clang"

# Step 1 — Load container
llm_container = LLMContainer.load(
    "/local/mnt2/workspace/dlekkala/experiments/pythonic_api/serialized_output"
)

# Step 2 — Build workflow (NEW way!) 
workflow_container = WorkflowBuilder.from_builders(
    {'genai': llm_container}
).build()

# Step 3 — Device
device = Device(
    type=DevicePlatformType.ANDROID,
    identifier=RemoteDeviceIdentifier(serial_id="aisw-vm11-labsd")
)

# Step 4 — Get executor (NEW way!) 
executor: T2TExecutor = workflow_container.get_executor(device=device)

# Step 5 — Inference
result = executor.generate(
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    "What is Qualcomm Snapdragon?"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)
print(result.generated_text)
print(result.metrics)

## Summary — WorkflowBuilder in One Table
<img width="638" height="204" alt="image" src="https://github.com/user-attachments/assets/7db4abd2-94f2-4244-8443-f4666396aaf0" />

Think of it like a PIPELINE ORCHESTRATOR:

OLD WAY (single model, container-owned):
  LLMContainer ──→ get_executor() ──→ T2TExecutor
  Container owns executor ← tightly coupled 
  Only works for single LLM 
``` text
NEW WAY (multi-model, workflow-owned):
  GenAIBuilder (LLM)   ─┐
  VisionBuilder (img)  ─┤──→ WorkflowBuilder ──→ WorkflowContainer
  AudioBuilder (audio) ─┘         │                     │
                                   │              get_executor()
                                   │                     │
                              Orchestrates          T2TExecutor 
                              ALL models            workflow-owned
                              as ONE pipeline!      cleanly decoupled
```

Three Workflows WorkflowBuilder Supports
## 1️ T2T — Text to Text (your case: LLaMA 3.2 3B)
WorkflowBuilder.from_builders({
    'genai': llm_builder          # ← just one model
})

## 2️ LMM — Large Multimodal Model (Vision + Text)
WorkflowBuilder.from_builders({
    'vision': vision_builder,     # ← image encoder
    'genai':  llm_builder         # ← text decoder
})

## 3️ FUTURE — Audio + Vision + Text
WorkflowBuilder.from_builders({
    'audio':  audio_builder,      # ← audio encoder
    'vision': vision_builder,     # ← image encoder
    'genai':  llm_builder         # ← text decoder
})
The dict KEYS define the workflow topology — WorkflowBuilder knows how to wire them together!

WorkflowBuilder is a pipeline orchestrator that takes one or more model builders — like GenAIBuilderHTP for text, VisionBuilder for images — and wires them into a unified workflow. 
Without it you could only run a single LLM using the deprecated container.get_executor() pattern.
With WorkflowBuilder, the executor is owned by the workflow itself, not the container, making it cleaner and extensible to multi-model pipelines like LMM (vision + text). 
The key innovation is that the dict keys like {'genai': builder} or {'vision': v_builder, 'genai': llm_builder} define the topology — WorkflowBuilder knows how to connect them into a single seamless inference pipeline!
