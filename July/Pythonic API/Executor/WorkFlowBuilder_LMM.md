## WorkflowBuilder + LMM — How It Works!
Based on general QAIRT design knowledge — no specific internal docs found for this topic.

## How WorkflowBuilder Knows Which Executor — The KEY!
The DICT KEYS are the topology declaration!
```  text
WorkflowBuilder.from_builders({
    'genai': llm_builder      ← KEY 'genai' → T2TExecutor topology
})
```
``` text
WorkflowBuilder.from_builders({
    'vision': vision_builder, ← KEY 'vision' → VisionExecutor
    'genai':  llm_builder     ← KEY 'genai'  → T2TExecutor
})
```
WorkflowBuilder internally has an ExecutorFactory:


# Inside WorkflowBuilder:

TOPOLOGY_MAP = {
    frozenset(['genai'])          → T2TExecutorFactory,       # T2T LLM
    frozenset(['vision', 'genai'])→ LMMExecutorFactory,       # LMM
    frozenset(['audio', 'genai']) → AudioLLMExecutorFactory,  # Audio+LLM
}

# When you call .build():
keys = frozenset(builders.keys())
executor_factory = TOPOLOGY_MAP[keys]   ← dispatch based on keys!
→ Returns correct executor type! 
LMM — Full Code Pattern

from qairt.gen_ai_api.workflow_builder import WorkflowBuilder
from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilderHTP
from qairt.gen_ai_api.builders.vision_builder import VisionBuilder  # ← vision builder

#  Step 1 — Build LLM (your LLaMA 3.2 3B — already done!)
llm_builder: GenAIBuilderHTP = cast(
    GenAIBuilderHTP,
    GenAIBuilderFactory.create(
        pretrained_model_path="./output_artifacts",
        backend_type=BackendType.HTP,
        cache_root="./cache_llm",
    )
)
llm_builder.set_targets(["dsp_arch:v81;soc_model:87"])

#  Step 2 — Build Vision Encoder
vision_builder = VisionBuilder.from_pretrained(
    pretrained_model_path="./vision_model",   # ← CLIP or SigLIP encoder
    backend_type=BackendType.HTP,
    cache_root="./cache_vision",
)
vision_builder.set_targets(["dsp_arch:v81;soc_model:87"])

#  Step 3 — Combine via WorkflowBuilder
workflow_container = WorkflowBuilder.from_builders({
    'vision': vision_builder,   # ← KEY tells topology!
    'genai':  llm_builder       # ← KEY tells topology!
}).build()

#  Step 4 — Get LMM executor
lmm_executor = workflow_container.get_executor(device=device)

# Step 5 — Run multimodal inference
result = lmm_executor.generate(
    prompt="Describe this image in detail",
    image="./image.jpg"          # ← image input for vision encoder!
)
print(result.generated_text)
## How Genie Executes LMM Internally
``` text
lmm_executor.generate(prompt, image)
        │
        ├── STEP 1: Vision Encoder
        │   image.jpg
        │       ↓ VisionBuilder .bin
        │   image_embeddings [1, 256, 4096]  ← visual tokens
        │
        ├── STEP 2: Combine with Text
        │   text tokens:  [128000, 3923, 374, ...]
        │   image tokens: [visual_emb_0, visual_emb_1, ...]
        │       ↓ concatenated
        │   combined: [visual_tokens + text_tokens]
        │
        └── STEP 3: LLM Processing
            combined_tokens
                ↓ embedding_table.bin
                ↓ split_0/model.bin  (layers 0-13)
                ↓ split_1/model.bin  (layers 14-27)
                ↓ split_2/model.bin  (lm_head)
                ↓
            generated text: "The image shows..."
``` 

## T2T vs LMM — How Genie Picks the Right Executor
Dict Keys       	 Workflow Type	          Executor Used
{'genai': builder}	T2T (text only)	       T2TExecutor 
{'vision', 'genai'}	LMM (image+text)	     LMMExecutor 
{'audio', 'genai'}	Audio+LLM	             AudioLLMExecutor 

WorkflowBuilder acts as an executor dispatcher — the dict keys you pass to from_builders() declare the workflow topology, and the internal ExecutorFactory maps those keys to the correct executor type. {'genai': builder} → T2TExecutor, {'vision', 'genai'} → LMMExecutor. Genie then orchestrates the execution pipeline — vision encoder runs first, produces image embeddings, which are concatenated with text tokens and fed into the LLM for generation"
