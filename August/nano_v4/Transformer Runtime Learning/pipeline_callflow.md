# From manual classes to `LLMPipeline`: a call-flow walkthrough

This doc is for anyone who already understands the manual flow in
`nano/models/gemma4_text/` (load checkpoint -> reauthor -> quantize -> build
container -> run on device, called by hand from a script) and wants to
understand exactly what changes -- and what doesn't -- when that same flow is
driven through qairt's `LLMPipeline` API instead, via `nano_pipeline/`.

The short version, stated up front so the rest of this doc has a frame to hang
on: **the pipeline never invents new model logic.** Every real unit of work
(`load_checkpoint`, `build_qc_config`, `reauthor_model`, `assert_reauthored`,
`quantize_model`, `resize_for_arn`, `GenAIBuilderFactory.create`,
`shrink_checkpoint`) is the exact same function from `nano/models/gemma4_text/`
and `nano/experiments/uniform_kv_dim/`, called unchanged from inside our three
custom stage classes. The pipeline only adds scheduling, caching/resume, and a
uniform `generate()` entrypoint around that existing code.

All line numbers below are exact citations against the installed SDK at
`qairt_env_nano/lib/python3.10/site-packages/qairt/experimental/pipeline/`.

---

## 0. Before any pipeline code runs: the recipe

`nano_pipeline/run.py`'s `build_pipeline()` loads
`nano_pipeline/recipes/gemma4_text/gemma4_text.yaml` as a plain dict and
mutates it in Python *before* handing it to `LLMPipeline`:

```python
with open(RECIPE_PATH) as f:
    recipe = yaml.safe_load(f)
recipe["stages"]["gemma4_model_loader"]["use_shrink_checkpoint"] = use_shrink_checkpoint
if use_shrink_checkpoint:
    recipe["stages"]["gemma4_quantization"]["prepare_path"] = PREPARE_PATH + "_shrink"
    recipe["stages"]["gemma4_quantization"]["export_path"] = EXPORT_PATH + "_shrink"
    recipe["stages"]["gemma4_genai_builder"]["onnx_path"] = EXPORT_PATH + "_shrink_v2_encodings/model.onnx"
    recipe["stages"]["gemma4_genai_builder"]["encodings_path"] = EXPORT_PATH + "_shrink_v2_encodings/model.encodings"
    recipe["stages"]["gemma4_genai_builder"]["resized_dir"] = RESIZED_DIR + "_shrink"
return LLMPipeline.from_pretrained(MODEL_PATH, recipe=recipe)
```

`LLMPipeline.from_pretrained(recipe=...)` accepts a plain dict directly (no
temp YAML file needed) -- confirmed at `torch/common/bases/pipeline.py:177-182`:

```python
if not isinstance(recipe, (str, Path, dict)):
    raise TypeError(f"recipe must be a file path or a dict, got {type(recipe).__name__!r}")
if isinstance(recipe, (str, Path)):
    recipe = dict(Recipe.from_file(recipe))
```

So from this point on, `recipe` *is* the config -- everything downstream reads
from this dict, including our `use_shrink_checkpoint`/`prepare_path`/etc.
overrides. This step is why the two flows (real model vs. shrink-checkpoint
experiment) can share one recipe file and one set of stage classes instead of
needing duplicated scripts.

---

## 1. `LLMPipeline.from_pretrained(MODEL_PATH, recipe=recipe)`

This single call does two very different things, in two different classes:

1. The **base** `Pipeline.from_pretrained()` (`torch/common/bases/pipeline.py:150-202`)
   builds every stage's config and instantiates the pipeline object. It runs
   **zero** stages.
2. **`LLMPipeline`'s own override** (`torch/llm/pipeline.py:242-292`) then runs
   `super().from_pretrained()` and, on top of that, executes exactly the
   *first* stage in the recipe -- in our case, `gemma4_model_loader`.

Both parts happen inside this one call. Let's take them in order.

### 1a. Base class: build configs, instantiate, but run nothing

```python
# pipeline.py:150-202 (abbreviated to the essential steps)
if recipe is None:
    recipe = cls._get_recipe(model_id_or_path)
if isinstance(recipe, (str, Path)):
    recipe = dict(Recipe.from_file(recipe))

recipe["model_id_or_path"] = model_id_or_path        # caller's model_id_or_path wins
config = cls._config_from_recipe(recipe)              # <-- builds EVERY stage's Config object

for key, value in kwargs.items():                      # extra from_pretrained(**kwargs) overrides
    if hasattr(config, key):
        setattr(config, key, value)

pipeline = cls(config=config)                          # <-- triggers __init__ (see below)
Recipe(recipe).to_yaml(pipeline._pipeline_state_dir / "recipe.yaml")
return pipeline
```

`config = cls._config_from_recipe(recipe)` is where every stage's `Config`
Pydantic model gets constructed from the recipe dict's per-stage block --
this is the moment `Gemma4ModelLoadingConfig(use_shrink_checkpoint=True, ...)`,
`Gemma4QuantizationConfig(prepare_path="...shrink", ...)`, and
`Gemma4GenAIBuilderConfig(onnx_path="...shrink...", ...)` are all built, purely
by validating the dict keys we injected in step 0 against each `Config`
class's fields.

`pipeline = cls(config=config)` runs `Pipeline.__init__`
(`pipeline.py:114-137`), which in turn calls, **in this exact order**:

```python
self._initialize_stage_configs()   # resolves stage classes by name, calls _validate_stage_order() at the end
self._initialize_stages()          # instantiates our 3 stage OBJECTS (Gemma4ModelLoadingStage(), etc.)
self._validate_checkpoint()
self._initialize_cache()           # StageCache at workspace/pipeline_state, IF enable_cache
self._initialize_observers()
self._initialize_manifest()        # reads any prior PipelineManifest from disk
self._build_runner()               # composes [Caching ->] Observing -> Direct runner chain
```

**Nothing here calls any stage's `_execute()`.** This is purely wiring: figure
out which stage classes to use, validate their declared order, build a cache
handle, build a manifest reader, compose the runner chain that will later
execute stages. If you're used to the manual flow, this whole block is the
equivalent of "import the right functions and check your config makes sense" --
no checkpoint has been touched yet.

### 1b. `LLMPipeline`'s override: bootstrap exactly the first stage

Immediately after `super().from_pretrained()` returns, `LLMPipeline.from_pretrained`
(`llm/pipeline.py:242-292`) does this:

```python
pipeline = super().from_pretrained(model_id_or_path, recipe=recipe, **kwargs)
start_stage_name = list(pipeline._stage_classes.keys())[0]   # "gemma4_model_loader" -- first in recipe

manifest_stages = pipeline._manifest.read().get("stages", {})
if any(entry.get("artifact_path") for entry in manifest_stages.values()):
    # A prior run already has cached artifacts -- construct()'s own resume
    # logic will restore them, so don't waste time re-running here.
    pipeline._logger.info(f"Stage '{start_stage_name}': skipping bootstrap ...")
    return pipeline

stage = pipeline._stage_instances[start_stage_name]
stage_config = pipeline._stage_configs[start_stage_name]
stage_input = pipeline._construct_stage_input(start_stage_name)   # no prev output -> Input defaults only
result = pipeline._runner.run(stage, stage_input, stage_config, upstream_key=None)
pipeline._stage_outputs[start_stage_name] = result.output
pipeline._manifest.update(start_stage_name, result.artifact_path, stage_config, cache_key=result.cache_key)
return pipeline
```

**This is where our real code first runs.** `pipeline._runner.run(...)`
eventually calls `Gemma4ModelLoadingStage.run()` (see section 3 for exactly
what that does) -- which is where `shrink_checkpoint()` /
`load_checkpoint()` + `build_qc_config()`, `reauthor_model()`,
`assert_reauthored()`, `TextModelQc(...)` actually execute, exactly the calls
you'd make by hand in a script.

**Why only the first stage, and why here at all?** The contract of
`from_pretrained()` across the whole `transformers`-like ecosystem is "give me
a live, loaded model I can inspect *right now*" -- you shouldn't need to call
`construct()` just to see what got loaded. So `LLMPipeline` bootstraps stage 1
eagerly; every other stage waits for an explicit `construct()` call.

If a **prior run's manifest already has artifacts on disk** (the `load()`-resume
case), this bootstrap is skipped entirely -- `construct()`'s own resume logic
(section 2) will restore stage 1's output from the cache instead, so
re-running it here would be redundant.

At this point, `from_pretrained()` returns a pipeline object with exactly one
stage already executed and its output sitting in `pipeline._stage_outputs["gemma4_model_loader"]`.

---

## 2. `pipe.construct()` -- run every remaining stage

`construct()` (`torch/common/bases/pipeline.py:434-527`, unmodified by
`LLMPipeline` -- it's pure base-class logic) does three things: resume
whatever it can from a prior run, then loop over stages in recipe order,
running whichever ones aren't already satisfied.

### 2a. Resume: `_preload_manifest_stages()`

```python
stages_to_skip = self._preload_manifest_stages()
```

Walks the recipe's stages in order and, for each one already recorded in the
on-disk `PipelineManifest`, recomputes its cache key
(`StageCache.compute_key(stage_name, stage_config, prev_key)`, see section 2c)
and compares it to the manifest's stored key:

- **Match** -> restore that stage's output from disk
  (`type(stage).load_from_cache(artifact_path, ...)`), mark it to skip, advance
  `prev_key`, keep walking.
- **Mismatch** ("key changed since last run") -> log it and **stop restoring
  further stages** (the chain breaks here; everything from this stage onward
  will actually execute).

This is exactly what you saw in the logs when flipping `use_shrink_checkpoint`:
`gemma4_model_loader`'s key changed (different config), so the manifest
restore stopped there and quantization/builder re-ran for real.

### 2b. The main loop

```python
prev_key = None
for stage_name, stage in self._stage_instances.items():     # recipe declaration order
    stage_config = self._stage_configs[stage_name]

    if stage_name in stages_to_skip:
        if self._cache is not None:
            prev_key = StageCache.compute_key(stage_name, stage_config, prev_key)  # keep the chain valid
        continue                                              # <-- no execution

    stage_input = self._construct_stage_input(stage_name)      # field-matching, see 2d
    result = self._runner.run(stage, stage_input, stage_config, upstream_key=prev_key)
    self._stage_outputs[stage_name] = result.output
    prev_key = result.cache_key
    self._manifest.update(stage_name, result.artifact_path, stage_config, cache_key=result.cache_key)
```

Iterates `self._stage_instances` -- an `OrderedDict` built in recipe
declaration order -- so stages **always** run in the order they're written in
the YAML `stages:` block. For each non-skipped stage: build its `Input`, run
it through the runner chain (which is where the `_pre_hook -> _execute ->
_post_hook` sequence actually fires -- see section 3), record the output, and
persist the manifest entry so a future run can resume from here.

### 2c. Cache keys: how "did anything change" is decided

`StageCache.compute_key` (`cache.py:92-118`):

```python
digest = hashlib.sha256(name_bytes + b"|" + config_bytes + b"|" + upstream_bytes).hexdigest()
```

A **chain**: `key_2 = hash(name_2 | config_2 | key_1)`. This is exactly why
changing `use_shrink_checkpoint` on stage 1 invalidates the cache for stage 2
and stage 3 too, even if their own configs stayed byte-identical -- their key
depends on the *upstream* key, which changed. Combined with us also
parameterizing `prepare_path`/`export_path`/`resized_dir` per-flow (see
section 4), this double-guarantees the real-model and shrink-checkpoint runs
never share a cache directory or silently reuse each other's artifacts.

Cache lookup itself happens inside `_CachingRunner.run()` (`runners.py:165-183`):
compute the key, check `self._cache.get(key, ...)` -- hit means skip
`_execute` entirely and return the cached output; miss means run the stage for
real and then `self._cache.put(key, output, ...)`.

### 2d. `_construct_stage_input()`: how fields cross stage boundaries

This is the mechanism that lets `qc_config` produced by stage 1 show up as an
input to stage 2 and stage 3 *without us writing any explicit wiring code*.
(`pipeline.py:1094-1148`, no-`io_bindings` branch, since our recipe doesn't
declare any):

```python
stage_names = list(self._stage_instances.keys())
stage_idx = stage_names.index(stage_name)
prev_output = None
for i in range(stage_idx - 1, -1, -1):          # walk BACKWARD in recipe order
    if stage_names[i] in self._stage_outputs:
        prev_output = self._stage_outputs[stage_names[i]]
        break

input_kwargs = {}
if prev_output is not None:
    for field_name in stage_class.Input.model_fields:      # every declared field on THIS stage's Input
        if hasattr(prev_output, field_name):                # does the PREVIOUS stage's output have it?
            value = getattr(prev_output, field_name)
            if value is not None:
                input_kwargs[field_name] = value

return stage_class.Input(**input_kwargs)
```

Plain field-name matching: for every field our stage's `Input` model declares,
check whether the previous stage's `Output` object has an attribute of that
exact name, and if so, copy it over. This is *why* we had to add a
`config: Optional[Any] = None` field to `Gemma4GenAIBuilderInput` --
`GenAIBuilderInput` doesn't declare a `config` field by default, so without
that addition, `qc_config` from `Gemma4QuantizationOutput.config` would never
have been picked up (`hasattr` would find nothing to copy into, since
`config` wasn't in `stage_class.Input.model_fields` at all).

Note: it walks backward looking for the *nearest* stage with an available
output, not literally "the stage listed right before this one" -- this
matters for resume scenarios where a downstream stage might already be
restored while an upstream one hasn't executed yet.

---

## 3. Inside a single stage: `Stage.run()` and the three hooks

Every `_runner.run(stage, ...)` call in sections 1b and 2b eventually reaches
`Stage.run()` (`torch/common/bases/stage.py:156-181`, decorated `@final` --
**no subclass may override this method itself**):

```python
@final
def run(self, input, config, *, artifact_path=None):
    self._validate_stage_types(input, config)                        # type-checks input/config
    prepared_input, prepared_config = self._pre_hook(input=input, config=config)
    raw_output = self._execute(input=prepared_input, config=prepared_config, artifact_path=artifact_path)
    final_output = self._post_hook(input=prepared_input, config=prepared_config, output=raw_output)
    return final_output
```

Fixed, non-negotiable sequence: **`_pre_hook` -> `_execute` -> `_post_hook`**.
This is the *only* customization surface a stage subclass has. The base class
gives each hook a real default:

| Hook | Base default (`stage.py`) | Meaning if you don't override |
|---|---|---|
| `_pre_hook(input, config)` | `return input, config` | identity -- no preprocessing |
| `_execute(input, config, *, artifact_path)` | `@abstractmethod` -- no default | **must** be overridden; this is the actual work |
| `_post_hook(input, config, output)` | `return output` | identity -- no postprocessing |

So overriding `_pre_hook`/`_post_hook` is optional (skip it, get a no-op);
overriding `_execute` is mandatory. `run()` itself has no try/except -- error
logging/re-raising happens one layer up, in `_DirectRunner.run()`
(`runners.py:60-74`), which wraps `stage.run(...)` in try/except and logs
`"Stage '{name}': successfully completed"` or `"Stage '{name}' failed: {e}"`.

### What each of our three stages actually does at each hook

**`Gemma4ModelLoadingStage`** (`nano_pipeline/stages/model_loading_stage.py`):

| Hook | What we do | Why |
|---|---|---|
| `_pre_hook` | `return input, config` (explicit no-op) | Base `ModelLoadingStage._pre_hook` builds an HF `AutoConfig`-style config assuming a plain single-model checkpoint -- wrong for our multi-modal one. We build `qc_config` ourselves in `_execute` instead. |
| `_execute` | `shrink_checkpoint(model_path, shrink_head_dim)` **or** `load_checkpoint(model_path)` + `build_qc_config(...)`, then `reauthor_model(...)`, `assert_reauthored(...)`, wrap in `TextModelQc(...)` | This is the entire manual "load + reauthor" flow, verbatim, just gated behind `config.use_shrink_checkpoint`. |
| `_post_hook` | `return output` (explicit no-op) | Base `ModelLoadingStage._post_hook` reads `config.model_reauthoring`, a field our minimal config doesn't declare -- would `AttributeError` if not overridden. |

**`Gemma4QuantizationStage`** (`nano_pipeline/stages/quantization_stage.py`):

| Hook | What we do | Why |
|---|---|---|
| `_pre_hook` | Keep the base class's `input.model is None` / `input.tokenizer is None` checks; drop its `RecipeRegistry.get(config.recipe_name)` lookup | We bypass the YAML-driven recipe/calibration system entirely -- our calibration dataloader is a live Python object that can't travel through a YAML `config_overrides` dict. |
| `_execute` | `quantize_model(input.model.qc_model, qc_config, input.tokenizer, prepare_path=config.prepare_path, export_path=config.export_path)` | Same function as the manual flow's `nano/models/gemma4_text/quantize.py`, just parameterized so the shrink-checkpoint flow can point at isolated `_shrink`-suffixed paths. |
| `_post_hook` | not overridden -> inherited identity | Nothing extra needed. |

**`Gemma4GenAIBuilderStage`** (`nano_pipeline/stages/gen_ai_builder/gemma4_text_builder_stage.py`):

| Hook | What we do | Why |
|---|---|---|
| `_pre_hook` | Does **not** call `super()._pre_hook()` at all. Instead: `resize_for_arn(...)` (our custom `per_layer_inputs` seed rule) -> `GenAIBuilderFactory.create(config_dict=build_gen_ai_config_dict(qc_config))` -> `skip_ar_cl_conversion=True` -> `set_targets([CHIPSET])` -> `set_transformation_options(...)` | The builder's own AR/CL resize pass has no way to accept a custom seed rule, and Gemma's `per_layer_inputs` tensor isn't covered by its built-in axis-denotation patterns. We resize ourselves first and tell the builder to skip its own pass. |
| `_execute` | Inherited unchanged from base `GenAIBuilderStage` -- just `builder.build()` on the `preconfigured_builder` we stashed on `input` in `_pre_hook` | Nothing model-specific left to do once the builder object itself is correctly configured. |
| `_post_hook` | not overridden -> inherited identity | Nothing extra needed. |

---

## 4. Config classes: where the recipe's extra keys land

Each stage declares a `Config` subclass of `StageConfig`
(`stage.py:90-105`), which itself carries base fields every stage inherits:
`execution_environment`, `generator_config`, `evaluator_config`,
`exporter_config`, and a private `_pipeline_context` (injected by
`_initialize_stage_configs()`, giving every stage access to
`config._pipeline_context.model_id_or_path`). All three base classes
(`StageInput`, `StageConfig`, `StageOutput`) share
`model_config = {"extra": "allow", "protected_namespaces": (), "arbitrary_types_allowed": True}`
-- `extra="allow"` is what lets our recipe-dict keys (`use_shrink_checkpoint`,
`prepare_path`, `onnx_path`, ...) simply become attributes on the parsed
`Config` object, and it's also what makes `hasattr(prev_output, field_name)`
in `_construct_stage_input` work correctly against extra fields, not just
explicitly declared ones.

Our three `Config` subclasses add exactly the fields we need, each with a
default matching the real-model flow, so leaving `use_shrink_checkpoint`
unset in the recipe reproduces the original manual behavior unchanged:

```python
class Gemma4ModelLoadingConfig(StageConfig):
    use_shrink_checkpoint: bool = False
    shrink_head_dim: int = TARGET_HEAD_DIM

class Gemma4QuantizationConfig(StageConfig):
    prepare_path: str = PREPARE_PATH
    export_path: str = EXPORT_PATH

class Gemma4GenAIBuilderConfig(GenAIBuilderConfig):
    arn: int = ARN
    context_length: int = CONTEXT_LENGTH
    onnx_path: str = ONNX_MODEL_PATH
    encodings_path: str = ENCODINGS_PATH
    resized_dir: str = RESIZED_DIR
```

---

## 5. `pipe.generate(prompt, device=...)` -- the very last call

`LLMPipeline.generate()` (`llm/pipeline.py:294-347`):

```python
last_stage_name = list(self._stage_instances.keys())[-1]     # LAST stage declared in the recipe
last_stage = self._stage_instances[last_stage_name]
last_output = self._stage_outputs.get(last_stage_name)        # must exist -- i.e. construct() ran it
if last_output is None:
    raise RuntimeError(f"No output found for last stage '{last_stage_name}'. ...")
if not getattr(last_stage, "can_generate", False):
    raise NotImplementedError(f"Last stage '{last_stage_name}' does not support generation ...")

last_stage_config = self._stage_configs.get(last_stage_name)
try:
    return last_stage.generate(last_output, last_stage_config, prompt, device=device, **kwargs)
except Exception as e:
    raise RuntimeError(f"Generation failed in stage '{last_stage_name}': {e}") from e
```

**Important, easy-to-miss detail:** "last stage" is determined purely by
position -- the last key in the recipe's `stages:` block -- **not** by
searching for whichever stage has `can_generate=True`. In our recipe,
`gemma4_genai_builder` happens to be both the last stage *and* the only
generation-capable one, so this works cleanly. If a stage were ever appended
after it without `can_generate=True`, `generate()` would raise
`NotImplementedError` rather than searching backward for a usable stage --
something to watch for if this recipe ever grows a fourth stage.

`last_stage.generate(...)` is `GenAIBuilderStage.generate()` (built-in, not
overridden by us) -- it resolves the `LLMContainer` we built in stage 3's
`_execute`, gets a `T2TExecutor` bound to `device`, pushes the container over
adb, and runs `genie-t2t-run`. This is the exact same code path
`runtime/executor.py` used manually -- the pipeline just gives it a uniform
entrypoint.

---

## 6. Side-by-side: manual script vs. pipeline

| | Manual (`nano/main.py` / `experiments/uniform_kv_dim/run.py`) | Pipeline (`nano_pipeline/run.py`) |
|---|---|---|
| Entry point | Linear script, top-to-bottom function calls | `LLMPipeline.from_pretrained(recipe=...)` + `pipe.construct()` + `pipe.generate(...)` |
| Stage boundaries | Implicit -- just sequential calls | Explicit `Stage` subclasses with declared `Input`/`Config`/`Output` |
| Where the real work happens | Top level of the script | Inside each stage's `_execute` -- same function calls, just relocated |
| Resume / caching | None -- rerun from scratch every time, or comment out lines by hand | Automatic -- `StageCache`/`PipelineManifest`, keyed on `sha256(name \| config \| upstream_key)` |
| Switching real vs. shrink flow | Two separate scripts with duplicated logic and separate hardcoded path constants | One recipe + one `use_shrink_checkpoint` config flag; same stage classes, parameterized paths |
| Field wiring between steps | Manual -- `model, qc_config, tokenizer = shrink_checkpoint(...)`, pass `qc_config` to the next call yourself | Automatic -- `_construct_stage_input`'s `hasattr`-based field-name matching |
| Device generation | Hand-rolled per script (`run_on_device()`) | Uniform `pipe.generate(prompt, device=...)` |

Nothing about *what* runs changed -- same weights, same quantization, same
on-device binary format. What changed is *how it's invoked and tracked*:
declarative stages with automatic caching/resume and one `generate()` call,
instead of a script you rerun from scratch every time.

---

## 7. A gotcha we hit for real: path collisions across model variants

`prepare_llm()` (the model-preparation step inside `quantize_model()`) decides
whether to skip re-preparing a model with a **pure path-existence check** --
no hash or fingerprint of the actual model/config, just "do
`converted_model.py`/`converted_model.safetensors` already exist at this
path?" The real-model flow and the shrink-checkpoint flow originally shared
the same hardcoded `PREPARE_PATH`/`EXPORT_PATH`/`RESIZED_DIR` constants in
`nano/models/gemma4_text/quantize.py` and
`nano/models/gemma4_text/builder/container.py`. Running one flow after the
other would have **silently reused the wrong model's prepared graph** --
wrong results, not a crash.

Fix: `quantize_model()` and `resize_for_arn()` both now accept
`prepare_path`/`export_path`/`onnx_path`/`encodings_path`/`resized_dir` as
parameters (defaulting to the original real-model constants), and
`nano_pipeline/run.py`'s `build_pipeline()` overrides all of them to
`_shrink`-suffixed siblings whenever `use_shrink_checkpoint=True`. Combined
with the cache-key chain in section 2c, the two flows can no longer step on
each other's artifacts, on disk or in the pipeline's own cache.

This is a good general lesson for extending this pipeline: **any hardcoded
path constant reused across model variants is a latent correctness bug**,
not just a tidiness issue -- the pipeline's caching makes the failure mode
worse (silent stale reuse) rather than better, precisely because it's *good*
at skipping work it thinks is already done.
