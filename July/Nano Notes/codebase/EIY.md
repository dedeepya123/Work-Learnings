# Open Questions

## Module (ArgParser)

- **purpose** - To register args object with (default ) or client specified args.
- **observation** - The args object shoudl be updated with all attribute details
- **Question** - Can we use Builder Pattern here? --> like constructing a parser object from client foields and also default once.


## Module (qmodel)

function - load_gemma4 - Is explicit , we are calling laod_gemma4 , what if tomorrow anotehr model comes --> Instead can we use LSP princple coz it violates OCP ?

Adding common interface and based on the model Config loads taht particular model?
``` text
def load_gemma4(config: Model_Config, adaptations: AdaptationFlags = AdaptationFlags(), logger: logging.Logger = None, qc: bool = True) -> 'Gemma4Context':
    """Construct a Gemma4Context: load + adapt the HF model and extract its tools.

    Replaces get_model_builder. When qc=True (default), eagerly runs the QC
    adaptation; when qc=False, loads the stock HF model instead (parity tests).
    """
    ctx = Gemma4Context(config, adaptations, logger or logging.getLogger(__name__))
    ctx.create_qc_model() if qc else ctx.create_gg_model()
    return ctx
```

Craeting object within parametrs is it good practice ?

In qmodel.py load_gemma_4 has Gemma4Context 

This Gemma4Context --> is initializing processor attribute 
``` text
@functools.lru_cache(maxsize=None)
def get_processor(model_id: str, vision_soft_tokens, use_fast=False, add_bos_token=False):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=use_fast,
        add_bos_token=add_bos_token,
        image_seq_length=vision_soft_tokens,
    )
    processor.image_processor.max_soft_tokens = vision_soft_tokens
    return processor
**Question** : What it means ? functools lru_cache?
Also Layer names how ill it be [ varies or modtly all llms will have similar names ] ? How to rember the prefix of safetensors [sps to extract particular layer tensor]
```

Understand what with event_maker() does?
``` text
 with event_marker("FP model adaptation & creation"), _patched_gemma4_classes(self.adaptations):
            model = modeling_gemma4.Gemma4ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_id,
                config=self.lmm_config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation='eager',
                ignore_mismatched_sizes=self.ignore_mismatched_sizes,
            ).eval()

event_maker : In common , profiler.py

def event_marker(event: str, device: Union[Device, int] = None, flush_ram: bool = False):
    """
    utility to mark time taken and memory usage before and after executing a section of code.
    :param event: marker string to use to identify the context.
    :param device: (torch.device or int, optional): selected device.
    :param flush_ram: invoke garbage collect for true estimates before profiling.
    """
    profiler = EventProfiler()
    # reset for start low-watermark
    if flush_ram:
        gc.collect()
        event = f'{event}[gc]'

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    profiler.reset_peak_memory_stats()
    start_marker = profiler.snapshot(f'{event} >> ', device, append=False)
    yield
    end_marker = profiler.snapshot(f'{event} << ', device, append=False)
    profile_marker = end_marker.delta(event, start_marker)
    logger.info('%s', profile_marker)
    profiler._markers.append(profile_marker)  # pylint: disable=protected-access

```
- Understand linear to Conv inside math
- 








  


