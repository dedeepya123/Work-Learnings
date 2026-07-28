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








  


