## Loading Text-Only Model from a Gemma4 Multimodal Checkpoint
### Load Full Config

```python
lmm_config = Gemma4Config.from_pretrained(model_path)

lmm_config is a Gemma4Config object containing both text and vision configurations.

```text
Gemma4Config
├── text_config
├── vision_config
└── multimodal settings
```

Extract Text Configuration
```python
text_config = lmm_config.text_config
text_config is of type Gemma4TextConfig.
```
```python
text_model = Gemma4ForCausalLM.from_pretrained(
    model_path,
    config=text_config,
    device_map="auto"
)
This loads only the LLM (text tower) from the multimodal checkpoint.
```
```python
tokenizer = AutoTokenizer.from_pretrained(model_path)

The tokenizer remains the same because it belongs to the language model vocabulary and is shared by the multimodal checkpoint.
```
```text
Gemma4Config
├── text_config (Gemma4TextConfig)
│   └── Gemma4ForCausalLM
│
└── vision_config (Gemma4VisionConfig)

Tokenizer ──► Text Model
Image Processor ──► Vision Model

```

```python
text_model = Gemma4ForCausalLM.from_pretrained(...)
Here:
- Gemma4ForCausalLM → Model class
- from_pretrained() → Class method/API used to load weights
- text_model → Model instance/object
```
- Gemma4ForCausalLM is a Hugging Face model class for text generation (causal language modeling).
- from_pretrained() loads the model architecture and pretrained weights from a checkpoint.

### With Config
```python
text_model = Gemma4ForCausalLM.from_pretrained(
            model_path,
            config=text_config
)
```

#### What happens?
- Loads weights from checkpoint.
- Uses the configuration object you explicitly provide.
- Builds the model according to that config.
```python
config=text_config
means:Use only the Gemma4TextConfig portion of the multimodal configuration.
```

### Without config
```python
text_model = Gemma4ForCausalLM.from_pretrained(model_path)
```

#### What happens?
```python
Hugging Face automatically does:
config = AutoConfig.from_pretrained(model_path)
internally and uses the config stored in the checkpoint.
```
### Why pass config explicitly?

Because  checkpoint is multimodal:
``` text
Gemma4Config
├── text_config
├── vision_config
├── audio_config
└── craft_config
```
and wanted:
```text
Gemma4TextConfig
        ↓
Gemma4ForCausalLM
```
so we are explicitly saying  Ignore the other modality configs and use the text configuration.

```text
Full Multimodal Config
            |
            v
      text_config
            |
            v
   Gemma4ForCausalLM
```

### Text Inference Workflow

1. Convert prompt to tokens using tokenizer.
2. Pass tokenized inputs to the model.
3. Generate output token IDs using `model.generate()`.
4. Decode output IDs back to text.

```python
prompt = "The capital of France is"
1. Tokenize:
inputs = tokenizer(prompt, return_tensors="pt")
2. Move to model device
inputs = {k: v.to(text_model.device) for k, v in inputs.items()}
3. Generate:
outputs = text_model.generate(
    **inputs,
    max_new_tokens=50
)
4. Decode:
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```python
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

**For any Text Model Flow is same**
Prompt
  ↓
Tokenizer
  ↓
input_ids
  ↓
Text Model (CausalLM)
  ↓
generate()
  ↓
output_ids
  ↓
decode()
  ↓
Generated Text
```
```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = XxxForCausalLM.from_pretrained(model_path)

inputs = tokenizer("Hello", return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=20
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
### What changes between models?

Usually only:
1. Gemma4ForCausalLM
2. LlamaForCausalLM
3. Qwen3ForCausalLM
4. MistralForCausalLM
and their corresponding tokenizer. The inference steps remain the same.

### What changes for multimodal models?
- For multimodal models, there is an extra preprocessing step:
```text
Text  ---> Tokenizer ---\
                         ---> Model
Image ---> Processor ---/

or

Text ---> Tokenizer ----\
Audio --> Processor -----+--> Model
Image --> Processor ----/
```
- Any Hugging Face *ForCausalLM model follows essentially the same inference workflow: Tokenize → Generate → Decode.
- The architecture and weights differ, but the inference API remains largely identical.

### Tokenization
```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)
```
Prompt: "What is AI?"
- Tokenizer converts text into tokens and token IDs.

Example:
"What"  -> 123
"is"    -> 456
"AI"    -> 789
"?"     -> 102
```python
print(inputs)
output:
{
    "input_ids": tensor([[123, 456, 789, 102]]),
    "attention_mask": tensor([[1, 1, 1, 1]])
}
```
### input_ids

- Contains token IDs representing the text.
- example : input_ids = [123, 456, 789, 102]

### attention_mask
- Indicates which tokens are valid.
- Example: attention_mask = [1, 1, 1, 1]
- 1 = Real token
0 = Padding token
If prompt length is 10 and no padding exists: then attention_mask is [1,1,1,1,1,1,1,1,1,1] All tokens are valid.

### Why return_tensors="pt"?
#### without pt
tokenizer(prompt) : returns Python lists:
{
"input_ids": [123,456,789],
"attention_mask": [1,1,1]
}
#### with return_tensors="pt"
returns PyTorch tensors:
{
"input_ids": tensor(...),
"attention_mask": tensor(...)
}
- pt stands for: PyTorch
- The model expects tensors, not Python lists.
### Moving Inputs to Model Device
```python

inputs = {
k: v.to(text_model.device) for k, v in inputs.items()
}
```
#### What is inputs.items()?
Since inputs behaves like a dictionary:
inputs.items() returns:

("input_ids", tensor(...))
("attention_mask", tensor(...))

We iterate through all key-value pairs and move every tensor to the same device (CPU/GPU) as the model.

#### What Does **inputs Mean?
```python
inputs = {
"input_ids": tensor(...),
"attention_mask": tensor(...)
}
now becomes
``` python
model.generate(**inputs)
get expands to

model.generate(
input_ids=tensor(...),
attention_mask=tensor(...)
)

It unpacks the dictionary. Keys become argument names. Values become argument values.
```
### Generate Output
```python 
outputs = text_model.generate(
**inputs,
max_new_tokens=50
)
The model predicts new tokens based on the prompt.
```
#### What Does max_new_tokens Mean?

max_new_tokens=50 ( Generate at most 50 new tokens after the prompt. )

Example: Prompt: What is AI?
Suppose prompt contains 4 tokens.
If model generates 20 additional tokens:
- Prompt Tokens = 4
- Generated Tokens = 20
- Total Tokens = 24

The generated part cannot exceed 50 tokens.
#### Convert Output Back to Text
```python
response = tokenizer.decode(
outputs[0],
skip_special_tokens=True
)
print(response)
```
Example:
What is AI? 
AI is a field of computer science focused on creating systems capable of performing tasks that normally require human intelligence.
