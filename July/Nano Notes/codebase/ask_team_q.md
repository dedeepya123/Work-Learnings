- Is this LongContext getting enbled? default 8k passed 15k?
- What is mask negative menas --> How do we decide on that number?
- In ModelConfig dataclasses (frozen=True) so can it be overrided ?
    a. Why datset(text dataset,vision_dataset, audio_dataset)  type is getting set to tuple ? --> qargparse.py
    b. Modatilty is of Type Modality ? --> qargpase.py --> why dont w ekeep it as string ? 
 - While loading model
     1. instailting Geemma4 Context --> why add_BOS , use_fast is set to false what they mean
     2. num_logits_to_keep = 0 - Does this mean [ no tokens returned ]
     3. We are kind of overriding lmm.text_config = llm_config paremtres like
        ``` text
          [
            transposed_key_cahe,
            return_new_key_value_only,
            input_tokens_per_infernece = arn,
            context_length = args.contextlength,
            eos_token_id = processor.tokenizer.eos_token_id
            pad_token_id = processor.tokenizer.pad_token
            pad_to_left = False
            sliding_window_pattern = "our computed pattern" - Finding index of the layer whose attention_type == "full_attention" + 1
          ]

        Question:
The model contains two embedding representations:
1. embed_tokens
   Shape: [vocab_size, hidden_size]
   Example: [vocab_size, 1536]

2. embed_tokens_per_layer
   Shape: [vocab_size, num_hidden_layers × hidden_size_per_layer_input]
   Example: [vocab_size, 35 × 256 = 8960]

My understanding is that embed_tokens_per_layer stores a concatenation of layer-specific embedding slices (256 dimensions per layer), potentially with separate quantization scales per layer.

Could you explain:
- Why is a layer-wise embedding representation required in Gemma4/QAIRT? so for a token why its paer layer mbedding needed and where and why?
- How is embed_tokens_per_layer consumed by the model?
- Is this primarily for quantization accuracy, export/runtime optimization, or a model architecture requirement?
- How does it relate to the standard embed_tokens embedding table?
```
   


  
