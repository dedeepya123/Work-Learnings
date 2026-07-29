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
The model contains two embedding representations in Gemma4Context (load_embed_tokens_paer_layer, load_embed_tokens)
1. embed_tokens
   Shape: [vocab_size, hidden_size]
   Example: [vocab_size, 1536]

2. embed_tokens_per_layer
   Shape: [vocab_size, num_hidden_layers × hidden_size_per_layer_input]
   Example: [vocab_size, 35 × 256 = 8960]

My understanding is that embed_tokens_per_layer stores a concatenation of layer-specific embedding slices (256 dimensions per layer), with separate quantization scales per layer.
Doubt is 
- Why is a layer-wise embedding representation required in Gemma4? so for a token why its paer layer mbedding needed and where and why?
- How is embed_tokens_per_layer consumed by the model?
- Why scale is sqrt of hidden_size per layer input.

Didnt completely understood _Adapted Enbedding 
- why dequnatize
- What excatly is happening
- so we have token and its embedding tensor we are craeting and scale == 1 , what is it ?

Gemma4Context class is big enough --> Single Responsibilty principle violation ? what excatly it does 
(dequantization, load embeddings (MTP, per layer, embedtokens), calibrate text, vision encoder, mtp, audio encoder, MPP( model preparer subgraphs (prefix, decode, ve, ae, mtp), Quantsim wrapped sub graphs for prepared model (prefix, decode, ve, ae, mtp)

```
2. In the create_qc_model [inside Gemma4Context]
- what does "model.qc" state represent? wht if its not None does it mean directly loading adapted model (if model is alredy adpated ?)
3. why explicit mha to sha for vision only ?
4. dequantize_e,beddings() --> called by create_qc method is misleading
    (It does dequnatize embeddig laeyrs , replace liner to conv, 
    why firstly we quntized and then now dequntized embedding weights of model)
5. why deleting and setting to None (why we instantiated)

 
   


  
