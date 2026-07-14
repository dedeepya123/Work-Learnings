# Introduction
Gemma is a family of lightweight, state-of-the art open models built from the same research and technology used to create the Gemini models.

Different variations of Gemma are designed for different use cases and modalities, such as:

- Single modality (Text in, Text out)
- Specialization for coding use cases
- Multi modality (Text and Image in, Text out)
- Varying sizes for different hardware types, inference needs, and other constraints. “Novel” architectures

Model(
  (model): Gemma4ForConditionalGeneration(
    (model): Gemma4Model(
      (language_model): Gemma4TextModel(
        (embed_tokens): Gemma4QuantizableEmbedding()
        (layers): ModuleList(
          (0-14): 15 x Gemma4TextDecoderLayer(
            (self_attn): Gemma4TextAttention(
              (q_proj): Gemma4QuantizableLinear()
              (q_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (k_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (v_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (k_proj): Gemma4QuantizableLinear()
              (v_proj): Gemma4QuantizableLinear()
              (o_proj): Gemma4QuantizableLinear()
              (attn_matmul_qk): Matmul(
                (output_sfq): StaticFakeQuant()
              )
              (attn_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (attn_add): Add(
                (output_sfq): StaticFakeQuant()
              )
              (attn_softmax): Softmax(
                (output_sfq): StaticFakeQuant()
              )
              (attn_matmul_av): Matmul(
                (output_sfq): StaticFakeQuant()
              )
              (softcap_div): Div(
                (output_sfq): StaticFakeQuant()
              )
              (softcap_tanh): Tanh(
                (output_sfq): StaticFakeQuant()
              )
              (softcap_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (q_rope_operator): RotaryEmbeddingOperator(
                (cos_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (sin_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (add): Add(
                  (output_sfq): StaticFakeQuant()
                )
                (cat): Concat(
                  (output_sfq): StaticFakeQuant()
                )
                (neg): Neg(
                  (output_sfq): StaticFakeQuant()
                )
              )
              (k_rope_operator): RotaryEmbeddingOperator(
                (cos_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (sin_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (add): Add(
                  (output_sfq): StaticFakeQuant()
                )
                (cat): Concat(
                  (output_sfq): StaticFakeQuant()
                )
                (neg): Neg(
                  (output_sfq): StaticFakeQuant()
                )
              )
            )
            (mlp): Gemma4TextMLP(
              (gate_proj): Gemma4QuantizableLinear()
              (up_proj): Gemma4QuantizableLinear()
              (down_proj): Gemma4QuantizableLinear()
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): GELUTanh()
              )
              (mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
            )
            (input_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (post_attention_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (pre_feedforward_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (post_feedforward_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (add1): Add(
              (output_sfq): StaticFakeQuant()
            )
            (add2): Add(
              (output_sfq): StaticFakeQuant()
            )
            (add3): Add(
              (output_sfq): StaticFakeQuant()
            )
            (mul1): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (mul2): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUTanh()
            )
            (per_layer_input_gate): Gemma4QuantizableLinear()
            (per_layer_projection): Gemma4QuantizableLinear()
            (post_per_layer_input_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
          )
          (15-34): 20 x Gemma4TextDecoderLayer(
            (self_attn): Gemma4TextAttention(
              (q_proj): Gemma4QuantizableLinear()
              (q_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (o_proj): Gemma4QuantizableLinear()
              (attn_matmul_qk): Matmul(
                (output_sfq): StaticFakeQuant()
              )
              (attn_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (attn_add): Add(
                (output_sfq): StaticFakeQuant()
              )
              (attn_softmax): Softmax(
                (output_sfq): StaticFakeQuant()
              )
              (attn_matmul_av): Matmul(
                (output_sfq): StaticFakeQuant()
              )
              (softcap_div): Div(
                (output_sfq): StaticFakeQuant()
              )
              (softcap_tanh): Tanh(
                (output_sfq): StaticFakeQuant()
              )
              (softcap_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (q_rope_operator): RotaryEmbeddingOperator(
                (cos_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (sin_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (add): Add(
                  (output_sfq): StaticFakeQuant()
                )
                (cat): Concat(
                  (output_sfq): StaticFakeQuant()
                )
                (neg): Neg(
                  (output_sfq): StaticFakeQuant()
                )
              )
              (k_rope_operator): RotaryEmbeddingOperator(
                (cos_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (sin_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (add): Add(
                  (output_sfq): StaticFakeQuant()
                )
                (cat): Concat(
                  (output_sfq): StaticFakeQuant()
                )
                (neg): Neg(
                  (output_sfq): StaticFakeQuant()
                )
              )
            )
            (mlp): Gemma4TextMLP(
              (gate_proj): Gemma4QuantizableLinear()
              (up_proj): Gemma4QuantizableLinear()
              (down_proj): Gemma4QuantizableLinear()
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): GELUTanh()
              )
              (mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
            )
            (input_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (post_attention_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (pre_feedforward_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (post_feedforward_layernorm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (add1): Add(
              (output_sfq): StaticFakeQuant()
            )
            (add2): Add(
              (output_sfq): StaticFakeQuant()
            )
            (add3): Add(
              (output_sfq): StaticFakeQuant()
            )
            (mul1): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (mul2): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUTanh()
            )
            (per_layer_input_gate): Gemma4QuantizableLinear()
            (per_layer_projection): Gemma4QuantizableLinear()
            (post_per_layer_input_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
          )
        )
        (norm): Gemma4RMSNorm(
          (output_sfq): StaticFakeQuant()
        )
        (rotary_emb): Gemma4TextRotaryEmbedding()
        (embed_outs): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
        (ple_in): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
        (ple_outs): ModuleList(
          (0-34): 35 x CraftModule(
            (output_sfq): StaticFakeQuant()
          )
        )
        (cos_in): ModuleDict(
          (sliding_attention): CraftModule(
            (output_sfq): StaticFakeQuant()
          )
          (full_attention): CraftModule(
            (output_sfq): StaticFakeQuant()
          )
        )
        (sin_in): ModuleDict(
          (sliding_attention): CraftModule(
            (output_sfq): StaticFakeQuant()
          )
          (full_attention): CraftModule(
            (output_sfq): StaticFakeQuant()
          )
        )
        (embed_tokens_per_layer): Gemma4QuantizableEmbedding()
        (per_layer_model_projection): Gemma4QuantizableLinear()
        (per_layer_model_projection_mul): Mul(
          (output_sfq): StaticFakeQuant()
        )
        (per_layer_projection_norm): Gemma4RMSNorm(
          (output_sfq): StaticFakeQuant()
        )
        (add): Add(
          (output_sfq): StaticFakeQuant()
        )
        (mul): Mul(
          (output_sfq): StaticFakeQuant()
        )
      )
      (vision_tower): Gemma4VisionModel(
        (patch_embedder): Gemma4VisionPatchEmbedder(
          (input_proj): Linear(
            in_features=768, out_features=768, bias=False
            (output_sfq): StaticFakeQuant()
          )
          (add): Add(
            (output_sfq): StaticFakeQuant()
          )
          (pixel_sfq): CraftModule(
            (output_sfq): StaticFakeQuant()
          )
        )
        (encoder): Gemma4VisionEncoder(
          (rotary_emb): Gemma4VisionRotaryEmbedding(
            (cos_outs): CraftModule(
              (output_sfq): StaticFakeQuant()
            )
            (sin_outs): CraftModule(
              (output_sfq): StaticFakeQuant()
            )
          )
          (layers): ModuleList(
            (0-15): 16 x Gemma4VisionEncoderLayer(
              (self_attn): Gemma4VisionAttention(
                (q_proj): Gemma4QuantizableLinear()
                (k_proj): Gemma4QuantizableLinear()
                (v_proj): Gemma4QuantizableLinear()
                (o_proj): Gemma4QuantizableLinear()
                (q_norm): Gemma4RMSNorm(
                  (output_sfq): StaticFakeQuant()
                )
                (k_norm): Gemma4RMSNorm(
                  (output_sfq): StaticFakeQuant()
                )
                (v_norm): Gemma4RMSNorm(
                  (output_sfq): StaticFakeQuant()
                )
                (attn_matmul_qk): Matmul(
                  (output_sfq): StaticFakeQuant()
                )
                (attn_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (attn_add): Add(
                  (output_sfq): StaticFakeQuant()
                )
                (attn_softmax): Softmax(
                  (output_sfq): StaticFakeQuant()
                )
                (attn_matmul_av): Matmul(
                  (output_sfq): StaticFakeQuant()
                )
                (softcap_div): Div(
                  (output_sfq): StaticFakeQuant()
                )
                (softcap_tanh): Tanh(
                  (output_sfq): StaticFakeQuant()
                )
                (softcap_mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
                (q_rope_operator1): RotaryEmbeddingOperator(
                  (cos_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (sin_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (add): Add(
                    (output_sfq): StaticFakeQuant()
                  )
                  (cat): Concat(
                    (output_sfq): StaticFakeQuant()
                  )
                  (neg): Neg(
                    (output_sfq): StaticFakeQuant()
                  )
                )
                (q_rope_operator2): RotaryEmbeddingOperator(
                  (cos_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (sin_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (add): Add(
                    (output_sfq): StaticFakeQuant()
                  )
                  (cat): Concat(
                    (output_sfq): StaticFakeQuant()
                  )
                  (neg): Neg(
                    (output_sfq): StaticFakeQuant()
                  )
                )
                (k_rope_operator1): RotaryEmbeddingOperator(
                  (cos_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (sin_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (add): Add(
                    (output_sfq): StaticFakeQuant()
                  )
                  (cat): Concat(
                    (output_sfq): StaticFakeQuant()
                  )
                  (neg): Neg(
                    (output_sfq): StaticFakeQuant()
                  )
                )
                (k_rope_operator2): RotaryEmbeddingOperator(
                  (cos_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (sin_mul): Mul(
                    (output_sfq): StaticFakeQuant()
                  )
                  (add): Add(
                    (output_sfq): StaticFakeQuant()
                  )
                  (cat): Concat(
                    (output_sfq): StaticFakeQuant()
                  )
                  (neg): Neg(
                    (output_sfq): StaticFakeQuant()
                  )
                )
              )
              (mlp): Gemma4VisionMLP(
                (gate_proj): Gemma4QuantizableLinear()
                (up_proj): Gemma4QuantizableLinear()
                (down_proj): Gemma4QuantizableLinear()
                (act_fn): Act2FN(
                  (output_sfq): StaticFakeQuant()
                  (act_fn): GELUTanh()
                )
                (mul): Mul(
                  (output_sfq): StaticFakeQuant()
                )
              )
              (input_layernorm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (post_attention_layernorm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (pre_feedforward_layernorm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (post_feedforward_layernorm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (add1): Add(
                (output_sfq): StaticFakeQuant()
              )
              (add2): Add(
                (output_sfq): StaticFakeQuant()
              )
            )
          )
        )
        (pooler): Gemma4VisionPooler(
          (mul): Mul(
            (output_sfq): StaticFakeQuant()
          )
          (matmul): Matmul(
            (output_sfq): StaticFakeQuant()
          )
          (patch_to_pool_weights_in): CraftModule(
            (output_sfq): StaticFakeQuant()
          )
        )
      )
      (embed_vision): Gemma4MultimodalEmbedder(
        (embedding_projection): Linear(
          in_features=768, out_features=1536, bias=False
          (output_sfq): StaticFakeQuant()
        )
        (embedding_pre_projection_norm): Gemma4RMSNorm(
          (output_sfq): StaticFakeQuant()
        )
      )
      (audio_tower): Gemma4AudioModel(
        (div): Div(
          (output_sfq): StaticFakeQuant()
        )
        (mul): Mul(
          (output_sfq): StaticFakeQuant()
        )
        (subsample_conv_projection): Gemma4AudioSubSampleConvProjection(
          (output_sfq): StaticFakeQuant()
          (layer0): Gemma4AudioSubSampleConvProjectionLayer(
            (conv): Conv2d(
              1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 1), bias=False
              (output_sfq): StaticFakeQuant()
            )
            (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True, bias=False)
            (act): ReLU(
              (output_sfq): StaticFakeQuant()
            )
          )
          (layer1): Gemma4AudioSubSampleConvProjectionLayer(
            (conv): Conv2d(
              128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 1), bias=False
              (output_sfq): StaticFakeQuant()
            )
            (norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True, bias=False)
            (act): ReLU(
              (output_sfq): StaticFakeQuant()
            )
          )
          (input_proj_linear): Linear(
            in_features=1024, out_features=1024, bias=False
            (output_sfq): StaticFakeQuant()
          )
        )
        (rel_pos_enc): Gemma4AudioRelPositionalEncoding()
        (layers): ModuleList(
          (0-11): 12 x Gemma4AudioLayer(
            (feed_forward1): Gemma4AudioFeedForward(
              (ffw_layer_1): Gemma4QuantizableLinear()
              (ffw_layer_2): Gemma4QuantizableLinear()
              (pre_layer_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (post_layer_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): SiLUActivation()
              )
              (mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (add): Add(
                (output_sfq): StaticFakeQuant()
              )
            )
            (feed_forward2): Gemma4AudioFeedForward(
              (ffw_layer_1): Gemma4QuantizableLinear()
              (ffw_layer_2): Gemma4QuantizableLinear()
              (pre_layer_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (post_layer_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): SiLUActivation()
              )
              (mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (add): Add(
                (output_sfq): StaticFakeQuant()
              )
            )
            (self_attn): Gemma4AudioAttention(
              (q_proj): Gemma4QuantizableLinear()
              (k_proj): Gemma4QuantizableLinear()
              (v_proj): Gemma4QuantizableLinear()
              (post): Gemma4QuantizableLinear()
              (relative_k_proj): Linear(
                in_features=1024, out_features=1024, bias=False
                (output_sfq): StaticFakeQuant()
              )
              (add): Add(
                (output_sfq): StaticFakeQuant()
              )
              (div): Div(
                (output_sfq): StaticFakeQuant()
              )
              (mul_logits): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (tanh): Tanh(
                (output_sfq): StaticFakeQuant()
              )
              (softplus_per_dim_scale): Softplus(
                (output_sfq): StaticFakeQuant()
              )
              (softmax): Softmax(
                (output_sfq): StaticFakeQuant()
              )
              (mul_query_states1): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (mul_query_states2): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (mul_key_states): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (matmul_ac): Matmul(
                (output_sfq): StaticFakeQuant()
              )
              (matmul_bd): Matmul(
                (output_sfq): StaticFakeQuant()
              )
              (matmul_output): Matmul(
                (output_sfq): StaticFakeQuant()
              )
            )
            (lconv1d): Gemma4AudioLightConv1d(
              (linear_start): Gemma4QuantizableLinear()
              (linear_end): Gemma4QuantizableLinear()
              (depthwise_conv1d): Gemma4AudioCausalConv1d(
                1024, 1024, kernel_size=(5,), stride=(1,), groups=1024, bias=False
                (output_sfq): StaticFakeQuant()
              )
              (pre_layer_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (conv_norm): Gemma4RMSNorm(
                (output_sfq): StaticFakeQuant()
              )
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): SiLUActivation()
              )
              (glu): GLU(
                (output_sfq): StaticFakeQuant()
              )
              (add): Add(
                (output_sfq): StaticFakeQuant()
              )
            )
            (norm_pre_attn): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (norm_post_attn): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (norm_out): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (add): Add(
              (output_sfq): StaticFakeQuant()
            )
          )
        )
        (output_proj): Linear(
          in_features=1024, out_features=1536, bias=True
          (output_sfq): StaticFakeQuant()
        )
      )
      (embed_audio): Gemma4MultimodalEmbedder(
        (embedding_projection): Linear(
          in_features=1536, out_features=1536, bias=False
          (output_sfq): StaticFakeQuant()
        )
        (embedding_pre_projection_norm): Gemma4RMSNorm(
          (output_sfq): StaticFakeQuant()
        )
      )
    )
    (lm_head): Gemma4QuantizableLinear()
    (lm_head_out): CraftModule(
      (output_sfq): StaticFakeQuant()
    )
    (logit_div): Div(
      (output_sfq): StaticFakeQuant()
    )
    (logit_tanh): Tanh(
      (output_sfq): StaticFakeQuant()
    )
    (logit_mul): Mul(
      (output_sfq): StaticFakeQuant()
    )
    (assistant_model): Gemma4AssistantModel(
      (pre_projection): Gemma4QuantizableLinear()
      (layers): ModuleList(
        (0-3): 4 x Gemma4AssistantDecoderLayer(
          (self_attn): Gemma4MtpCrossAttention(
            (q_proj): Gemma4QuantizableLinear()
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): Gemma4QuantizableLinear()
            (v_proj): Gemma4QuantizableLinear()
            (o_proj): Gemma4QuantizableLinear()
            (attn_matmul_qk): Matmul(
              (output_sfq): StaticFakeQuant()
            )
            (attn_mul): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (attn_add): Add(
              (output_sfq): StaticFakeQuant()
            )
            (attn_softmax): Softmax(
              (output_sfq): StaticFakeQuant()
            )
            (attn_matmul_av): Matmul(
              (output_sfq): StaticFakeQuant()
            )
            (softcap_div): Div(
              (output_sfq): StaticFakeQuant()
            )
            (softcap_tanh): Tanh(
              (output_sfq): StaticFakeQuant()
            )
            (softcap_mul): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (q_rope_operator): RotaryEmbeddingOperator(
              (cos_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (sin_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (add): Add(
                (output_sfq): StaticFakeQuant()
              )
              (cat): Concat(
                (output_sfq): StaticFakeQuant()
              )
              (neg): Neg(
                (output_sfq): StaticFakeQuant()
              )
            )
            (k_rope_operator): RotaryEmbeddingOperator(
              (cos_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (sin_mul): Mul(
                (output_sfq): StaticFakeQuant()
              )
              (add): Add(
                (output_sfq): StaticFakeQuant()
              )
              (cat): Concat(
                (output_sfq): StaticFakeQuant()
              )
              (neg): Neg(
                (output_sfq): StaticFakeQuant()
              )
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Gemma4QuantizableLinear()
            (up_proj): Gemma4QuantizableLinear()
            (down_proj): Gemma4QuantizableLinear()
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUTanh()
            )
            (mul): Mul(
              (output_sfq): StaticFakeQuant()
            )
          )
          (input_layernorm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (post_attention_layernorm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (pre_feedforward_layernorm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (post_feedforward_layernorm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (add1): Add(
            (output_sfq): StaticFakeQuant()
          )
          (add2): Add(
            (output_sfq): StaticFakeQuant()
          )
          (mul1): Mul(
            (output_sfq): StaticFakeQuant()
          )
        )
      )
      (final_norm): Gemma4RMSNorm(
        (output_sfq): StaticFakeQuant()
      )
      (post_projection): Gemma4QuantizableLinear()
      (lm_head): Gemma4QuantizableLinear()
      (lm_head_out): CraftModule(
        (output_sfq): StaticFakeQuant()
      )
      (matmul_centroid1): Matmul(
        (output_sfq): StaticFakeQuant()
      )
      (lm_head_einsum): Gemma4QuantizableEinsum()
      (lm_head_einsum_out): CraftModule(
        (output_sfq): StaticFakeQuant()
      )
      (rotary_emb): Gemma4TextRotaryEmbedding()
      (cat): Concat(
        (output_sfq): StaticFakeQuant()
      )
      (cos_in): ModuleDict(
        (sliding_attention): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
        (full_attention): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
      )
      (sin_in): ModuleDict(
        (sliding_attention): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
        (full_attention): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
      )
    )
  )
)
