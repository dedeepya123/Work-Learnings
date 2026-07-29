``` text
Gemma4ForConditionalGeneration(
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Gemma4QuantizableLinear()
            (up_proj): Gemma4QuantizableLinear()
            (down_proj): Gemma4QuantizableLinear()
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Gemma4QuantizableLinear()
            (up_proj): Gemma4QuantizableLinear()
            (down_proj): Gemma4QuantizableLinear()
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (0-14): 15 x Gemma4VisionEncoderLayer(
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
              (q_proj_0): Gemma4QuantizableLinear()
              (q_proj_1): Gemma4QuantizableLinear()
              (q_proj_2): Gemma4QuantizableLinear()
              (q_proj_3): Gemma4QuantizableLinear()
              (q_proj_4): Gemma4QuantizableLinear()
              (q_proj_5): Gemma4QuantizableLinear()
              (q_proj_6): Gemma4QuantizableLinear()
              (q_proj_7): Gemma4QuantizableLinear()
              (q_proj_8): Gemma4QuantizableLinear()
              (q_proj_9): Gemma4QuantizableLinear()
              (q_proj_10): Gemma4QuantizableLinear()
              (q_proj_11): Gemma4QuantizableLinear()
              (k_proj_0): Gemma4QuantizableLinear()
              (k_proj_1): Gemma4QuantizableLinear()
              (k_proj_2): Gemma4QuantizableLinear()
              (k_proj_3): Gemma4QuantizableLinear()
              (k_proj_4): Gemma4QuantizableLinear()
              (k_proj_5): Gemma4QuantizableLinear()
              (k_proj_6): Gemma4QuantizableLinear()
              (k_proj_7): Gemma4QuantizableLinear()
              (k_proj_8): Gemma4QuantizableLinear()
              (k_proj_9): Gemma4QuantizableLinear()
              (k_proj_10): Gemma4QuantizableLinear()
              (k_proj_11): Gemma4QuantizableLinear()
              (v_proj_0): Gemma4QuantizableLinear()
              (v_proj_1): Gemma4QuantizableLinear()
              (v_proj_2): Gemma4QuantizableLinear()
              (v_proj_3): Gemma4QuantizableLinear()
              (v_proj_4): Gemma4QuantizableLinear()
              (v_proj_5): Gemma4QuantizableLinear()
              (v_proj_6): Gemma4QuantizableLinear()
              (v_proj_7): Gemma4QuantizableLinear()
              (v_proj_8): Gemma4QuantizableLinear()
              (v_proj_9): Gemma4QuantizableLinear()
              (v_proj_10): Gemma4QuantizableLinear()
              (v_proj_11): Gemma4QuantizableLinear()
            )
            (mlp): Gemma4VisionMLP(
              (gate_proj): Gemma4QuantizableLinear()
              (up_proj): Gemma4QuantizableLinear()
              (down_proj): Gemma4QuantizableLinear()
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): GELUActivation()
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
          (15): Gemma4VisionEncoderLayer(
            (self_attn): Gemma4VisionAttention(
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
              (q_proj_0): Gemma4QuantizableLinear()
              (q_proj_1): Gemma4QuantizableLinear()
              (q_proj_2): Gemma4QuantizableLinear()
              (q_proj_3): Gemma4QuantizableLinear()
              (q_proj_4): Gemma4QuantizableLinear()
              (q_proj_5): Gemma4QuantizableLinear()
              (q_proj_6): Gemma4QuantizableLinear()
              (q_proj_7): Gemma4QuantizableLinear()
              (q_proj_8): Gemma4QuantizableLinear()
              (q_proj_9): Gemma4QuantizableLinear()
              (q_proj_10): Gemma4QuantizableLinear()
              (q_proj_11): Gemma4QuantizableLinear()
              (k_proj_0): Gemma4QuantizableLinear()
              (k_proj_1): Gemma4QuantizableLinear()
              (k_proj_2): Gemma4QuantizableLinear()
              (k_proj_3): Gemma4QuantizableLinear()
              (k_proj_4): Gemma4QuantizableLinear()
              (k_proj_5): Gemma4QuantizableLinear()
              (k_proj_6): Gemma4QuantizableLinear()
              (k_proj_7): Gemma4QuantizableLinear()
              (k_proj_8): Gemma4QuantizableLinear()
              (k_proj_9): Gemma4QuantizableLinear()
              (k_proj_10): Gemma4QuantizableLinear()
              (k_proj_11): Gemma4QuantizableLinear()
              (v_proj_0): Gemma4QuantizableLinear()
              (v_proj_1): Gemma4QuantizableLinear()
              (v_proj_2): Gemma4QuantizableLinear()
              (v_proj_3): Gemma4QuantizableLinear()
              (v_proj_4): Gemma4QuantizableLinear()
              (v_proj_5): Gemma4QuantizableLinear()
              (v_proj_6): Gemma4QuantizableLinear()
              (v_proj_7): Gemma4QuantizableLinear()
              (v_proj_8): Gemma4QuantizableLinear()
              (v_proj_9): Gemma4QuantizableLinear()
              (v_proj_10): Gemma4QuantizableLinear()
              (v_proj_11): Gemma4QuantizableLinear()
            )
            (mlp): Gemma4VisionMLP(
              (gate_proj): Gemma4QuantizableLinear()
              (up_proj): Gemma4QuantizableLinear()
              (down_proj): Gemma4QuantizableLinear()
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): GELUActivation()
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
          (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (act): ReLU(
            (output_sfq): StaticFakeQuant()
          )
        )
        (layer1): Gemma4AudioSubSampleConvProjectionLayer(
          (conv): Conv2d(
            128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 1), bias=False
            (output_sfq): StaticFakeQuant()
          )
          (norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
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
            (value_states_mul): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (key_states_mul): Mul(
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
      (embedding_projection): Gemma4QuantizableLinear()
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
          (apply_rope_fn): ApplyRopeSingle(
            (mul_x_real_rope_real): MulModule()
            (mul_x_im_rope_im): MulModule()
            (mul_x_real_rope_im): MulModule()
            (mul_x_im_rope_real): MulModule()
          )
        )
        (mlp): Gemma4TextMLP(
          (gate_proj): Gemma4QuantizableLinear()
          (up_proj): Gemma4QuantizableLinear()
          (down_proj): Gemma4QuantizableLinear()
          (act_fn): Act2FN(
            (output_sfq): StaticFakeQuant()
            (act_fn): GELUActivation()
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
    (matmul_centroid1): EinsumOp()
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
```

## Adapted Model

``` text
Gemma4ForConditionalGeneration(
  (model): Gemma4Model(
    (language_model): Gemma4TextModel(
      (embed_tokens): Gemma4QuantizableEmbedding()
      (layers): ModuleList(
        (0-3): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(6144, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (4): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): ConvInplaceLinear(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(6144, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (5-8): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(6144, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (9): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): ConvInplaceLinear(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(6144, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (10-13): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(6144, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (14): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (v_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (k_proj): ConvInplaceLinear(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 6144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(6144, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (15-18): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (19): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (20-23): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (24): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (25-28): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (29): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (30-33): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(2048, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
        )
        (34): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): ConvInplaceLinear(1536, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (q_norm): Gemma4RMSNorm(
              (output_sfq): StaticFakeQuant()
            )
            (o_proj): ConvInplaceLinear(4096, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (apply_rope_fn): ApplyRopeSingle(
              (mul_x_real_rope_real): MulModule()
              (mul_x_im_rope_im): MulModule()
              (mul_x_real_rope_im): MulModule()
              (mul_x_im_rope_real): MulModule()
            )
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (up_proj): ConvInplaceLinear(1536, 12288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (down_proj): ConvInplaceLinear(12288, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (act_fn): Act2FN(
              (output_sfq): StaticFakeQuant()
              (act_fn): GELUActivation()
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
          (per_layer_input_gate): ConvInplaceLinear(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (per_layer_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
      (per_layer_model_projection): ConvInplaceLinear(1536, 8960, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
        (input_proj): ConvInplaceLinear(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (add): Add(
          (output_sfq): StaticFakeQuant()
        )
        (pixel_sfq): CraftModule(
          (output_sfq): StaticFakeQuant()
        )
        (pe_table_0): Embedding(3360, 768)
        (pe_table_1): Embedding(3360, 768)
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
          (0-14): 15 x Gemma4VisionEncoderLayer(
            (self_attn): Gemma4VisionAttention(
              (q_proj): ConvInplaceLinear(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj): ConvInplaceLinear(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj): ConvInplaceLinear(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (o_proj): ConvInplaceLinear(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
              (q_proj_0): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_1): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_2): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_3): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_4): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_5): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_6): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_7): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_8): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_9): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_10): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_11): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_0): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_1): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_2): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_3): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_4): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_5): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_6): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_7): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_8): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_9): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_10): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_11): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_0): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_1): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_2): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_3): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_4): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_5): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_6): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_7): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_8): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_9): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_10): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_11): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (mlp): Gemma4VisionMLP(
              (gate_proj): ConvInplaceLinear(768, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (up_proj): ConvInplaceLinear(768, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (down_proj): ConvInplaceLinear(3072, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): GELUActivation()
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
          (15): Gemma4VisionEncoderLayer(
            (self_attn): Gemma4VisionAttention(
              (o_proj): ConvInplaceLinear(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
              (q_proj_0): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_1): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_2): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_3): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_4): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_5): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_6): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_7): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_8): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_9): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_10): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (q_proj_11): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_0): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_1): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_2): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_3): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_4): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_5): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_6): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_7): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_8): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_9): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_10): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (k_proj_11): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_0): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_1): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_2): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_3): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_4): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_5): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_6): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_7): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_8): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_9): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_10): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (v_proj_11): ConvInplaceLinear(768, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (mlp): Gemma4VisionMLP(
              (gate_proj): ConvInplaceLinear(768, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (up_proj): ConvInplaceLinear(768, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (down_proj): ConvInplaceLinear(3072, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (act_fn): Act2FN(
                (output_sfq): StaticFakeQuant()
                (act_fn): GELUActivation()
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
      (embedding_projection): ConvInplaceLinear(768, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
          (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (act): ReLU(
            (output_sfq): StaticFakeQuant()
          )
        )
        (layer1): Gemma4AudioSubSampleConvProjectionLayer(
          (conv): Conv2d(
            128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 1), bias=False
            (output_sfq): StaticFakeQuant()
          )
          (norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
          (act): ReLU(
            (output_sfq): StaticFakeQuant()
          )
        )
        (input_proj_linear): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (rel_pos_enc): Gemma4AudioRelPositionalEncoding()
      (layers): ModuleList(
        (0-11): 12 x Gemma4AudioLayer(
          (feed_forward1): Gemma4AudioFeedForward(
            (ffw_layer_1): ConvInplaceLinear(1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (ffw_layer_2): ConvInplaceLinear(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (ffw_layer_1): ConvInplaceLinear(1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (ffw_layer_2): ConvInplaceLinear(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (q_proj): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (k_proj): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (v_proj): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (post): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (relative_k_proj): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (value_states_mul): Mul(
              (output_sfq): StaticFakeQuant()
            )
            (key_states_mul): Mul(
              (output_sfq): StaticFakeQuant()
            )
          )
          (lconv1d): Gemma4AudioLightConv1d(
            (linear_start): ConvInplaceLinear(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (linear_end): ConvInplaceLinear(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
      (output_proj): ConvInplaceLinear(1024, 1536, kernel_size=(1, 1), stride=(1, 1))
    )
    (embed_audio): Gemma4MultimodalEmbedder(
      (embedding_projection): ConvInplaceLinear(1536, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (embedding_pre_projection_norm): Gemma4RMSNorm(
        (output_sfq): StaticFakeQuant()
      )
    )
  )
  (lm_head): ConvInplaceLinear(1536, 262144, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
    (pre_projection): ConvInplaceLinear(3072, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (layers): ModuleList(
      (0-2): 3 x Gemma4AssistantDecoderLayer(
        (self_attn): Gemma4MtpCrossAttention(
          (q_proj): ConvInplaceLinear(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (q_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (k_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (v_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (k_proj): ConvInplaceLinear(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (v_proj): ConvInplaceLinear(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (o_proj): ConvInplaceLinear(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
          (apply_rope_fn): ApplyRopeSingle(
            (mul_x_real_rope_real): MulModule()
            (mul_x_im_rope_im): MulModule()
            (mul_x_real_rope_im): MulModule()
            (mul_x_im_rope_real): MulModule()
          )
        )
        (mlp): Gemma4TextMLP(
          (gate_proj): ConvInplaceLinear(256, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (up_proj): ConvInplaceLinear(256, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (down_proj): ConvInplaceLinear(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): Act2FN(
            (output_sfq): StaticFakeQuant()
            (act_fn): GELUActivation()
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
      (3): Gemma4AssistantDecoderLayer(
        (self_attn): Gemma4MtpCrossAttention(
          (q_proj): ConvInplaceLinear(256, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (q_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (k_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (v_norm): Gemma4RMSNorm(
            (output_sfq): StaticFakeQuant()
          )
          (k_proj): ConvInplaceLinear(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (v_proj): ConvInplaceLinear(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (o_proj): ConvInplaceLinear(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
          (apply_rope_fn): ApplyRopeSingle(
            (mul_x_real_rope_real): MulModule()
            (mul_x_im_rope_im): MulModule()
            (mul_x_real_rope_im): MulModule()
            (mul_x_im_rope_real): MulModule()
          )
        )
        (mlp): Gemma4TextMLP(
          (gate_proj): ConvInplaceLinear(256, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (up_proj): ConvInplaceLinear(256, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (down_proj): ConvInplaceLinear(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): Act2FN(
            (output_sfq): StaticFakeQuant()
            (act_fn): GELUActivation()
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
    (post_projection): ConvInplaceLinear(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (lm_head): ConvInplaceLinear(256, 262144, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (lm_head_out): CraftModule(
      (output_sfq): StaticFakeQuant()
    )
    (matmul_centroid1): EinsumOp()
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
```
