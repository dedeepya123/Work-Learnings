# AIMET issue
- QAIRT and AIMET compatibility ?

# Try to export HF Floating point model to onnx error 
# exporting HF to HF no error

# Does GenAI Builder support CPU backend ?


  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/transformers/modeling_utils.py", line 5144, in from_pretrained
    hf_quantizer.postprocess_model(model, config=config)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/transformers/quantizers/base.py", line 238, in postprocess_model
    return self._process_model_after_weight_loading(model, **kwargs)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/qairt/experimental/pipeline/torch/llm/quantization/techniques/hf/base.py", line 104, in _wrapped
    return original(self, model, **kw)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/qairt/experimental/pipeline/torch/llm/quantization/techniques/hf/lpbq_seqmse_hf_quantizer.py", line 164, in _process_model_after_weight_loading
    result = LPBQ_SeqMSE_Recipe().apply(
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/qairt/experimental/pipeline/torch/llm/quantization/recipes/defaults.py", line 677, in apply
    with place_model(quantsim.model, DEFAULT_QUANT_DEVICE):
  File "/usr/lib/python3.10/contextlib.py", line 135, in __enter__
    return next(self.gen)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/aimet_torch/utils.py", line 1028, in place_model
    model.to(device=device)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
  File "/local/mnt2/workspace/dlekkala/qairt_env/lib/python3.10/site-packages/torch/cuda/__init__.py", line 319, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

