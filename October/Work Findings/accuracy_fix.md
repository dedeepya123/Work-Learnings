# VoiceDM On-Device Accuracy Investigation — Summary

## Problem statement
Running the VoiceDM TTS pipeline with the encoder, decoder, and SpectroStream decoder all executing on-device (Qualcomm HTP)
produced broken/garbled audio, while the same pipeline running on host (FP32 / QuantSim) produced correct audio. Goal: find why,
and bring on-device accuracy to an acceptable level relative to host.

## Investigation approach
Used the device-multi-tap-bsearch skill to binary-search the ONNX computation graph by "generation" (dependency depth), tapping
internal tensors and comparing host vs. on-device values at each point, to localize exactly where device output first diverges from
host.
First false lead (corrected): Initially assumed the decoder was broken, based on an input sample that turned out to be a leftover
calibration-time dummy tensor (torch.randn noise), not real pipeline data. Rebuilt the real input by actually running the encoder
on the test sentence, and reran the comparison — the decoder turned out to be clean on its own. Re-ran a full 3-stage
host-vs-device comparison (encoder → decoder → SpectroStream decoder) and found the encoder was the actual source: its output
(cond) was already badly wrong before the decoder even touched it.

## Bug 1: Attention-mask constant overflows fp16

Where: VoiceDMEncoder.forward() in gvoicedm/voicedm_modules/encoders.py, used by the speaker_encoder, audio_ctx_encoder, and
text_encoder sub-modules.

## What's wrong: Padding masks for attention are built as:
mask = paddings * get_large_negative_number(dtype)   # ≈ -0.7 * finfo(float32).max ≈ -2.38e38
This value is added to attention scores so padded positions get ~zero probability after softmax. It's a valid FP32 number (FP32's
max is ~3.4e38), so it works fine on host. But the encoder compiles to fp16 for on-device execution, and fp16's max representable
value is only ~65,504. Casting -2.38e38 to fp16 doesn't just lose precision — it overflows to actual -Infinity, which then poisons
every downstream computation it touches (softmax, residual adds, etc.) with Inf/NaN.

This was especially severe in our case because the real TTS use case never provides a speaker prompt or audio context —
speaker_paddings/audio_ctx_paddings are fully padded, so this overflow fires on essentially every element of those two branches,
every single call.

Why this wasn't already fixed project-wide: The exact same bug pattern already existed once in the codebase — for the decoder's
cross-attention mask — and was already fixed there (DiTDecoder override using -10000.0, with a comment describing this exact
failure mode). It had simply never been applied to the encoder's three masks (text_mask, speaker_mask, audio_ctx_mask).

Fix: Added a VoiceDMEncoder override in voicedm/qvoicedm/qadaptations.py (same monkey-patch pattern as the existing decoder fix)
that builds all three masks using -10000.0 instead of get_large_negative_number(...). -10000 is still "effectively negative
infinity" for softmax purposes (any real attention score is orders of magnitude smaller), but it's well within fp16's safe range,
so it never overflows.

## Bug 2: RMSNorm's variance computation overflows fp16

Where: RMSNorm.forward() in gvoicedm/voicedm_modules/layers.py, used throughout every transformer layer in text_encoder,
speaker_encoder, and audio_ctx_encoder.

What's wrong: Standard RMSNorm computes:
  variance = x.pow(2).mean(-1, keepdim=True)   # mean of x squared
  normed = x * (variance + eps) ** -0.5 * (1 + weight)
  This model's attention output projections legitimately produce large activation values — we measured attn_out_proj_MatMul output
  reaching into the thousands (e.g. ~6000) at just the first text-encoder layer. Squaring a value like 6000 gives 36,000,000 — again,
  fine for FP32 (host), but it blows straight past fp16's ~65,504 ceiling on-device, overflowing to Inf inside the very first
  normalization layer. This is a different kind of bug than the mask issue: it's not a bad constant, it's the model's own natural
  activation magnitudes outgrowing fp16's dynamic range.

  Fix: Added an RMSNorm override in qadaptations.py that computes the same math in an overflow-safe way: normalize by the
  per-position max absolute value before squaring, so the squared term is always bounded in [-1, 1] regardless of the input's actual
  magnitude, then rescale afterward. Verified numerically (unit-tested against the original formula across normal/large/zero/tiny
  inputs) that this produces bit-identical results up to ordinary float rounding noise (~5e-7), while guaranteeing no intermediate
  value can overflow fp16.

  Supporting fix: per-model compile precision

  voicedm/run_pipeline.py hardcoded a single global precision (a16w8) for compiling all three models, but the encoder and
  SpectroStream decoder are actually meant to compile as fp16 (no int quantization), only the decoder uses a16w8. Added a
  _MODEL_PRECISION dict so each model compiles with its correct, intended precision instead of accidentally being forced to the wrong
  one when re-running the pipeline.

  Files changed

  - voicedm/qvoicedm/qadaptations.py — added RMSNorm and VoiceDMEncoder override classes; wired both into _apply_qc_adaptations().
  - voicedm/run_pipeline.py — replaced the single global _PRECISION constant with a per-model _MODEL_PRECISION dict
  (VoiceDMEncoder→fp16, VoiceDMDecoder→a16w8, SoundStreamDecoder→fp16).

  Verification / result

  Re-exported and recompiled just the encoder with both fixes, then re-ran the full 3-stage host-vs-device comparison on the real
  "Hello world. This is a test." input:

  ┌──────────────────────────┬──────────────────────┬─────────────────┬─────────────┐
  │          Tensor          │   Before fix (NSR)   │ After fix (NSR) │ Improvement │
  ├──────────────────────────┼──────────────────────┼─────────────────┼─────────────┤
  │ cond (encoder output)    │ ~2.4 – 4.4 (broken)  │ ~5×10⁻⁴         │ ~5,000x     │
  ├──────────────────────────┼──────────────────────┼─────────────────┼─────────────┤
  │ pred_x0 (decoder output) │ ~0.6 – 0.98 (broken) │ ~2–5×10⁻⁴       │ ~2,000x     │
  └──────────────────────────┴──────────────────────┴─────────────────┴─────────────┘

  Finally ran the actual end-to-end pipeline (qbuilder.py) with all three stages on-device (fixed encoder bin + existing decoder bin
  + existing SSD bin), produced out_fp_qs_fp_device_fixed.wav, and confirmed by listening that the audio now sounds correct/perfect —
  matching the quality of the known-good host reference.


