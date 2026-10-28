# VoiceDM 1.1 Drop: Architectural Diff Notes

**Scope:** Comparing the new drop at `/prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/voicedm_1.1/` (code + checkpoints + calibration pickle) against the `gvoicedm` package currently vendored in this repo (`voicedm/gvoicedm/`), which backs all the accuracy and perf adaptation work done so far.

**Method:** Direct `diff -rq` + `diff -u` across every file in both module trees (every file differs — confirmed line-by-line below), plus direct `torch.load()` inspection of both checkpoints' `state_dict` keys to distinguish "just a rename" from "actual new tensors/behavior."

---

## 1. Package rename (cosmetic, zero behavior change)

Every single file differs at the top because the package was renamed `gvoicedm` → `voicedm` (e.g. `from gvoicedm.voicedm_modules import craft_modules` → `from voicedm.voicedm_modules import craft_modules`). This alone accounts for 100% of the diff in:

- `voicedm_modules/layers.py`
- `voicedm_modules/encoders.py`
- `voicedm_decoder.py`, `voicedm_encoder.py`, `voicedm_inference.py`
- `soundstream_decoder.py`, `soundstream_encoder.py`
- `soundstream_modules/blocks.py`, `soundstream_modules/soundstream.py`

`voicedm_modules/decoder.py` has one extra one-line change: a stray `# pyrefly: ignore[unsupported-operation]` lint-suppression comment was removed above `DiffusionReverse`'s final `return`. No logic change.

**None of these files have any architectural/behavioral difference.** `RMSNorm`, `MultiHeadAttention`, `RotaryPositionEmbedding`, `TransformerEncoderLayer`, `ContinuousEncoder`, `TextEncoder`, `VoiceDMEncoder`, `VoiceDMDecoder`, `DiTDecoder`, `DiTBlock`, `BandSplitResidualUnit2d` are all byte-identical in logic to what our `qadaptations.py` patches today.

**Implication:** all of our monkey-patch overrides in `qadaptations.py` (`RMSNorm`, `VoiceDMEncoder`, `MultiHeadAttention`, `DiTDecoder`, `DiTBlock`, `BandSplitResidualUnit2d`, `TransformerEncoderLayer`, `ContinuousEncoder`, `TextEncoder`) subclass unchanged base classes and remain structurally valid against the new source. The accuracy fixes (masking-overflow, native-RmsNorm) and perf changes (Conv-Linear, 4D carrier, RoPE) do **not** need to be redesigned for architecture reasons.

**Action needed regardless:** our `qadaptations.py` currently imports `from gvoicedm...` everywhere. If/when this repo switches to the 1.1 drop's package, every `gvoicedm` import in `qadaptations.py` (and `qmain.py`, `qbuilder.py`, `qgenerator.py`, `onnx_adaptation.py`, etc. — anywhere referencing the old package name) must be updated to `voicedm`. This is mechanical, not architectural.

---

## 2. Real architectural change: `craft_modules.Linear` → `QuantizableLinear`

This is the one substantive change, in `voicedm_modules/craft_modules.py`.

### 2.1 What changed

**Old (`gvoicedm`, what our code is built against):**
```python
class Linear(nn.Linear):
  """Linear projection with CRAFT output fake quantization."""
  def __init__(self, in_features, out_features, bias=True, craft_config=None):
    super().__init__(in_features, out_features, bias=bias)
    self.craft_config = craft_config
    self.output_sfq = StaticFakeQuant(craft_config=craft_config)

  def forward(self, x):
    result = super().forward(x)
    return self.output_sfq(result)
```
One optional `output_sfq` (`StaticFakeQuant`) submodule. No-op when `craft_config=None` (always the case in our pipeline — confirmed `qbuilder.py` never passes `craft_config`).

**New (`voicedm_1.1`):**
```python
class QuantizableLinear(nn.Linear):
  """Quantizable Linear layer for VoiceDM supporting mixed-precision W/A."""
  def __init__(self, in_features, out_features, bias=True, craft_config=None):
    super().__init__(in_features, out_features, bias=bias)
    self.craft_config = craft_config
    self.register_buffer("input_scale", torch.ones((1,1,1)))
    self.register_buffer("input_bits", torch.tensor(0, dtype=torch.int64))
    self.register_buffer("output_scale", torch.ones((1,1,1)))
    self.register_buffer("output_bits", torch.tensor(0, dtype=torch.int64))
    self.register_buffer("weight_bits", torch.tensor(0, dtype=torch.int64))
    self.is_logging = False  # ephemeral, not in state_dict

  def forward(self, x):
    if self.is_logging:
      # record abs-max for calibration, run unquantized
      ...
      return out
    in_bits = int(self.input_bits.item())
    if in_bits > 0:
      x = fake_quant_activation(x, self.input_scale, in_bits)
    out = F.linear(x, self.weight, self.bias)
    out_bits = int(self.output_bits.item())
    if out_bits > 0:
      out = fake_quant_activation(out, self.output_scale, out_bits)
    return out

Linear = QuantizableLinear  # alias, so call sites (q_proj = craft_modules.Linear(...)) are unchanged
```
`QuantizableLinear._load_from_state_dict` explicitly strips legacy `output_sfq.min_val`/`output_sfq.max_val` keys if present (backward-compat with old-style checkpoints) and back-fills the five new buffers with defaults if absent (forward-compat with unquantized checkpoints).

### 2.2 Why this exists: per-layer mixed-precision quantization, independent of CRAFT

This is a **completely separate quantization mechanism** from `CraftConfig`/`StaticFakeQuant`. It uses:
- **Symmetric** (zero_point = 0) fake quantization, not the min/max-clip style `StaticFakeQuant` uses.
- **Per-layer, not uniform**: `quantization_utils.configure_quantizable_linears()` (new function) assigns FFN linears (`.ffn.0`, `.ffn.3`) a different bit-width than attention projections.
- **Baked into the checkpoint**, not computed at runtime from a `CraftConfig` flag.

### 2.3 Confirmed via direct checkpoint inspection

```
voicedm_encoder.pth: 2519 keys (old) -> 2963 keys (new), +444 keys
444 / 3 ≈ 148  ->  matches exactly the 148 Linear instances in the encoder
  (96 attention q/k/v/out + 48 FFN + 4 input/output projections -- same
  count our perf work's Conv-Linear replacement touches)
```
Actual bit-width values found in the checkpoint (not just placeholder zeros):
```
weight_bits distribution across 148 layers: {16: 200, 8: 196, 4: 48}
  -> attention q/k/v/out_proj:  weight_bits=8,  input_bits=16, output_bits=16
  -> FFN (ffn.0, ffn.3):        weight_bits=4,  input_bits=8,  output_bits=8
```
This is a **real W8A16 (attention) / W4A8 (FFN) mixed-precision scheme already calibrated and shipped in the checkpoint** — not dead/placeholder metadata.

Also confirmed the **weight tensors themselves differ** between old and new checkpoints (not just new metadata appended to the same weights) — e.g. `text_encoder.transformer.layers.0.attn.q_proj.weight` has a different hash and `torch.allclose() == False` between the two checkpoints, same shape. **This is a newly-trained/retrained model, not a re-export of the same weights.**

### 2.4 Direct impact on our adaptation code — confirmed broken

Our perf adaptation's `ConvLinearEnc` (in `qadaptations.py`) mimics the **old** `Linear` interface (`weight`, `bias`, `output_sfq` only). Tested directly:

```python
model.load_state_dict(new_voicedm_1.1_encoder_checkpoint)
# RuntimeError: Unexpected key(s) in state_dict:
#   "...q_proj.input_scale", "...q_proj.input_bits",
#   "...q_proj.output_scale", "...q_proj.output_bits", "...q_proj.weight_bits", ...
#   (repeated for every one of the 148 QuantizableLinear instances)
```
**`strict=True` load_state_dict fails immediately** against the new checkpoint, for both our `ConvLinearEnc` (perf path) and the plain `craft_modules.Linear` (accuracy-only path) — neither has the five new buffers `QuantizableLinear` requires.

### 2.5 Second-order question: is the baked-in quantization even desirable for our pipeline?

Our pipeline's own quantization (AIMET QuantSim / qairt-quantizer, driven through `qmain.py`/Stage A → E) is a **separate, independent quantization pass** from whatever `QuantizableLinear`'s `input_bits`/`output_bits`/`weight_bits` buffers encode. If we load this checkpoint as-is (with `in_bits>0`/`out_bits>0` active), the model would be **double-quantized**: once by `QuantizableLinear.forward()`'s baked-in fake-quant (W8A16/W4A8, calibrated on Google's own calibration set), and again by our own AIMET/QNN quantization pipeline (a16w8, calibrated on our own inputs). This is very likely undesirable — it would compound quantization error rather than letting our pipeline's calibration be the only source of precision loss.

**Likely correct approach** (not yet implemented, flagged for next step): load weights with the baked-in activation quantization **disabled** — i.e. set every `QuantizableLinear.input_bits`/`output_bits` to 0 after loading (keeping only the plain float weights, which can still be loaded since `weight`/`bias` have unchanged shapes), so the forward pass behaves like the old plain `nn.Linear` and our own AIMET/QNN pipeline remains the sole quantizer. Needs confirmation with the team on whether `weight_bits`-driven fake-quantization of the stored *weight values themselves* (as opposed to activations) was already baked into the saved `weight` tensors at save time (i.e. are the floats already fake-quantized to 8/4-bit precision, making the weights themselves lower-fidelity regardless of what we do with `input_bits`/`output_bits`) — this needs a direct check (see Open Items).

---

## 3. Calibration data: real speaker prompts, not always-padded

From `voicedm_1.1/README.md`:

> **Base model calibration**: Run calibration across all samples in `calibration/voicedm_calibration.pickle`, synthesizing each text prompt with its paired speaker prompt so that the speaker encoder branch is calibrated on real speaker conditioning rather than an empty (all-padding) prompt.

This is a direct, explicit acknowledgment (from the new drop's own documentation) of the exact same degenerate-input pattern our masking-overflow investigation flagged earlier: our prior testing (and seemingly Google's prior calibration) always used `speaker_prompt`/`audio_context` as fully padded (all-ones `speaker_paddings`/`audio_ctx_paddings`), which is what made `get_large_negative_number()`'s overflow bug maximally and uniformly exercised in every call. The new calibration set pairs each text prompt with a **real, non-padded speaker embedding** (`calibration/voicedm_calibration.pickle`, format: list of `[text, speaker_embedding]`, speaker_embedding shape `[frames<=83, 96]`).

**Implication for our masking fix:** our `VoiceDMEncoder` override's `-10000.0` mask fix was derived and verified only under the always-fully-padded case (documented as an open item in `ENCODER_FP16_ACCURACY_INVESTIGATION.md` §7: "Have not verified behavior with a real (non-padded) speaker prompt"). This new calibration set is the first concrete opportunity to close that open item — worth re-running the masking-overflow verification with a real (partially-active, partially-padded) speaker prompt from this pickle, since partial masks exercise a different code path (mix of 0 and -10000 values in the same mask tensor) than the all-same-value case tested so far.

---

## 4. Everything else: confirmed unchanged

Checked explicitly and confirmed **identical** (beyond the package-name import line):
- `RotaryPositionEmbedding`, `PerDimScale`, `MultiHeadAttention`, `get_large_negative_number()` — `voicedm_modules/layers.py`
- `TransformerEncoderLayer`, `TransformerEncoder`, `ContinuousEncoder`, `TextEncoder`, `MetaEncoder`, `VoiceDMEncoder` — `voicedm_modules/encoders.py`
- `VoiceDMDecoder`, `DiTDecoder`, `DiTBlock`, `DiffusionReverse`, `cosine_logsnr` — `voicedm_modules/decoder.py`
- `BandSplitResidualUnit2d`, `ResidualUnit2d` and all of `soundstream_modules/blocks.py`
- `SpectroStreamEncoder`, `SpectroStreamDecoder` and all of `soundstream_modules/soundstream.py`
- `soundstream_decoder.py`'s early/full-decode split and `xf_frames_to_wav` ISTFT logic
- `voicedm_inference.py`'s end-to-end orchestration, including the early-stop-on-silence logic
- All `StaticFakeQuant`, `CraftModule`, and the full catalog of CRAFT op wrappers (`Matmul`, `Mul`, `Add`, `Sin`, `Cos`, `Softmax`, `GELU`, etc.) in `craft_modules.py` — unchanged aside from the `Linear`→`QuantizableLinear` section
- `LayerNorm`, `Embedding`, `Dropout` in `craft_modules.py` — unchanged, still use the old `output_sfq`-only pattern (only `Linear` was special-cased)
- File set is identical — no files added or removed, confirmed via `diff -rq`

No new files, no removed files, no changes to tensor shapes/hyperparameters (`model_dims`, `num_heads`, `num_layers`, etc. all identical) anywhere outside of `Linear`.

---

## 5. Summary / what to do before pointing our pipeline at this checkpoint

| Area | Status | Action |
|---|---|---|
| Package import rename (`gvoicedm`→`voicedm`) | Mechanical only | Update imports across `qadaptations.py`, `qmain.py`, `qbuilder.py`, `qgenerator.py`, `onnx_adaptation.py`, etc. if/when switching to this drop |
| Accuracy-fix classes (`RMSNorm`, `VoiceDMEncoder` mask fix, etc.) | Structurally valid, no redesign needed | Re-verify numerically against new weights once loadable (see below) |
| Perf classes (`ConvLinearEnc`, 4D `TransformerEncoderLayer`/`ContinuousEncoder`/`TextEncoder`) | Structurally valid, no redesign needed | Same — re-verify once loadable |
| `craft_modules.Linear` → `QuantizableLinear` | **Breaking**: checkpoint load currently fails `strict=True` | `ConvLinearEnc` and the plain accuracy-only path both need an `input_scale`/`input_bits`/`output_scale`/`output_bits`/`weight_bits` buffer set added (matching shapes/dtypes) so `load_state_dict` succeeds — see Open Items for whether to then zero out `*_bits` to disable the baked-in fake-quant, or keep it active |
| Masking-overflow fix, all-padded vs. real speaker prompt | Open item, now testable | New calibration pickle has real speaker embeddings — re-run the fp16 overflow check with a real (partial) mask, not just all-padded |
| Model weights | **Different model** — retrained, not just re-exported | Full re-verification (NSR/listening) required once checkpoint is loadable; cannot assume old verification numbers carry over |

## 6. Open items / needs follow-up

1. **Are the stored `weight` tensors themselves already fake-quantized** to their `weight_bits` precision (i.e. is weight quantization baked into the `.pth` at save time, independent of the `input_bits`/`output_bits` activation quantization toggle), or are they full-precision floats with quantization applied only at `forward()` time via the separate `quantization_utils.apply_fake_quantization()` path (which is opt-in via `apply_fake_quant=True`, off by default in `voicedm_encoder.py`/`voicedm_decoder.py`)? This determines whether disabling `input_bits`/`output_bits` alone is sufficient to get "clean" float weights, or whether the weights need separate handling.
2. **Decide and confirm with the team**: should our pipeline disable `QuantizableLinear`'s baked-in activation quantization entirely (treat it as a plain `nn.Linear` with 5 extra inert buffers) and let AIMET/QNN be the sole quantizer, or is there a reason to keep Google's calibrated W8A16/W4A8 scheme active underneath our own a16w8 pass? Double-quantization risk makes "disable theirs, keep ours" the likely default, but this should be an explicit decision, not an assumption.
3. Once checkpoint loading is fixed, **full re-verification is required** (not optional) — this is confirmed to be different trained weights, so none of the previously-measured NSR numbers (encoder/decoder/ssd, fp16 or a16w8) can be assumed to hold.
4. Re-test the masking-overflow fix against a **real, partially-padded** speaker prompt from `calibration/voicedm_calibration.pickle`, closing the open item from `ENCODER_FP16_ACCURACY_INVESTIGATION.md` §7.
