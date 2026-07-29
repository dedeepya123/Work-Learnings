# Gemma4Generator

## Purpose:
## Input :
Gemma4Generator acts as a wrapper/orchestrator around multiple Qualcomm-specific model components.

It is initialized with:
- QC Vision Encoder (VE)
- QC Audio Encoder (AE)
- QC Prefix model
- QC Decode model
- QC MTP model

The actual modules are created/retrieved through the Gemma4Context (`ctx`) methods and then injected into Gemma4Generator.

## Design Pattern:
- Dependency Injection / Composition
- Gemma4Generator composes multiple QC model components into a complete inference pipeline.
