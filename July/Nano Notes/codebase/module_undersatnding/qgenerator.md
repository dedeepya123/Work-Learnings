# Gemma4Generator

## Purpose:
## Input :
Gemma4Generator acts as a wrapper/orchestrator around multiple Qualcomm-specific model components.

It is initialized with:
- QC Vision Encoder (VE) ( returns vision encoder subgraph )
- QC Audio Encoder (AE)   ( returns Audio Encoder subgraph )
- QC Prefix model          ( returns QC text subgraph)
- QC Decode model         (return QC text subgraph)
- QC MTP model            (returns MTP subgraph)

The actual modules are created/retrieved through the Gemma4Context (`ctx`) methods and then injected into Gemma4Generator.

## Design Pattern:
- Dependency Injection / Composition
- Gemma4Generator composes multiple QC model components into a complete inference pipeline.
