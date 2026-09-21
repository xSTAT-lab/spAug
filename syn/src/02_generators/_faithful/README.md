# Published-design Python generator implementations

This directory stores canonical Python implementations of published generator
mechanisms before they are copied or wrapped by task-specific pipelines.

Current canonical cores:

- `srtsim_core.py`: SRTsim tissue/domain fitting and rank-preserving count generation.
- `splatter_core.py`: Splatter Splat single-population parameter chain with region simAdaptor.
- `sparsim_core.py`: SPARsim-style Gamma-Multivariate-Hypergeometric composition simulator with region simAdaptor.

Validation helpers in the public release:

- `scripts/check_public_repo.py`: layout, path, compilation, and optional mechanism gate.
- `scripts/check_generator_entrypoint_contracts.py`: static contract checks for every task-local generator entry point.
- `scripts/check_deep_generator_task_cli_smoke.py`: task-entry smoke test for `scGAN` and `scDiffusion`. It creates a tiny DLPFC supervised AnnData file, runs representative `train.py -> generate.py` commands on CPU, and checks synthetic h5ad schema, label provenance, spatial coordinates, and published-design provenance metadata.
- `scripts/check_srtsim_faithful_core.py`: tiny synthetic checks for SRTsim all-zero fitted-gene rescue behavior and rank-preserving output schema.
- `scripts/check_splatter_faithful_core.py`: tiny synthetic checks for the Splatter Python core region wrapper, count schema, spatial sampling, and serialization.
- `scripts/check_sparsim_faithful_core.py`: tiny synthetic checks for the SPARsim GMH/composition core, integer count schema, spatial condition wrapper, and serialization.

Task-local implementations:

- `*/scDiffusion/`: task-local implementation follows the published
  `EperLuo/scDiffusion` design. The local code uses a published-design 1024x3
  VAE, `Cell_Unet` diffusion backbone, `Cell_classifier`-style guidance
  classifier, classifier training on `t=0..T/2`, and sampling guidance gated
to `t<=T/2` with `beta_t * guidance_scale` weighting.
- `*/scGAN/`: task-local WGAN-GP implementation follows the published
  `imsb-uke/scGAN` design. The conditional path uses cscGAN-style conditional
  batch normalization in the generator and a projected conditional critic.

Rules:

- Implementations follow the published model descriptions or the SpatialSimBench call contract.
- Task-specific generator folders share the canonical model logic.
- Formal outputs record `generator_backend=python_reimplementation`.
- Published package names identify runs that call a published package or repository implementation.
