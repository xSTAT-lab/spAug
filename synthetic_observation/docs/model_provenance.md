# Model provenance

The generator names in this repository identify published statistical or deep
generative ideas. The project-local implementations follow the corresponding
public descriptions and are exercised by toy mechanism checks.

| Generator | Local implementation | Main mechanism |
| --- | --- | --- |
| SRTsim | `src/02_generators/_faithful/srtsim_core.py` | Marginal count-family selection and rank-preserving sampling |
| Splatter | `src/02_generators/_faithful/splatter_core.py` | Region-aware Splat-style library, mean, BCV, and dropout simulation |
| SPARsim | `src/02_generators/_faithful/sparsim_core.py` | Gamma composition, multivariate hypergeometric sampling, and dropout |
| scGAN | task-local `scGAN/model.py` files | Conditional batch normalization and projection critic with WGAN-GP training |
| scDiffusion | task-local `scDiffusion/model.py` and `diffusion.py` files | VAE latent diffusion with optional classifier guidance |

The development source-review checkout is maintained separately from this
public distribution. Generator metadata records published project names and
source locations for clear provenance. Published projects retain their own
licenses; consult their terms when extending or redistributing related work.

## Validation scope

The synthetic mechanism checks exercise model structure, array shapes, count
constraints, coordinate variation, and checkpoint serialization. The CLI smoke
checks exercise representative CPU training and generation commands.
Dataset-level scientific validation uses the full workflow, the selected data,
and recorded runtime versions. Numerical equivalence to published results
requires model-specific validation under matching experimental conditions.
