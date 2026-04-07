# InducedFitNet

**Protein Conformational Adaptation Diffusion Model**

InducedFitNet is a joint SE(3) diffusion model that co-diffuses ligand pose and protein backbone simultaneously, capturing induced-fit effects and cryptic pocket opening — including PPI pockets invisible to rigid-receptor docking.

## Key Capability

| Benchmark | Standard Docking | InducedFitNet |
|-----------|-----------------|---------------|
| NKG2D homodimer PPI (8EA6, 0–20% similarity) | ❌ Fails (rigid receptor) | ✅ Predicted |
| Cryptic pocket opening | ❌ Not modeled | ✅ Co-diffused |
| Induced fit (holo→apo) | ❌ Fixed backbone | ✅ Joint distribution |

## Architecture

```
Joint distribution:  p(R_protein ∈ SE(3)^L, x_ligand ∈ R^3)

Denoising network
├── Protein backbone score network   ← conditions on ligand position (cross-attention)
├── Ligand pose score network        ← conditions on protein frames (cross-attention)
└── Sidechain rotamer sampler        ← Cython-compiled, called 200× per generation

Diffusion integrator
└── Neural ODE via CuPy cupyx.scipy.integrate.odeint  (CUDA-managed, GC-free)
```

## Installation

```bash
# Clone
git clone https://github.com/yourorg/inducedfitnet.git
cd inducedfitnet

# Create environment
conda create -n ifnet python=3.10
conda activate ifnet

# Install dependencies
pip install -r requirements.txt

# Compile Cython extension
cd inducedfitnet/cython_ext
python setup.py build_ext --inplace
cd ../..

# Install package
pip install -e .
```

### Requirements

- Python ≥ 3.10
- CUDA ≥ 11.8
- CuPy (matching CUDA version)
- PyTorch ≥ 2.0
- BioPython, e3nn, einops

## Quick Start

```python
from inducedfitnet import InducedFitNet
from inducedfitnet.data import ProteinLigandComplex

# Load receptor + ligand
complex_ = ProteinLigandComplex.from_pdb("8EA6.pdb", ligand_sdf="ligand.sdf")

# Load model
model = InducedFitNet.from_pretrained("checkpoints/ifnet_v1.pt")

# Sample — co-diffuses protein backbone + ligand pose over 200 steps
poses = model.sample(complex_, n_samples=10, n_steps=200)

# poses[0].ligand_coords  -> (N_atoms, 3)  predicted ligand coordinates
# poses[0].backbone_coords -> (L, 4, 3)   N, CA, C, O per residue
```

## Training

```bash
python scripts/train.py \
    --config configs/train_default.yaml \
    --data_dir /data/pdbbind \
    --output_dir checkpoints/
```

## Repository Structure

```
inducedfitnet/
├── inducedfitnet/
│   ├── models/
│   │   ├── score_network.py       # Joint SE(3) denoising network
│   │   ├── cross_attention.py     # Ligand↔protein cross-attention
│   │   ├── protein_encoder.py     # IPA / backbone frame encoder
│   │   └── ligand_encoder.py      # Atom-level ligand encoder
│   ├── diffusion/
│   │   ├── se3_diffusion.py       # SE(3) diffusion process (protein)
│   │   ├── r3_diffusion.py        # R³ diffusion process (ligand)
│   │   ├── schedule.py            # Noise schedules
│   │   └── ode_integrator.py      # CuPy neural ODE integrator
│   ├── data/
│   │   ├── complex.py             # ProteinLigandComplex dataclass
│   │   ├── dataset.py             # PDBBind / custom dataset
│   │   └── featurizer.py          # Atom & residue featurization
│   ├── cython_ext/
│   │   ├── rotamer.pyx            # Cython rotamer sampler
│   │   └── setup.py               # Cython build
│   └── utils/
│       ├── geometry.py            # SO(3) / SE(3) math utilities
│       ├── frames.py              # Backbone rigid-frame construction
│       └── metrics.py             # RMSD, pocket volume, success rate
├── scripts/
│   ├── train.py
│   ├── sample.py
│   └── evaluate.py
├── configs/
│   └── train_default.yaml
└── tests/
```

## Citation

```bibtex
@software{inducedfitnet2024,
  title   = {InducedFitNet: Protein Conformational Adaptation Diffusion Model},
  year    = {2024},
}
```

## License

MIT
