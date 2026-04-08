"""
InducedFitNet: Protein Conformational Adaptation Diffusion Model.

Joint SE(3) diffusion over protein backbone (R_protein ∈ SE(3)^L)
and ligand pose (x_ligand ∈ R^3), enabling induced-fit docking and
cryptic pocket prediction without fixing the receptor.
"""

from inducedfitnet.models.score_network import JointScoreNetwork
from inducedfitnet.diffusion.se3_diffusion import SE3Diffusion
from inducedfitnet.diffusion.r3_diffusion import R3Diffusion
from inducedfitnet.data.complex import ProteinLigandComplex

__version__ = "0.1.0"
__all__ = [
    "JointScoreNetwork",
    "SE3Diffusion",
    "R3Diffusion",
    "ProteinLigandComplex",
    "InducedFitNet",
]


class InducedFitNet:
    """
    High-level API for InducedFitNet inference.

    Wraps the joint score network and diffusion processes into a single
    object suitable for sampling conformations from an apo receptor + ligand.
    """

    def __init__(
        self,
        score_network: JointScoreNetwork,
        se3_diffusion: SE3Diffusion,
        r3_diffusion: R3Diffusion,
        device: str = "cuda",
    ):
        self.score_network = score_network.to(device)
        self.se3_diffusion = se3_diffusion
        self.r3_diffusion = r3_diffusion
        self.device = device

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cuda") -> "InducedFitNet":
        """Load model from a saved checkpoint."""
        import torch
        from omegaconf import OmegaConf

        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = OmegaConf.create(ckpt["config"])

        score_network = JointScoreNetwork(cfg.model)
        score_network.load_state_dict(ckpt["model_state_dict"])
        score_network.eval()

        se3_diff = SE3Diffusion(cfg.diffusion)
        r3_diff = R3Diffusion(cfg.diffusion)

        return cls(score_network, se3_diff, r3_diff, device=device)

    def sample(
        self,
        complex_: "ProteinLigandComplex",
        n_samples: int = 10,
        n_steps: int = 200,
        temperature: float = 1.0,
    ) -> list:
        """
        Run reverse diffusion to generate n_samples joint conformations.

        Returns a list of ProteinLigandComplex objects with predicted coords.
        """
        from inducedfitnet.diffusion.ode_integrator import CuPyODEIntegrator

        integrator = CuPyODEIntegrator(
            score_fn=self.score_network,
            se3_diffusion=self.se3_diffusion,
            r3_diffusion=self.r3_diffusion,
            n_steps=n_steps,
            temperature=temperature,
        )
        return integrator.sample(complex_, n_samples=n_samples)
