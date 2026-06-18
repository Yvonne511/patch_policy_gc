import torch.nn as nn


class DummyEncoder(nn.Module):
    """Pass-through encoder for state-based observations.

    Expects input of shape (..., state_dim) and returns (..., 1, state_dim),
    matching the V P E convention used by patch-based encoders with n_patches=1.
    """

    def __init__(self, output_dim: int, n_patches: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.n_patches = n_patches

    def forward(self, x):
        # x: (..., state_dim)  →  (..., 1, state_dim)
        return x.unsqueeze(-2)
