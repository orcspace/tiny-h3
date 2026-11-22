"""pi-Flow sampling schedule (LakonLab ``FlowEulerODEScheduler`` with ``num_train_timesteps=1``).

Sigma is the noise level (1 = noise). MiniMax-H3's transformer takes ``t = 1 - sigma`` and predicts the negated
velocity; ``noise_pred = -output`` and the Euler step is ``x += noise_pred * (sigma_next - sigma)``.
"""
from __future__ import annotations

import numpy as np
import torch


def sigmas(steps: int, shift: float, max_raw_t: float, final_step_size_scale: float, min_raw_t: float = 0.0):
    raw = np.linspace(max_raw_t,
                      (max_raw_t - min_raw_t) * final_step_size_scale / (steps - 1 + final_step_size_scale)
                      + min_raw_t,
                      steps, dtype=np.float32)
    s = torch.from_numpy(raw)
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([s, torch.zeros(1)])


class FlowEuler:
    def __init__(self, steps: int, shift: float, max_raw_t: float, final_step_size_scale: float):
        self.sigmas = sigmas(steps, shift, max_raw_t, final_step_size_scale)
        self.steps = steps

    def model_t(self, i: int) -> float:
        """The transformer's timestep for step ``i``."""
        return 1.0 - float(self.sigmas[i])

    def step(self, i: int, noise_pred: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        dt = self.sigmas[i + 1] - self.sigmas[i]
        return (sample.float() + noise_pred.float() * dt).to(sample.dtype)
