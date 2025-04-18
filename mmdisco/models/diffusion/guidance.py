from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor


@dataclass
class NoisePredictorOutputs:
    pred_noise_uncond: Tensor
    pred_noise_cond: Tensor


class CFGuidanceNoisePredictor(object):
    guidance_scale: float

    def __init__(
        self,
        cond_predictor: Callable[[Tensor, Tensor], Tensor],
        uncond_predictor: Callable[[Tensor, Tensor], Tensor],
        guidance_scale: float,
    ) -> None:
        self.cond_predictor = cond_predictor
        self.uncond_predictor = uncond_predictor
        self.guidance_scale = guidance_scale

    def __call__(self, noisy_sample: Tensor, timesteps: Tensor, **g_args):
        pred_uncond = self.uncond_predictor(
            noisy_sample=noisy_sample, timesteps=timesteps
        )
        if self.guidance_scale == 0:
            # return unconditional prediction without guidance
            return pred_uncond

        pred_cond = self.cond_predictor(noisy_sample=noisy_sample, timesteps=timesteps)
        if self.guidance_scale == 1:
            # return conditional prediction without guidance
            return pred_cond

        # return (1 + self.guidance_scale) * pred_cond - self.guidance_scale * pred_uncond
        return pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)


class RMSPropCHGuidanceNoisePredictor(CFGuidanceNoisePredictor):
    def __init__(
        self,
        cond_predictor: Callable[[Tensor, Tensor], Tensor],
        uncond_predictor: Callable[[Tensor, Tensor], Tensor],
        guidance_scale: float,
        lr: float,
        alpha: float,
        max_iters: int,
        exit_threshold_factor: float,
    ):
        self.cond_predictor = cond_predictor
        self.uncond_predictor = uncond_predictor
        self.guidance_scale = self.w = guidance_scale

        # hyper parameters for FPI by RMSProp
        self.lr = lr
        self.alpha = alpha
        self.max_iters = max_iters
        self.eta = exit_threshold_factor

    def __call__(self, noisy_sample: Tensor, timesteps: Tensor, **g_args):
        # sigma = sqrt(1 - alpha_bar)
        assert "sigma" in g_args, "argument 'sigma' must be passed."

        delta = torch.zeros(
            noisy_sample.shape, requires_grad=True, device=noisy_sample.device
        )
        opt = torch.optim.RMSprop([delta], lr=self.lr, alpha=self.alpha)

        for cnt in range(self.max_iters + 1):
            opt.zero_grad()

            # x1 for conditional pass
            x1 = noisy_sample + self.w * delta
            eps1 = self.cond_predictor(x1, timesteps)

            # x2 for unconditional pass
            x2 = noisy_sample + (1 + self.w) * delta
            eps2 = self.uncond_predictor(x2, timesteps)

            # compute first order gradient
            g: Tensor = delta - (eps2 - eps1) * g_args["sigma"]

            # compute mask for the elems already converged
            norm_dims = list(range(1, g.ndim))
            numel = np.prod(g.shape[norm_dims])
            mask = g.norm(dim=norm_dims, keepdim=True) > (self.eta * (numel**0.5))
            g = g * mask

            if mask.sum() == 0 or cnt == self.max_iters:
                # All elems are converged or reach max iters.
                # Do guidance with eps1 and eps2.
                break

            # update delta
            delta.grad = g
            opt.step()

        return (1 + self.w) * eps1 - self.w * eps2
