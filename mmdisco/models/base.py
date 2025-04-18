from dataclasses import dataclass
from typing import Optional, Union
from functools import partial
import numpy as np
import torch
from torch import Tensor

from .diffusion import guidance
from .diffusion.base import PromptEmbs
from .residual_predictor.base import VAResidualPredictorBase


@dataclass
class GuidanceConfigMixin:
    guidance_scale: float = 1.0

    ch_guidance: bool = False
    lr: Optional[float] = None
    alpha: Optional[float] = None
    max_iters: Optional[int] = None
    exit_threshold_factor: Optional[float] = None

    @property
    def do_cf_guide(self):
        return self.guidance_scale != 0.0

    @property
    def noise_predictor_kls(self):
        if self.do_cf_guide and self.ch_guidance:
            return partial(
                guidance.RMSPropCHGuidanceNoisePredictor,
                lr=self.lr,
                alpha=self.alpha,
                max_iters=self.max_iters,
                exit_threshold_factor=self.exit_threshold_factor,
            )

        return partial(
            guidance.CFGuidanceNoisePredictor, guidance_scale=self.guidance_scale
        )


@dataclass
class AudioInferenceConfig(GuidanceConfigMixin):
    length_in_sec: float = 1.6


@dataclass
class VideoInferenceConfig(GuidanceConfigMixin):
    height: int = 256
    width: int = 256
    num_frames: int = 16


@dataclass
class BaseAudioOutputs:
    a_0: Tensor
    a_t: Tensor
    noise: Tensor
    pred_noise: Union[Tensor, None]
    text_embs: PromptEmbs


@dataclass
class BaseVideoOutputs:
    v_0: Tensor
    v_t: Tensor
    noise: Tensor
    pred_noise: Union[Tensor, None]
    text_embs: PromptEmbs


@dataclass
class MMInferenceConfig:
    batch_size: int
    sample_fn: str
    num_samples: int
    video_fps: int
    audio_sr: int

    joint: bool
    joint_scale_audio: float = 1.0
    joint_scale_video: float = 1.0


class JointDiffusionBase(torch.nn.Module):
    residual_predictor: Union[VAResidualPredictorBase, None]
    max_timesteps: int

    def __init__(self):
        super().__init__()

        self._generator = None
        self._rng = None
        self.initial_seed = 803

    def state_dict(self, keep_vars=False):
        # keep only residual_predictor's weight
        if self.residual_predictor is None:
            return {}
        return self.residual_predictor.state_dict(
            prefix="residual_predictor.", keep_vars=keep_vars
        )

    def set_seed(self, seed):
        self.initial_seed = seed

    @property
    def generator(self):
        if self._generator is None:
            try:
                self._generator = torch.Generator(device=self.device)
                # seed = self._generator.initial_seed()
                self._generator.manual_seed(self.initial_seed)
            except Exception as e:
                self._generator = None
                raise ValueError(f"Creating a new generator is failed due to `{e}`")

        return self._generator

    @property
    def rng(self):
        if self._rng is None:
            try:
                self._rng = np.random.RandomState(
                    seed=self.initial_seed + int(self.device.index)
                )
            except Exception as e:
                print(f"Creating a new rng is failed due to {e}")
                self._rng = None
                raise ValueError(f"Creating a new rng is failed due to {e}")

        return self._rng

    @staticmethod
    def expand_dims_except_batch(x, target, cast_dtype=True):
        if isinstance(x, (int, float)):
            return x

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        assert isinstance(x, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        assert x.shape[0] == target.shape[0]
        shape = (target.shape[0],) + tuple([1] * len(target.shape[1:]))

        return x.reshape(*shape).to(
            device=target.device, dtype=target.dtype if cast_dtype else x.dtype
        )

    @staticmethod
    def norm(x: Tensor):
        return x.flatten(start_dim=1).norm(dim=1)
