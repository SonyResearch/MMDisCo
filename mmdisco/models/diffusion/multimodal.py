import io
import os
from dataclasses import asdict, dataclass
from typing import List

import blobfile as bf
import torch as th
from einops import rearrange, repeat
from torch import nn

from .mmdiffusion.multimodal_script_util import create_model_and_diffusion
from .mmdiffusion.script_util import image_sr_create_model_and_diffusion


@dataclass
class ModelFlags:
    video_size: List
    audio_size: List
    num_channels: int
    num_res_blocks: int
    num_heads: int
    num_heads_upsample: int
    num_head_channels: int
    cross_attention_resolutions: str
    cross_attention_windows: str
    cross_attention_shift: bool
    video_attention_resolutions: str
    audio_attention_resolutions: str
    channel_mult: str
    dropout: int
    class_cond: bool
    use_checkpoint: bool
    use_scale_shift_norm: bool
    resblock_updown: bool
    use_fp16: bool
    video_type: str
    audio_type: str


@dataclass
class DiffusionFlags:
    learn_sigma: bool = False
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False


class MMDiffusion(nn.Module):
    def __init__(
        self, model_flags: ModelFlags, diffusion_flags: DiffusionFlags, ckpt_path: str
    ):
        super().__init__()

        self.model_flags = model_flags
        self.diffusion_flags = diffusion_flags

        multimodal_model, multimodal_diffusion = create_model_and_diffusion(
            **asdict(model_flags), **asdict(diffusion_flags)
        )

        self.multimodal_model = multimodal_model
        self.multimodal_diffusion = multimodal_diffusion

        # load model weight from ckpt
        assert os.path.exists(ckpt_path)
        with bf.BlobFile(ckpt_path, "rb") as f:
            data = f.read()

        multimodal_model.load_state_dict_(
            th.load(io.BytesIO(data), map_location="cpu"), is_strict=True
        )

    def forward(self, video, audio, timestep):
        timestep = self.multimodal_diffusion._scale_timesteps(timestep)
        return self.multimodal_model(video, audio, timestep)


@dataclass
class ImageSRModelFlags:
    large_size: int
    small_size: int
    num_channels: int
    num_res_blocks: int
    class_cond: bool
    use_checkpoint: bool
    attention_resolutions: List[int]
    num_heads: int
    num_head_channels: int
    num_heads_upsample: int
    use_scale_shift_norm: bool
    dropout: float
    resblock_updown: bool
    use_fp16: bool


class MMDiffusionImageSR(nn.Module):
    def __init__(
        self,
        model_flags: ImageSRModelFlags,
        diffusion_flags: DiffusionFlags,
        sample_fn: str,
        ckpt_path: str,
    ):
        super().__init__()

        self.model_flags = model_flags
        self.diffusion_flags = diffusion_flags
        self.sample_fn = sample_fn

        sr_model, sr_diffusion = image_sr_create_model_and_diffusion(
            **asdict(model_flags), **asdict(diffusion_flags)
        )

        self.sr_model = sr_model
        self.sr_diffusion = sr_diffusion

        # load model weight from ckpt
        assert os.path.exists(ckpt_path)
        with bf.BlobFile(ckpt_path, "rb") as f:
            data = f.read()

        self.sr_model.load_state_dict_(
            th.load(io.BytesIO(data), map_location="cpu"), is_strict=True
        )

    def forward(self, video):
        # stack frames to form a batch and sample noise
        b, t, c, h, w = video.shape
        video = rearrange(video, "b t c h w -> (b t) c h w")
        noise = th.randn(
            (b, c, self.model_flags.large_size, self.model_flags.large_size),
            dtype=video.dtype,
            device=video.device,
        )
        noise = repeat(noise, "b c h w -> (b t) c h w", t=t)

        # perform super-resolution
        if self.sample_fn == "ddim":
            sr_video = self.sr_diffusion.ddim_sample_loop(
                self.sr_model,
                shape=(
                    b * t,
                    c,
                    self.model_flags.large_size,
                    self.model_flags.large_size,
                ),
                noise=noise,
                clip_denoised=True,
                model_kwargs={"low_res": video},
                progress=True,
            )

        return rearrange(sr_video, "(b t) c h w -> b t c h w", b=b, t=t)
