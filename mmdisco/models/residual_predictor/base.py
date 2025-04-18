from abc import abstractmethod

import torch
from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn

from ..scheduler.scheduling_ddpm_extcuda import DDPMScheduler


@dataclass
class VAEmbs:
    video: Tensor
    audio: Tensor


@dataclass
class VAResidualPredictorOutputs:
    pred_video: Tensor
    pred_audio: Tensor

    # for discriminator style training
    logit: Optional[Tensor] = None
    logit_exp: Optional[Tensor] = None
    prob: Optional[Tensor] = None

    # gradient scale
    pred_grad_scale_audio: Tensor = None
    pred_grad_scale_video: Tensor = None


class VATimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, video, audio, emb: VAEmbs):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass


class VATimestepEmbedSequential(nn.Sequential, VATimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, video: Tensor, audio: Tensor, emb: VAEmbs):
        for layer in self:
            if isinstance(layer, VATimestepBlock):
                video, audio = layer(video, audio, emb)
            else:
                video, audio = layer(video, audio)
        return video, audio


class VAResidualPredictorBase(nn.Module):
    video_input_type: str
    audio_input_type: str

    video_noise_scheduler: DDPMScheduler = None
    audio_noise_scheduler: DDPMScheduler = None

    def set_noise_scheduler(self, video_scheduler, audio_scheduler):
        self.video_noise_scheduler = video_scheduler
        self.audio_noise_scheduler = audio_scheduler

    @property
    def video_input_channels_factor(self):
        if self.video_input_type == "cat_noise":
            return 2

        return 1

    @property
    def audio_input_channels_factor(self):
        if self.audio_input_type == "cat_noise":
            return 2

        return 1

    def setup_inputs(
        self,
        video: Tensor,
        pred_video_noise: Tensor,
        audio: Tensor,
        pred_audio_noise: Tensor,
        timesteps: Tensor,
        video_channel_time_transpose: bool,
        audio_spatial_transpose: bool,
    ):
        # video should be (b, t, c, h, w), audio should be (b, c, l, m) for the input of main network

        if video_channel_time_transpose:
            # In this case, the input video has the shape of (b, c, t, h, w). need to perform transpose.
            video = video.permute(0, 2, 1, 3, 4)
            if pred_video_noise is not None:
                pred_video_noise = pred_video_noise.permute(0, 2, 1, 3, 4)

        if self.video_input_type == "cat_noise":
            assert pred_video_noise is not None
            video = torch.cat([video, pred_video_noise], dim=2)

        elif self.video_input_type == "pred_t0":
            assert pred_video_noise is not None
            assert self.video_noise_scheduler is not None

            video = self.video_noise_scheduler.step(
                pred_video_noise, timesteps.cpu(), video
            ).pred_original_sample

        else:
            assert self.video_input_type == "sample_only"

        if audio_spatial_transpose:
            # In this case, the input audio has the shape of (b, c, m, l). need to perform transpose.
            audio = audio.permute(0, 1, 3, 2)
            if pred_audio_noise is not None:
                pred_audio_noise = pred_audio_noise.permute(0, 1, 3, 2)

        if self.audio_input_type == "cat_noise":
            assert pred_audio_noise is not None
            audio = torch.cat([audio, pred_audio_noise], dim=1)

        elif self.audio_input_type == "pred_t0":
            assert pred_audio_noise is not None
            assert self.audio_noise_scheduler is not None

            audio = self.audio_noise_scheduler.step(
                pred_audio_noise, timesteps.cpu(), audio
            ).pred_original_sample

        else:
            assert self.audio_input_type == "sample_only"

        return video, audio

    @abstractmethod
    def forward(
        self,
        video: Tensor,
        pred_video_noise: Tensor,
        audio: Tensor,
        pred_audio_noise: Tensor,
        timesteps: Tensor,
    ) -> VAResidualPredictorOutputs:
        pass


class FeatureExtractorBase(nn.Module):
    out_channels: int

    @abstractmethod
    def forward(self, v: Tensor, a: Tensor, emb: VAEmbs) -> tuple[Tensor, Tensor]:
        pass
