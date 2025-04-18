from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UNet3DConditionModel
from easydict import EasyDict as edict
from torch import Tensor, nn
from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan

from ..scheduler.scheduling_ddpm_extcuda import DDPMScheduler
from .mel_extractor import AudioLDMLogMelExtractor, spectral_normalize


@dataclass
class PromptEmbs:
    prompt_embeds: Tensor
    attention_mask: Union[None, Tensor] = None
    generated_prompt_embeds: Union[None, Tensor] = None


class LatentAudioDiffusion(nn.Module):
    vae: AutoencoderKL
    scheduler: DDPMScheduler
    unet: UNet2DConditionModel
    mel_extractor: Union[SpeechT5FeatureExtractor, AudioLDMLogMelExtractor]
    vocoder: SpeechT5HifiGan

    sampling_rate: int
    is_mel_channel_last: bool
    default_guidance_scale: float

    def __init__(self, default_guidance_scale: float = 1.0, *args, **kwargs):
        super().__init__()
        self.max_length = 10
        self.is_mel_channel_last = True
        self.default_guidance_scale = default_guidance_scale

    def enable_finetune(self, enable: bool):
        self.unet.requires_grad_(enable)

    def get_latents_from_wave(self, audio: Tensor):
        # raw audio wave -> mel spectrogram
        if isinstance(self.mel_extractor, SpeechT5FeatureExtractor):
            mel_feats = self.mel_extractor(
                audio_target=audio.cpu().numpy().astype(np.float32),
                sampling_rate=self.vocoder.config.sampling_rate,
                max_length=self.vocoder.config.sampling_rate // 160 * self.max_length,
                truncation=True,
            )["input_values"]
            mel_feats = torch.from_numpy(mel_feats).to(audio.device, dtype=audio.dtype)
            # Note: SpeechT5FeatrueExtractor computes stft spec by log10 but AudioLDM and AudioLDM2 supports log.
            log_mel_feats = spectral_normalize(10**mel_feats)
        elif isinstance(self.mel_extractor, AudioLDMLogMelExtractor):
            log_mel_feats, _ = self.mel_extractor(audio)

        # shape: [B, T, M]

        # mel -> latent
        latents = self.vae.encode(log_mel_feats.unsqueeze(1)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents

    def get_noisy_latents(
        self,
        timesteps,
        audio: Union[Tensor, None] = None,
        latents: Union[Tensor, None] = None,
        generator=None,
    ):
        assert (audio is not None) or (latents is not None)
        device = audio.device if audio is not None else latents.device

        # raw wave audio -> latent
        if latents is None:
            latents = self.get_latents_from_wave(audio)

        # latent -> noisy latent
        noise = torch.randn(latents.shape, device=device, generator=generator)
        noisy_latent = self.scheduler.add_noise(latents, noise, timesteps)

        return edict(
            {
                "latent": latents,
                "noisy_latent": noisy_latent,
                "noise": noise,
            }
        )

    def sample_init_latents(
        self, batch_size, target_length_in_sec, dtype, device, generator=None
    ):
        # compute vocoder latent dims
        voc_conf = self.vocoder.config
        voc_upsample_factor = np.prod(voc_conf.upsample_rates)
        num_audio_samples = target_length_in_sec * voc_conf.sampling_rate
        voc_latent_time_dims = int(np.ceil(num_audio_samples / voc_upsample_factor))
        voc_latnet_channel_dims = voc_conf.model_in_dim

        # compute vae latent dims
        vae_conf = self.vae.config
        vae_upsample_factor = 2 ** (len(vae_conf.block_out_channels) - 1)
        vae_latent_time_dims = int(np.ceil(voc_latent_time_dims / vae_upsample_factor))
        vae_latent_channel_dims = voc_latnet_channel_dims // vae_upsample_factor

        # sample latents
        shape = (
            batch_size,
            self.unet.config.in_channels,  # channel for unet
            vae_latent_time_dims,  # time
            vae_latent_channel_dims,  # n_mels
        )
        init_scale = self.scheduler.init_noise_sigma
        latents = init_scale * torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )

        return latents

    def decode_latents(self, latents, to_int16, **kwargs):
        # latents -> mel spec
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample

        # mel spec -> wave
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()

        if to_int16:
            MAX_WAV_VALUE = 32768.0
            waveform = (waveform * MAX_WAV_VALUE).numpy().astype(np.int16)

        return waveform

    def convert_model_prediction_to_noise(
        self, model_output: torch.Tensor, scheduler: DDPMScheduler
    ):
        if scheduler.config.prediction_type == "epsilon":
            if scheduler.config.variance_type in ["learned", "learned_range"]:
                model_output = model_output[:, :3]
            return model_output

        raise NotImplementedError(
            f"Only support noise prediction models"
            f"but the model is trained on prediction_type == '{scheduler.config.prediction_type}'"
        )

    def encode_prompt(
        self, prompt, device, do_classifier_free_guidance=False
    ) -> tuple[PromptEmbs, PromptEmbs]:
        raise NotImplementedError()

    def predict_noise(
        self, text_emb: PromptEmbs, noisy_sample: Tensor, timesteps: Tensor
    ):
        raise NotImplementedError()


class LatentVideoDiffusion(nn.Module):
    vae: AutoencoderKL
    unet: UNet3DConditionModel
    scheduler: DDPMScheduler

    default_guidance_scale: float

    def __init__(self, bs_micro: int = -1, default_guidance_scale: float = 1.0) -> None:
        super().__init__()

        self.bs_micro = bs_micro
        self.default_guidance_scale = default_guidance_scale

    def enable_finetune(self, enable: bool):
        self.unet.requires_grad_(enable)

    def get_latents_from_video(self, video: Tensor):
        B, C, T, H, W = video.shape
        video = video.permute((0, 2, 1, 3, 4))  # (B, T, C, H, W)
        latents = self.vae.encode(
            video.flatten(start_dim=0, end_dim=1)
        ).latent_dist.sample()  # (BT, C, H, W)
        latents = latents * self.vae.config.scaling_factor
        latents = latents.reshape((B, T) + latents.shape[1:]).permute(
            (0, 2, 1, 3, 4)
        )  # (B, C, T, H, W)

        return latents

    def get_noisy_latents(
        self, timesteps, video: Tensor = None, latents: Tensor = None, generator=None
    ):
        assert (video is not None) or (latents is not None)
        device = video.device if video is not None else latents.device

        # raw video frames -> latent
        if latents is None:
            latents = self.get_latents_from_video(video)

        # latent -> noisy latent
        noise = torch.randn(latents.shape, device=device, generator=generator)
        noisy_latent = self.scheduler.add_noise(latents, noise, timesteps)

        return edict({"latent": latents, "noisy_latent": noisy_latent, "noise": noise})

    def sample_init_latents(
        self,
        batch_size,
        num_frames,
        target_height,
        target_width,
        dtype,
        device,
        generator=None,
    ):
        # compute vae latent dims
        vae_conf = self.vae.config
        vae_upsample_factor = 2 ** (len(vae_conf.block_out_channels) - 1)

        # sample latents
        shape = (
            batch_size,
            self.unet.config.in_channels,  # channel for unet
            num_frames,
            target_height // vae_upsample_factor,  # height for latent
            target_width // vae_upsample_factor,  # width for latent
        )
        init_scale = self.scheduler.init_noise_sigma
        latents = init_scale * torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )

        return latents

    def decode_latents(self, latents: Tensor, to_uint8):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()

        if to_uint8:
            mean = torch.tensor([0.5, 0.5, 0.5], device=video.device).reshape(
                1, -1, 1, 1, 1
            )
            std = torch.tensor([0.5, 0.5, 0.5], device=video.device).reshape(
                1, -1, 1, 1, 1
            )
            # unnormalize back to [0,1]
            video = video.mul_(std).add_(mean)
            video.clamp_(0, 1)
            video = (video * 255).to(dtype=torch.uint8)

        return video

    def convert_model_prediction_to_noise(
        self, model_output: torch.Tensor, scheduler: DDPMScheduler
    ):
        if scheduler.config.prediction_type == "epsilon":
            if scheduler.config.variance_type in ["learned", "learned_range"]:
                model_output = model_output[:, :3]
            return model_output

        raise NotImplementedError(
            f"Only support noise prediction models"
            f"but the model is trained on prediction_type == '{scheduler.config.prediction_type}'"
        )

    def encode_prompt(
        self, prompt: List[str], device, do_classifier_free_guidance=False
    ):
        # todo: improve prompt engineering
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        return PromptEmbs(prompt_embeds=prompt_embeds), PromptEmbs(
            prompt_embeds=negative_prompt_embeds
        )

    def predict_noise(
        self, text_emb: PromptEmbs, noisy_sample: Tensor, timesteps: Tensor
    ):
        # prediction by unet
        b = noisy_sample.shape[0]
        bs_micro = (
            self.bs_micro if self.bs_micro is not None and self.bs_micro >= 1 else b
        )
        n_loops = 1 if bs_micro == b else (b + self.bs_micro - 1) // self.bs_micro
        pred = []
        for li in range(n_loops):
            pred_micro = self.unet(
                noisy_sample[li * bs_micro : (li + 1) * bs_micro],
                timesteps[li * bs_micro : (li + 1) * bs_micro],
                encoder_hidden_states=text_emb.prompt_embeds[
                    li * bs_micro : (li + 1) * bs_micro
                ],
                return_dict=False,
            )[0]
            pred.append(pred_micro)
        pred = torch.concat(pred, dim=0)

        # convert prediction output to noise estimation
        pred_noise = self.convert_model_prediction_to_noise(pred, self.scheduler)

        return pred_noise
