import numpy as np
import torch
from diffusers import (
    AudioLDM2Pipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from torch import Tensor
from transformers import CLIPImageProcessor, SpeechT5HifiGan, SpeechT5FeatureExtractor

from ..scheduler.scheduling_ddpm_extcuda import DDPMScheduler
from .auffusion import converter as au_converter
from .auffusion.auffusion_pipeline import AuffusionPipeline, Generator
from .base import LatentAudioDiffusion, PromptEmbs
from .mel_extractor import AudioLDMLogMelExtractor


class AudioLDM(LatentAudioDiffusion):
    def __init__(
        self,
        mel_extractor: AudioLDMLogMelExtractor,
        default_guidance_scale: float = 2.5,
    ):
        super().__init__(default_guidance_scale=default_guidance_scale)

        from diffusers import AudioLDMPipeline, UNet2DConditionModel

        model_id = "cvssp/audioldm-m-full"
        pipe: AudioLDMPipeline = AudioLDMPipeline.from_pretrained(
            model_id,
            #   torch_dtype=torch.float16
        )

        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # self.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.unet: UNet2DConditionModel = pipe.unet
        self.vae: AutoencoderKL = pipe.vae
        self.vocoder: SpeechT5HifiGan = pipe.vocoder
        self.mel_extractor: AudioLDMLogMelExtractor = mel_extractor

        self.vae.enable_slicing()

        self.pipe: AudioLDMPipeline = pipe

        # set text related encoders to move them to the device
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

    def encode_prompt(self, prompt, device, do_classifier_free_guidance=False):
        # todo: improve prompt engineering
        # AudioLDMPipeline._encode_prompt always return cat([negative_prompt, prompt])
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        neg_prompt_embeds = None
        if do_classifier_free_guidance:
            neg_prompt_embeds, prompt_embeds = prompt_embeds.chunk(2, dim=0)

        return PromptEmbs(prompt_embeds=prompt_embeds), PromptEmbs(
            prompt_embeds=neg_prompt_embeds
        )

    def predict_noise(
        self, text_emb: PromptEmbs, noisy_sample: Tensor, timesteps: Tensor
    ):
        # prediction by unet
        pred = self.unet(
            noisy_sample,
            timesteps,
            encoder_hidden_states=None,
            class_labels=text_emb.prompt_embeds,
            return_dict=False,
        )[0]

        # convert prediction output to noise estimation
        pred_noise = self.convert_model_prediction_to_noise(pred, self.scheduler)

        return pred_noise


class AudioLDM2(LatentAudioDiffusion):
    def __init__(
        self,
        mel_extractor: SpeechT5FeatureExtractor,
        default_guidance_scale: float = 2.5,
    ):
        super().__init__(default_guidance_scale=default_guidance_scale)

        from diffusers.pipelines.audioldm2.modeling_audioldm2 import (
            AudioLDM2UNet2DConditionModel,
        )

        pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            # torch_dtype=dtype
        )
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.unet: AudioLDM2UNet2DConditionModel = pipe.unet
        self.vae: AutoencoderKL = pipe.vae
        self.vocoder: SpeechT5HifiGan = pipe.vocoder
        self.mel_extractor: SpeechT5FeatureExtractor = mel_extractor

        self.vae.enable_slicing()

        self.val_scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        self.sampling_rate = mel_extractor.sampling_rate

        self.pipe: AudioLDM2Pipeline = pipe

        # set text related encoders to move them to the device
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.projection_model = pipe.projection_model
        self.language_model = pipe.language_model

    def encode_prompt(self, prompt, device, do_classifier_free_guidance=False):
        prompt_embeds, attention_mask, generated_prompt_embeds = (
            self.pipe.encode_prompt(
                prompt,
                device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        )

        neg_prompt_embeds = neg_attention_mask = neg_generated_prompt_embeds = None
        if do_classifier_free_guidance:
            neg_prompt_embeds, prompt_embeds = prompt_embeds.chunk(2, dim=0)
            neg_attention_mask, attention_mask = attention_mask.chunk(2, dim=0)
            neg_generated_prompt_embeds, generated_prompt_embeds = (
                generated_prompt_embeds.chunk(2, dim=0)
            )

        pos = PromptEmbs(
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            generated_prompt_embeds=generated_prompt_embeds,
        )

        neg = PromptEmbs(
            prompt_embeds=neg_prompt_embeds,
            attention_mask=neg_attention_mask,
            generated_prompt_embeds=neg_generated_prompt_embeds,
        )

        return pos, neg

    def predict_noise(
        self,
        text_emb: PromptEmbs,
        noisy_sample: Tensor,
        timesteps: Tensor,
    ):
        # prediction by unet
        pred = self.unet(
            noisy_sample,
            timesteps,
            encoder_hidden_states=text_emb.generated_prompt_embeds,
            encoder_hidden_states_1=text_emb.prompt_embeds,
            encoder_attention_mask_1=text_emb.attention_mask,
            return_dict=False,
        )[0]

        # convert prediction output to noise estimation
        pred_noise = self.convert_model_prediction_to_noise(pred, self.scheduler)

        return pred_noise


class Auffusion(LatentAudioDiffusion):
    def __init__(self, default_guidance_scale: float = 8.0):
        super().__init__(default_guidance_scale=default_guidance_scale)

        model_id = "auffusion/auffusion-full"
        pipe: AuffusionPipeline = AuffusionPipeline.from_pretrained(
            model_id, dtype=torch.float32, device="cpu"
        )

        # NOTE: Auffusion assume the audio shape (B, C, M, L) (the last axis is the time)
        self.is_mel_channel_last = False

        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # self.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.vae: AutoencoderKL = pipe.vae
        self.unet: UNet2DConditionModel = pipe.unet
        self.feature_extractor: CLIPImageProcessor = pipe.feature_extractor
        self.vocoder: Generator = pipe.vocoder

        self.vocoder.config.model_in_dim = self.vocoder.config.num_mels

        # self.vae.enable_slicing()

        self.pipe: AuffusionPipeline = pipe

        # set text related encoders to move them to the device
        self.tokenizer_list = pipe.tokenizer_list
        self.text_encoder_list = pipe.text_encoder_list

        # for padding
        self.zero_emb = None

    def create_zero_emb(self, device):
        # Auffusion needs 10.24 sec audio to process
        zero_audio = torch.zeros(
            (1, int(10.24 * self.vocoder.config.sampling_rate)), device=device
        )
        self.zero_emb = self.get_latents_from_wave(zero_audio)

    def to(self, device, dtype=None):
        super().to(device, dtype)

        self.vocoder.to(device, dtype)

        for text_encoder in self.text_encoder_list:
            text_encoder.to(device, dtype)

        self.pipe.to(device, dtype)

        return self

    def encode_prompt(self, prompt, device, do_classifier_free_guidance=False):
        # todo: improve prompt engineering
        # AudioLDMPipeline._encode_prompt always return cat([negative_prompt, prompt])
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        neg_prompt_embeds = None
        if do_classifier_free_guidance:
            neg_prompt_embeds, prompt_embeds = prompt_embeds.chunk(2, dim=0)

        return PromptEmbs(prompt_embeds=prompt_embeds), PromptEmbs(
            prompt_embeds=neg_prompt_embeds
        )

    def predict_noise(
        self, text_emb: PromptEmbs, noisy_sample: Tensor, timesteps: Tensor
    ):
        T = noisy_sample.shape[-1]
        if T != 128:  # Auffusion needs to have 10.24 sec audio -> 128 in latents
            if self.zero_emb is None:
                self.create_zero_emb(device=noisy_sample.device)
            pad_sample: Tensor = self.get_noisy_latents(
                timesteps,
                latents=self.zero_emb.repeat(len(noisy_sample), 1, 1, 1),
                generator=None,
            ).noisy_latent
            noisy_sample = torch.cat([noisy_sample, pad_sample[..., T:]], dim=-1)
            assert noisy_sample.shape[-1] == 128

        # prediction by unet
        pred = self.unet(
            noisy_sample,
            timesteps,
            encoder_hidden_states=text_emb.prompt_embeds,
            class_labels=None,
            return_dict=False,
        )[0]

        # convert prediction output to noise estimation
        pred_noise = self.convert_model_prediction_to_noise(pred, self.scheduler)

        return pred_noise[..., :T]

    def decode_latents(self, latents, to_int16=True, target_length_sec=None):
        assert to_int16

        # latents -> spec
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image, has_nsfw_concept = self.pipe.run_safety_checker(
            image, latents.device, latents.dtype
        )

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.pipe.image_processor.postprocess(
            image, output_type="pt", do_denormalize=do_denormalize
        )

        # Generate audio
        audios = []
        for img in image:
            spectrogram = au_converter.denormalize_spectrogram(img)
            audio = self.vocoder.inference(
                spectrogram,
                lengths=target_length_sec * self.vocoder.config.sampling_rate,
            )[0]
            audios.append(audio)

        return audios

    @staticmethod
    def pad_spec(spec, spec_length, pad_value=0):  # spec: [3, mel_dim, spec_len]
        assert spec_length % 8 == 0, "spec_length must be divisible by 8"
        import torch.nn.functional as F

        if spec.shape[-1] < spec_length:
            # pad spec to spec_length
            spec = F.pad(spec, (0, spec_length - spec.shape[-1]), value=pad_value)
        else:
            spec = spec[:, :, :spec_length]
        return spec

    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1].
        """
        if images.min() >= 0:
            return 2.0 * images - 1.0
        else:
            return images

    def sample_init_latents(
        self, batch_size, target_length_in_sec, dtype, device, generator=None
    ):
        # this should be (B, C, L, M)
        latents = super().sample_init_latents(
            batch_size, target_length_in_sec, dtype, device, generator=generator
        )

        # change it to (B, C, M, L)
        latents = latents.permute(0, 1, 3, 2)

        return latents

    def get_latents_from_wave(self, audio: Tensor):
        assert audio.ndim == 2

        # raw audio wave -> mel spectrogram
        audio_list = []
        spec_list = []
        spec_length = None
        for a in audio:
            a, s = au_converter.get_mel_spectrogram_from_audio(
                (a * au_converter.MAX_WAV_VALUE).cpu().numpy().astype(np.int16),
                device=audio.device,
            )
            if spec_length is None:
                spec_length = s.shape[-1]
            else:
                assert spec_length == s.shape[-1]

            norm_spec = au_converter.normalize_spectrogram(s)
            norm_spec = self.pad_spec(norm_spec, 1024)
            norm_spec = self.normalize(norm_spec)

            # spec: (B, M, T)

            audio_list.append(a)
            spec_list.append(norm_spec)

        audio = torch.stack(audio_list)
        norm_spec = torch.stack(spec_list)

        # mel -> latent: (B, C, M, T)
        latents = self.vae.encode(
            norm_spec.to(dtype=self.vae.dtype)
        ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # remove padded values once
        # note: padding is also performed in pred_noise
        vae_conf = self.vae.config
        vae_upsample_factor = 2 ** (len(vae_conf.block_out_channels) - 1)
        need_length = spec_length // vae_upsample_factor

        return latents[:, :, :, :need_length]
