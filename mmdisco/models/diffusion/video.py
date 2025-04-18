from typing import List, Union

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet3DConditionModel
from easydict import EasyDict as edict
from torch import Tensor, nn

from ..scheduler.scheduling_ddpm_extcuda import DDPMScheduler
from .base import LatentVideoDiffusion, PromptEmbs
from .videocrafter.models.ddpm3d import LatentDiffusion as VC_LatentDiffusion
from .videocrafter.utils import instantiate_from_config as instantiate_vc
from .videocrafter.utils import load_model_checkpoint as load_vc_ckpt


class ModelScope(LatentVideoDiffusion):
    def __init__(self, bs_micro: int = -1, default_guidance_scale=8.0) -> None:
        super().__init__(bs_micro, default_guidance_scale)

        from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import (
            TextToVideoSDPipeline,
        )

        # using ModelScope's Text2Video module
        pipe = TextToVideoSDPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",  # cspell:disable-line
            # torch_dtype=dtype,
            # variant="fp16" if dtype is torch.half else None
        )

        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.vae.enable_slicing()

        self.val_scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )


class ZeroScope(LatentVideoDiffusion):
    def __init__(self, bs_micro: int = -1, default_guidance_scale=8.0) -> None:
        super().__init__(bs_micro, default_guidance_scale)

        from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import (
            TextToVideoSDPipeline,
        )

        # using ModelScope's Text2Video module
        pipe = TextToVideoSDPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",  # cspell:disable-line
            # torch_dtype=dtype,
            # variant="fp16" if dtype is torch.half else None
        )

        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.unet: UNet3DConditionModel = pipe.unet
        self.vae: AutoencoderKL = pipe.vae
        self.vae.enable_slicing()

        self.val_scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        self.pipe: TextToVideoSDPipeline = pipe

        # set text related encoders to move them to the device
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder


class AnimateDiff(LatentVideoDiffusion):
    def __init__(
        self,
        bs_micro: int = -1,
        default_guidance_scale=7.5,
        use_linear_schedule: bool = False,
    ):
        super().__init__(bs_micro, default_guidance_scale)

        # load AnimateDiff
        # TODO:
        #    Since seems the model is continuously updated, using the latest one would improve generated video quality.
        #    To do that, we should use github implementation rather than diffusers. (loading v3_sd15_mm.ckpt to MotionAdapter doesn't work.)
        from diffusers import AnimateDiffPipeline, MotionAdapter, UNetMotionModel

        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",  # cspell:disable-line
            # torch_dtype=torch.float16
        )
        # load SD 1.5 based finetuned model
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        self.pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pretrained(
            model_id,
            motion_adapter=adapter,
            # torch_dtype=torch.float16
        )

        # linear is used in the paper
        # this scheduler is configured as clip_sample=False and thresholding=False
        self.scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_schedule="linear"
            if use_linear_schedule
            else self.pipe.scheduler.config.beta_schedule,
        )
        # self.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
        #                                                          algorithm_type="dpmsolver++",
        #                                                          solver_type="midpoint",
        #                                                          beta_schedule="linear" if use_linear_schedule \
        #                                                             else self.pipe.scheduler.config.beta_schedule
        #                                                          )

        self.vae: AutoencoderKL = self.pipe.vae
        self.unet: UNetMotionModel = self.pipe.unet
        self.vae.enable_slicing()

        # set text related encoders to move them to the device
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder


class VideoCrafter(nn.Module):
    def __init__(
        self,
        orig_conf,
        fps,
        ckpt_path,
        bs_micro: Union[int, None] = None,
        default_guidance_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.default_guidance_scale = default_guidance_scale

        # instantiate VideoCrafter and load weights
        vc = instantiate_vc(orig_conf)
        vc = load_vc_ckpt(vc, ckpt_path)
        self.vc: VC_LatentDiffusion = vc

        self.fps = fps
        self.bs_micro = bs_micro

        # scheduler is defined by diffusers API for compatibility
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.vc.num_timesteps,
            beta_start=self.vc.linear_start,
            beta_end=self.vc.linear_end,
            beta_schedule="linear",
            variance_type="fixed_small",
            prediction_type="epsilon",
            clip_sample=False,
            thresholding=False,
            steps_offset=1,  # make sure the last timestep is 1.
        )

    def enable_finetune(self, enable: bool):
        self.vc.model.requires_grad_(enable)

    def get_latents_from_video(self, video: Tensor):
        latents = self.vc.encode_first_stage_2DAE(video)
        return latents

    def get_noisy_latents(
        self,
        timesteps,
        video: Union[Tensor, None] = None,
        latents: Union[Tensor, None] = None,
        generator=None,
    ):
        assert (video is not None) or (latents is not None)
        device = video.device if video is not None else latents.device

        # raw video frames -> latent
        if latents is None:
            latents = self.get_latents_from_video(video)

        # latent -> noisy latent
        noise = torch.randn(latents.shape, device=device, generator=generator)
        noisy_latent = self.vc.q_sample(latents, timesteps, noise)

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
        h, w = target_height // 8, target_width // 8

        # sample latents
        shape = (
            batch_size,
            self.vc.channels,  # channel for unet
            num_frames,
            h,
            w,
        )
        latents = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        return latents

    def decode_latents(self, latents: Tensor, to_uint8):
        video = self.vc.decode_first_stage_2DAE(latents)
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
        prompt_embeds = self.vc.get_learned_conditioning(prompt)

        if do_classifier_free_guidance:
            if self.vc.uncond_type == "empty_seq":
                neg_prompt = len(prompt) * [""]
                negative_prompt_embeds = self.vc.get_learned_conditioning(neg_prompt)
            elif self.vc.uncond_type == "zero_embed":
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            else:
                raise ValueError(f"{self.vc.uncond_type} is not supported.")
        else:
            negative_prompt_embeds = None

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
            cond = {
                "c_crossattn": [
                    text_emb.prompt_embeds[li * bs_micro : (li + 1) * bs_micro]
                ],
                "fps": torch.tensor([self.fps] * b).to(noisy_sample.device).long(),
            }
            pred_micro = self.vc.apply_model(
                x_noisy=noisy_sample, t=timesteps, cond=cond
            )
            pred.append(pred_micro)
        pred = torch.concat(pred, dim=0)

        # convert prediction output to noise estimation
        pred_noise = self.convert_model_prediction_to_noise(pred, self.scheduler)

        return pred_noise
