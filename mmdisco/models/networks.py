from dataclasses import dataclass
from typing import Callable, Union, Optional

import torch
from easydict import EasyDict as edict
from torch import Tensor
from tqdm import trange

from mmdisco.utils.misc import set_torch_flags

from .base import (
    AudioInferenceConfig,
    BaseAudioOutputs,
    BaseVideoOutputs,
    JointDiffusionBase,
    MMInferenceConfig,
    VideoInferenceConfig,
)
from .diffusion import LatentAudioDiffusion, LatentVideoDiffusion
from .diffusion.multimodal import MMDiffusion
from .residual_predictor import VAResidualPredictorBase, VAResidualPredictorOutputs
from .scheduler.scheduling_ddpm_extcuda import DDPMScheduler


class MMDisCo(JointDiffusionBase):
    def __init__(
        self,
        audio_diffusion: LatentAudioDiffusion,
        video_diffusion: LatentVideoDiffusion,
        residual_predictor: Union[VAResidualPredictorBase, None],
    ):
        super().__init__()

        set_torch_flags(allow_matmul_tf32=False, allow_cudnn_tf32=False)

        # set input argument to self
        self.audio_diffusion = audio_diffusion
        self.video_diffusion = video_diffusion
        self.residual_predictor = residual_predictor

        # setup training diffusion scheduler
        # currently only support the same max timestep for t2a and t2v
        assert (
            audio_diffusion.scheduler.config.num_train_timesteps
            == video_diffusion.scheduler.config.num_train_timesteps
        ), "two scheduler must have the same max time steps."
        self.max_timesteps = self.audio_diffusion.scheduler.config.num_train_timesteps
        self.train_diffusion_scheduler_audio = DDPMScheduler.from_config(
            self.audio_diffusion.scheduler.config
        )
        self.train_diffusion_scheduler_video = DDPMScheduler.from_config(
            self.video_diffusion.scheduler.config
        )
        self.train_diffusion_scheduler_audio.set_timesteps(
            num_inference_steps=self.max_timesteps
        )
        self.train_diffusion_scheduler_video.set_timesteps(
            num_inference_steps=self.max_timesteps
        )

        self.audio_diffusion.requires_grad_(False)
        self.video_diffusion.requires_grad_(False)

        if self.residual_predictor is not None:
            self.residual_predictor.requires_grad_(False)

        # default device
        self.device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.dtype = torch.float32

    def to(self, device, dtype):
        super().to(device, dtype)

        if self.audio_diffusion is not None:
            self.audio_diffusion.to(device=device, dtype=dtype)

        if self.video_diffusion is not None:
            self.video_diffusion.to(device=device, dtype=dtype)

        if self.residual_predictor is not None:
            self.residual_predictor.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        return self

    def get_score_factor_from_timesteps(self, timesteps, *, audio, video):
        alphas_cumprod_audio = self.train_diffusion_scheduler_audio.alphas_cumprod[
            timesteps.cpu()
        ]
        alphas_cumprod_video = self.train_diffusion_scheduler_video.alphas_cumprod[
            timesteps.cpu()
        ]
        score_factor = {
            "audio": self.expand_dims_except_batch(
                -1 * (1 - alphas_cumprod_audio) ** 0.5, audio
            ),
            "video": self.expand_dims_except_batch(
                -1 * (1 - alphas_cumprod_video) ** 0.5, video
            ),
        }

        return score_factor

    #### methods for inference ####
    @torch.no_grad()
    def sample(
        self,
        a_conf: AudioInferenceConfig,
        v_conf: VideoInferenceConfig,
        prompt: str,
        num_inference_steps: int,
        joint: bool,
        joint_scale_audio: float = 1.0,
        joint_scale_video: float = 1.0,
        joint_num_steps: int = 1,
        num_samples_per_prompt: int = 1,
    ) -> tuple[Tensor, Tensor]:
        if joint:
            assert joint_num_steps > 0

        assert isinstance(prompt, str)
        prompt = [prompt] * num_samples_per_prompt
        num_samples = len(prompt)

        self.to(device=self.device, dtype=self.dtype)

        # get first latents
        latent_a = self.audio_diffusion.sample_init_latents(
            batch_size=num_samples,
            target_length_in_sec=a_conf.length_in_sec,
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )
        latent_v = self.video_diffusion.sample_init_latents(
            batch_size=num_samples,
            num_frames=v_conf.num_frames,
            target_height=v_conf.height,
            target_width=v_conf.width,
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )

        # set inference timesteps
        self.audio_diffusion.scheduler.set_timesteps(num_inference_steps, self.device)
        self.video_diffusion.scheduler.set_timesteps(num_inference_steps, self.device)

        # compute prompt embs
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                audio_text_embs, audio_text_embs_neg = (
                    self.audio_diffusion.encode_prompt(
                        prompt, self.device, do_classifier_free_guidance=True
                    )
                )
                video_text_embs, video_text_embs_neg = (
                    self.video_diffusion.encode_prompt(
                        prompt, self.device, do_classifier_free_guidance=True
                    )
                )
        except Exception as e:
            print(e)
            raise ValueError(f"encode_prompt raises for the text caption: {prompt}")

        # construct audio noise predictor with guidance
        a_conf.guidance_scale = self.audio_diffusion.default_guidance_scale

        def audio_cond_predict(*, noisy_sample, timesteps):
            return self.audio_diffusion.predict_noise(
                noisy_sample=noisy_sample, timesteps=timesteps, text_emb=audio_text_embs
            )

        def audio_uncond_predict(*, noisy_sample, timesteps):
            return self.audio_diffusion.predict_noise(
                noisy_sample=noisy_sample,
                timesteps=timesteps,
                text_emb=audio_text_embs_neg,
            )

        audio_predictor = a_conf.noise_predictor_kls(
            cond_predictor=audio_cond_predict, uncond_predictor=audio_uncond_predict
        )

        # construct video noise predictor with guidance
        v_conf.guidance_scale = self.video_diffusion.default_guidance_scale

        def video_cond_predict(*, noisy_sample, timesteps):
            return self.video_diffusion.predict_noise(
                noisy_sample=noisy_sample, timesteps=timesteps, text_emb=video_text_embs
            )

        def video_uncond_predict(*, noisy_sample, timesteps):
            return self.video_diffusion.predict_noise(
                noisy_sample=noisy_sample,
                timesteps=timesteps,
                text_emb=video_text_embs_neg,
            )

        video_predictor = v_conf.noise_predictor_kls(
            cond_predictor=video_cond_predict, uncond_predictor=video_uncond_predict
        )

        # construct residual noise predictor with guidance
        def residual_predictor(**args):
            res_pred_cond: VAResidualPredictorOutputs = self.residual_predictor(
                video_text_emb=video_text_embs.prompt_embeds.mean(dim=1),
                audio_text_emb=audio_text_embs.prompt_embeds,
                **args,
            )
            res_pred_uncond: VAResidualPredictorOutputs = self.residual_predictor(
                video_text_emb=video_text_embs_neg.prompt_embeds.mean(dim=1),
                audio_text_emb=audio_text_embs_neg.prompt_embeds,
                **args,
            )

            # audio
            def get_audio_prediction(x: VAResidualPredictorOutputs):
                return x.pred_audio * x.pred_grad_scale_audio

            res_pred_audio_cond = get_audio_prediction(res_pred_cond)
            res_pred_audio_uncond = get_audio_prediction(res_pred_uncond)
            res_audio_predictor = a_conf.noise_predictor_kls(
                cond_predictor=lambda *x, **y: res_pred_audio_cond,
                uncond_predictor=lambda *x, **y: res_pred_audio_uncond,
            )

            # video
            def get_video_prediction(x: VAResidualPredictorOutputs):
                return x.pred_video * x.pred_grad_scale_video

            res_pred_video_cond = get_video_prediction(res_pred_cond)
            res_pred_video_uncond = get_video_prediction(res_pred_uncond)
            res_video_predictor = v_conf.noise_predictor_kls(
                cond_predictor=lambda *x, **y: res_pred_video_cond,
                uncond_predictor=lambda *x, **y: res_pred_video_uncond,
            )

            return edict(
                res_pred_audio=res_audio_predictor(None, None),
                res_pred_video=res_video_predictor(None, None),
                prob_uncond=res_pred_uncond.prob,
                prob_cond=res_pred_cond.prob,
            )

        # denoising loop
        pbar = trange(num_inference_steps)
        for i in pbar:
            # predict noise for audio
            t_a = self.audio_diffusion.scheduler.timesteps[i]
            t_a_tensor = t_a.tile(len(latent_a))

            # predict noise for video
            t_v = self.video_diffusion.scheduler.timesteps[i]
            t_v_tensor = t_v.tile(
                len(latent_a),
            )

            # joint guidance by covariance estimator
            if joint and self.residual_predictor is not None:
                score_factor = self.get_score_factor_from_timesteps(
                    t_a_tensor, audio=latent_a, video=latent_v
                )

                for j in range(joint_num_steps):
                    if j > 0:
                        latent_a = self.audio_diffusion.scheduler.rev_step(
                            t_a_tensor.cpu(), latent_a, generator=self.generator
                        )
                        latent_v = self.video_diffusion.scheduler.rev_step(
                            t_v_tensor.cpu(), latent_v, generator=self.generator
                        )

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_audio_noise = audio_predictor(latent_a, t_a_tensor)
                        pred_video_noise = video_predictor(latent_v, t_v_tensor)

                    res_pred = residual_predictor(
                        video=latent_v,
                        pred_video_noise=pred_video_noise,
                        audio=latent_a,
                        pred_audio_noise=pred_audio_noise,
                        timesteps=t_a_tensor,  # == t_v_tensor
                        score_factor=score_factor,
                        inference=True,
                        video_channel_time_transpose=True,
                        audio_spatial_transpose=not self.audio_diffusion.is_mel_channel_last,
                    )

                    pred_joint_audio_noise = (
                        pred_audio_noise + joint_scale_audio * res_pred.res_pred_audio
                    )
                    pred_joint_video_noise = (
                        pred_video_noise + joint_scale_video * res_pred.res_pred_video
                    )

                    latent_a = self.audio_diffusion.scheduler.step(
                        pred_joint_audio_noise,
                        t_a_tensor.cpu(),
                        latent_a,
                        generator=self.generator,
                    ).prev_sample
                    latent_v = self.video_diffusion.scheduler.step(
                        pred_joint_video_noise,
                        t_v_tensor.cpu(),
                        latent_v,
                        generator=self.generator,
                    ).prev_sample
            else:
                # no joint
                # compute previous noisy sample x_t, y_t -> x_t-1, y_t-1
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_audio_noise = audio_predictor(latent_a, t_a_tensor)
                    pred_video_noise = video_predictor(latent_v, t_v_tensor)

                latent_a = self.audio_diffusion.scheduler.step(
                    pred_audio_noise, t_a_tensor, latent_a, generator=self.generator
                ).prev_sample
                latent_v = self.video_diffusion.scheduler.step(
                    pred_video_noise, t_v_tensor, latent_v, generator=self.generator
                ).prev_sample

        # decode latents
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            audio = self.audio_diffusion.decode_latents(
                latent_a, to_int16=True, target_length_sec=a_conf.length_in_sec
            )
            video = self.video_diffusion.decode_latents(latent_v, to_uint8=True)

        return audio, video


class MMDisCoMMDiffusion(JointDiffusionBase):
    @dataclass
    class BaseOutputs:
        audio: BaseAudioOutputs
        video: BaseVideoOutputs

    def __init__(
        self,
        mm_diffusion: MMDiffusion,
        mm_diffusion_image_sr: Optional[MMDiffusion],
        residual_predictor: Optional[VAResidualPredictorBase],
    ):
        super().__init__()

        set_torch_flags(allow_matmul_tf32=False, allow_cudnn_tf32=False)

        # set input argument to self
        self.mm_diffusion = mm_diffusion
        self.mm_diffusion_image_sr = mm_diffusion_image_sr
        self.residual_predictor = residual_predictor

        self.mm_diffusion.requires_grad_(False)

        if self.residual_predictor is not None:
            self.residual_predictor.requires_grad_(False)

        self.device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.dtype = torch.float32

    def to(self, device, dtype):
        super().to(device, dtype)

        self.mm_diffusion.to(device=device, dtype=dtype)

        if self.mm_diffusion_image_sr is not None:
            self.mm_diffusion_image_sr.to(device=device, dtype=dtype)

        if self.residual_predictor is not None:
            self.residual_predictor.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        return self

    def get_score_factor_from_timesteps(self, timesteps, *, audio, video):
        alphas_cumprod_video = self.mm_diffusion.multimodal_diffusion.alphas_cumprod[
            timesteps.cpu()
        ]
        alphas_cumprod_audio = self.mm_diffusion.multimodal_diffusion.alphas_cumprod[
            timesteps.cpu()
        ]
        score_factor = {
            "audio": self.expand_dims_except_batch(
                -1 * (1 - alphas_cumprod_audio) ** 0.5, audio
            ),
            "video": self.expand_dims_except_batch(
                -1 * (1 - alphas_cumprod_video) ** 0.5, video
            ),
        }

        return score_factor

    def get_guided_model(self, joint_scale_audio, joint_scale_video) -> Callable:
        class wrapped_model:
            def __init__(self, video_out_channels, audio_out_channels):
                self.video_out_channels = video_out_channels
                self.audio_out_channels = audio_out_channels

            def __call__(self, *args, **kwargs):
                return self.call(*args, **kwargs)

            @staticmethod
            def call(video, audio, timestep, *args, **kwargs):
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    # base prediction
                    pred_video_base, pred_audio_base = self.mm_diffusion(
                        video, audio, timestep
                    )

                    # joint guidance
                    score_factor = self.get_score_factor_from_timesteps(
                        timestep, audio=audio, video=video
                    )

                    pred_res: VAResidualPredictorOutputs = self.residual_predictor(
                        video=video,
                        pred_video_noise=pred_video_base,
                        audio=audio,
                        pred_audio_noise=pred_audio_base,
                        timesteps=timestep,
                        video_text_emb=None,
                        audio_text_emb=None,
                        score_factor=score_factor,
                        prob_only=False,
                        inference=True,
                        video_channel_time_transpose=False,
                        audio_spatial_transpose=False,
                    )

                    pred_res_video_noise = pred_res.pred_video * joint_scale_video
                    pred_res_audio_noise = pred_res.pred_audio * joint_scale_audio

                    return (
                        pred_video_base + pred_res_video_noise,
                        pred_audio_base + pred_res_audio_noise,
                    )

        return wrapped_model(
            video_out_channels=self.mm_diffusion.multimodal_model.video_out_channels,
            audio_out_channels=self.mm_diffusion.multimodal_model.audio_out_channels,
        )

    #### methods for inference ####

    @torch.no_grad()
    def sample(self, mm_inference_conf: MMInferenceConfig) -> tuple[Tensor, Tensor]:
        self.to(device=self.device, dtype=self.dtype)

        shape = {
            "video": (
                mm_inference_conf.batch_size,
                *self.mm_diffusion.model_flags.video_size,
            ),
            "audio": (
                mm_inference_conf.batch_size,
                *self.mm_diffusion.model_flags.audio_size,
            ),
        }

        model = self.mm_diffusion.multimodal_model
        if mm_inference_conf.joint:
            model = self.get_guided_model(
                joint_scale_audio=mm_inference_conf.joint_scale_audio,
                joint_scale_video=mm_inference_conf.joint_scale_video,
            )

        if mm_inference_conf.sample_fn == "dpm_solver":
            from .diffusion.mmdiffusion.multimodal_dpm_solver_plus import DPM_Solver

            dpm_solver = DPM_Solver(
                model=model,
                alphas_cumprod=torch.tensor(
                    self.mm_diffusion.multimodal_diffusion.alphas_cumprod,
                    dtype=torch.float32,
                ),
                predict_x0=True,
                thresholding=True,
            )

            x_T = {
                "video": torch.randn(
                    shape["video"],
                    generator=self.generator,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "audio": torch.randn(
                    shape["audio"],
                    generator=self.generator,
                    device=self.device,
                    dtype=self.dtype,
                ),
            }

            sample = dpm_solver.sample(
                x_T, steps=20, order=3, skip_type="logSNR", method="singlestep"
            )
        else:
            raise NotImplementedError(
                f"sample_fn '{mm_inference_conf.sample_fn}' is not supported."
            )

        if self.mm_diffusion_image_sr is not None:
            # image super resolution
            video = sample["video"]
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                video = self.mm_diffusion_image_sr(video)
            sample["video"] = video

        # (B, F, C, H, W) and (B, 1, L) here
        audio = sample["audio"]
        video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(dtype=torch.uint8)
        video = video.permute(0, 1, 3, 4, 2).contiguous()

        return audio, video
