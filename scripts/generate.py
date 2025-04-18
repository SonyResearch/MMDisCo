import os
from pathlib import Path

import hydra
import rootutils
import lightning as L
import torch
import numpy as np

from lightning.pytorch.utilities.rank_zero import rank_zero_info
from omegaconf import DictConfig, OmegaConf

from mmdisco.models.networks import MMDisCo, AudioInferenceConfig, VideoInferenceConfig
from mmdisco.utils.save import save_audio_video

# avoid transformers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# setup project root
root = rootutils.setup_root(
    __file__, dotenv=True, pythonpath=False, cwd=False, project_root_env_var=True
)


@hydra.main(version_base=None, config_path=f"{root}/conf", config_name="generate.yaml")
def main(conf: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    outdir = Path(conf.paths.output_dir)

    # setup output dir
    rank_zero_info(f"output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "wav").mkdir(parents=True, exist_ok=True)
    (outdir / "mp4").mkdir(parents=True, exist_ok=True)

    # show all configuration
    rank_zero_info("All Hyperparameters")
    rank_zero_info(OmegaConf.to_yaml(conf, resolve=True))

    # model
    model: MMDisCo = hydra.utils.instantiate(conf.model)
    model.to(device=device, dtype=dtype)

    # setup seed
    if conf.seed is not None:
        L.seed_everything(seed=conf.seed, workers=True)
    else:
        L.seed_everything(seed=np.random.randint(2**20))
        model.set_seed(seed=np.random.randint(2**20))

    # inference config
    a_conf: AudioInferenceConfig = hydra.utils.instantiate(conf.inference.audio)
    v_conf: VideoInferenceConfig = hydra.utils.instantiate(conf.inference.video)

    video_fps = conf.inference.video.num_frames / conf.inference.audio.length_in_sec

    for i, prompt in enumerate(conf.inference.prompts):
        rank_zero_info(
            f"[{i}/{len(conf.inference.prompts)}] Generating {conf.inference.num_samples_per_prompt} video and audio with the prompt: {prompt}"
        )
        audio, video = model.sample(
            a_conf,
            v_conf,
            prompt,
            num_inference_steps=conf.inference.num_inference_steps,
            joint=conf.inference.joint,
            joint_scale_audio=conf.inference.joint_scale_audio,
            joint_scale_video=conf.inference.joint_scale_video,
            joint_num_steps=conf.inference.joint_num_steps,
            num_samples_per_prompt=conf.inference.num_samples_per_prompt,
        )

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy().astype("int16")

        video = video.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

        for j in range(conf.inference.num_samples_per_prompt):
            save_audio_video(
                audio[j],
                video[j],
                outdir=outdir,
                filename=prompt.replace(" ", "_") + f"_{j:0>3d}",
                fps=video_fps,
            )

    rank_zero_info(f"Save all videos and audios to {outdir}")


if __name__ == "__main__":
    main()
