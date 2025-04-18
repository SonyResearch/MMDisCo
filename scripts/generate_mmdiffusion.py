import os
from pathlib import Path

import hydra
import rootutils
import lightning as L
import torch
import numpy as np

from lightning.pytorch.utilities.rank_zero import rank_zero_info
from omegaconf import DictConfig, OmegaConf

from mmdisco.models.networks import (
    MMDisCoMMDiffusion,
    MMInferenceConfig,
)
from mmdisco.models.diffusion.mmdiffusion.common import save_multimodal, save_audio

# avoid transformers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# setup project root
root = rootutils.setup_root(
    __file__, dotenv=True, pythonpath=False, cwd=False, project_root_env_var=True
)


@hydra.main(
    version_base=None,
    config_path=f"{root}/conf",
    config_name="generate_mmdiffusion.yaml",
)
def main(conf: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # this must be float32

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
    model: MMDisCoMMDiffusion = hydra.utils.instantiate(conf.model)
    model.to(device=device, dtype=dtype)

    # setup seed
    if conf.seed is not None:
        L.seed_everything(seed=conf.seed, workers=True)
    else:
        L.seed_everything(seed=np.random.randint(2**20))
        model.set_seed(seed=np.random.randint(2**20))

    # inference config
    inference_config: MMInferenceConfig = hydra.utils.instantiate(conf.inference)
    num_saved = 0
    num_samples = inference_config.num_samples
    batch_size = inference_config.batch_size
    num_loops = (num_samples + batch_size - 1) // batch_size
    for i in range(num_loops):
        rank_zero_info(f"[{i+1}/{num_loops}] Generating video and audio...")
        audio, video = model.sample(inference_config)

        audio = audio.detach().cpu().numpy()
        video = video.detach().cpu().numpy()

        for j in range(batch_size):
            if num_saved >= num_samples:
                break
            num_saved += 1

            # save video and audio
            save_multimodal(
                video[j],
                audio[j],
                output_path=outdir / "mp4" / f"{num_saved:0>3d}.mp4",
                args=inference_config,
            )
            save_audio(
                audio[j],
                output_path=outdir / "wav" / f"{num_saved:0>3d}.wav",
                audio_fps=inference_config.audio_sr,
            )

    rank_zero_info(f"All videos and audios are saved in {outdir}.")


if __name__ == "__main__":
    main()
