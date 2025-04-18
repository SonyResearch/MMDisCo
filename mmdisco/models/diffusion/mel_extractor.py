import torch
from torch import Tensor


### logmel extractor from https://github.com/haoheliu/AudioLDM-training-finetuning/blob/main/audioldm_train/utilities/data/dataset.py
def spectral_normalize(magnitude, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(magnitude, min=clip_val) * C)


class AudioLDMLogMelExtractor(object):
    def __init__(
        self,
        sampling_rate,
        filter_length,
        n_mel,
        mel_fmin,
        mel_fmax,
        win_length,
        hop_length,
    ):
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.n_mel = n_mel
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.win_length = win_length
        self.hop_length = hop_length

        # cache
        self.mel_basis = {}
        self.hann_window = {}

    def mel_spectrogram_train(self, wave: Tensor):
        if torch.min(wave) < -1.0 or torch.max(wave) > 1.0:
            print(
                f"train min value is {torch.min(wave)} and max value is {torch.max(wave)}"
            )

        device = wave.device

        # setup filters at the first execution
        mel_key = f"{self.mel_fmax}_{str(device)}"
        if mel_key not in self.mel_basis:
            from librosa.filters import mel as librosa_mel_fn

            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )

            self.mel_basis[mel_key] = torch.from_numpy(mel).float().to(device)
            self.hann_window[str(device)] = torch.hann_window(self.win_length).to(
                device
            )

        # padding wave
        pad_size = int((self.filter_length - self.hop_length) / 2)
        wave = torch.nn.functional.pad(wave, (pad_size, pad_size), mode="reflect")

        # wave -> stft spec
        stft_spec = torch.stft(
            wave,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stft_spec = torch.abs(stft_spec)

        # stft spec -> log mel spec
        mel_spec = torch.matmul(self.mel_basis[mel_key], stft_spec)
        log_mel_spec = spectral_normalize(mel_spec)

        return log_mel_spec, stft_spec

    def __call__(self, wave: Tensor):
        log_mel_spec, stft = self.mel_spectrogram_train(wave)

        # (B, C, T) -> (B, T, C)
        log_mel_spec = log_mel_spec.mT
        stft = stft.mT

        # # padding to get desired length.
        # # Not needed for our case, because we don't clip length of the wave.
        # log_mel_spec = self.pad_spec(log_mel_spec)
        # stft = self.pad_spec(stft)

        return log_mel_spec, stft
