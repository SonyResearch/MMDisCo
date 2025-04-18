import os
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


def save_video(video: np.ndarray, video_path: Path, fps: float) -> None:
    """
    Save a video to a file.
    Args:
        video (np.ndarray): Video frames of shape (T, H, W, C).
        video_path (Path): Path to save the video.
        fps (float): Frames per second for the video.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W, C = video[0].shape
    video_writer = cv2.VideoWriter(video_path, fourcc, fps=fps, frameSize=(W, H))
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def save_audio_video(
    audio: np.ndarray,
    video: np.ndarray,
    outdir: Path,
    filename: str,
    *,
    sampling_rate=16000,
    fps: float = 8,
    subprocess_prefix: str = "",
) -> None:
    """
    Save audio and video.
    This function save two files: A wav file having the input audio and a mp4 file having the input video with the input audio.

    Args:
        audio (np.ndarray): Audio signal of shape (T,).
        video (np.ndarray): Video frames of shape (T, H, W, C).
        outdir (Path): Directory to save the files.
        filename (str): Filename to save the files.
        sampling_rate (int): Sampling rate for the audio. Default is 16000.
        fps (float): Frames per second for the video. Default is 8.
        subprocess_prefix (str): Prefix for subprocess commands. Default is "".
    """

    # save audio
    assert audio.dtype == np.int16
    audio_path = outdir / "wav" / f"{filename}.wav"
    sf.write(audio_path, audio, samplerate=sampling_rate)

    # save video
    assert video.dtype == np.uint8
    video_path = outdir / "mp4" / f"{filename}_1.mp4"
    save_video(video, video_path, fps)

    import subprocess

    new_video_path = outdir / "mp4" / f"{filename}_no_audio.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-y",
        "-i",
        video_path,
        "-an",
        "-vcodec",
        "libx264",
        new_video_path,
    ]
    subprocess.check_call(subprocess_prefix.split() + cmd)
    os.remove(video_path)

    # combine both to a single video file
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-i",
        new_video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "mp3",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-y",
        "-strict",
        "experimental",
        outdir / "mp4" / f"{filename}.mp4",
    ]
    subprocess.check_call(subprocess_prefix.split() + cmd)

    # remove "no_audio" and wav file
    os.remove(new_video_path)
