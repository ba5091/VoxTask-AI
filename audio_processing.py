from __future__ import annotations

import os
from dataclasses import dataclass

import av
import numpy as np
import torch
import whisper

SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 600
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


@dataclass
class AudioValidationResult:
    source_path: str
    audio_samples: np.ndarray
    duration: float
    sample_rate: int
    channels: int
    rms: float
    is_silent: bool


def _decode_audio(file_path: str) -> tuple[np.ndarray, int, int]:
    """Decode and resample audio with PyAV's bundled codecs (no FFmpeg executable)."""
    chunks: list[np.ndarray] = []
    sample_count = 0
    max_samples = SAMPLE_RATE * MAX_DURATION_SECONDS

    try:
        with av.open(file_path, mode="r") as container:
            audio_stream = next(
                (stream for stream in container.streams if stream.type == "audio"),
                None,
            )
            if audio_stream is None:
                raise ValueError("The uploaded file does not contain an audio stream.")

            layout = audio_stream.codec_context.layout
            channels = len(layout.channels) if layout is not None else 1
            resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=SAMPLE_RATE,
            )

            for frame in container.decode(audio_stream):
                for resampled_frame in resampler.resample(frame):
                    samples = resampled_frame.to_ndarray().reshape(-1)
                    sample_count += samples.size
                    if sample_count > max_samples:
                        raise ValueError(
                            "Please use an audio recording shorter than 10 minutes."
                        )
                    chunks.append(samples.astype(np.float32, copy=False))

            for resampled_frame in resampler.resample(None):
                samples = resampled_frame.to_ndarray().reshape(-1)
                sample_count += samples.size
                if sample_count > max_samples:
                    raise ValueError(
                        "Please use an audio recording shorter than 10 minutes."
                    )
                chunks.append(samples.astype(np.float32, copy=False))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "The uploaded file could not be decoded. Use a readable WAV, MP3, M4A, or FLAC audio file."
        ) from exc

    if not chunks:
        raise ValueError("The audio file is empty or contains no usable samples.")

    return np.concatenate(chunks), SAMPLE_RATE, channels


def validate_audio_file(file_path: str) -> AudioValidationResult:
    if file_path is None or str(file_path).strip() == "":
        raise FileNotFoundError("Please upload an audio file before processing.")

    file_path = os.path.abspath(str(file_path))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError("The selected audio path is not a file.")

    audio_samples, sample_rate, channels = _decode_audio(file_path)

    if audio_samples.size == 0:
        raise ValueError("The audio file is empty or contains no usable samples.")

    if not np.isfinite(audio_samples).all():
        raise ValueError("The audio contains invalid samples and is not usable.")

    duration = audio_samples.size / float(sample_rate)
    if duration <= 0:
        raise ValueError("The audio file is empty.")

    rms = float(np.sqrt(np.mean(np.square(audio_samples))))
    is_silent = rms < 1e-4
    if is_silent:
        raise ValueError("The audio appears to be silent. Please upload a clearer recording.")

    return AudioValidationResult(
        source_path=file_path,
        audio_samples=audio_samples,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        rms=rms,
        is_silent=is_silent,
    )


def load_whisper_model(model_name: str = "base", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"Whisper model '{model_name}' could not be loaded. Install the model or use a smaller option like 'base'."
        ) from exc

    return model


def transcribe_audio(audio_samples: np.ndarray, model=None, model_name: str = "base") -> str:
    if audio_samples is None or not isinstance(audio_samples, np.ndarray) or audio_samples.size == 0:
        raise ValueError("No decoded audio samples were provided for transcription.")

    if model is None:
        model = load_whisper_model(model_name=model_name)

    try:
        result = model.transcribe(audio_samples, task="transcribe")
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("Transcription failed. Please check the audio and the Whisper setup.") from exc

    return str(result.get("text", "")).strip()
