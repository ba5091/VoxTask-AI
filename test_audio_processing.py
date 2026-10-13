import os
import wave
import tempfile

import numpy as np
import pytest

from audio_processing import transcribe_audio, validate_audio_file


def test_validate_audio_file_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_audio_file("missing_file.wav")


def test_validate_audio_file_rejects_silent_recording():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        path = temp_file.name

    try:
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(bytes(16000 * 2))
        with pytest.raises(ValueError, match="silent|usable"):
            validate_audio_file(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_transcription_uses_decoded_samples_without_a_file_decoder():
    audio = np.ones(16000, dtype=np.float32) * 0.1

    class FakeWhisperModel:
        def transcribe(self, audio_input, task):
            assert audio_input is audio
            assert task == "transcribe"
            return {"text": "hello"}

    assert transcribe_audio(audio, model=FakeWhisperModel()) == "hello"
