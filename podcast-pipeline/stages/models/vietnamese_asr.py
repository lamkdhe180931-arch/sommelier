from __future__ import annotations

import tempfile
from typing import Any


PHOWHISPER_MODEL_NAME = "vinai/PhoWhisper-large"
CHUNKFORMER_MODEL_NAME = "khanhld/chunkformer-ctc-large-vie"
SAMPLE_RATE = 16000


def extract_transcription_text(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"]).strip()
        if "transcription" in result:
            return extract_transcription_text(result["transcription"])
        if "segments" in result:
            return extract_transcription_text(result["segments"])
        return ""

    if isinstance(result, (list, tuple)):
        if len(result) >= 3 and isinstance(result[-1], str):
            return result[-1].strip()

        parts = [extract_transcription_text(item) for item in result]
        return " ".join(part for part in parts if part)

    text = getattr(result, "text", None)
    if text is not None:
        return str(text).strip()

    return str(result).strip()


def _as_float32_mono(audio_16k):
    try:
        import numpy as np
    except ModuleNotFoundError:
        if isinstance(audio_16k, (list, tuple)):
            return [float(sample) for sample in audio_16k]
        return audio_16k

    audio = np.asarray(audio_16k, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return audio


def _pipeline_device(device) -> int:
    if device is None:
        return -1
    device_str = str(device)
    if not device_str.startswith("cuda"):
        return -1
    if ":" not in device_str:
        return 0
    return int(device_str.split(":", 1)[1])


class PhoWhisperTranscriber:
    def __init__(self, pipe) -> None:
        self.pipe = pipe

    @classmethod
    def from_pretrained(cls, model_name: str = PHOWHISPER_MODEL_NAME, device=None):
        import torch
        from transformers import pipeline

        kwargs = {
            "model": model_name,
            "device": _pipeline_device(device),
        }
        if str(device).startswith("cuda") and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        return cls(pipeline("automatic-speech-recognition", **kwargs))

    def transcribe(self, audio_16k) -> str:
        result = self.pipe(
            {"array": _as_float32_mono(audio_16k), "sampling_rate": SAMPLE_RATE},
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
        return extract_transcription_text(result)


class ChunkFormerTranscriber:
    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def from_pretrained(cls, model_name: str = CHUNKFORMER_MODEL_NAME, device=None):
        from chunkformer import ChunkFormerModel

        model = ChunkFormerModel.from_pretrained(model_name)
        if device is not None and str(device) != "cpu" and hasattr(model, "to"):
            model = model.to(str(device))
        if hasattr(model, "eval"):
            model.eval()
        return cls(model)

    def transcribe(self, audio_16k) -> str:
        import soundfile as sf

        audio = _as_float32_mono(audio_16k)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            sf.write(temp_wav.name, audio, SAMPLE_RATE)
            temp_wav.flush()
            result = self.model.batch_decode(
                audio_paths=[temp_wav.name],
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=1800,
            )
        return extract_transcription_text(result)


def load_phowhisper_model(device=None) -> PhoWhisperTranscriber:
    return PhoWhisperTranscriber.from_pretrained(device=device)


def load_chunkformer_model(device=None) -> ChunkFormerTranscriber:
    return ChunkFormerTranscriber.from_pretrained(device=device)


def transcribe(model: Any, audio_16k) -> str:
    if model is None:
        raise RuntimeError("Vietnamese ASR model is not loaded.")
    if hasattr(model, "transcribe"):
        return extract_transcription_text(model.transcribe(audio_16k))
    return extract_transcription_text(model(audio_16k))
