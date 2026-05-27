from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def dump_json(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_stage_dir(audio_path: str | os.PathLike[str]) -> Path:
    audio = Path(audio_path)
    return audio.parent / "_staged" / audio.stem


def audio_name_from_path(audio_path: str | os.PathLike[str]) -> str:
    return Path(audio_path).stem


def clean_segment_for_json(segment: dict[str, Any]) -> dict[str, Any]:
    skipped_keys = {
        "enhanced_audio",
        "audio",
        "waveform",
        "audio_segment",
    }
    return {
        key: json_safe(value)
        for key, value in segment.items()
        if key not in skipped_keys and not key.startswith("_")
    }


def clean_segments_for_json(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clean_segment_for_json(segment) for segment in segments]


def load_audio_info(audio_path: str | os.PathLike[str], sample_rate: int) -> dict[str, Any]:
    import numpy as np
    from pydub import AudioSegment

    audio_segment = AudioSegment.from_file(str(audio_path))
    audio_segment = audio_segment.set_frame_rate(sample_rate).set_sample_width(2).set_channels(1)

    target_dbfs = -20
    gain = target_dbfs - audio_segment.dBFS
    normalized_audio = audio_segment.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.flatten()

    max_amplitude = np.max(np.abs(waveform)) if waveform.size else 0
    if max_amplitude > 0:
        waveform = waveform / max_amplitude

    return {
        "waveform": waveform.astype(np.float32),
        "name": Path(audio_path).name,
        "sample_rate": sample_rate,
        "audio_segment": normalized_audio,
    }


def load_wav_mono(path: str | os.PathLike[str], target_sample_rate: int | None = None):
    import librosa

    return librosa.load(str(path), sr=target_sample_rate, mono=True)


def write_wav(path: str | os.PathLike[str], waveform, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(waveform), int(sample_rate))


def audio_segment_from_waveform(waveform, sample_rate: int):
    import numpy as np
    from pydub import AudioSegment

    clipped = np.clip(waveform, -1.0, 1.0)
    int16_waveform = (clipped * 32767).astype(np.int16)
    return AudioSegment(
        int16_waveform.tobytes(),
        frame_rate=int(sample_rate),
        sample_width=2,
        channels=1,
    )


def normalized_index(index: int) -> str:
    return f"{index:05d}"

