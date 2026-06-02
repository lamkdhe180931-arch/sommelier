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


def segment_duration(segment: dict[str, Any]) -> float:
    try:
        return max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _similarity_margin(best_similarity: float, second_best_similarity: float) -> float:
    if second_best_similarity < 0:
        return float("inf")
    return best_similarity - second_best_similarity


def resolve_speaker_identity_decision(
    *,
    best_id: str | None,
    best_similarity: float,
    second_best_similarity: float,
    has_clean_identity_evidence: bool,
    next_global_id: str,
    similarity_threshold: float,
    similarity_margin: float,
    weak_match_threshold: float,
    centroid_update_threshold: float = 0.85,
    review_speaker_label: str = "SPEAKER_REVIEW",
) -> dict[str, Any]:
    """
    Decide how a local speaker track should map into global speaker identity.

    Clean identity evidence means the local track has enough non-overlap speech to
    trust its embedding. Weak tracks can match a clearly similar existing speaker,
    but they must not create or update global centroids.
    """
    margin = _similarity_margin(best_similarity, second_best_similarity)
    has_clear_best = best_id is not None and margin >= similarity_margin

    if has_clear_best and best_similarity >= similarity_threshold:
        return {
            "mapped_speaker": best_id,
            "action": "matched_existing",
            "reason": "matched",
            "should_create_new": False,
            "should_update_centroid": bool(
                has_clean_identity_evidence and best_similarity >= centroid_update_threshold
            ),
            "identity_low_confidence": not has_clean_identity_evidence,
        }

    if has_clean_identity_evidence:
        reason = "margin_too_small" if best_id is not None and best_similarity >= similarity_threshold else "below_threshold"
        return {
            "mapped_speaker": next_global_id,
            "action": "created_new",
            "reason": reason,
            "should_create_new": True,
            "should_update_centroid": True,
            "identity_low_confidence": False,
        }

    if has_clear_best and best_similarity >= weak_match_threshold:
        return {
            "mapped_speaker": best_id,
            "action": "matched_weak_context",
            "reason": "weak_context_match",
            "should_create_new": False,
            "should_update_centroid": False,
            "identity_low_confidence": True,
        }

    return {
        "mapped_speaker": review_speaker_label,
        "action": "review_weak_identity",
        "reason": "weak_identity_review",
        "should_create_new": False,
        "should_update_centroid": False,
        "identity_low_confidence": True,
    }


DUPLEX_GROUP_CLEAN = "clean_duplex_2speaker"
DUPLEX_GROUP_REVIEW = "overlap_review"
DUPLEX_GROUP_EXCLUDE = "exclude_or_extra_speaker"
DUPLEX_GROUPS = (DUPLEX_GROUP_CLEAN, DUPLEX_GROUP_REVIEW, DUPLEX_GROUP_EXCLUDE)


def speaker_duration_totals(segments: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for segment in segments:
        speaker = str(segment.get("speaker", "")).strip()
        if not speaker:
            continue
        totals[speaker] = totals.get(speaker, 0.0) + segment_duration(segment)
    return totals


def infer_main_speakers(
    segments: list[dict[str, Any]],
    *,
    expected_main_speakers: int = 2,
    main_speakers: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if main_speakers:
        return [str(speaker) for speaker in main_speakers][:expected_main_speakers]

    totals = speaker_duration_totals(segments)
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [speaker for speaker, _duration in ranked[:expected_main_speakers]]


def duplex_group_for_segment(segment: dict[str, Any], main_speakers: list[str]) -> tuple[str, str]:
    speaker = str(segment.get("speaker", "")).strip()
    if speaker not in set(main_speakers):
        return DUPLEX_GROUP_EXCLUDE, "extra_speaker"

    if str(segment.get("separation_status", "")).strip() == "low_confidence":
        return DUPLEX_GROUP_REVIEW, "low_confidence_overlap"
    if int(segment.get("low_confidence_overlap_count") or 0) > 0:
        return DUPLEX_GROUP_REVIEW, "low_confidence_overlap"
    if bool(segment.get("has_independent_osd_overlap")):
        return DUPLEX_GROUP_REVIEW, "independent_osd_overlap"
    if bool(segment.get("has_overlap")) and not bool(segment.get("is_separated")):
        return DUPLEX_GROUP_REVIEW, "unseparated_overlap"
    if bool(segment.get("needs_manual_review")):
        return DUPLEX_GROUP_REVIEW, "manual_review"

    return DUPLEX_GROUP_CLEAN, "main_speaker_clean"


def assign_duplex_train_groups(
    segments: list[dict[str, Any]],
    *,
    expected_main_speakers: int = 2,
    main_speakers: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_main_speakers = infer_main_speakers(
        segments,
        expected_main_speakers=expected_main_speakers,
        main_speakers=main_speakers,
    )
    counts = {group: 0 for group in DUPLEX_GROUPS}
    grouped: list[dict[str, Any]] = []

    for segment in segments:
        clean_segment = dict(segment)
        group, reason = duplex_group_for_segment(clean_segment, resolved_main_speakers)
        counts[group] += 1
        clean_segment["duplex_train_group"] = group
        clean_segment["duplex_group_reason"] = reason
        clean_segment["is_main_speaker"] = str(clean_segment.get("speaker", "")).strip() in set(resolved_main_speakers)
        clean_segment["main_speakers"] = resolved_main_speakers
        grouped.append(clean_segment)

    return grouped, {
        "expected_main_speakers": expected_main_speakers,
        "main_speakers": resolved_main_speakers,
        "speaker_duration_seconds": {
            speaker: round(duration, 6)
            for speaker, duration in sorted(speaker_duration_totals(segments).items(), key=lambda item: (-item[1], item[0]))
        },
        "group_counts": counts,
    }


def _merge_segment_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["start"] = min(float(left.get("start", 0.0)), float(right.get("start", 0.0)))
    merged["end"] = max(float(left.get("end", 0.0)), float(right.get("end", 0.0)))
    merged["speaker"] = left.get("speaker", right.get("speaker", ""))
    source_indices: list[str] = []
    for segment in (left, right):
        existing = segment.get("source_indices")
        if isinstance(existing, list):
            source_indices.extend(str(item) for item in existing)
        elif segment.get("index") is not None:
            source_indices.append(str(segment["index"]))
    if source_indices:
        merged["source_indices"] = source_indices
    return merged


def postprocess_diarization_segments(
    segments: list[dict[str, Any]],
    *,
    same_speaker_merge_gap: float = 0.3,
    short_backchannel_seconds: float = 1.0,
    max_segment_duration: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Merge only same-speaker micro gaps and mark short podcast backchannels.

    Different-speaker turns are intentionally preserved because short interjections
    such as "vâng", "ừ", or "dạ" are useful for full-duplex training.
    """
    ordered = sorted((dict(segment) for segment in segments), key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    merged_segments: list[dict[str, Any]] = []
    merged_count = 0

    for segment in ordered:
        if not merged_segments:
            merged_segments.append(segment)
            continue

        prev = merged_segments[-1]
        same_speaker = str(prev.get("speaker", "")) == str(segment.get("speaker", ""))
        gap = float(segment.get("start", 0.0)) - float(prev.get("end", 0.0))
        merged_duration = max(float(prev.get("end", 0.0)), float(segment.get("end", 0.0))) - min(
            float(prev.get("start", 0.0)), float(segment.get("start", 0.0))
        )

        if same_speaker and gap <= same_speaker_merge_gap and merged_duration <= max_segment_duration:
            merged_segments[-1] = _merge_segment_pair(prev, segment)
            merged_count += 1
        else:
            merged_segments.append(segment)

    short_backchannel_count = 0
    for idx, segment in enumerate(merged_segments):
        duration = segment_duration(segment)
        segment["index"] = normalized_index(idx)
        segment["duration"] = round(duration, 6)
        
        is_short = duration < short_backchannel_seconds
        
        is_real_backchannel = False
        if is_short:
            speaker = segment.get("speaker")
            start = float(segment.get("start", 0.0))
            
            # Find if the other speaker spoke recently in the conversation
            other_speaker_active_nearby = False
            for prev_seg in reversed(merged_segments[:idx]):
                if prev_seg.get("speaker") != speaker:
                    gap_to_other = start - float(prev_seg.get("end", 0.0))
                    if gap_to_other <= 3.0:
                        other_speaker_active_nearby = True
                    break
                
                # If we encounter the same speaker and the gap is small, they were just pausing their monologue
                gap_to_same = start - float(prev_seg.get("end", 0.0))
                if gap_to_same <= 2.0:
                    break
            
            is_real_backchannel = other_speaker_active_nearby

        segment["is_short_backchannel"] = bool(is_real_backchannel)
        if is_real_backchannel:
            short_backchannel_count += 1
            segment.setdefault("train_quality_label", "short_backchannel_review")
            segment.setdefault("needs_manual_review", True)
        else:
            segment.setdefault("train_quality_label", "clean_candidate")
            segment.setdefault("needs_manual_review", False)

    return merged_segments, {
        "same_speaker_merge_gap_seconds": same_speaker_merge_gap,
        "short_backchannel_seconds": short_backchannel_seconds,
        "input_segment_count": len(segments),
        "output_segment_count": len(merged_segments),
        "merged_same_speaker_gap_count": merged_count,
        "short_backchannel_count": short_backchannel_count,
    }


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
