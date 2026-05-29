from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import wave
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from utils.asr_quality import choose_asr_text


SHORT_THRESHOLDS = (0.5, 1.0, 2.0)
LOG_PATTERNS = {
    "traceback": re.compile(r"\btraceback\b", re.IGNORECASE),
    "error": re.compile(r"\berror\b", re.IGNORECASE),
    "warning": re.compile(r"\bwarning\b|\bwarn\b", re.IGNORECASE),
    "oom": re.compile(r"\bout of memory\b|\boom\b|\bcuda out\b", re.IGNORECASE),
    "killed": re.compile(r"\bkilled\b|exit code 137", re.IGNORECASE),
}
BOILERPLATE_MARKERS = (
    "thank you",
    "thanks for watching",
    "subscribe",
    "la la school",
    "ghien mi go",
    "ghiền mì gõ",
    "ini mah",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 06: evaluate staged pipeline outputs.")
    parser.add_argument("--run_dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--out_dir", default="", help="Eval output directory. Defaults to RUN_DIR/06_eval.")
    parser.add_argument("--print_markdown", action="store_true", help="Print the markdown report.")
    return parser.parse_args()


def read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(run_dir: Path, *relative_paths: str) -> Path | None:
    for rel_path in relative_paths:
        path = run_dir / rel_path
        if path.exists():
            return path
    return None


def resolve_paths(run_dir: Path) -> dict[str, Path | None]:
    return {
        "input_audio": first_existing(run_dir, "00_input/full.wav", "full.wav", "audio.wav", "input.wav"),
        "input_preview": first_existing(run_dir, "00_input/preview_input_30s.wav", "preview_input_30s.wav"),
        "diarization": first_existing(run_dir, "01_diarization/diarization.json", "diarization.json"),
        "vad_trace": first_existing(run_dir, "01_diarization/trace_vad_chunks.json", "trace_vad_chunks.json"),
        "cleaned_audio": first_existing(run_dir, "02_music_clean/cleaned_audio.wav", "cleaned_audio.wav"),
        "segment_flags": first_existing(run_dir, "02_music_clean/segment_flags.json", "segment_flags.json"),
        "segments": first_existing(run_dir, "03_overlap/segments.json", "segments.json"),
        "transcript": first_existing(run_dir, "04_asr/transcript.json", "transcript.json"),
        "export_json": first_existing(run_dir, "05_export/final/data_audio.json", "final/data_audio.json"),
        "logs_dir": first_existing(run_dir, "logs", "_logs"),
        "separated_dir": first_existing(run_dir, "03_overlap/separated_segments", "separated_segments"),
        "export_dir": first_existing(run_dir, "05_export/final", "final"),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dump_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def duration_of(segment: dict[str, Any]) -> float:
    return max(0.0, as_float(segment.get("end")) - as_float(segment.get("start")))


def round_float(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def stats_for(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round_float(min(values)),
        "median": round_float(statistics.median(values)),
        "mean": round_float(statistics.fmean(values)),
        "max": round_float(max(values)),
        "sum": round_float(sum(values)),
    }


def segment_stats(segments: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [duration_of(segment) for segment in segments]
    sorted_segments = sorted(segments, key=lambda item: (as_float(item.get("start")), as_float(item.get("end"))))
    gaps: list[float] = []
    overlaps: list[float] = []
    for prev, cur in zip(sorted_segments, sorted_segments[1:]):
        prev_end = as_float(prev.get("end"))
        cur_start = as_float(cur.get("start"))
        diff = cur_start - prev_end
        if diff > 0:
            gaps.append(diff)
        elif diff < 0:
            overlaps.append(abs(diff))

    return {
        "count": len(segments),
        "duration": stats_for(durations),
        "short_counts": {f"lt_{threshold:g}s": sum(duration < threshold for duration in durations) for threshold in SHORT_THRESHOLDS},
        "invalid_duration_count": sum(as_float(segment.get("end")) <= as_float(segment.get("start")) for segment in segments),
        "speaker_counts": dict(Counter(str(segment.get("speaker", "")) for segment in segments)),
        "gap": stats_for(gaps),
        "overlap": stats_for(overlaps),
    }


def wav_info(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"exists": False}
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            info.update(
                {
                    "sample_rate": sample_rate,
                    "channels": wav.getnchannels(),
                    "sample_width": wav.getsampwidth(),
                    "frames": frames,
                    "duration_sec": round_float(frames / sample_rate if sample_rate else 0.0),
                }
            )
    except wave.Error as exc:
        info["error"] = str(exc)
    return info


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^\w\sà-ỹ]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a: Any, b: Any) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def contains_boilerplate(text: Any) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in BOILERPLATE_MARKERS)


def contains_foreign_script(text: Any) -> bool:
    value = str(text or "")
    return any(
        "\u0e00" <= char <= "\u0e7f" or "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff"
        for char in value
    )


def text_len_stats(segments: list[dict[str, Any]], field: str) -> dict[str, Any]:
    lengths = [len(str(segment.get(field) or "").strip()) for segment in segments]
    empty_count = sum(length == 0 for length in lengths)
    stats = stats_for([float(length) for length in lengths])
    stats["empty_count"] = empty_count
    return stats


def suspicious_asr_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suspicious: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        final_text = segment.get("text", "")
        whisper_text = segment.get("text_whisper", "")
        phowhisper_text = segment.get("text_phowhisper", "")
        chunkformer_text = segment.get("text_chunkformer", "")
        duration = duration_of(segment)
        reasons: list[str] = []

        if duration < 0.5 and final_text:
            reasons.append("duration_lt_0_5_with_text")
        elif duration < 1.0 and final_text:
            reasons.append("duration_lt_1_with_text")
        if not normalize_text(final_text):
            reasons.append("empty_final_text")
        if not normalize_text(whisper_text):
            reasons.append("empty_whisper")
        if not normalize_text(phowhisper_text):
            reasons.append("empty_phowhisper")
        if not normalize_text(chunkformer_text):
            reasons.append("empty_chunkformer")
        if contains_boilerplate(final_text):
            reasons.append("final_boilerplate_marker")
        if contains_foreign_script(final_text):
            reasons.append("final_foreign_script")

        pho_chunk_sim = similarity(phowhisper_text, chunkformer_text)
        final_whisper_sim = similarity(final_text, whisper_text)
        whisper_pho_sim = similarity(whisper_text, phowhisper_text)
        final_pho_sim = similarity(final_text, phowhisper_text)
        if pho_chunk_sim >= 0.75 and final_whisper_sim >= 0.85 and final_pho_sim < 0.55 and whisper_pho_sim < 0.55:
            reasons.append("final_follows_whisper_against_vi_models")

        if reasons:
            error_reasons = {
                "empty_final_text",
                "final_boilerplate_marker",
                "final_foreign_script",
                "final_follows_whisper_against_vi_models",
            }
            severity = "likely_error" if any(reason in error_reasons for reason in reasons) else "review"
            suspicious.append(
                {
                    "index": segment.get("index", f"{idx:05d}"),
                    "start": round_float(as_float(segment.get("start"))),
                    "end": round_float(as_float(segment.get("end"))),
                    "duration": round_float(duration),
                    "speaker": segment.get("speaker", ""),
                    "severity": severity,
                    "reasons": reasons,
                    "text": final_text,
                    "text_whisper": whisper_text,
                    "text_phowhisper": phowhisper_text,
                    "text_chunkformer": chunkformer_text,
                }
            )
    return suspicious


def asr_quality_counts(segments: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for segment in segments:
        source = str(segment.get("asr_quality_source") or "")
        if source:
            source_counts[source] += 1
        actions = segment.get("asr_quality_actions") or []
        if isinstance(actions, str):
            actions = [actions]
        for action in actions:
            if action:
                action_counts[str(action)] += 1
    return {
        "source_counts": dict(source_counts),
        "action_counts": dict(action_counts),
        "changed_count": sum(action_counts.values()),
    }


def asr_quality_preview(segments: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    for segment in segments:
        decision = choose_asr_text(
            rover_text=str(segment.get("text") or ""),
            text_whisper=str(segment.get("text_whisper") or ""),
            text_phowhisper=str(segment.get("text_phowhisper") or ""),
            text_chunkformer=str(segment.get("text_chunkformer") or ""),
            duration_sec=duration_of(segment),
        )
        for action in decision["actions"]:
            action_counts[action] += 1
        source_counts[decision["source"]] += 1
        if decision["actions"] and decision["text"] != str(segment.get("text") or "").strip():
            changes.append(
                {
                    "index": segment.get("index", ""),
                    "start": round_float(as_float(segment.get("start"))),
                    "end": round_float(as_float(segment.get("end"))),
                    "duration": round_float(duration_of(segment)),
                    "old_text": segment.get("text", ""),
                    "new_text": decision["text"],
                    "source": decision["source"],
                    "actions": decision["actions"],
                }
            )
    return {
        "would_change_count": len(changes),
        "source_counts": dict(source_counts),
        "action_counts": dict(action_counts),
        "changes": changes,
    }


def count_log_issues(logs_dir: Path | None) -> dict[str, Any]:
    if not logs_dir or not logs_dir.exists():
        return {"exists": False, "files": 0, "issue_counts": {}, "samples": []}
    counts = {key: 0 for key in LOG_PATTERNS}
    samples: list[dict[str, Any]] = []
    files = sorted(logs_dir.glob("*.log"))
    for log_path in files:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            matched = [key for key, pattern in LOG_PATTERNS.items() if pattern.search(line)]
            if not matched:
                continue
            for key in matched:
                counts[key] += 1
            if len(samples) < 20:
                samples.append({"file": str(log_path), "line": line_no, "kind": matched, "text": line[:300]})
    return {"exists": True, "files": len(files), "issue_counts": counts, "samples": samples}


def resolve_enhanced_path(path_value: Any, run_dir: Path, separated_dir: Path | None) -> Path | None:
    if not path_value:
        return None
    raw_path = Path(str(path_value))
    candidates = [raw_path]
    if separated_dir:
        candidates.append(separated_dir / raw_path.name)
    candidates.append(run_dir / raw_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_path


def evaluate(run_dir: Path) -> dict[str, Any]:
    paths = resolve_paths(run_dir)

    diarization_data = read_json(paths["diarization"])
    flags_data = read_json(paths["segment_flags"])
    overlap_data = read_json(paths["segments"])
    transcript_data = read_json(paths["transcript"])
    export_data = read_json(paths["export_json"])
    vad_data = read_json(paths["vad_trace"])

    diarization_segments = diarization_data.get("segments", [])
    music_segments = flags_data.get("segments", [])
    overlap_segments = overlap_data.get("segments", [])
    transcript_segments = transcript_data.get("segments", [])
    exported_segments = export_data.get("segments", []) if isinstance(export_data, dict) else []
    flags = flags_data.get("segment_demucs_flags", [])
    vad_chunks = vad_data.get("chunks", [])

    separated_segments = [segment for segment in overlap_segments if segment.get("is_separated")]
    separated_files = []
    for segment in separated_segments:
        resolved = resolve_enhanced_path(segment.get("enhanced_audio_path"), run_dir, paths["separated_dir"])
        separated_files.append(
            {
                "index": segment.get("index"),
                "duration": round_float(duration_of(segment)),
                "path": str(resolved) if resolved else "",
                "exists": bool(resolved and resolved.exists()),
                "wav": wav_info(resolved) if resolved and resolved.suffix.lower() == ".wav" else None,
            }
        )

    mp3_files = []
    export_dir = paths["export_dir"]
    if export_dir and export_dir.exists():
        mp3_files = sorted(export_dir.glob("**/*.mp3"))

    suspicious = suspicious_asr_segments(transcript_segments)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "paths": {key: str(value) if value else "" for key, value in paths.items()},
        "audio": {
            "input_audio": wav_info(paths["input_audio"]),
            "input_preview": wav_info(paths["input_preview"]),
            "cleaned_audio": wav_info(paths["cleaned_audio"]),
        },
        "logs": count_log_issues(paths["logs_dir"]),
        "stage_01_diarization": {
            "json_exists": bool(paths["diarization"] and paths["diarization"].exists()),
            "stats": segment_stats(diarization_segments),
            "vad_chunk_count": len(vad_chunks),
            "vad_chunk_duration": stats_for([duration_of(chunk) for chunk in vad_chunks]),
            "metadata": diarization_data.get("metadata", {}),
        },
        "stage_02_music_clean": {
            "json_exists": bool(paths["segment_flags"] and paths["segment_flags"].exists()),
            "stats": segment_stats(music_segments),
            "demucs_flagged_count": sum(bool(flag) for flag in flags),
            "demucs_flagged_indices": [idx for idx, flag in enumerate(flags) if flag],
            "metadata": flags_data.get("metadata", {}),
        },
        "stage_03_overlap": {
            "json_exists": bool(paths["segments"] and paths["segments"].exists()),
            "stats": segment_stats(overlap_segments),
            "separated_count": len(separated_segments),
            "separated_too_short_count": sum(duration_of(segment) < 1.0 for segment in separated_segments),
            "separated_files": separated_files,
            "metadata": overlap_data.get("metadata", {}),
        },
        "stage_04_asr": {
            "json_exists": bool(paths["transcript"] and paths["transcript"].exists()),
            "stats": segment_stats(transcript_segments),
            "text_lengths": {
                "text": text_len_stats(transcript_segments, "text"),
                "text_whisper": text_len_stats(transcript_segments, "text_whisper"),
                "text_phowhisper": text_len_stats(transcript_segments, "text_phowhisper"),
                "text_chunkformer": text_len_stats(transcript_segments, "text_chunkformer"),
            },
            "suspicious_count": len(suspicious),
            "likely_error_count": sum(item.get("severity") == "likely_error" for item in suspicious),
            "review_only_count": sum(item.get("severity") == "review" for item in suspicious),
            "suspicious_segments": suspicious,
            "asr_quality": asr_quality_counts(transcript_segments),
            "asr_quality_preview": asr_quality_preview(transcript_segments),
            "metadata": transcript_data.get("metadata", {}),
        },
        "stage_05_export": {
            "json_exists": bool(paths["export_json"] and paths["export_json"].exists()),
            "json_segment_count": len(exported_segments),
            "mp3_count": len(mp3_files),
            "metadata": export_data.get("metadata", {}) if isinstance(export_data, dict) else {},
        },
    }
    report["recommendations"] = recommendations(report)
    return report


def recommendations(report: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    input_audio = report["audio"]["input_audio"]
    if not input_audio.get("exists"):
        recs.append("Store the full input WAV under 00_input/full.wav so future reviews can trace the exact source.")

    diar_stats = report["stage_01_diarization"]["stats"]
    short_lt_1 = diar_stats.get("short_counts", {}).get("lt_1s", 0)
    if short_lt_1:
        recs.append(
            f"Track short diarization segments carefully; current stage 01 has {short_lt_1} segments shorter than 1s, but only merge/drop them if ASR review still flags real errors."
        )

    overlap_report = report["stage_03_overlap"]
    too_short_sep = overlap_report.get("separated_too_short_count", 0)
    if too_short_sep:
        recs.append(
            f"Skip overlap separation when the segment/overlap is shorter than 1s; current run has {too_short_sep} separated segment(s) below 1s."
        )

    asr_report = report["stage_04_asr"]
    suspicious_count = asr_report.get("suspicious_count", 0)
    likely_error_count = asr_report.get("likely_error_count", 0)
    preview_changes = asr_report.get("asr_quality_preview", {}).get("would_change_count", 0)
    quality_changed = asr_report.get("asr_quality", {}).get("changed_count", 0)
    if likely_error_count:
        if preview_changes and not quality_changed:
            recs.append(
                f"Rerun stage 04 with --asr_quality_guard to apply the previewed ASR fixes; it would change {preview_changes} segment(s)."
            )
        elif quality_changed:
            recs.append(
                f"Review remaining ASR likely-error segments after guard; current run still has {likely_error_count} likely error segment(s)."
            )
        else:
            recs.append(
                f"Enable ASR guard rules for short or boilerplate outputs; current run has {likely_error_count} likely error transcript segment(s)."
            )

    chunk_empty = asr_report.get("text_lengths", {}).get("text_chunkformer", {}).get("empty_count", 0)
    if chunk_empty:
        recs.append(
            f"Handle empty ChunkFormer outputs explicitly in ensemble voting; current run has {chunk_empty} empty ChunkFormer segment(s)."
        )

    if quality_changed:
        recs.append(f"Review ASR quality guard decisions; it changed {quality_changed} segment(s) in this run.")

    export_report = report["stage_05_export"]
    if export_report.get("json_segment_count") != export_report.get("mp3_count"):
        recs.append(
            "Check stage 05 export consistency; final JSON segment count does not match exported MP3 count."
        )

    log_counts = report.get("logs", {}).get("issue_counts", {})
    if log_counts.get("oom") or log_counts.get("killed"):
        recs.append("Reduce batch/model load or split stages further; logs contain OOM/killed signals.")
    return recs


def md_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[col])) for row in rows) for col in range(len(rows[0]))]
    lines = []
    for row_idx, row in enumerate(rows):
        line = "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        lines.append(line)
        if row_idx == 0:
            lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(row))) + " |")
    return "\n".join(lines)


def markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Run Full Eval Report")
    lines.append("")
    lines.append(f"- Run dir: `{report['run_dir']}`")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append("")

    audio_rows = [["file", "exists", "duration_sec", "sample_rate", "channels"]]
    for name, info in report["audio"].items():
        audio_rows.append(
            [
                name,
                info.get("exists", False),
                info.get("duration_sec", ""),
                info.get("sample_rate", ""),
                info.get("channels", ""),
            ]
        )
    lines.append("## Audio")
    lines.append(md_table(audio_rows))
    lines.append("")

    stage_rows = [["stage", "segments", "short<0.5s", "short<1s", "gaps", "overlaps", "extra"]]
    for key, label in [
        ("stage_01_diarization", "01 diarization"),
        ("stage_02_music_clean", "02 music clean"),
        ("stage_03_overlap", "03 overlap"),
        ("stage_04_asr", "04 asr"),
    ]:
        item = report[key]
        stats = item.get("stats", {})
        short_counts = stats.get("short_counts", {})
        extra = ""
        if key == "stage_02_music_clean":
            extra = f"demucs={item.get('demucs_flagged_count', 0)}"
        elif key == "stage_03_overlap":
            extra = f"separated={item.get('separated_count', 0)}"
        elif key == "stage_04_asr":
            extra = f"likely_error={item.get('likely_error_count', 0)}, review={item.get('review_only_count', 0)}"
        stage_rows.append(
            [
                label,
                stats.get("count", 0),
                short_counts.get("lt_0.5s", 0),
                short_counts.get("lt_1s", 0),
                stats.get("gap", {}).get("count", 0),
                stats.get("overlap", {}).get("count", 0),
                extra,
            ]
        )
    lines.append("## Stage Summary")
    lines.append(md_table(stage_rows))
    lines.append("")

    asr = report["stage_04_asr"]
    text_rows = [["field", "empty", "median_chars", "mean_chars", "max_chars"]]
    for field, stats in asr.get("text_lengths", {}).items():
        text_rows.append(
            [
                field,
                stats.get("empty_count", 0),
                stats.get("median", ""),
                stats.get("mean", ""),
                stats.get("max", ""),
            ]
        )
    lines.append("## ASR Text")
    lines.append(md_table(text_rows))
    lines.append("")

    quality = asr.get("asr_quality", {})
    quality_preview = asr.get("asr_quality_preview", {})
    lines.append("## ASR Quality Guard")
    lines.append(f"- Source counts: `{quality.get('source_counts', {})}`")
    lines.append(f"- Action counts: `{quality.get('action_counts', {})}`")
    lines.append(f"- Preview changes if guard is applied now: `{quality_preview.get('would_change_count', 0)}`")
    if quality_preview.get("changes"):
        preview_rows = [["index", "time", "actions", "old_text", "new_text"]]
        for change in quality_preview["changes"][:10]:
            old_text = str(change.get("old_text", "")).replace("\n", " ")
            new_text = str(change.get("new_text", "")).replace("\n", " ")
            if len(old_text) > 80:
                old_text = old_text[:77] + "..."
            if len(new_text) > 80:
                new_text = new_text[:77] + "..."
            preview_rows.append(
                [
                    change.get("index", ""),
                    f"{change.get('start')} - {change.get('end')}",
                    ", ".join(change.get("actions", [])),
                    old_text,
                    new_text,
                ]
            )
        lines.append(md_table(preview_rows))
    lines.append("")

    lines.append("## ASR Review Segments")
    suspicious = asr.get("suspicious_segments", [])
    if not suspicious:
        lines.append("- None")
    else:
        suspicious_rows = [["index", "severity", "time", "dur", "reasons", "text"]]
        for segment in suspicious[:20]:
            text = str(segment.get("text", "")).replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            suspicious_rows.append(
                [
                    segment.get("index", ""),
                    segment.get("severity", ""),
                    f"{segment.get('start')} - {segment.get('end')}",
                    segment.get("duration", ""),
                    ", ".join(segment.get("reasons", [])),
                    text,
                ]
            )
        lines.append(md_table(suspicious_rows))
        if len(suspicious) > 20:
            lines.append(f"- Showing first 20 of {len(suspicious)} suspicious segments.")
    lines.append("")

    export = report["stage_05_export"]
    lines.append("## Export")
    lines.append(f"- JSON segments: `{export.get('json_segment_count', 0)}`")
    lines.append(f"- MP3 files: `{export.get('mp3_count', 0)}`")
    lines.append("")

    logs = report["logs"]
    lines.append("## Logs")
    lines.append(f"- Log files: `{logs.get('files', 0)}`")
    lines.append(f"- Issue counts: `{logs.get('issue_counts', {})}`")
    if logs.get("samples"):
        lines.append("- First log samples:")
        for sample in logs["samples"][:5]:
            lines.append(f"  - `{Path(sample['file']).name}:{sample['line']}` {sample['text']}")
    lines.append("")

    lines.append("## Recommendations")
    recs = report.get("recommendations", [])
    if not recs:
        lines.append("- No automatic recommendation.")
    else:
        lines.extend(f"- {rec}" for rec in recs)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "06_eval"
    report = evaluate(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(report, out_dir / "eval_report.json")
    markdown = markdown_report(report)
    (out_dir / "eval_report.md").write_text(markdown, encoding="utf-8")
    print(f"Stage 06 eval complete: {out_dir / 'eval_report.md'}")
    if args.print_markdown:
        print()
        print(markdown)


if __name__ == "__main__":
    main()
