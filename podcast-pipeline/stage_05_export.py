from __future__ import annotations

import argparse
from pathlib import Path

import stage_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 05: export final JSON and MP3 segment files.")
    parser.add_argument("--transcript_json", required=True, help="Transcript JSON from stage_04_asr.py.")
    parser.add_argument("--audio", default="", help="Override full audio path. Defaults to transcript audio_path.")
    parser.add_argument("--out_dir", default="", help="Final output directory.")
    parser.add_argument("--audio_name", default="", help="Override output audio name.")
    parser.add_argument("--keep_internal_paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcript = stage_common.load_json(args.transcript_json)
    audio_path = Path(args.audio or transcript["audio_path"]).expanduser().resolve()
    audio_name = args.audio_name or transcript.get("audio_name") or stage_common.audio_name_from_path(audio_path)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.transcript_json).resolve().parent / "final"
    segments_dir = out_dir / audio_name
    segments_dir.mkdir(parents=True, exist_ok=True)

    from pydub import AudioSegment

    full_audio = AudioSegment.from_file(str(audio_path)).set_channels(1)
    exported_segments = []

    for idx, segment in enumerate(transcript["segments"]):
        seg_index = segment.get("index") or stage_common.normalized_index(idx)
        speaker = segment.get("speaker", "Unknown")
        out_file = segments_dir / f"{seg_index}_{speaker}.mp3"

        enhanced_path = segment.get("enhanced_audio_path")
        target_audio = None
        if enhanced_path and Path(enhanced_path).exists():
            enhanced_audio = AudioSegment.from_file(enhanced_path).set_channels(1)
            source_start = segment.get("source_start")
            if source_start is not None:
                rel_start_ms = max(0, int((float(segment["start"]) - float(source_start)) * 1000))
                rel_end_ms = max(rel_start_ms, int((float(segment["end"]) - float(source_start)) * 1000))
                target_audio = enhanced_audio[rel_start_ms:rel_end_ms]
                if len(target_audio) == 0:
                    target_audio = enhanced_audio
            else:
                target_audio = enhanced_audio
        else:
            start_ms = int(float(segment["start"]) * 1000)
            end_ms = max(start_ms, int(float(segment["end"]) * 1000))
            target_audio = full_audio[start_ms:end_ms]

        target_audio.export(out_file, format="mp3")

        clean_segment = dict(segment)
        if not args.keep_internal_paths:
            clean_segment.pop("enhanced_audio_path", None)
        clean_segment["audio_file"] = str(out_file.relative_to(out_dir))
        exported_segments.append(clean_segment)

    metadata = dict(transcript.get("metadata", {}))
    metadata["exported_segment_count"] = len(exported_segments)
    metadata["export_audio_path"] = str(audio_path)

    final_data = {
        "metadata": metadata,
        "segments": exported_segments,
    }

    final_json = out_dir / f"{audio_name}.json"
    stage_common.dump_json(final_data, final_json)
    print(f"Stage 05 complete: {final_json}")


if __name__ == "__main__":
    main()
