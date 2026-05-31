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
    parser.add_argument(
        "--partition_by_duplex_group",
        action="store_true",
        help="Export segment MP3 files under clean/review/exclude duplex train-group folders.",
    )
    parser.add_argument(
        "--expected_main_speakers",
        type=int,
        default=2,
        help="Number of main speakers to infer when transcript segments do not already contain duplex_train_group.",
    )
    parser.add_argument(
        "--main_speakers",
        default="",
        help="Optional comma-separated speaker IDs to use as main speakers instead of duration-based inference.",
    )
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
    transcript_segments = transcript["segments"]
    grouping_stats = transcript.get("metadata", {}).get("duplex_train_grouping", {})

    if args.partition_by_duplex_group and any("duplex_train_group" not in segment for segment in transcript_segments):
        configured_main_speakers = [item.strip() for item in args.main_speakers.split(",") if item.strip()] or None
        transcript_segments, grouping_stats = stage_common.assign_duplex_train_groups(
            transcript_segments,
            expected_main_speakers=args.expected_main_speakers,
            main_speakers=configured_main_speakers,
        )

    for idx, segment in enumerate(transcript_segments):
        seg_index = segment.get("index") or stage_common.normalized_index(idx)
        speaker = segment.get("speaker", "Unknown")
        if args.partition_by_duplex_group:
            group = str(segment.get("duplex_train_group") or stage_common.DUPLEX_GROUP_EXCLUDE)
            if group not in stage_common.DUPLEX_GROUPS:
                group = stage_common.DUPLEX_GROUP_EXCLUDE
            out_file = segments_dir / group / f"{seg_index}_{speaker}.mp3"
        else:
            out_file = segments_dir / f"{seg_index}_{speaker}.mp3"
        out_file.parent.mkdir(parents=True, exist_ok=True)

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
    if args.partition_by_duplex_group:
        group_counts = {group: 0 for group in stage_common.DUPLEX_GROUPS}
        for segment in exported_segments:
            group = segment.get("duplex_train_group") or stage_common.DUPLEX_GROUP_EXCLUDE
            if group not in group_counts:
                group = stage_common.DUPLEX_GROUP_EXCLUDE
            group_counts[group] += 1
        if not grouping_stats:
            main_speakers = []
            for segment in exported_segments:
                segment_main_speakers = segment.get("main_speakers")
                if isinstance(segment_main_speakers, list) and segment_main_speakers:
                    main_speakers = [str(speaker) for speaker in segment_main_speakers]
                    break
            grouping_stats = {
                "expected_main_speakers": args.expected_main_speakers,
                "main_speakers": main_speakers,
                "group_counts": group_counts,
            }
        metadata["partition_by_duplex_group"] = True
        metadata["duplex_train_grouping"] = grouping_stats
        metadata["duplex_train_group_counts"] = group_counts

    final_data = {
        "metadata": metadata,
        "segments": exported_segments,
    }

    final_json = out_dir / f"{audio_name}.json"
    stage_common.dump_json(final_data, final_json)
    print(f"Stage 05 complete: {final_json}")


if __name__ == "__main__":
    main()
