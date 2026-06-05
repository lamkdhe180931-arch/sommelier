import json
import re
import sys
from pathlib import Path


STAGES = [
    ("01_diarization", Path("01_diarization") / "diarization.json"),
    ("04_asr", Path("04_asr") / "transcript.json"),
    ("05_export", Path("05_export") / "final" / "data_audio.json"),
]


def has_trace_stages(run_dir: Path) -> bool:
    return any((run_dir / rel_path).exists() for _, rel_path in STAGES)


def discover_run_dirs(targets: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()

    for target in targets:
        target = target.expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Thư mục {target} không tồn tại!")

        candidates = [target] if has_trace_stages(target) else []
        if not candidates:
            candidates.extend(
                path
                for path in sorted(target.rglob("run_full*"))
                if path.is_dir() and has_trace_stages(path)
            )

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                run_dirs.append(resolved)

    return run_dirs


def load_stage_data(run_dir: Path) -> dict:
    stage_data = {}

    for stage_name, relative_path in STAGES:
        json_path = run_dir / relative_path
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            segments = data.get("segments", data) if isinstance(data, dict) else data

            # Fix audio paths for export stage
            if stage_name == "05_export":
                for seg in segments:
                    if "audio_file" in seg and seg["audio_file"].startswith("data_audio/"):
                        seg["audio_file"] = f"05_export/final/{seg['audio_file']}"

            # Ensure every segment has an index
            for i, seg in enumerate(segments):
                if "index" not in seg:
                    seg["index"] = f"{i:05d}"

            stage_data[stage_name] = segments

    return stage_data


def build_trace_html(run_dir: Path, template_html: str) -> Path:
    html = template_html
    stage_data = load_stage_data(run_dir)
    if not stage_data:
        raise FileNotFoundError(f"Không tìm thấy file JSON nào trong các stage của {run_dir}")

    stage_data_json = json.dumps(stage_data, ensure_ascii=False)

    # Load VAD chunks if available
    vad_chunks_path = run_dir / "01_diarization" / "vad_chunks.json"
    vad_chunks = []
    if vad_chunks_path.exists():
        try:
            vad_data = json.loads(vad_chunks_path.read_text(encoding="utf-8"))
            vad_chunks = vad_data.get("vad_chunks", [])
        except Exception as e:
            print(f"Error loading vad_chunks: {e}")
    vad_chunks_json = json.dumps(vad_chunks, ensure_ascii=False)

    # Replace the SEGMENTS block with STAGE_DATA and VAD_CHUNKS
    replacement = f'const STAGE_DATA = {stage_data_json};\nconst VAD_CHUNKS = {vad_chunks_json};\nlet current_stage = "05_export" in STAGE_DATA ? "05_export" : Object.keys(STAGE_DATA)[0];\nlet SEGMENTS = STAGE_DATA[current_stage] || [];\n'

    # Try the old SEGMENTS way first
    html, n = re.subn(r'^const SEGMENTS = \[.*\].*$', replacement, html, count=1, flags=re.MULTILINE)

    if n == 0:
        # Try replacing the fallback block instead
        fallback_pattern = r'// Initialize Stage Selector.*?\n\}'
        html, n2 = re.subn(fallback_pattern, replacement, html, count=1, flags=re.DOTALL)
        if n2 == 0:
            print("⚠️ Cảnh báo: Không tìm thấy chỗ để inject STAGE_DATA trong template.")

    # Replace the title/header with the run directory name.
    run_name = run_dir.name
    html = html.replace("Run_Full", run_name)
    html = re.sub(r'Run Full \d+', run_name, html, flags=re.IGNORECASE)

    # Detect and replace input and clean audio paths dynamically based on existing files
    audio_extensions = [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"]

    # 1. Full input audio
    full_input_audio = "00_input/full.wav"
    input_dir = run_dir / "00_input"
    if input_dir.exists():
        for ext in audio_extensions:
            if (input_dir / f"full{ext}").exists():
                full_input_audio = f"00_input/full{ext}"
                break
    html = html.replace('src="00_input/full.wav"', f'src="{full_input_audio}"')

    # 2. Preview input audio
    preview_input_audio = "00_input/preview_input_30s.wav"
    if input_dir.exists():
        for ext in audio_extensions:
            if (input_dir / f"preview_input_30s{ext}").exists():
                preview_input_audio = f"00_input/preview_input_30s{ext}"
                break
    html = html.replace('src="00_input/preview_input_30s.wav"', f'src="{preview_input_audio}"')

    # 3. Cleaned full audio
    cleaned_audio = "02_music_clean/cleaned_audio.wav"
    clean_dir = run_dir / "02_music_clean"
    if clean_dir.exists():
        for ext in audio_extensions:
            if (clean_dir / f"cleaned_audio{ext}").exists():
                cleaned_audio = f"02_music_clean/cleaned_audio{ext}"
                break
    html = html.replace('src="02_music_clean/cleaned_audio.wav"', f'src="{cleaned_audio}"')

    # 4. Preview cleaned audio
    preview_cleaned_audio = "02_music_clean/preview_cleaned_30s.wav"
    if clean_dir.exists():
        for ext in audio_extensions:
            if (clean_dir / f"preview_cleaned_30s{ext}").exists():
                preview_cleaned_audio = f"02_music_clean/preview_cleaned_30s{ext}"
                break
    html = html.replace('src="02_music_clean/preview_cleaned_30s.wav"', f'src="{preview_cleaned_audio}"')

    # Save the new file INSIDE the run_full directory so relative audio paths work
    out_path = run_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    return out_path


def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python tools/build_trace_html.py <run_full_hoặc_folder_chứa_nhiều_run_full> [...]")
        sys.exit(1)

    try:
        run_dirs = discover_run_dirs([Path(arg) for arg in sys.argv[1:]])
    except FileNotFoundError as exc:
        print(f"Lỗi: {exc}")
        sys.exit(1)

    if not run_dirs:
        print("Lỗi: Không tìm thấy thư mục run_full nào có stage JSON.")
        sys.exit(1)

    # Read the beautiful template (index.html at root)
    root_dir = Path(__file__).parent.parent
    template_path = root_dir / "index.html"
    if not template_path.exists():
        print(f"Lỗi: Không tìm thấy file {template_path} để làm template.")
        sys.exit(1)

    template_html = template_path.read_text(encoding="utf-8")
    print(f"Đang tạo HTML cho {len(run_dirs)} run_full...")

    failures = []
    for run_dir in run_dirs:
        try:
            out_path = build_trace_html(run_dir, template_html)
        except Exception as exc:
            failures.append((run_dir, exc))
            print(f"❌ Lỗi khi tạo HTML cho {run_dir.name}: {exc}")
            continue

        print(f"✅ {run_dir.name}: {out_path}")

    if failures:
        print(f"Hoàn tất với {len(failures)} lỗi.")
        sys.exit(1)

    print(f"Hoàn tất: đã tạo {len(run_dirs)} file index.html.")

if __name__ == "__main__":
    main()
