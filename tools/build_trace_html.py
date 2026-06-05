from __future__ import annotations

import argparse
import html
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


STAGES = [
    ("diarization", "01_diarization/diarization.json"),
    ("music_clean", "02_music_clean/segment_flags.json"),
    ("overlap", "03_overlap/segments.json"),
    ("asr", "04_asr/transcript.json"),
    ("export", "05_export/final/data_audio.json"),
]


@dataclass
class LoadedStage:
    name: str
    rel_path: str
    path: Path
    exists: bool
    data: dict[str, Any]
    load_error: str = ""

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = self.data.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    @property
    def segments(self) -> list[dict[str, Any]]:
        segments = self.data.get("segments", [])
        return segments if isinstance(segments, list) else []


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def json_block(value: Any) -> str:
    return esc(json.dumps(value, ensure_ascii=False, indent=2))


def rel_url(path: Path | None, html_dir: Path) -> str:
    if path is None:
        return ""
    rel = os.path.relpath(path.resolve(), html_dir.resolve()).replace(os.sep, "/")
    return quote(rel, safe="/:._-~()")


def resolve_run_path(run_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return run_dir / path


def find_run_dir(path: Path) -> Path:
    if (path / "01_diarization" / "diarization.json").exists() or path.name.startswith("run_full"):
        return path
    candidates = sorted(path.rglob("01_diarization/diarization.json"))
    if candidates:
        return candidates[0].parents[1]
    raise FileNotFoundError(f"Could not find run_full trace artifacts under {path}")


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(extract_dir)
    return find_run_dir(extract_dir)


def input_to_run_dir(input_path: Path, out_path: Path, extract_dir: Path | None) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        assets_dir = extract_dir or out_path.with_name(out_path.stem + "_assets")
        return safe_extract_zip(input_path, assets_dir.expanduser().resolve())
    if input_path.is_dir():
        return find_run_dir(input_path)
    raise ValueError(f"Input must be a run_full folder, a folder containing run_full, or a zip: {input_path}")


def load_stage(run_dir: Path, name: str, rel_path: str) -> LoadedStage:
    path = run_dir / rel_path
    if not path.exists():
        return LoadedStage(name=name, rel_path=rel_path, path=path, exists=False, data={})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return LoadedStage(name=name, rel_path=rel_path, path=path, exists=True, data={}, load_error="json root is not object")
        return LoadedStage(name=name, rel_path=rel_path, path=path, exists=True, data=data)
    except Exception as exc:
        return LoadedStage(name=name, rel_path=rel_path, path=path, exists=True, data={}, load_error=repr(exc))


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def check_row(rows: list[dict[str, Any]], status: str, stage: str, code: str, path: str, segment: Any = "", value: Any = "") -> None:
    rows.append(
        {
            "status": status,
            "stage": stage,
            "code": code,
            "path": path,
            "segment": segment,
            "value": value,
        }
    )


def build_checks(stages: dict[str, LoadedStage], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for stage in stages.values():
        if not stage.exists:
            check_row(rows, "ERROR", stage.name, "missing_json", stage.rel_path)
            continue
        if stage.load_error:
            check_row(rows, "ERROR", stage.name, "json_load_error", stage.rel_path, value=stage.load_error)
            continue
        metadata = stage.metadata
        if metadata.get("proxy") is True:
            raise ValueError(f"proxy artifact is not allowed: {stage.rel_path}")
        check_row(rows, "OK", stage.name, "json_exists", stage.rel_path)
        check_row(rows, "OK" if metadata.get("trace") is True else "ERROR", stage.name, "metadata.trace", stage.rel_path, value=metadata.get("trace"))
        check_row(rows, "OK" if metadata.get("proxy") is False else "ERROR", stage.name, "metadata.proxy", stage.rel_path, value=metadata.get("proxy"))
        check_row(rows, "OK", stage.name, "segment_count", stage.rel_path, value=len(stage.segments))

    diarization = stages["diarization"]
    for seg in diarization.segments:
        idx = seg.get("index", "")
        start = to_float(seg.get("start"))
        end = to_float(seg.get("end"))
        duration = None if start is None or end is None else end - start
        if duration is None or duration <= 0:
            check_row(rows, "ERROR", "diarization", "segment_duration", diarization.rel_path, idx, duration)
        if not seg.get("speaker"):
            check_row(rows, "ERROR", "diarization", "speaker", diarization.rel_path, idx, seg.get("speaker"))

    music = stages["music_clean"]
    if music.exists and not music.load_error:
        flags = music.data.get("segment_demucs_flags", [])
        status = "OK" if isinstance(flags, list) and len(flags) == len(music.segments) else "ERROR"
        check_row(rows, status, "music_clean", "segment_demucs_flags_count", music.rel_path, value=f"{len(flags) if isinstance(flags, list) else 'not_list'}/{len(music.segments)}")
        cleaned_audio = resolve_run_path(run_dir, music.data.get("audio_path") or "02_music_clean/cleaned_audio.wav")
        check_row(rows, "OK" if cleaned_audio and cleaned_audio.exists() else "ERROR", "music_clean", "cleaned_audio", "02_music_clean/cleaned_audio.wav", value=cleaned_audio)

    overlap = stages["overlap"]
    for seg in overlap.segments:
        enhanced_path = seg.get("enhanced_audio_path")
        if enhanced_path:
            resolved = resolve_run_path(run_dir, enhanced_path)
            check_row(rows, "OK" if resolved and resolved.exists() else "ERROR", "overlap", "enhanced_audio_path", str(enhanced_path), seg.get("index", ""), resolved)

    asr = stages["asr"]
    for seg in asr.segments:
        if not str(seg.get("text") or "").strip():
            check_row(rows, "ERROR", "asr", "text", asr.rel_path, seg.get("index", ""), seg.get("text"))

    export = stages["export"]
    for seg in export.segments:
        audio_file = seg.get("audio_file")
        audio_path = run_dir / "05_export" / "final" / str(audio_file or "")
        check_row(rows, "OK" if audio_file and audio_path.exists() else "ERROR", "export", "audio_file", str(audio_file or ""), seg.get("index", ""), audio_path if audio_file else "")

    return rows


def stage_summary_rows(stages: dict[str, LoadedStage]) -> str:
    rows = []
    for stage in stages.values():
        metadata = stage.metadata
        status = "MISSING"
        if stage.exists and stage.load_error:
            status = "INVALID_JSON"
        elif stage.exists:
            status = "OK"
        rows.append(
            "<tr>"
            f"<td>{esc(stage.name)}</td>"
            f"<td>{esc(status)}</td>"
            f"<td class='mono'>{esc(stage.rel_path)}</td>"
            f"<td>{esc(len(stage.segments))}</td>"
            f"<td><pre>{json_block({'trace': metadata.get('trace'), 'proxy': metadata.get('proxy'), 'stage': metadata.get('stage')})}</pre></td>"
            "</tr>"
        )
    return "\n".join(rows)


def check_rows_html(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        cls = "ok" if row["status"] == "OK" else "error"
        out.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(row['status'])}</td>"
            f"<td>{esc(row['stage'])}</td>"
            f"<td>{esc(row['code'])}</td>"
            f"<td class='mono'>{esc(row['path'])}</td>"
            f"<td class='mono'>{esc(row['segment'])}</td>"
            f"<td class='mono'>{esc(row['value'])}</td>"
            "</tr>"
        )
    return "\n".join(out)


def segment_matrix_html(stages: dict[str, LoadedStage], run_dir: Path, html_dir: Path, max_segments: int) -> str:
    export_segments = stages["export"].segments
    asr_by_idx = {str(seg.get("index")): seg for seg in stages["asr"].segments if seg.get("index") is not None}
    overlap_by_idx = {str(seg.get("index")): seg for seg in stages["overlap"].segments if seg.get("index") is not None}
    music_flags = stages["music_clean"].data.get("segment_demucs_flags", [])
    rows = []

    for i, seg in enumerate(export_segments[:max_segments]):
        idx = str(seg.get("index", f"{i:05d}"))
        asr_seg = asr_by_idx.get(idx, {})
        overlap_seg = overlap_by_idx.get(idx, {})
        audio_file = seg.get("audio_file")
        audio_url = ""
        if audio_file:
            audio_url = rel_url(run_dir / "05_export" / "final" / str(audio_file), html_dir)
        enhanced_url = ""
        if overlap_seg.get("enhanced_audio_path"):
            enhanced_url = rel_url(resolve_run_path(run_dir, overlap_seg.get("enhanced_audio_path")), html_dir)
        demucs_value = music_flags[i] if isinstance(music_flags, list) and i < len(music_flags) else ""
        rows.append(
            "<tr>"
            f"<td class='mono'>{esc(idx)}</td>"
            f"<td class='mono'>{esc(seg.get('start'))}</td>"
            f"<td class='mono'>{esc(seg.get('end'))}</td>"
            f"<td>{esc(seg.get('speaker'))}</td>"
            f"<td>{esc(demucs_value)}</td>"
            f"<td>{esc(overlap_seg.get('has_overlap'))}</td>"
            f"<td>{esc(overlap_seg.get('is_separated'))}</td>"
            f"<td>{esc(asr_seg.get('text_whisper'))}</td>"
            f"<td>{esc(asr_seg.get('text_phowhisper'))}</td>"
            f"<td>{esc(asr_seg.get('text_chunkformer'))}</td>"
            f"<td>{esc(seg.get('text'))}</td>"
            f"<td class='mono'>{esc(audio_file)}"
            + (f"<br><audio controls preload='metadata' src='{esc(audio_url)}'></audio>" if audio_url else "")
            + "</td>"
            f"<td class='mono'>{esc(overlap_seg.get('enhanced_audio_path'))}"
            + (f"<br><audio controls preload='metadata' src='{esc(enhanced_url)}'></audio>" if enhanced_url else "")
            + "</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_html(input_path: Path, out_path: Path, extract_dir: Path | None = None, max_segments: int = 2000) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = input_to_run_dir(input_path, out_path, extract_dir)
    html_dir = out_path.parent
    stages = {name: load_stage(run_dir, name, rel_path) for name, rel_path in STAGES}
    checks = build_checks(stages, run_dir)
    error_count = sum(1 for row in checks if row["status"] != "OK")
    ok_count = sum(1 for row in checks if row["status"] == "OK")
    export_count = len(stages["export"].segments)

    html_text = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sommelier True Trace Report</title>
  <style>
    :root {{ --bg:#f7f8fb; --panel:#fff; --text:#172033; --muted:#667085; --line:#d8dee9; --ok:#067647; --err:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line); padding:16px 20px; }}
    h1 {{ margin:0 0 6px; font-size:22px; }}
    .sub {{ color:var(--muted); font-size:13px; }}
    main {{ padding:16px 20px 28px; max-width:1600px; margin:auto; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:14px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:12px; }}
    .card b {{ display:block; font-size:24px; margin-bottom:3px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:7px; margin-bottom:14px; overflow:hidden; }}
    h2 {{ font-size:15px; margin:0; padding:11px 12px; border-bottom:1px solid var(--line); }}
    .table-wrap {{ overflow:auto; max-height:70vh; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border-bottom:1px solid #edf0f5; padding:7px 8px; text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#fbfcfe; z-index:1; }}
    tr.ok td:first-child {{ color:var(--ok); font-weight:700; }}
    tr.error td:first-child {{ color:var(--err); font-weight:700; }}
    .mono, pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
    pre {{ white-space:pre-wrap; margin:0; max-width:460px; }}
    audio {{ width:240px; height:32px; }}
  </style>
</head>
<body>
<header>
  <h1>Sommelier True Trace Report</h1>
  <div class="sub">run_dir={esc(run_dir)} | generated_at={esc(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
</header>
<main>
  <div class="cards">
    <div class="card"><b>{esc(export_count)}</b>export segments</div>
    <div class="card"><b>{esc(ok_count)}</b>OK checks</div>
    <div class="card"><b>{esc(error_count)}</b>ERROR checks</div>
    <div class="card"><b>{esc(sum(1 for stage in stages.values() if stage.exists))}/5</b>stage JSON files</div>
  </div>
  <section>
    <h2>Stage Artifacts</h2>
    <div class="table-wrap"><table><thead><tr><th>stage</th><th>status</th><th>json</th><th>segments</th><th>metadata subset</th></tr></thead><tbody>
      {stage_summary_rows(stages)}
    </tbody></table></div>
  </section>
  <section>
    <h2>Factual Checks</h2>
    <div class="table-wrap"><table><thead><tr><th>status</th><th>stage</th><th>code</th><th>path</th><th>segment</th><th>value</th></tr></thead><tbody>
      {check_rows_html(checks)}
    </tbody></table></div>
  </section>
  <section>
    <h2>Segment Matrix</h2>
    <div class="table-wrap"><table><thead><tr><th>index</th><th>start</th><th>end</th><th>speaker</th><th>demucs flag</th><th>has_overlap</th><th>is_separated</th><th>text_whisper</th><th>text_phowhisper</th><th>text_chunkformer</th><th>final text</th><th>mp3</th><th>enhanced wav</th></tr></thead><tbody>
      {segment_matrix_html(stages, run_dir, html_dir, max_segments)}
    </tbody></table></div>
  </section>
  <section>
    <h2>Raw Metadata</h2>
    <div class="table-wrap"><table><thead><tr><th>stage</th><th>metadata</th></tr></thead><tbody>
      {"".join(f"<tr><td>{esc(stage.name)}</td><td><pre>{json_block(stage.metadata)}</pre></td></tr>" for stage in stages.values())}
    </tbody></table></div>
  </section>
</main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def default_out_path(input_path: Path) -> Path:
    input_path = input_path.expanduser()
    if input_path.is_file():
        return input_path.with_name(input_path.stem + "_trace_report.html")
    return input_path / "trace_report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a factual static HTML report from a run_full folder or zip.")
    parser.add_argument("input", help="run_full folder, folder containing run_full, or run_full_download.zip")
    parser.add_argument("--out", default="", help="Output HTML path.")
    parser.add_argument("--extract-dir", default="", help="Directory used when input is a zip.")
    parser.add_argument("--max-segments", type=int, default=2000, help="Maximum export segments to render in the matrix.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out).expanduser() if args.out else default_out_path(input_path)
    extract_dir = Path(args.extract_dir).expanduser() if args.extract_dir else None
    try:
        result = build_html(input_path, out_path, extract_dir=extract_dir, max_segments=args.max_segments)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(result)


if __name__ == "__main__":
    main()
