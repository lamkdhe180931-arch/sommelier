from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote


STAGE_DEFS = [
    {
        "id": "1",
        "title": "Stage 1 - Diarization",
        "json": "01_diarization/diarization.json",
        "audio": "00_input/full.wav",
        "log": "logs/18_stage_01_diarize.log",
    },
    {
        "id": "2",
        "title": "Stage 2 - Music Clean",
        "json": "02_music_clean/segment_flags.json",
        "audio": "02_music_clean/cleaned_audio.wav",
        "log": "logs/19_stage_02_music_clean.log",
    },
    {
        "id": "3",
        "title": "Stage 3 - Overlap Separation",
        "json": "03_overlap/segments.json",
        "audio": "02_music_clean/cleaned_audio.wav",
        "log": "logs/20_stage_03_overlap_separate.log",
    },
    {
        "id": "4",
        "title": "Stage 4 - ASR",
        "json": "04_asr/transcript.json",
        "audio": "02_music_clean/cleaned_audio.wav",
        "log": "logs/22_stage_04_asr.log",
    },
    {
        "id": "5",
        "title": "Stage 5 - Export",
        "json": "05_export/final/data_audio.json",
        "audio": "02_music_clean/cleaned_audio.wav",
        "log": "logs/23_stage_05_export.log",
        "final_dir": "05_export/final",
    },
    {
        "id": "6",
        "title": "Stage 6 - Eval",
        "json": "06_eval/eval_report.json",
        "markdown": "06_eval/eval_report.md",
        "audio": "02_music_clean/cleaned_audio.wav",
        "log": "logs/24_stage_06_eval.log",
    },
]


SEGMENT_FIELDS = [
    "index",
    "start",
    "end",
    "duration",
    "speaker",
    "text",
    "correct_script",
    "text_whisper",
    "text_phowhisper",
    "text_chunkformer",
    "train_quality_label",
    "duplex_train_group",
    "duplex_group_reason",
    "is_main_speaker",
    "needs_manual_review",
    "is_short_backchannel",
    "has_overlap",
    "overlap_count",
    "overlap_total_seconds",
    "separation_status",
    "assignment_confidence",
    "assignment_margin",
    "low_confidence_overlap_count",
    "is_separated",
    "demucs",
    "asr_quality_source",
    "asr_quality_actions",
    "audio_file",
]


PAIR_AUDIO_FIELDS = [
    "window_audio_path",
    "raw_stem1_audio_path",
    "raw_stem2_audio_path",
    "stem1_audio_path",
    "stem2_audio_path",
    "assigned_seg1_audio_path",
    "assigned_seg2_audio_path",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": repr(exc)}


def read_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def natural_run_key(path: Path) -> tuple[int, str]:
    name = path.name
    suffix = name.replace("run_full", "").strip()
    if suffix.isdigit():
        return (int(suffix), name)
    if suffix == "":
        return (0, name)
    return (10_000, name)


def rel_url(path: Path | None, html_dir: Path) -> str:
    if path is None or not path.exists():
        return ""
    rel = os.path.relpath(path.resolve(), html_dir.resolve()).replace(os.sep, "/")
    return quote(rel, safe="/:._-~()")


def resolve_artifact_path(value: str | None, run_root: Path) -> Path | None:
    if not value:
        return None
    raw = str(value)
    candidates: list[Path] = []
    path = Path(raw)
    if path.exists():
        candidates.append(path)
    if not path.is_absolute():
        candidates.append(run_root / path)

    kaggle_prefixes = ["/kaggle/working/run_full/", "/kaggle/working/run_full"]
    for prefix in kaggle_prefixes:
        if raw.startswith(prefix):
            suffix = raw[len(prefix) :].lstrip("/")
            candidates.append(run_root / suffix)

    for marker in ("/00_input/", "/01_diarization/", "/02_music_clean/", "/03_overlap/", "/04_asr/", "/05_export/", "/06_eval/"):
        if marker in raw:
            candidates.append(run_root / raw[raw.index(marker) + 1 :])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_known_path(run_root: Path, rel: str) -> Path | None:
    path = run_root / rel
    return path if path.exists() else None


def resolve_stage_audio(run_root: Path, stage: dict, data: dict) -> Path | None:
    candidates: list[Path | None] = [
        resolve_known_path(run_root, stage["audio"]),
        resolve_artifact_path(data.get("audio_path"), run_root) if isinstance(data, dict) else None,
        resolve_artifact_path(data.get("source_audio_path"), run_root) if isinstance(data, dict) else None,
        resolve_known_path(run_root, "00_input/full.wav"),
        resolve_known_path(run_root, "02_music_clean/cleaned_audio.wav"),
        resolve_known_path(run_root, "00_input/preview_input_30s.wav"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def clean_segment(segment: dict, run_root: Path, html_dir: Path, stage: dict) -> dict:
    row = {key: segment.get(key) for key in SEGMENT_FIELDS if key in segment}
    row["start"] = segment.get("start", row.get("start"))
    row["end"] = segment.get("end", row.get("end"))

    enhanced = resolve_artifact_path(segment.get("enhanced_audio_path"), run_root)
    row["enhanced_audio_url"] = rel_url(enhanced, html_dir)

    audio_file = segment.get("audio_file")
    final_dir = stage.get("final_dir")
    if audio_file and final_dir:
        mp3 = run_root / str(final_dir) / str(audio_file)
        row["segment_audio_url"] = rel_url(mp3, html_dir)

    return row


def stage_summary(data: dict, segments: list[dict]) -> dict:
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    speakers: dict[str, int] = {}
    groups: dict[str, int] = {}
    for segment in segments:
        speaker = str(segment.get("speaker", "") or "")
        group = str(segment.get("duplex_train_group", "") or "")
        if speaker:
            speakers[speaker] = speakers.get(speaker, 0) + 1
        if group:
            groups[group] = groups.get(group, 0) + 1
    return {
        "segment_count": len(segments),
        "speakers": speakers,
        "groups": groups,
        "metadata": metadata,
    }


def build_stage(run_root: Path, stage: dict, html_dir: Path) -> dict:
    json_path = run_root / stage["json"]
    data = load_json(json_path)
    segments_raw = data.get("segments", []) if isinstance(data, dict) else []

    if stage["id"] == "2" and isinstance(data, dict):
        flags = data.get("segment_demucs_flags", [])
        patched = []
        for idx, segment in enumerate(segments_raw):
            clean = dict(segment)
            if idx < len(flags):
                clean["demucs"] = bool(flags[idx])
            patched.append(clean)
        segments_raw = patched

    segments = [clean_segment(segment, run_root, html_dir, stage) for segment in segments_raw[:2000]]

    pairs = []
    if stage["id"] == "3" and isinstance(data, dict):
        for pair in data.get("overlap_pairs", [])[:1000]:
            clean_pair = {
                "seg1_index": pair.get("seg1_index"),
                "seg2_index": pair.get("seg2_index"),
                "seg1_speaker": pair.get("seg1_speaker"),
                "seg2_speaker": pair.get("seg2_speaker"),
                "overlap_start": pair.get("overlap_start"),
                "overlap_end": pair.get("overlap_end"),
                "overlap_duration": pair.get("overlap_duration"),
                "separation_status": pair.get("separation_status"),
                "assignment_method": pair.get("assignment_method"),
                "assignment_accepted": pair.get("assignment_accepted"),
                "assignment_confidence": pair.get("assignment_confidence"),
                "assignment_margin": pair.get("assignment_margin"),
                "assignment_reject_reasons": pair.get("assignment_reject_reasons"),
            }
            for field in PAIR_AUDIO_FIELDS:
                clean_pair[field.replace("_path", "_url")] = rel_url(resolve_artifact_path(pair.get(field), run_root), html_dir)
            pairs.append(clean_pair)

    markdown = ""
    if stage.get("markdown"):
        md_path = run_root / stage["markdown"]
        if md_path.exists():
            markdown = md_path.read_text(encoding="utf-8", errors="replace")

    audio_path = resolve_stage_audio(run_root, stage, data)
    return {
        "id": stage["id"],
        "title": stage["title"],
        "exists": json_path.exists(),
        "json_url": rel_url(json_path, html_dir),
        "audio_url": rel_url(audio_path, html_dir),
        "log_tail": read_tail(run_root / stage["log"]),
        "segments": segments,
        "pairs": pairs,
        "markdown": markdown,
        "summary": stage_summary(data, segments),
    }


def build_runs(root: Path, html_dir: Path) -> list[dict]:
    runs = []
    run_roots: list[Path] = []
    if (root / "01_diarization" / "diarization.json").exists():
        run_roots.append(root)
    for candidate in sorted(root.glob("run_full*"), key=natural_run_key):
        if candidate not in run_roots:
            run_roots.append(candidate)

    for run_root in run_roots:
        if not run_root.is_dir():
            continue
        stages = [build_stage(run_root, stage, html_dir) for stage in STAGE_DEFS]
        runs.append(
            {
                "name": run_root.name,
                "root_url": rel_url(run_root, html_dir),
                "stages": stages,
            }
        )
    return runs


HTML_TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Run Full Viewer</title>
  <style>
    :root { color-scheme: light; --border:#d8dee8; --muted:#667085; --bg:#f7f8fb; --text:#172033; --blue:#1f6feb; --red:#b42318; --green:#067647; --amber:#b54708; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }
    header { min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 8px 18px; background: white; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 5; }
    h1 { font-size: 18px; margin: 0; font-weight: 700; }
    .header-actions { display: flex; align-items: center; justify-content: flex-end; gap: 10px; flex-wrap: wrap; }
    .layout { display: grid; grid-template-columns: 260px 1fr; min-height: calc(100vh - 56px); }
    aside { background: white; border-right: 1px solid var(--border); padding: 14px; overflow: auto; }
    main { padding: 16px; overflow: auto; }
    .run-btn, .stage-btn, button { border: 1px solid var(--border); background: white; color: var(--text); border-radius: 7px; padding: 8px 10px; cursor: pointer; font-size: 13px; }
    .run-btn { width: 100%; text-align: left; margin-bottom: 8px; display: flex; justify-content: space-between; gap: 8px; }
    .run-btn.active, .stage-btn.active { border-color: var(--blue); color: var(--blue); background: #eef5ff; }
    .stage-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    .panel { background: white; border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .summary { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .chip { display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px; border-radius: 999px; background: #f2f4f7; font-size: 12px; color: #344054; }
    .chip.green { color: var(--green); background: #ecfdf3; }
    .chip.red { color: var(--red); background: #fef3f2; }
    .chip.amber { color: var(--amber); background: #fffaeb; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; align-items: center; }
    input, select { height: 34px; border: 1px solid var(--border); border-radius: 7px; padding: 0 10px; min-width: 180px; background: white; }
    .correct-script { width: 100%; min-width: 260px; min-height: 68px; resize: vertical; border: 1px solid var(--border); border-radius: 7px; padding: 7px 8px; font: inherit; line-height: 1.35; color: var(--text); background: white; }
    audio { width: 100%; height: 36px; margin-top: 8px; }
    .table-wrap { overflow: auto; max-height: 65vh; border: 1px solid var(--border); border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; }
    th, td { border-bottom: 1px solid #eef0f4; padding: 7px 8px; vertical-align: top; text-align: left; }
    th { position: sticky; top: 0; background: #f9fafb; z-index: 1; font-weight: 700; white-space: nowrap; }
    td { max-width: 340px; }
    .text-cell { white-space: normal; min-width: 280px; line-height: 1.35; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .muted { color: var(--muted); }
    .row-review { background: #fffbeb; }
    .row-exclude { background: #fff5f5; }
    .row-clean { background: #f6fef9; }
    .actions { display: flex; gap: 5px; flex-wrap: wrap; min-width: 130px; }
    .small { font-size: 12px; padding: 5px 7px; }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 260px; overflow: auto; background: #101828; color: #f2f4f7; padding: 10px; border-radius: 7px; font-size: 12px; }
    details summary { cursor: pointer; font-weight: 700; margin-bottom: 8px; }
    .empty { color: var(--muted); padding: 24px; text-align: center; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--border); } }
  </style>
</head>
<body>
  <header>
    <h1>Run Full Viewer</h1>
    <div class="header-actions">
      <button class="small" type="button" onclick="saveAnnotatedHtml()">Save annotated HTML</button>
      <span class="muted" id="annotationStatus"></span>
      <span class="muted" id="generatedAt"></span>
    </div>
  </header>
  <div class="layout">
    <aside>
      <div class="muted" style="margin-bottom:8px">Runs</div>
      <div id="runList"></div>
    </aside>
    <main>
      <div class="stage-tabs" id="stageTabs"></div>
      <section class="panel" id="summaryPanel"></section>
      <section class="panel">
        <div class="muted">Audio player</div>
        <audio id="player" controls preload="metadata"></audio>
        <div class="muted mono" id="nowPlaying"></div>
      </section>
      <section class="panel" id="content"></section>
      <section class="panel" id="pairPanel" style="display:none"></section>
      <section class="panel">
        <details>
          <summary>Log tail</summary>
          <pre id="logTail"></pre>
        </details>
      </section>
    </main>
  </div>
  <script>
    const DATA = __DATA__;
    let currentRun = 0;
    let currentStage = "1";
    let stopTimer = null;
    const ANNOTATION_PREFIX = "sommelier-run-viewer-correct-script:";

    const player = document.getElementById("player");
    const nowPlaying = document.getElementById("nowPlaying");
    document.getElementById("generatedAt").textContent = `Generated ${DATA.generated_at}`;

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
    }

    function fmtTime(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(2) : "";
    }

    function activeRun() { return DATA.runs[currentRun]; }
    function activeStage() { return activeRun().stages.find(s => s.id === currentStage) || activeRun().stages[0]; }

    function annotationStorageKey(runName, stageId, row, i) {
      return `${ANNOTATION_PREFIX}${runName}:${stageId}:${row.index ?? i}`;
    }

    function hydrateAnnotationsFromStorage() {
      DATA.runs.forEach(run => {
        run.stages.forEach(stage => {
          (stage.segments || []).forEach((row, i) => {
            const stored = localStorage.getItem(annotationStorageKey(run.name, stage.id, row, i));
            if (stored !== null) row.correct_script = stored;
          });
        });
      });
    }

    function setAnnotationStatus(message) {
      const el = document.getElementById("annotationStatus");
      if (!el) return;
      el.textContent = message;
      if (message) {
        window.setTimeout(() => {
          if (el.textContent === message) el.textContent = "";
        }, 2500);
      }
    }

    function autosizeCorrectScript(area) {
      area.style.height = "auto";
      area.style.height = `${Math.max(68, area.scrollHeight)}px`;
    }

    function updateCorrectScript(area) {
      const rowIndex = Number(area.dataset.rowIndex);
      const stage = activeStage();
      const row = (stage.segments || [])[rowIndex];
      if (!row) return;
      row.correct_script = area.value;
      localStorage.setItem(annotationStorageKey(activeRun().name, stage.id, row, rowIndex), area.value);
      autosizeCorrectScript(area);
      setAnnotationStatus("Draft saved");
    }

    function currentFileName() {
      const name = decodeURIComponent((location.pathname.split("/").pop() || "").trim());
      return name || "index.html";
    }

    function annotatedHtml() {
      hydrateAnnotationsFromStorage();
      const replacement = `const DATA = ${JSON.stringify(DATA)};\n    let currentRun`;
      const html = "<!doctype html>\n" + document.documentElement.outerHTML + "\n";
      return html.replace(/const DATA = [\s\S]*?;\n    let currentRun/, replacement);
    }

    async function saveAnnotatedHtml() {
      const blob = new Blob([annotatedHtml()], { type: "text/html;charset=utf-8" });
      const suggestedName = currentFileName();

      if (window.showSaveFilePicker) {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName,
            types: [
              {
                description: "HTML viewer",
                accept: { "text/html": [".html", ".htm"] },
              },
            ],
          });
          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
          setAnnotationStatus("Annotated HTML saved");
          return;
        } catch (error) {
          if (error && error.name === "AbortError") {
            setAnnotationStatus("Save canceled");
            return;
          }
        }
      }

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = suggestedName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setAnnotationStatus("Annotated HTML downloaded");
    }

    function renderRuns() {
      const el = document.getElementById("runList");
      el.innerHTML = DATA.runs.map((run, i) => {
        const ready = run.stages.filter(s => s.exists).length;
        return `<button class="run-btn ${i === currentRun ? "active" : ""}" onclick="selectRun(${i})"><span>${esc(run.name)}</span><span class="muted">${ready}/6</span></button>`;
      }).join("");
    }

    function renderStages() {
      const el = document.getElementById("stageTabs");
      el.innerHTML = activeRun().stages.map(stage =>
        `<button class="stage-btn ${stage.id === currentStage ? "active" : ""}" onclick="selectStage('${stage.id}')">${esc(stage.title)} ${stage.exists ? "" : "· missing"}</button>`
      ).join("");
    }

    function selectRun(i) {
      currentRun = i;
      const firstExisting = activeRun().stages.find(s => s.exists);
      currentStage = firstExisting ? firstExisting.id : "1";
      render();
    }

    function selectStage(id) {
      currentStage = id;
      render();
    }

    function chip(label, value, cls="") {
      if (value === undefined || value === null || value === "") return "";
      return `<span class="chip ${cls}">${esc(label)}: <b>${esc(value)}</b></span>`;
    }

    function renderSummary(stage) {
      const s = stage.summary || {};
      const meta = s.metadata || {};
      const groups = s.groups || {};
      const speakers = s.speakers || {};
      const topSpeakers = Object.entries(speakers).slice(0, 8).map(([k,v]) => `${k}:${v}`).join("  ");
      const groupText = Object.entries(groups).map(([k,v]) => `${k}:${v}`).join("  ");
      document.getElementById("summaryPanel").innerHTML = `
        <div class="summary">
          ${chip("run", activeRun().name)}
          ${chip("stage", stage.title)}
          ${chip("segments", s.segment_count ?? 0)}
          ${chip("speakers", topSpeakers || "-")}
          ${chip("groups", groupText || "-")}
          ${chip("rt", meta.rt_factor ? Number(meta.rt_factor).toFixed(3) : "")}
          ${chip("separated", meta.separated_segments_count)}
          ${chip("low-conf", meta.low_confidence_pairs_count, meta.low_confidence_pairs_count ? "amber" : "")}
        </div>
      `;
    }

    function rowClass(row) {
      const group = row.duplex_train_group || "";
      if (group.includes("clean")) return "row-clean";
      if (group.includes("exclude")) return "row-exclude";
      if (group || row.needs_manual_review || row.has_overlap || row.separation_status === "low_confidence") return "row-review";
      return "";
    }

    function playRange(url, start, end, label) {
      if (!url) return;
      if (stopTimer) clearInterval(stopTimer);
      const startNum = Number(start) || 0;
      const endNum = Number(end);
      nowPlaying.textContent = `${label || url}  [${fmtTime(start)} - ${fmtTime(end)}]`;
      const startPlayback = () => {
        try {
          player.currentTime = Math.max(0, startNum);
        } catch (err) {
          console.warn("seek failed", err);
        }
        const playPromise = player.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(err => {
            nowPlaying.textContent = `Không play được: ${err.message || err}`;
          });
        }
      };
      if (player.getAttribute("src") !== url) {
        player.onloadedmetadata = () => {
          player.onloadedmetadata = null;
          startPlayback();
        };
        player.src = url;
        player.load();
      } else {
        startPlayback();
      }
      if (Number.isFinite(endNum) && endNum > startNum) {
        stopTimer = setInterval(() => {
          if (player.currentTime >= endNum) {
            player.pause();
            clearInterval(stopTimer);
            stopTimer = null;
          }
        }, 80);
      }
    }

    function playFile(url, label) {
      if (!url) return;
      if (stopTimer) clearInterval(stopTimer);
      nowPlaying.textContent = label || url;
      const startPlayback = () => {
        try {
          player.currentTime = 0;
        } catch (err) {
          console.warn("seek failed", err);
        }
        const playPromise = player.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(err => {
            nowPlaying.textContent = `Không play được: ${err.message || err}`;
          });
        }
      };
      if (player.getAttribute("src") !== url) {
        player.onloadedmetadata = () => {
          player.onloadedmetadata = null;
          startPlayback();
        };
        player.src = url;
        player.load();
      } else {
        startPlayback();
      }
    }

    function renderRows(stage) {
      const rows = stage.segments || [];
      if (!stage.exists) return `<div class="empty">Stage output chưa tồn tại.</div>`;
      if (!rows.length) return `<div class="empty">Không có segment để hiển thị.</div>`;
      const columns = [
        ["actions", ""],
        ["index", "idx"],
        ["start", "start"],
        ["end", "end"],
        ["speaker", "speaker"],
        ["duplex_train_group", "group"],
        ["duplex_group_reason", "reason"],
        ["train_quality_label", "label"],
        ["separation_status", "sep"],
        ["has_overlap", "ov"],
        ["text", "text"],
        ["correct_script", "correct script"],
        ["text_whisper", "whisper"],
        ["text_phowhisper", "pho"],
        ["text_chunkformer", "chunk"],
      ];
      const head = columns.map(([, title]) => `<th>${esc(title)}</th>`).join("");
      const body = rows.map((row, i) => {
        const actions = `
          <div class="actions">
            ${stage.audio_url ? `<button class="small" onclick="playRange('${stage.audio_url}', ${Number(row.start)||0}, ${Number(row.end)||0}, '${esc(row.index || i)}')">source</button>` : ""}
            ${row.enhanced_audio_url ? `<button class="small" onclick="playFile('${row.enhanced_audio_url}', 'enhanced ${esc(row.index || i)}')">enh</button>` : ""}
            ${row.segment_audio_url ? `<button class="small" onclick="playFile('${row.segment_audio_url}', 'mp3 ${esc(row.index || i)}')">mp3</button>` : ""}
          </div>`;
        const cells = columns.map(([key]) => {
          if (key === "actions") return `<td>${actions}</td>`;
          if (key === "start" || key === "end") return `<td class="mono">${fmtTime(row[key])}</td>`;
          if (key === "correct_script") {
            return `<td class="text-cell"><textarea class="correct-script" data-row-index="${i}" data-segment-index="${esc(row.index || i)}" oninput="updateCorrectScript(this)" onfocus="autosizeCorrectScript(this)" spellcheck="false">${esc(row.correct_script || "")}</textarea></td>`;
          }
          const cls = key.startsWith("text") ? "text-cell" : "";
          return `<td class="${cls}">${esc(Array.isArray(row[key]) ? row[key].join(", ") : row[key])}</td>`;
        }).join("");
        return `<tr class="${rowClass(row)}">${cells}</tr>`;
      }).join("");
      return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody id="segmentBody">${body}</tbody></table></div>`;
    }

    function renderPairs(stage) {
      const panel = document.getElementById("pairPanel");
      const pairs = stage.pairs || [];
      if (!pairs.length) {
        panel.style.display = "none";
        panel.innerHTML = "";
        return;
      }
      panel.style.display = "";
      const rows = pairs.map((p, i) => `
        <tr>
          <td class="mono">${i}</td>
          <td>${esc(p.seg1_index)} / ${esc(p.seg2_index)}</td>
          <td>${esc(p.seg1_speaker)} / ${esc(p.seg2_speaker)}</td>
          <td class="mono">${fmtTime(p.overlap_start)}-${fmtTime(p.overlap_end)} (${fmtTime(p.overlap_duration)})</td>
          <td>${esc(p.separation_status)} ${esc(p.assignment_method || "")}</td>
          <td class="mono">${esc(p.assignment_confidence ?? "")} / ${esc(p.assignment_margin ?? "")}</td>
          <td><div class="actions">
            ${p.window_audio_url ? `<button class="small" onclick="playFile('${p.window_audio_url}', 'pair ${i} mix')">mix</button>` : ""}
            ${p.raw_stem1_audio_url ? `<button class="small" onclick="playFile('${p.raw_stem1_audio_url}', 'pair ${i} raw stem1')">raw1</button>` : ""}
            ${p.raw_stem2_audio_url ? `<button class="small" onclick="playFile('${p.raw_stem2_audio_url}', 'pair ${i} raw stem2')">raw2</button>` : ""}
            ${p.assigned_seg1_audio_url ? `<button class="small" onclick="playFile('${p.assigned_seg1_audio_url}', 'pair ${i} assigned seg1')">as1</button>` : ""}
            ${p.assigned_seg2_audio_url ? `<button class="small" onclick="playFile('${p.assigned_seg2_audio_url}', 'pair ${i} assigned seg2')">as2</button>` : ""}
          </div></td>
        </tr>
      `).join("");
      panel.innerHTML = `<details open><summary>Overlap pairs / stems</summary><div class="table-wrap"><table><thead><tr><th>#</th><th>segments</th><th>speakers</th><th>overlap</th><th>status</th><th>conf/margin</th><th>audio</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
    }

    function applyFilter() {
      const q = document.getElementById("filterText").value.toLowerCase().trim();
      const group = document.getElementById("filterGroup").value;
      document.querySelectorAll("#segmentBody tr").forEach(tr => {
        const text = tr.textContent.toLowerCase();
        const okText = !q || text.includes(q);
        const okGroup = !group || text.includes(group.toLowerCase());
        tr.style.display = okText && okGroup ? "" : "none";
      });
    }

    function renderContent(stage) {
      const content = document.getElementById("content");
      const controls = `
        <div class="controls">
          <input id="filterText" placeholder="lọc speaker/text/label..." oninput="applyFilter()" />
          <select id="filterGroup" onchange="applyFilter()">
            <option value="">all groups</option>
            <option value="clean_duplex_2speaker">clean_duplex_2speaker</option>
            <option value="overlap_review">overlap_review</option>
            <option value="exclude_or_extra_speaker">exclude_or_extra_speaker</option>
            <option value="low_confidence">low_confidence</option>
          </select>
          ${stage.json_url ? `<a href="${stage.json_url}" target="_blank">open json</a>` : ""}
        </div>`;
      const markdown = stage.markdown ? `<details><summary>Eval markdown</summary><pre>${esc(stage.markdown)}</pre></details>` : "";
      content.innerHTML = controls + markdown + renderRows(stage);
      document.querySelectorAll("textarea.correct-script").forEach(autosizeCorrectScript);
      document.getElementById("logTail").textContent = stage.log_tail || "No log";
      renderPairs(stage);
    }

    function render() {
      renderRuns();
      renderStages();
      const stage = activeStage();
      renderSummary(stage);
      renderContent(stage);
    }

    hydrateAnnotationsFromStorage();

    if (!DATA.runs.length) {
      document.getElementById("content").innerHTML = '<div class="empty">Không tìm thấy run_full nào.</div>';
    } else {
      render();
    }
  </script>
</body>
</html>
"""


def build_html(root: Path, out_path: Path) -> None:
    html_dir = out_path.parent
    html_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": build_runs(root, html_dir),
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML viewer for run_full pipeline outputs.")
    parser.add_argument("--root", default=".", help="Repository/workspace root containing run_full* directories.")
    parser.add_argument("--out", default=".run_viewer/index.html", help="Output HTML path.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    build_html(root, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
