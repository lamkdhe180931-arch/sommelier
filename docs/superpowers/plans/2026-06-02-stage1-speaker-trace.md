# Stage 1 Speaker Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add segment-level and file-level diagnostics so Stage 1 speaker decisions can be traced from chunk output through speaker linking, re-clustering, splitting, and postprocessing.

**Architecture:** Keep the runtime change local to `podcast-pipeline/stages/stage_01_diarize.py`. Add compact `stage1_trace` data on each segment and heavier aggregate details in `speaker_diagnostics.json`; preserve existing public metadata so downstream stages keep working.

**Tech Stack:** Python, pandas DataFrame metadata columns, `unittest`, existing `stage_common.dump_json` JSON-safe serialization.

---

### Task 1: Split Trace Preservation

**Files:**
- Modify: `tests/test_stage_common.py`
- Modify: `podcast-pipeline/stages/stage_01_diarize.py`

- [ ] **Step 1: Write the failing test**

Add a test that imports `stage_01_diarize.split_long_segments`, splits one long segment containing `stage1_trace`, and asserts each split part keeps metadata plus `split_from_index`, `split_part`, and `split_part_count`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=podcast-pipeline/stages python3 -m unittest tests.test_stage_common`
Expected: fail because split parts currently rebuild a minimal dict and drop metadata.

- [ ] **Step 3: Preserve metadata in split implementation**

Change `split_long_segments` so split parts are copied from the original segment before replacing `index`, `start`, and `end`, and augment `stage1_trace["split"]`.

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=podcast-pipeline/stages python3 -m unittest tests.test_stage_common`
Expected: all tests pass.

### Task 2: Speaker Link Trace Fields

**Files:**
- Modify: `tests/test_stage_common.py`
- Modify: `podcast-pipeline/stages/stage_01_diarize.py`

- [ ] **Step 1: Write the failing test**

Add a test around `df_to_list` using a DataFrame row with `stage1_trace` and speaker-link columns, asserting the nested trace survives JSON-clean conversion.

- [ ] **Step 2: Run test to verify it fails or expose current gap**

Run: `PYTHONPATH=podcast-pipeline/stages python3 -m unittest tests.test_stage_common`
Expected: pass for preservation if already supported, then implementation focuses on producing the fields; if it fails, fix preservation.

- [ ] **Step 3: Add trace creation in chunk alignment**

In `align_speakers_across_chunks`, create `_stage1_trace_id` and `stage1_trace` per segment with chunk/local identity, link decision, thresholds, candidate similarities, clean evidence, and centroid update status.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=podcast-pipeline/stages python3 -m unittest tests.test_stage_common`
Expected: all tests pass.

### Task 3: Re-cluster Diagnostics

**Files:**
- Modify: `tests/test_stage_common.py`
- Modify: `podcast-pipeline/stages/stage_01_diarize.py`

- [ ] **Step 1: Write the failing test**

Add a unit-level test for a new helper that attaches recluster trace to segment records, asserting `pre_recluster_speaker`, `final_speaker`, `recluster_action`, and best candidate similarity are present.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=podcast-pipeline/stages python3 -m unittest tests.test_stage_common`
Expected: fail because no helper exists yet.

- [ ] **Step 3: Implement recluster decisions and trace attachment**

Make `re_cluster_speakers` return per-speaker `decisions` with merge/keep reason and top candidates. Attach compact recluster fields to `stage1_trace` before converting DataFrame to segment list.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=podcast-pipeline/stages python3 -m unittest tests.test_stage_common`
Expected: all tests pass.

### Task 4: Diagnostics JSON Output

**Files:**
- Modify: `podcast-pipeline/stages/stage_01_diarize.py`
- Modify: `podcast-pipeline/build_notebooks.py`

- [ ] **Step 1: Add diagnostics payload**

Write `speaker_diagnostics.json` next to `diarization.json` containing speaker-linking stats, recluster stats, and a bottleneck summary derived from final segments.

- [ ] **Step 2: Add CLI output path if needed**

Use the default adjacent path; no new user-facing argument is required unless tests show a need.

- [ ] **Step 3: Rebuild notebooks**

Run: `python3 build_notebooks.py` from `podcast-pipeline`.
Expected: Stage 1 notebook reflects updated source.

- [ ] **Step 4: Final verification**

Run unit tests, py_compile for touched Python files, and `git diff --check`.
Expected: all commands exit 0.
