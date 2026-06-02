import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGES_DIR = ROOT / "podcast-pipeline" / "stages"
sys.path.insert(0, str(STAGES_DIR))


def import_stage_01_diarize():
    """Import Stage 01 helpers without requiring heavy audio/model packages."""
    librosa = types.ModuleType("librosa")
    librosa.resample = lambda waveform, orig_sr, target_sr: waveform
    librosa.get_duration = lambda path: 0.0
    sys.modules.setdefault("librosa", librosa)

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    sys.modules.setdefault("torch", torch)

    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = type("AudioSegment", (), {})
    sys.modules.setdefault("pydub", pydub)
    sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

    models = types.ModuleType("models")
    silero = types.ModuleType("models.silero_vad")
    silero.SAMPLING_RATE = 16000
    silero.SileroVAD = type("SileroVAD", (), {})
    models.silero_vad = silero
    sys.modules.setdefault("models", models)
    sys.modules.setdefault("models.silero_vad", silero)

    utils = types.ModuleType("utils")
    tool = types.ModuleType("utils.tool")
    tool.load_cfg = lambda path: {}
    logger_mod = types.ModuleType("utils.logger")

    class FakeLogger:
        @staticmethod
        def get_logger():
            return types.SimpleNamespace(
                info=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            )

    logger_mod.Logger = FakeLogger
    utils.tool = tool
    utils.logger = logger_mod
    sys.modules.setdefault("utils", utils)
    sys.modules.setdefault("utils.tool", tool)
    sys.modules.setdefault("utils.logger", logger_mod)

    return importlib.import_module("stage_01_diarize")


class Stage01TraceTests(unittest.TestCase):
    def test_split_long_segments_preserves_trace_metadata(self):
        stage_01 = import_stage_01_diarize()
        segments = [
            {
                "index": "00007",
                "start": 0.0,
                "end": 65.0,
                "speaker": "SPEAKER_04",
                "speaker_link_action": "created_new",
                "stage1_trace": {
                    "trace_id": "chunk002:local:SPEAKER_01:row000",
                    "link": {"action": "created_new"},
                },
            }
        ]

        split_segments = stage_01.split_long_segments(segments, max_duration=30.0)

        self.assertEqual(len(split_segments), 3)
        self.assertEqual(split_segments[0]["speaker_link_action"], "created_new")
        self.assertEqual(split_segments[1]["stage1_trace"]["trace_id"], "chunk002:local:SPEAKER_01:row000")
        self.assertEqual(split_segments[2]["stage1_trace"]["split"]["split_from_index"], "00007")
        self.assertEqual(split_segments[2]["stage1_trace"]["split"]["split_part"], 2)
        self.assertEqual(split_segments[2]["stage1_trace"]["split"]["split_part_count"], 3)

    def test_apply_recluster_trace_records_pre_and_final_speaker(self):
        stage_01 = import_stage_01_diarize()
        df = pd.DataFrame(
            [
                {"start": 10.0, "end": 14.0, "speaker": "SPEAKER_02", "stage1_trace": {"trace_id": "a"}},
                {"start": 20.0, "end": 25.0, "speaker": "SPEAKER_04", "stage1_trace": {"trace_id": "b"}},
            ]
        )
        decisions = {
            "SPEAKER_02": {
                "speaker": "SPEAKER_02",
                "mapped_speaker": "SPEAKER_04",
                "action": "merged",
                "reason": "above_threshold",
                "best_recluster_candidate": "SPEAKER_04",
                "best_recluster_similarity": 0.812345,
                "similarity_threshold": 0.7,
                "top_candidates": [{"speaker": "SPEAKER_04", "similarity": 0.812345}],
            },
            "SPEAKER_04": {
                "speaker": "SPEAKER_04",
                "mapped_speaker": "SPEAKER_04",
                "action": "kept",
                "reason": "cluster_representative",
                "best_recluster_candidate": None,
                "best_recluster_similarity": -1.0,
                "similarity_threshold": 0.7,
                "top_candidates": [],
            },
        }

        traced = stage_01._apply_recluster_trace(
            df,
            mapping={"SPEAKER_02": "SPEAKER_04", "SPEAKER_04": "SPEAKER_04"},
            decisions=decisions,
        )

        self.assertEqual(traced.loc[0, "speaker"], "SPEAKER_04")
        self.assertEqual(traced.loc[0, "pre_recluster_speaker"], "SPEAKER_02")
        self.assertEqual(traced.loc[0, "stage1_trace"]["recluster"]["action"], "merged")
        self.assertEqual(traced.loc[0, "stage1_trace"]["recluster"]["best_candidate"], "SPEAKER_04")
        self.assertEqual(traced.loc[1, "stage1_trace"]["recluster"]["reason"], "cluster_representative")

    def test_build_speaker_diagnostics_summarizes_bottlenecks(self):
        stage_01 = import_stage_01_diarize()
        segments = [
            {
                "speaker": "SPEAKER_04",
                "start": 0.0,
                "end": 3.0,
                "speaker_link_action": "created_new",
                "speaker_link_reason": "below_threshold",
                "identity_low_confidence": False,
                "speaker_identity_clean_candidate": True,
            },
            {
                "speaker": "SPEAKER_REVIEW",
                "start": 3.1,
                "end": 3.7,
                "speaker_link_action": "review_weak_identity",
                "speaker_link_reason": "weak_identity_review",
                "identity_low_confidence": True,
                "is_short_backchannel": True,
                "needs_manual_review": True,
            },
        ]

        diagnostics = stage_01._build_speaker_diagnostics(
            segments,
            speaker_linking_stats={"skipped": False},
            recluster_stats={"skipped": False},
        )

        summary = diagnostics["bottleneck_summary"]
        self.assertEqual(summary["segments_total"], 2)
        self.assertEqual(summary["speaker_link_actions"]["created_new"], 1)
        self.assertEqual(summary["speaker_link_reasons"]["weak_identity_review"], 1)
        self.assertEqual(summary["short_backchannel_segments"], 1)
        self.assertEqual(summary["manual_review_segments"], 1)
        self.assertEqual(summary["clean_identity_candidate_segments"], 1)

    def test_align_speakers_adds_stage1_link_trace(self):
        stage_01 = import_stage_01_diarize()
        frames = [
            pd.DataFrame(
                [
                    {
                        "start": 0.0,
                        "end": 3.0,
                        "speaker": "SPEAKER_00",
                        "_stage1_chunk_offset": 0.0,
                        "_stage1_chunk_duration": 180.0,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "start": 180.0,
                        "end": 184.0,
                        "speaker": "SPEAKER_00",
                        "_stage1_chunk_offset": 180.0,
                        "_stage1_chunk_duration": 180.0,
                    }
                ]
            ),
        ]

        original_profile_fn = stage_01._compute_chunk_speaker_identity_profiles
        try:
            def fake_profiles(df, audio_info, embedder, identity_min_duration):
                return (
                    {
                        "SPEAKER_00": {
                            "embedding": np.array([1.0, 0.0]),
                            "has_clean_identity_evidence": True,
                        }
                    },
                    pd.Series([True], index=df.index),
                )

            stage_01._compute_chunk_speaker_identity_profiles = fake_profiles
            aligned, stats = stage_01.align_speakers_across_chunks(
                frames,
                audio_info={},
                embedder=object(),
                similarity_threshold=0.75,
                similarity_margin=0.08,
                weak_match_threshold=0.55,
                centroid_update_threshold=0.85,
                return_stats=True,
            )
        finally:
            stage_01._compute_chunk_speaker_identity_profiles = original_profile_fn

        second_row = aligned[1].iloc[0]
        trace = second_row["stage1_trace"]
        self.assertEqual(second_row["pre_link_speaker"], "SPEAKER_00")
        self.assertEqual(second_row["speaker_link_action"], "matched_existing")
        self.assertEqual(trace["trace_id"], "chunk001:local:SPEAKER_00:row00000")
        self.assertEqual(trace["chunk"]["chunk_offset"], 180.0)
        self.assertEqual(trace["link"]["best_candidate"], "SPEAKER_00")
        self.assertEqual(trace["link"]["best_similarity"], 1.0)
        self.assertTrue(trace["link"]["centroid_updated"])
        self.assertEqual(stats["chunks"][1]["decisions"][0]["action"], "matched_existing")

    def test_recluster_merge_is_traceable_without_main_logger(self):
        stage_01 = import_stage_01_diarize()
        if hasattr(stage_01, "logger"):
            delattr(stage_01, "logger")
        df = pd.DataFrame(
            [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "speaker": "SPEAKER_00",
                    "speaker_identity_clean_candidate": True,
                    "identity_low_confidence": False,
                    "stage1_trace": {"trace_id": "left"},
                },
                {
                    "start": 10.0,
                    "end": 14.0,
                    "speaker": "SPEAKER_01",
                    "speaker_identity_clean_candidate": True,
                    "identity_low_confidence": False,
                    "stage1_trace": {"trace_id": "right"},
                },
            ]
        )

        original_extract = stage_01._extract_speaker_embedding
        try:
            def fake_extract(audio_info, start, end, embedder):
                return np.array([1.0, 0.0]) if float(start) < 5.0 else np.array([0.98, 0.02])

            stage_01._extract_speaker_embedding = fake_extract
            reclustered, stats = stage_01.re_cluster_speakers(
                df,
                audio_info={},
                embedder=object(),
                similarity_threshold=0.7,
                identity_min_duration=2.0,
            )
        finally:
            stage_01._extract_speaker_embedding = original_extract

        self.assertEqual(reclustered.loc[1, "speaker"], "SPEAKER_00")
        self.assertEqual(reclustered.loc[1, "pre_recluster_speaker"], "SPEAKER_01")
        self.assertEqual(reclustered.loc[1, "stage1_trace"]["recluster"]["action"], "merged")
        self.assertEqual(stats["decisions"]["SPEAKER_01"]["reason"], "above_threshold")


if __name__ == "__main__":
    unittest.main()
