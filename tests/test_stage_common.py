import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class FakeNumpyScalar:
    def item(self):
        return 3.5


class FakeNumpyArray:
    def tolist(self):
        return [1, 2, 3]


class StageCommonTests(unittest.TestCase):
    def test_dump_json_converts_numpy_like_values(self):
        import stage_common

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "nested" / "data.json"
            stage_common.dump_json(
                {
                    "scalar": FakeNumpyScalar(),
                    "array": FakeNumpyArray(),
                    "path": Path("/tmp/audio.wav"),
                },
                out_path,
            )

            data = json.loads(out_path.read_text())

        self.assertEqual(data["scalar"], 3.5)
        self.assertEqual(data["array"], [1, 2, 3])
        self.assertEqual(data["path"], "/tmp/audio.wav")

    def test_clean_segments_for_json_removes_audio_payloads(self):
        import stage_common

        cleaned = stage_common.clean_segments_for_json(
            [
                {
                    "index": "00000",
                    "start": 0.0,
                    "end": 1.0,
                    "enhanced_audio": FakeNumpyArray(),
                    "_private": "skip",
                    "speaker": "SPEAKER_00",
                }
            ]
        )

        self.assertEqual(cleaned, [{"index": "00000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    def test_default_stage_dir_uses_audio_stem(self):
        import stage_common

        out_dir = stage_common.default_stage_dir("/data/podcast/sample.wav")

        self.assertEqual(out_dir, Path("/data/podcast/_staged/sample"))

    def test_postprocess_merges_only_same_speaker_micro_gap_and_marks_backchannel(self):
        import stage_common

        segments = [
            {"index": "00000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"index": "00001", "start": 1.2, "end": 2.0, "speaker": "SPEAKER_00"},
            {"index": "00002", "start": 2.1, "end": 2.45, "speaker": "SPEAKER_01"},
            {"index": "00003", "start": 2.55, "end": 3.0, "speaker": "SPEAKER_00"},
        ]

        postprocessed, stats = stage_common.postprocess_diarization_segments(
            segments,
            same_speaker_merge_gap=0.3,
            short_backchannel_seconds=1.0,
        )

        self.assertEqual(len(postprocessed), 3)
        self.assertEqual(postprocessed[0]["speaker"], "SPEAKER_00")
        self.assertEqual(postprocessed[0]["start"], 0.0)
        self.assertEqual(postprocessed[0]["end"], 2.0)
        self.assertFalse(postprocessed[0]["is_short_backchannel"])
        self.assertTrue(postprocessed[1]["is_short_backchannel"])
        self.assertEqual(postprocessed[1]["train_quality_label"], "short_backchannel_review")
        self.assertEqual(postprocessed[2]["speaker"], "SPEAKER_00")
        self.assertEqual(stats["merged_same_speaker_gap_count"], 1)
        self.assertEqual(stats["short_backchannel_count"], 2)

    def test_speaker_identity_policy_reviews_weak_unmatched_segment(self):
        import stage_common

        decision = stage_common.resolve_speaker_identity_decision(
            best_id="SPEAKER_00",
            best_similarity=0.29,
            second_best_similarity=0.18,
            has_clean_identity_evidence=False,
            next_global_id="SPEAKER_05",
            similarity_threshold=0.75,
            similarity_margin=0.08,
            weak_match_threshold=0.55,
            review_speaker_label="SPEAKER_REVIEW",
        )

        self.assertEqual(decision["mapped_speaker"], "SPEAKER_REVIEW")
        self.assertEqual(decision["action"], "review_weak_identity")
        self.assertFalse(decision["should_create_new"])
        self.assertFalse(decision["should_update_centroid"])
        self.assertTrue(decision["identity_low_confidence"])

    def test_speaker_identity_policy_maps_clear_weak_segment_without_update(self):
        import stage_common

        decision = stage_common.resolve_speaker_identity_decision(
            best_id="SPEAKER_01",
            best_similarity=0.62,
            second_best_similarity=0.40,
            has_clean_identity_evidence=False,
            next_global_id="SPEAKER_05",
            similarity_threshold=0.75,
            similarity_margin=0.08,
            weak_match_threshold=0.55,
            review_speaker_label="SPEAKER_REVIEW",
        )

        self.assertEqual(decision["mapped_speaker"], "SPEAKER_01")
        self.assertEqual(decision["action"], "matched_weak_context")
        self.assertFalse(decision["should_create_new"])
        self.assertFalse(decision["should_update_centroid"])
        self.assertTrue(decision["identity_low_confidence"])

    def test_speaker_identity_policy_allows_clean_new_and_clean_update(self):
        import stage_common

        new_decision = stage_common.resolve_speaker_identity_decision(
            best_id="SPEAKER_00",
            best_similarity=0.30,
            second_best_similarity=0.10,
            has_clean_identity_evidence=True,
            next_global_id="SPEAKER_05",
            similarity_threshold=0.75,
            similarity_margin=0.08,
            weak_match_threshold=0.55,
            review_speaker_label="SPEAKER_REVIEW",
        )
        update_decision = stage_common.resolve_speaker_identity_decision(
            best_id="SPEAKER_00",
            best_similarity=0.90,
            second_best_similarity=0.20,
            has_clean_identity_evidence=True,
            next_global_id="SPEAKER_05",
            similarity_threshold=0.75,
            similarity_margin=0.08,
            weak_match_threshold=0.55,
            centroid_update_threshold=0.85,
            review_speaker_label="SPEAKER_REVIEW",
        )

        self.assertEqual(new_decision["mapped_speaker"], "SPEAKER_05")
        self.assertEqual(new_decision["action"], "created_new")
        self.assertTrue(new_decision["should_create_new"])
        self.assertTrue(new_decision["should_update_centroid"])
        self.assertFalse(new_decision["identity_low_confidence"])
        self.assertEqual(update_decision["mapped_speaker"], "SPEAKER_00")
        self.assertEqual(update_decision["action"], "matched_existing")
        self.assertFalse(update_decision["should_create_new"])
        self.assertTrue(update_decision["should_update_centroid"])
        self.assertFalse(update_decision["identity_low_confidence"])

    def test_assign_duplex_train_groups_selects_two_main_speakers(self):
        import stage_common

        segments = [
            {"index": "00000", "start": 0.0, "end": 10.0, "speaker": "SPEAKER_A"},
            {"index": "00001", "start": 11.0, "end": 19.0, "speaker": "SPEAKER_B", "has_overlap": True, "is_separated": False},
            {"index": "00002", "start": 20.0, "end": 23.0, "speaker": "SPEAKER_C"},
            {"index": "00003", "start": 24.0, "end": 28.0, "speaker": "SPEAKER_B", "separation_status": "low_confidence"},
            {"index": "00004", "start": 29.0, "end": 34.0, "speaker": "SPEAKER_A", "is_separated": True},
        ]

        grouped, stats = stage_common.assign_duplex_train_groups(segments, expected_main_speakers=2)

        self.assertEqual(stats["main_speakers"], ["SPEAKER_A", "SPEAKER_B"])
        self.assertEqual(grouped[0]["duplex_train_group"], "clean_duplex_2speaker")
        self.assertEqual(grouped[1]["duplex_train_group"], "overlap_review")
        self.assertEqual(grouped[1]["duplex_group_reason"], "unseparated_overlap")
        self.assertEqual(grouped[2]["duplex_train_group"], "exclude_or_extra_speaker")
        self.assertEqual(grouped[2]["duplex_group_reason"], "extra_speaker")
        self.assertEqual(grouped[3]["duplex_train_group"], "overlap_review")
        self.assertEqual(grouped[3]["duplex_group_reason"], "low_confidence_overlap")
        self.assertEqual(grouped[4]["duplex_train_group"], "clean_duplex_2speaker")
        self.assertEqual(stats["group_counts"]["clean_duplex_2speaker"], 2)
        self.assertEqual(stats["group_counts"]["overlap_review"], 2)
        self.assertEqual(stats["group_counts"]["exclude_or_extra_speaker"], 1)


if __name__ == "__main__":
    unittest.main()
