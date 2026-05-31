import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class StemAssignmentTests(unittest.TestCase):
    def test_accepts_direct_mapping_when_embedding_scores_are_clear(self):
        from utils.stem_assignment import choose_stem_assignment

        decision = choose_stem_assignment(
            "SPEAKER_00",
            "SPEAKER_01",
            source_scores=[
                {"SPEAKER_00": 0.82, "SPEAKER_01": 0.31},
                {"SPEAKER_00": 0.27, "SPEAKER_01": 0.79},
            ],
            min_confidence=0.55,
            min_margin=0.08,
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["mapping"], "direct")
        self.assertEqual(decision["assignment_method"], "embedding_direct")

    def test_accepts_swapped_mapping_when_embedding_scores_are_clear(self):
        from utils.stem_assignment import choose_stem_assignment

        decision = choose_stem_assignment(
            "SPEAKER_00",
            "SPEAKER_01",
            source_scores=[
                {"SPEAKER_00": 0.24, "SPEAKER_01": 0.83},
                {"SPEAKER_00": 0.81, "SPEAKER_01": 0.29},
            ],
            min_confidence=0.55,
            min_margin=0.08,
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["mapping"], "swapped")
        self.assertEqual(decision["assignment_method"], "embedding_swapped")

    def test_rejects_when_both_stems_match_the_same_speaker(self):
        from utils.stem_assignment import choose_stem_assignment

        decision = choose_stem_assignment(
            "SPEAKER_00",
            "SPEAKER_01",
            source_scores=[
                {"SPEAKER_00": 0.78, "SPEAKER_01": 0.32},
                {"SPEAKER_00": 0.76, "SPEAKER_01": 0.30},
            ],
            min_confidence=0.55,
            min_margin=0.08,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["mapping"], None)
        self.assertIn("same_best_speaker", decision["reject_reasons"])

    def test_rejects_when_direct_and_swapped_scores_are_too_close(self):
        from utils.stem_assignment import choose_stem_assignment

        decision = choose_stem_assignment(
            "SPEAKER_00",
            "SPEAKER_01",
            source_scores=[
                {"SPEAKER_00": 0.61, "SPEAKER_01": 0.58},
                {"SPEAKER_00": 0.57, "SPEAKER_01": 0.60},
            ],
            min_confidence=0.55,
            min_margin=0.08,
        )

        self.assertFalse(decision["accepted"])
        self.assertIn("margin_lt_min", decision["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
