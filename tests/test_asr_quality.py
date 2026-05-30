import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class ASRQualityTests(unittest.TestCase):
    def test_replaces_whisper_subscribe_hallucination_with_vietnamese_candidate(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn",
            text_whisper="Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn",
            text_phowhisper="năm.",
            text_chunkformer="luật năm",
            duration_sec=0.72,
        )

        self.assertEqual(decision["text"], "luật năm")
        self.assertEqual(decision["source"], "chunkformer")
        self.assertIn("replace_bad_whisper_with_vi_candidate", decision["actions"])

    def test_drops_micro_segment_boilerplate_when_vietnamese_models_do_not_agree(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="Thank you.",
            text_whisper="Thank you.",
            text_phowhisper="và.",
            text_chunkformer="",
            duration_sec=0.15,
        )

        self.assertEqual(decision["text"], "")
        self.assertEqual(decision["source"], "quality_guard")
        self.assertIn("drop_micro_boilerplate", decision["actions"])

    def test_prefers_agreed_vietnamese_models_over_foreign_whisper(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="Ini mah...",
            text_whisper="Ini mah...",
            text_phowhisper="nhưng mà",
            text_chunkformer="nhưng mà",
            duration_sec=0.96,
        )

        self.assertEqual(decision["text"], "nhưng mà")
        self.assertEqual(decision["source"], "vi_consensus")
        self.assertIn("replace_whisper_outlier_with_vi_consensus", decision["actions"])

    def test_prefers_vietnamese_diacritic_consensus_for_micro_segment(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="So.",
            text_whisper="So.",
            text_phowhisper="sợ",
            text_chunkformer="sợ",
            duration_sec=0.27,
        )

        self.assertEqual(decision["text"], "sợ")
        self.assertEqual(decision["source"], "vi_consensus")

    def test_keeps_rover_text_without_quality_signal(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="à đúng rồi",
            text_whisper="à đúng rồi",
            text_phowhisper="à đúng rồi",
            text_chunkformer="đúng rồi",
            duration_sec=0.8,
        )

        self.assertEqual(decision["text"], "à đúng rồi")
        self.assertEqual(decision["source"], "rover")
        self.assertEqual(decision["actions"], [])

    def test_drops_long_text_on_micro_segment_as_hallucination(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="mình nghĩ là chuyện này rất quan trọng",
            text_whisper="mình nghĩ là chuyện này rất quan trọng",
            text_phowhisper="ừ",
            text_chunkformer="",
            duration_sec=0.22,
        )

        self.assertEqual(decision["text"], "")
        self.assertEqual(decision["source"], "quality_guard")
        self.assertIn("drop_micro_hallucination", decision["actions"])

    def test_replaces_known_confusion_when_vietnamese_models_agree(self):
        from utils.asr_quality import choose_asr_text

        decision = choose_asr_text(
            rover_text="thành ra mình không đi nữa",
            text_whisper="thành ra mình không đi nữa",
            text_phowhisper="thật ra mình không đi nữa",
            text_chunkformer="thật ra mình không đi nữa",
            duration_sec=2.4,
        )

        self.assertEqual(decision["text"], "thật ra mình không đi nữa")
        self.assertEqual(decision["source"], "vi_consensus")
        self.assertIn("replace_known_confusion_with_vi_consensus", decision["actions"])

    def test_skips_sepreformer_for_tiny_overlap_or_segment(self):
        from utils.asr_quality import should_skip_sepreformer_pair

        pair = {
            "overlap_start": 271.17,
            "overlap_end": 271.57,
            "seg1": {"start": 269.41, "end": 271.57},
            "seg2": {"start": 271.17, "end": 271.89},
        }

        skip, reasons = should_skip_sepreformer_pair(pair, min_overlap_seconds=1.0, min_segment_seconds=1.0)

        self.assertTrue(skip)
        self.assertIn("overlap_lt_min", reasons)
        self.assertIn("segment_lt_min", reasons)

    def test_allows_sepreformer_for_stable_overlap_pair(self):
        from utils.asr_quality import should_skip_sepreformer_pair

        pair = {
            "overlap_start": 10.0,
            "overlap_end": 11.4,
            "seg1": {"start": 8.0, "end": 12.0},
            "seg2": {"start": 10.0, "end": 13.0},
        }

        skip, reasons = should_skip_sepreformer_pair(pair, min_overlap_seconds=1.0, min_segment_seconds=1.0)

        self.assertFalse(skip)
        self.assertEqual(reasons, [])

    def test_eval_preview_reports_guard_changes_without_rerunning_asr(self):
        from stage_06_eval import asr_quality_preview

        preview = asr_quality_preview(
            [
                {
                    "index": "00010",
                    "start": 0.0,
                    "end": 0.15,
                    "text": "Thank you.",
                    "text_whisper": "Thank you.",
                    "text_phowhisper": "và.",
                    "text_chunkformer": "",
                },
                {
                    "index": "00022",
                    "start": 1.0,
                    "end": 1.96,
                    "text": "Ini mah...",
                    "text_whisper": "Ini mah...",
                    "text_phowhisper": "nhưng mà",
                    "text_chunkformer": "nhưng mà",
                },
            ]
        )

        self.assertEqual(preview["would_change_count"], 2)
        self.assertEqual(preview["changes"][0]["new_text"], "")
        self.assertEqual(preview["changes"][1]["new_text"], "nhưng mà")


if __name__ == "__main__":
    unittest.main()
