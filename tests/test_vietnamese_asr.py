import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class FakePhoWhisperPipeline:
    def __init__(self):
        self.payload = None
        self.generate_kwargs = None

    def __call__(self, payload, generate_kwargs=None):
        self.payload = payload
        self.generate_kwargs = generate_kwargs
        return {"text": " xin chao "}


class VietnameseASRTests(unittest.TestCase):
    def test_phowhisper_transcriber_passes_16khz_audio_and_vietnamese_task(self):
        from models.vietnamese_asr import PhoWhisperTranscriber

        pipe = FakePhoWhisperPipeline()
        transcriber = PhoWhisperTranscriber(pipe)

        text = transcriber.transcribe([0.0, 0.1, -0.1])

        self.assertEqual(text, "xin chao")
        self.assertEqual(pipe.payload["sampling_rate"], 16000)
        self.assertEqual(pipe.generate_kwargs, {"language": "vi", "task": "transcribe"})

    def test_extract_transcription_text_handles_chunkformer_segment_outputs(self):
        from models.vietnamese_asr import extract_transcription_text

        result = [
            {"start": 0.0, "end": 1.0, "text": "xin chao"},
            (1.0, 2.0, "cac ban"),
        ]

        self.assertEqual(extract_transcription_text(result), "xin chao cac ban")


if __name__ == "__main__":
    unittest.main()
