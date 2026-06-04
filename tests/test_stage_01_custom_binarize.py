import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGES_DIR = ROOT / "podcast-pipeline" / "stages"
sys.path.insert(0, str(STAGES_DIR))


def import_stage_01_diarize():
    librosa = types.ModuleType("librosa")
    librosa.resample = lambda waveform, orig_sr, target_sr: waveform
    librosa.get_duration = lambda path: 0.0
    sys.modules["librosa"] = librosa

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    sys.modules["torch"] = torch

    nemo = types.ModuleType("nemo")
    nemo_collections = types.ModuleType("nemo.collections")
    nemo_asr = types.ModuleType("nemo.collections.asr")
    nemo_models = types.ModuleType("nemo.collections.asr.models")
    nemo_models.SortformerEncLabelModel = None
    nemo_asr.models = nemo_models
    nemo_collections.asr = nemo_asr
    nemo.collections = nemo_collections
    sys.modules["nemo"] = nemo
    sys.modules["nemo.collections"] = nemo_collections
    sys.modules["nemo.collections.asr"] = nemo_asr
    sys.modules["nemo.collections.asr.models"] = nemo_models

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Inference = None
    pyannote.audio = pyannote_audio
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio

    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = type("AudioSegment", (), {})
    sys.modules["pydub"] = pydub
    sys.modules["soundfile"] = types.ModuleType("soundfile")

    models = types.ModuleType("models")
    silero = types.ModuleType("models.silero_vad")
    silero.SAMPLING_RATE = 16000
    silero.SileroVAD = type("SileroVAD", (), {})
    models.silero_vad = silero
    sys.modules["models"] = models
    sys.modules["models.silero_vad"] = silero

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
    sys.modules["utils"] = utils
    sys.modules["utils.tool"] = tool
    sys.modules["utils.logger"] = logger_mod

    return importlib.import_module("stage_01_diarize")


class Stage01CustomBinarizeTests(unittest.TestCase):
    def test_custom_binarize_handles_hysteresis_merge_and_min_duration(self):
        stage_01 = import_stage_01_diarize()
        probs = np.array(
            [
                [0.10, 0.60, 0.10],
                [0.60, 0.70, 0.70],
                [0.55, 0.20, 0.20],
                [0.10, 0.10, 0.10],
                [0.58, 0.10, 0.10],
                [0.20, 0.10, 0.10],
            ],
            dtype=float,
        )

        segments = stage_01.custom_binarize(
            probs,
            frame_shift=0.5,
            onset=0.53,
            offset=0.49,
            min_duration_on=0.42,
            min_duration_off=0.34,
        )

        self.assertEqual(
            segments,
            [
                {"speaker": "SPEAKER_00", "start": 0.5, "end": 1.5},
                {"speaker": "SPEAKER_00", "start": 2.0, "end": 2.5},
                {"speaker": "SPEAKER_01", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_02", "start": 0.5, "end": 1.0},
            ],
        )

    def test_custom_binarize_rejects_non_matrix_probs(self):
        stage_01 = import_stage_01_diarize()

        segments = stage_01.custom_binarize(
            np.array([0.1, 0.6, 0.7]),
            frame_shift=0.5,
            onset=0.53,
            offset=0.49,
            min_duration_on=0.42,
            min_duration_off=0.34,
        )

        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
