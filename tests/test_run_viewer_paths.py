import tempfile
import unittest
from pathlib import Path

from tools import build_run_viewer


class RunViewerPathTests(unittest.TestCase):
    def test_resolve_artifact_path_maps_run_full_paths_to_local_run_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run_full"
            target = run_root / "01_diarization" / "sortformer_tuning.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}", encoding="utf-8")

            cases = [
                "01_diarization/sortformer_tuning.json",
                "/kaggle/working/run_full/01_diarization/sortformer_tuning.json",
                "/Users/lam/anything/run_full 2/01_diarization/sortformer_tuning.json",
            ]

            resolved = [build_run_viewer.resolve_artifact_path(value, run_root) for value in cases]

            self.assertEqual(resolved, [target, target, target])

    def test_rel_url_keeps_viewer_links_relative_to_html_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            html_dir = tmp / "viewer"
            target = tmp / "run_full" / "01_diarization" / "sortformer_tuning.json"
            html_dir.mkdir()
            target.parent.mkdir(parents=True)
            target.write_text("{}", encoding="utf-8")

            url = build_run_viewer.rel_url(target, html_dir)

            self.assertEqual(url, "../run_full/01_diarization/sortformer_tuning.json")


if __name__ == "__main__":
    unittest.main()
