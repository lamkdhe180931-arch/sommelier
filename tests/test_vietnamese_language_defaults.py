import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module_tree(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def _function_default(tree, function_name, argument_name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            defaults_by_arg = dict(
                zip(
                    [arg.arg for arg in node.args.args[-len(node.args.defaults) :]],
                    node.args.defaults,
                )
            )
            default_node = defaults_by_arg.get(argument_name)
            if isinstance(default_node, ast.Constant):
                return default_node.value
    raise AssertionError(f"Could not find default for {function_name}.{argument_name}")


def _has_language_assignment_to_en(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if node.value.value != "en":
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"language", "detected_language"}:
                    return True
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "language"
                    and isinstance(value, ast.Constant)
                    and value.value == "en"
                ):
                    return True
    return False


class VietnameseLanguageDefaultsTests(unittest.TestCase):
    def test_pipeline_config_only_advertises_vietnamese(self):
        config = json.loads((ROOT / "podcast-pipeline" / "config.json").read_text())

        self.assertFalse(config["language"]["multilingual"])
        self.assertEqual(config["language"]["supported"], ["vi"])

    def test_whisper_wrapper_defaults_to_vietnamese(self):
        tree = _module_tree("podcast-pipeline/models/whisper_asr.py")

        self.assertEqual(_function_default(tree, "__init__", "language"), "vi")
        self.assertEqual(_function_default(tree, "load_asr_model", "language"), "vi")
        self.assertFalse(_has_language_assignment_to_en(tree))

    def test_legacy_asr_helper_has_no_english_language_fallbacks(self):
        tree = _module_tree("podcast-pipeline/utils/asr_ensemble.py")

        self.assertFalse(_has_language_assignment_to_en(tree))


if __name__ == "__main__":
    unittest.main()
