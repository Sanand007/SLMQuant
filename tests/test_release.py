import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release", ROOT / "reproduce.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in (ROOT / "src").rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))

    def test_fresh_arrays_match_frozen_manifest(self):
        base = ROOT / "inputs/v3" / release.STUDY
        manifest = release.read(base / "fresh_data_manifest.json")
        for name, expected in manifest["arrays_sha256"].items():
            self.assertEqual(release.digest(base / "data" / name), expected)
        ids = [row["id"] for row in release.read(base / "data/arc_easy.json")]
        self.assertEqual(hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                         manifest["arc_ids_sha256"])

    def test_prepare_isolated_and_resume_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sub in ["src", "inputs"]:
                shutil.copytree(ROOT / sub, root / sub)
            models = {}
            for key in release.MODELS:
                model = root / "fake_models" / key
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}")
                models[key] = str(model)
            release.write(root / ".local_models.json", models)
            with patch.object(release, "ROOT", root):
                for version in ["v2", "v3"]:
                    run = release.prepare(version, version + "_test")
                    self.assertEqual(run, release.prepare(version, version + "_test"))
                    self.assertFalse(list(run.rglob("search_*.json")))
                    self.assertFalse(list(run.rglob("completed_stages.json")))
                    p = release.read(run / "v2/protocol.json")
                    self.assertEqual(p["evaluations_per_method_seed"], 90)
                    inp = release.read(run / "v3" / release.STUDY / "inputs.json")
                    self.assertEqual(inp["model_path"], models["control"])
                frozen = root / "runs/v3_test/v3" / release.STUDY / "data/train_tokens.npy"
                with frozen.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(RuntimeError, "Frozen runtime file changed"):
                    release.prepare("v3", "v3_test")
                with self.assertRaisesRegex(RuntimeError, "uncapped"):
                    release.prepare("v3", "limits", max_hours=12)

    def test_corrected_route_and_fixed_v2_allowance(self):
        source = (ROOT / "src/v3/mlx_affine_study.py").read_text()
        self.assertIn('path = STUDY / "heldout_text_fresh.json"', source)
        self.assertIn('folder = STUDY / "data"', source)
        self.assertIn('stage("heldout_text_fresh", text_tests)', source)
        v2 = (ROOT / "src/v2/runner.py").read_text()
        self.assertIn("if budget != 90:", v2)


if __name__ == "__main__":
    unittest.main()
