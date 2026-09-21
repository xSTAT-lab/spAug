"""Validate release layout and independent module boundaries."""
from pathlib import Path
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_public_repo import validate_repository


class PublicLayoutTest(unittest.TestCase):
    def test_combined_source_distribution(self):
        self.assertEqual(validate_repository(ROOT), [])

    def test_feature_imports_resolve_locally(self):
        package = ROOT / "feature/src"
        for path in (ROOT / "feature/notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
            self.assertNotIn("SYN_ROOT", source)
            local_imports = 0
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("spaug_feature"):
                    local_imports += 1
                    module = package.joinpath(*node.module.split("."))
                    self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file())
            self.assertGreater(local_imports, 0)

    def test_module_requirements_stay_local(self):
        for name in ("syn", "feature"):
            module = ROOT / name
            for path in module.rglob("requirements*.txt"):
                for line in path.read_text().splitlines():
                    if line.startswith("-r "):
                        include = (path.parent / line[3:].strip()).resolve()
                        self.assertTrue(include.is_relative_to(module))
                        self.assertTrue(include.is_file())

    def test_release_rejects_local_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("LOCAL_SETTING=example\n")
            (root / ".DS_Store").write_bytes(b"local metadata")
            (root / "nested/.git").mkdir(parents=True)
            (root / "credential.txt").write_text("gh" + "p_" + "x" * 24)
            errors = validate_repository(root)
            for marker in (".env", ".DS_Store", "nested/.git", "credential.txt"):
                self.assertTrue(any(marker in error for error in errors), marker)

    def test_synthetic_module_standalone(self):
        with tempfile.TemporaryDirectory() as temp:
            module = Path(temp) / "syn"
            shutil.copytree(ROOT / "syn", module, ignore=shutil.ignore_patterns("data", "__pycache__", ".venv", ".git"))
            for script in ("check_public_repo.py", "check_generator_entrypoint_contracts.py"):
                result = subprocess.run([sys.executable, str(module / "scripts" / script)], cwd=module, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_sibling_paths_in_module_sources(self):
        for name, sibling in (("feature", "syn"), ("syn", "feature")):
            for path in (ROOT / name).rglob("*"):
                if path.suffix not in {".py", ".ipynb", ".yaml", ".yml", ".txt"}:
                    continue
                text = path.read_text()
                if path.suffix == ".ipynb":
                    text = json.dumps(json.loads(text), ensure_ascii=False)
                self.assertIsNone(re.search(r"(?:\.\./|/)(?:" + sibling + r")/", text), str(path))


if __name__ == "__main__":
    unittest.main()
