from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicLayoutTest(unittest.TestCase):
    def test_required_paths_exist(self):
        required = (
            "README.md",
            "LICENSE",
            "requirements.txt",
            "src/01_data_prep",
            "src/02_generators",
            "src/04_paradigm",
            "src/05_downstream",
            "src/06_evaluation",
            "configs",
            "docs/data_contract.md",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).exists(), relative)

    def test_repository_has_no_machine_specific_paths(self):
        forbidden = ("/" + "home/", "/" + "Users/", "/" + "mnt/", "/" + "workspace/", "C:\\Users\\")
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt", ".sh", ".cff", ".toml", ".ini"}:
                continue
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8")
                for marker in forbidden:
                    self.assertNotIn(marker, content)


if __name__ == "__main__":
    unittest.main()
