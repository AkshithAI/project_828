import tempfile
import unittest
from pathlib import Path

from src.scripts.repository_evaluator import RepositoryEvaluator, to_markdown


class RepositoryEvaluatorTests(unittest.TestCase):
    def _create_sample_repo(self) -> Path:
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        (root / "README.md").write_text("# Demo\n", encoding="utf-8")
        (root / "requirements.txt").write_text("torch\n", encoding="utf-8")
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "model.py").write_text(
            "import torch\nclass GPT: pass\ndef train():\n    return 1\n",
            encoding="utf-8",
        )
        (root / "tests").mkdir(parents=True, exist_ok=True)
        (root / "tests" / "test_basic.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        return root

    def test_report_contains_required_sections(self):
        repo = self._create_sample_repo()
        report = RepositoryEvaluator(str(repo)).evaluate()

        self.assertIn("repository_summary", report)
        self.assertIn("file_by_file_analysis", report)
        self.assertIn("category_scores_out_of_10", report)
        self.assertIn("overall_score_out_of_100", report)
        self.assertIn("overall_verdict", report)
        self.assertIn("hireability_assessment", report)
        self.assertIn("concrete_improvements_ranked_by_impact", report)

    def test_scores_are_in_range(self):
        repo = self._create_sample_repo()
        report = RepositoryEvaluator(str(repo)).evaluate()

        for score in report["category_scores_out_of_10"].values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 10)

        self.assertGreaterEqual(report["overall_score_out_of_100"], 0)
        self.assertLessEqual(report["overall_score_out_of_100"], 100)

    def test_markdown_output_has_numbered_sections(self):
        repo = self._create_sample_repo()
        report = RepositoryEvaluator(str(repo)).evaluate()
        md = to_markdown(report)

        self.assertIn("1. Repository summary", md)
        self.assertIn("2. File-by-file analysis", md)
        self.assertIn("3. Category-wise scores", md)
        self.assertIn("4. Overall verdict", md)
        self.assertIn("5. Hireability assessment", md)
        self.assertIn("6. Concrete improvements ranked by impact", md)


if __name__ == "__main__":
    unittest.main()
