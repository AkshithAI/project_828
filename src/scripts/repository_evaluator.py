#!/usr/bin/env python3
"""Repository evaluator for AI/ML relevance and engineering quality."""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh",
    ".js", ".ts", ".tsx", ".jsx", ".ipynb", ".csv", ".sql", ".rst", ".env",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pt", ".pth", ".ckpt", ".pack", ".idx", ".rev",
}

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "node_modules"}
USABILITY_SCRIPTS = ("init.sh", "launch_distributed.sh")


@dataclass
class FileEvidence:
    path: str
    kind: str
    notes: str


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name.endswith(".example"):
        return True
    if path.name in {"Dockerfile", "Makefile", "requirements.txt", ".gitignore", ".gitattributes"}:
        return True
    return False


def _read_text_with_limit(path: Path, max_bytes: int = 300_000) -> Tuple[str, bool]:
    size = path.stat().st_size
    truncated = size > max_bytes
    with path.open("rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace"), truncated


class RepositoryEvaluator:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.text_files: List[Path] = []
        self.binary_files: List[Path] = []
        self.unread_files: List[FileEvidence] = []
        self.file_notes: List[FileEvidence] = []
        self.findings: Dict[str, bool] = {}
        self.metrics: Dict[str, int] = {}

    def _collect_files(self) -> None:
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                p = Path(root) / name
                if _is_text_file(p):
                    self.text_files.append(p)
                elif p.suffix.lower() in BINARY_EXTENSIONS:
                    self.binary_files.append(p)
                else:
                    self.unread_files.append(FileEvidence(str(p.relative_to(self.repo_path)), "unknown", "non-text or unsupported extension"))

    def _scan_file(self, path: Path) -> None:
        rel = str(path.relative_to(self.repo_path))
        content, truncated = _read_text_with_limit(path)
        lowered = content.lower()
        if truncated:
            self.unread_files.append(FileEvidence(rel, "large", "partially analyzed (file too large, scanned first chunk only)"))

        has_defs = "def " in content or "class " in content
        if path.suffix == ".py":
            purpose = "python module"
            if any(part.startswith("test") for part in path.parts):
                purpose = "test file"
            elif "train" in path.name:
                purpose = "training script"
            elif "inference" in path.name:
                purpose = "inference script"
            elif "dataloader" in path.name or "preprocess" in path.name or "packing" in path.name:
                purpose = "data pipeline"
            elif "model" in path.name:
                purpose = "model architecture"
            elif "config" in path.name:
                purpose = "configuration"
            note = purpose
            if has_defs:
                try:
                    tree = ast.parse(content)
                    funcs = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
                    classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
                    note += f" ({classes} classes, {funcs} functions)"
                except SyntaxError:
                    note += " (parse error while analyzing)"
            self.file_notes.append(FileEvidence(rel, "text", note))
        else:
            note = "documentation/config/script"
            self.file_notes.append(FileEvidence(rel, "text", note))

        def mark(key: str, cond: bool) -> None:
            self.findings[key] = self.findings.get(key, False) or cond

        mark("has_readme", path.name.lower() == "readme.md")
        mark("has_tests", "tests" in path.parts and path.suffix == ".py")
        mark("has_notebooks", path.suffix == ".ipynb")
        mark("has_ci", ".github" in path.parts and "workflows" in path.parts)
        mark("has_docs", path.suffix == ".md")
        mark("has_requirements", path.name in {"requirements.txt", "pyproject.toml", "environment.yml"})
        mark("uses_pytorch", "import torch" in content or "from torch" in content)
        mark("uses_transformers", "transformers" in lowered)
        mark("uses_deepspeed", "deepspeed" in lowered)
        mark("uses_wandb", "wandb" in lowered)
        mark("has_training", "def train" in content or "training" in path.parts)
        mark("has_inference", "def generate" in content or "inference" in path.name)
        mark("has_preprocessing", "preprocess" in path.name or "dataloader" in path.name or "dataset" in lowered)
        mark("has_validation", "validation" in lowered or "pytest" in lowered or "unittest" in lowered)
        mark("has_distributed", "distributed" in lowered or "ddp" in lowered or "deepspeed" in lowered)
        mark("has_model_arch", "class gpt" in lowered or "attention" in lowered or "moe" in lowered)

    def _scan_all(self) -> None:
        self._collect_files()
        for p in sorted(self.text_files):
            self._scan_file(p)

    def _score_categories(self) -> Dict[str, float]:
        # Lightweight heuristic weights tuned for evidence-first scoring.
        f = self.findings
        n_files = len(self.text_files)
        n_tests = sum(1 for x in self.file_notes if "test" in x.path and x.path.endswith(".py"))
        has_moe_path = False
        has_data_path = False
        has_packing_file = False
        has_phase_path = False
        has_validate_domains = False
        has_flash_attn = False
        has_tests_path = False
        python_file_count = 0
        md_file_count = 0
        for item in self.file_notes:
            path_l = item.path.lower()
            if "moe" in path_l:
                has_moe_path = True
            if "data" in path_l:
                has_data_path = True
            if "packing.py" in path_l:
                has_packing_file = True
            if "phase" in path_l:
                has_phase_path = True
            if "validate_domains.py" in path_l:
                has_validate_domains = True
            if "flash_attn" in path_l:
                has_flash_attn = True
            if "tests" in path_l:
                has_tests_path = True
            if path_l.endswith(".py"):
                python_file_count += 1
            if path_l.endswith(".md"):
                md_file_count += 1
        many_python_files = python_file_count > 10
        multiple_md_files = md_file_count > 1
        has_init_script = (self.repo_path / USABILITY_SCRIPTS[0]).exists()
        has_launch_script = (self.repo_path / USABILITY_SCRIPTS[1]).exists()

        scores = {
            "Project clarity and purpose": 3 + 4 * f.get("has_readme", False) + 1 * f.get("has_docs", False) + (1 if n_files > 10 else 0),
            "Technical depth": 2 + 2 * f.get("has_model_arch", False) + 2 * f.get("has_distributed", False) + 2 * f.get("uses_deepspeed", False) + (2 if has_moe_path else 0),
            "Code quality and architecture": 2 + 3 * (n_files > 8) + 2 * f.get("has_model_arch", False) + 1 * f.get("has_docs", False) + 2 * (n_tests > 0),
            "AI/ML relevance and sophistication": 1 + 3 * f.get("uses_pytorch", False) + 2 * f.get("uses_transformers", False) + 2 * f.get("has_training", False) + 2 * f.get("has_model_arch", False),
            "Data handling and preprocessing": 1 + 4 * f.get("has_preprocessing", False) + (2 if has_data_path else 0) + (3 if has_packing_file else 0),
            "Model design / training / inference quality": 1 + 3 * f.get("has_model_arch", False) + 3 * f.get("has_training", False) + 2 * f.get("has_inference", False) + 1 * f.get("has_validation", False),
            "Experimentation quality": 1 + 2 * f.get("uses_wandb", False) + 2 * f.get("has_validation", False) + (2 if has_phase_path else 0) + (3 if has_validate_domains else 0),
            "Testing, validation, and reproducibility": 1 + 3 * (n_tests > 0) + 2 * f.get("has_requirements", False) + 2 * f.get("has_validation", False) + 2 * f.get("has_ci", False),
            "Documentation quality": 1 + 4 * f.get("has_readme", False) + (2 if multiple_md_files else 0) + 3 * f.get("has_docs", False),
            "Engineering practices": 1 + 2 * f.get("has_requirements", False) + 2 * f.get("has_distributed", False) + 2 * f.get("uses_wandb", False) + (3 if has_tests_path else 0),
            "Real-world usefulness": 1 + 3 * f.get("has_training", False) + 2 * f.get("has_inference", False) + 2 * f.get("has_distributed", False) + 2 * f.get("has_preprocessing", False),
            "Originality / creativity": 2 + (3 if has_moe_path else 0) + (2 if has_flash_attn else 0) + (3 if has_validate_domains else 0),
            "Deployment or usability readiness": 1 + 2 * f.get("has_inference", False) + 2 * f.get("has_requirements", False) + 2 * has_init_script + 3 * has_launch_script,
            "Maintainability and scalability": 1 + 2 * f.get("has_distributed", False) + 2 * (n_files > 15) + (2 if many_python_files else 0) + 3 * (n_tests > 0),
            "Overall learning value for an AI student": 2 + 3 * f.get("has_model_arch", False) + 2 * f.get("has_preprocessing", False) + 2 * f.get("has_training", False) + 1 * (n_tests > 0),
            "Hiring signal for AI-related internships, projects, or entry-level roles": 1 + 3 * f.get("has_model_arch", False) + 2 * f.get("has_training", False) + 2 * f.get("has_validation", False) + 2 * (n_tests > 0),
        }

        clipped = {k: float(max(0, min(10, v))) for k, v in scores.items()}
        return clipped

    def _build_dynamic_summary(self) -> str:
        f = self.findings
        parts = []
        if f.get("has_model_arch"):
            parts.append("model architecture code")
        if f.get("has_training"):
            parts.append("training pipeline")
        if f.get("has_preprocessing"):
            parts.append("data preprocessing")
        if f.get("has_inference"):
            parts.append("inference utilities")
        if f.get("has_distributed"):
            parts.append("distributed execution support")

        if not parts:
            return (
                "This repository contains project files but limited directly verifiable AI/ML implementation evidence "
                "from the scanned sources."
            )

        if len(parts) == 1:
            return f"This repository includes {parts[0]}."
        return "This repository includes " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

    def _derive_strengths(self) -> List[str]:
        f = self.findings
        strengths: List[str] = []
        if f.get("has_model_arch"):
            strengths.append("Includes explicit model architecture implementation rather than only wrappers.")
        if f.get("has_training"):
            strengths.append("Contains train-time code paths for optimization and learning workflow.")
        if f.get("has_preprocessing"):
            strengths.append("Provides data handling/preprocessing components for ML workflows.")
        if f.get("has_validation"):
            strengths.append("Shows validation or testing-oriented code paths for quality tracking.")
        if f.get("has_distributed"):
            strengths.append("Includes distributed/scaling-related logic, signaling systems awareness.")
        if f.get("has_docs"):
            strengths.append("Contains documentation files that help explain usage and structure.")
        return strengths[:3] if strengths else ["Repository has a coherent structure with identifiable modules."]

    def _derive_weaknesses(self) -> List[str]:
        f = self.findings
        weaknesses: List[str] = []
        if not f.get("has_ci"):
            weaknesses.append("No CI workflow detected for automatic regression checks on changes.")
        if not f.get("has_tests"):
            weaknesses.append("Limited or no automated test files, increasing regression risk.")
        if not f.get("has_requirements"):
            weaknesses.append("No dependency manifest detected, reducing reproducibility.")
        if not f.get("has_inference"):
            weaknesses.append("No clear inference entry point for practical model usage.")
        if not f.get("has_preprocessing"):
            weaknesses.append("Limited visible preprocessing/data pipeline support.")
        if not f.get("has_docs"):
            weaknesses.append("Sparse documentation limits clarity for reviewers and users.")
        if not weaknesses:
            weaknesses.append("Main risks are environment-specific reproducibility and external data/service dependencies.")
        return weaknesses[:3]

    def _derive_improvements(self) -> List[str]:
        f = self.findings
        improvements: List[str] = []
        if not f.get("has_ci"):
            improvements.append("Add CI workflows for linting/tests/smoke checks on each pull request.")
        if not f.get("has_tests"):
            improvements.append("Increase automated unit/integration tests for core model and data paths.")
        if not f.get("has_requirements"):
            improvements.append("Add and pin dependencies for a reproducible local/CI environment.")
        if not f.get("has_inference"):
            improvements.append("Expose a clear inference CLI/API for easier practical usage.")
        if not f.get("has_validation"):
            improvements.append("Add richer evaluation/validation metrics and reproducible benchmark scripts.")
        if not f.get("has_preprocessing"):
            improvements.append("Provide explicit preprocessing scripts and data-format assumptions.")
        if not f.get("has_docs"):
            improvements.append("Expand technical documentation with setup, design rationale, and limitations.")
        if not improvements:
            improvements = [
                "Add deterministic benchmark scripts with fixed seeds and expected outputs.",
                "Package runtime entry points (CLI or service) for easier deployment.",
                "Add richer integration tests for end-to-end training/evaluation flows.",
            ]
        return improvements[:5]

    @staticmethod
    def _hireability_verdict(score_out_of_100: float) -> str:
        if score_out_of_100 >= 85:
            return "Excellent"
        if score_out_of_100 >= 75:
            return "Strong"
        if score_out_of_100 >= 65:
            return "Good"
        if score_out_of_100 >= 50:
            return "Fair"
        return "Poor"

    def evaluate(self) -> Dict[str, object]:
        self._scan_all()
        scores = self._score_categories()
        overall = round(sum(scores.values()) / len(scores) * 10, 1)
        hireability = self._hireability_verdict(overall)
        summary = self._build_dynamic_summary()
        strongest = self._derive_strengths()
        weakest = self._derive_weaknesses()
        improvements = self._derive_improvements()

        maturity = (
            "job-ready"
            if overall >= 85
            else "advanced"
            if overall >= 75
            else "intermediate"
            if overall >= 60
            else "beginner"
        )
        if overall >= 80 and self.findings.get("has_training") and self.findings.get("has_model_arch"):
            interview_note = (
                "Strong interview potential: repository shows implementer-level AI code paths; "
                "candidate should be ready to justify architecture, data, and evaluation decisions."
            )
        elif overall >= 65:
            interview_note = (
                "Moderate interview potential: technical depth is visible, but reproducibility/testing "
                "and design-tradeoff explanations may be probed heavily."
            )
        else:
            interview_note = (
                "Limited interview signal from repository evidence alone; deeper technical discussion may expose gaps "
                "in experimentation rigor and engineering completeness."
            )

        return {
            "repository_summary": {
                "short_summary": summary,
                "files_scanned": len(self.text_files),
                "binary_or_unparsed_files": len(self.binary_files) + len(self.unread_files),
                "not_fully_analyzed": [vars(x) for x in self.unread_files[:50]],
            },
            "file_by_file_analysis": [vars(x) for x in self.file_notes],
            "category_scores_out_of_10": scores,
            "overall_score_out_of_100": overall,
            "overall_verdict": {
                "strongest_parts": strongest,
                "weakest_parts_or_risks": weakest,
                "missing_pieces_highest_impact": improvements,
                "project_level": maturity,
            },
            "hireability_assessment": {
                "verdict": hireability,
                "interview_readiness_note": interview_note,
            },
            "concrete_improvements_ranked_by_impact": [
                {"rank": i + 1, "improvement": txt} for i, txt in enumerate(improvements)
            ],
        }


def to_markdown(report: Dict[str, object]) -> str:
    scores = report["category_scores_out_of_10"]
    lines: List[str] = []
    lines.append("1. Repository summary")
    rs = report["repository_summary"]
    lines.append(f"- {rs['short_summary']}")
    lines.append(f"- Files scanned: {rs['files_scanned']}")
    lines.append(f"- Binary/unparsed files: {rs['binary_or_unparsed_files']}")
    if rs["not_fully_analyzed"]:
        lines.append("- Files not fully analyzed:")
        for item in rs["not_fully_analyzed"]:
            lines.append(f"  - {item['path']}: {item['notes']}")

    lines.append("\n2. File-by-file analysis")
    for item in report["file_by_file_analysis"]:
        lines.append(f"- {item['path']}: {item['notes']}")

    lines.append("\n3. Category-wise scores")
    for k, v in scores.items():
        lines.append(f"- {k}: {v:.1f}/10")

    lines.append("\n4. Overall verdict")
    lines.append(f"- Overall score: {report['overall_score_out_of_100']}/100")
    ov = report["overall_verdict"]
    lines.append(f"- Project level: {ov['project_level']}")
    lines.append("- Strongest parts:")
    for p in ov["strongest_parts"]:
        lines.append(f"  - {p}")
    lines.append("- Weakest parts or risks:")
    for p in ov["weakest_parts_or_risks"]:
        lines.append(f"  - {p}")
    lines.append("- Missing pieces that would improve it most:")
    for p in ov["missing_pieces_highest_impact"]:
        lines.append(f"  - {p}")

    lines.append("\n5. Hireability assessment")
    ha = report["hireability_assessment"]
    lines.append(f"- Hireability verdict: {ha['verdict']}")
    lines.append(f"- Interview readiness note: {ha['interview_readiness_note']}")

    lines.append("\n6. Concrete improvements ranked by impact")
    for item in report["concrete_improvements_ranked_by_impact"]:
        lines.append(f"- {item['rank']}. {item['improvement']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a repository for AI/ML and engineering quality")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default="", help="Optional output file path")
    args = parser.parse_args()

    evaluator = RepositoryEvaluator(args.repo_path)
    report = evaluator.evaluate()
    rendered = to_markdown(report) if args.format == "markdown" else json.dumps(report, indent=2)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
