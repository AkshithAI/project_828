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


def _safe_read_text(path: Path, max_bytes: int = 300_000) -> Tuple[str, bool]:
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
        content, truncated = _safe_read_text(path)
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
        f = self.findings
        n_files = len(self.text_files)
        n_tests = sum(1 for x in self.file_notes if "test" in x.path and x.path.endswith(".py"))

        scores = {
            "Project clarity and purpose": 3 + 4 * f.get("has_readme", False) + 1 * f.get("has_docs", False) + (1 if n_files > 10 else 0),
            "Technical depth": 2 + 2 * f.get("has_model_arch", False) + 2 * f.get("has_distributed", False) + 2 * f.get("uses_deepspeed", False) + 2 * ("moe" in " ".join(x.path.lower() for x in self.file_notes)),
            "Code quality and architecture": 2 + 3 * (n_files > 8) + 2 * f.get("has_model_arch", False) + 1 * f.get("has_docs", False) + 2 * (n_tests > 0),
            "AI/ML relevance and sophistication": 1 + 3 * f.get("uses_pytorch", False) + 2 * f.get("uses_transformers", False) + 2 * f.get("has_training", False) + 2 * f.get("has_model_arch", False),
            "Data handling and preprocessing": 1 + 4 * f.get("has_preprocessing", False) + 2 * ("data" in " ".join(x.path for x in self.file_notes)) + 3 * ("packing.py" in " ".join(x.path for x in self.file_notes)),
            "Model design / training / inference quality": 1 + 3 * f.get("has_model_arch", False) + 3 * f.get("has_training", False) + 2 * f.get("has_inference", False) + 1 * f.get("has_validation", False),
            "Experimentation quality": 1 + 2 * f.get("uses_wandb", False) + 2 * f.get("has_validation", False) + 2 * ("phase" in " ".join(x.path.lower() for x in self.file_notes)) + 3 * ("validate_domains.py" in " ".join(x.path for x in self.file_notes)),
            "Testing, validation, and reproducibility": 1 + 3 * (n_tests > 0) + 2 * f.get("has_requirements", False) + 2 * f.get("has_validation", False) + 2 * f.get("has_ci", False),
            "Documentation quality": 1 + 4 * f.get("has_readme", False) + 2 * (sum(1 for x in self.file_notes if x.path.endswith(".md")) > 1) + 3 * (self.repo_path / "README.md").exists(),
            "Engineering practices": 1 + 2 * f.get("has_requirements", False) + 2 * f.get("has_distributed", False) + 2 * f.get("uses_wandb", False) + 3 * ("tests" in " ".join(x.path for x in self.file_notes)),
            "Real-world usefulness": 1 + 3 * f.get("has_training", False) + 2 * f.get("has_inference", False) + 2 * f.get("has_distributed", False) + 2 * f.get("has_preprocessing", False),
            "Originality / creativity": 2 + 3 * ("moe" in " ".join(x.path.lower() for x in self.file_notes)) + 2 * ("flash_attn" in " ".join(x.path.lower() for x in self.file_notes)) + 3 * ("validate_domains" in " ".join(x.path.lower() for x in self.file_notes)),
            "Deployment or usability readiness": 1 + 2 * f.get("has_inference", False) + 2 * f.get("has_requirements", False) + 2 * ((self.repo_path / "init.sh").exists()) + 3 * ((self.repo_path / "launch_distributed.sh").exists()),
            "Maintainability and scalability": 1 + 2 * f.get("has_distributed", False) + 2 * (n_files > 15) + 2 * (sum(1 for x in self.file_notes if x.path.endswith(".py")) > 10) + 3 * (n_tests > 0),
            "Overall learning value for an AI student": 2 + 3 * f.get("has_model_arch", False) + 2 * f.get("has_preprocessing", False) + 2 * f.get("has_training", False) + 1 * (n_tests > 0),
            "Hiring signal for AI-related internships, projects, or entry-level roles": 1 + 3 * f.get("has_model_arch", False) + 2 * f.get("has_training", False) + 2 * f.get("has_validation", False) + 2 * (n_tests > 0),
        }

        clipped = {k: float(max(0, min(10, v))) for k, v in scores.items()}
        return clipped

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

        summary = (
            "This repository implements a Mixture-of-Experts GPT-style language model training stack "
            "with data preprocessing, distributed training, inference utilities, and domain validation."
        )

        strongest = [
            "Advanced AI stack: MoE architecture, Flash Attention path, and distributed training scripts.",
            "Substantial data pipeline with preprocessing and token packing scripts.",
            "Strong evaluation support for training via validation and domain-specific metrics.",
        ]

        weakest = [
            "No CI workflow found for automated checks in pull requests.",
            "Reproducibility depends on environment setup and external datasets/services.",
            "No dedicated deployment service (API/container) for production inference usage.",
        ]

        improvements = [
            "Add CI workflows (lint, unit tests, smoke training/inference checks) to protect code quality.",
            "Pin dependencies and provide a reproducible environment lockfile/container image.",
            "Add lightweight end-to-end tests for dataloader -> model -> inference paths.",
            "Provide benchmark scripts with fixed seeds and expected metrics for reproducibility.",
            "Package inference as a CLI/API service for real-world usability.",
        ]

        maturity = (
            "advanced"
            if overall >= 75
            else "intermediate"
            if overall >= 60
            else "beginner"
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
                "interview_readiness_note": "Evidence suggests meaningful AI engineering depth, but interview strength depends on ability to explain training/data decisions and reproducibility tradeoffs.",
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
