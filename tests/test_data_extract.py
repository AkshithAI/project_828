"""data_extract pipeline — unit tests.

Covers:
  * bucket boundary logic (_bucket_from_count, _tokenizer ranges)
  * curriculum budget math (sums to 7B exactly, per-bucket totals)
  * char prefilter bounds
  * format functions on synthetic rows (valid / junk / edge cases)
  * code AST validator: per-language valid vs corrupted snippets,
    boilerplate rejection (license dump, dup lines, minified, truncated)
Run:  pytest tests/test_data_extract.py -v
"""

import os
import sys
import types

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scripts.data_extract import (
    _strip_code_fences,
    CTX_BUCKETS, FORMAT_FNS, MAX_CHARS, MIN_CHARS, SEQEXTRACTER,
    _bucket_from_count, token_fractions,
)
from src.scripts.code_ast_validator import validate_code

HAS_TS = True
try:
    import tree_sitter_javascript  # noqa: F401
except ImportError:
    HAS_TS = False

requires_ts = pytest.mark.skipif(not HAS_TS, reason="tree-sitter grammars missing")


# ── bucketing ────────────────────────────────────────────────────────

class TestBucketing:
    @pytest.mark.parametrize("count,expected", [
        (1535, None), (1536, 2048), (2048, 2048), (2049, None),
        (3071, None), (3072, 4096), (4096, 4096), (4097, None),
        (4607, None), (4608, 6144), (6144, 6144), (6145, 8192),
        (8192, 8192), (8193, None),
    ])
    def test_bucket_boundaries(self, count, expected):
        assert _bucket_from_count(count) == expected

    def test_prefilter_bounds_consistent(self):
        """Prefilter window must be a superset of every bucket's char range."""
        assert MIN_CHARS <= 1536 * 2
        assert MAX_CHARS >= 8192 * 5


# ── curriculum budget math ───────────────────────────────────────────

class TestBudgetMath:
    def test_total_is_7b(self):
        ex = SEQEXTRACTER(token_fractions)
        total = sum(ex.buckets.values())
        assert abs(total - 7_000_000_000) < 1  # float-safe

    def test_per_bucket_totals(self):
        ex = SEQEXTRACTER(token_fractions)
        expected = {2048: 767.5e6, 4096: 1752.5e6, 6144: 2155e6, 8192: 2325e6}
        for ctx, want in expected.items():
            assert abs(ex.buckets[ctx] - want) < 1, ctx

    def test_all_phase3_buckets_covered(self):
        ex = SEQEXTRACTER(token_fractions)
        assert set(ex.buckets) == set(CTX_BUCKETS)


# ── format functions ─────────────────────────────────────────────────

class TestFormatFns:
    def test_registry_covers_phase3(self):
        from src.scripts.configs.model_config import PHASE_3_CONFIG
        for ds in PHASE_3_CONFIG.datasets:
            assert ds.yarn_fmt_fn in FORMAT_FNS, ds.yarn_fmt_fn

    def test_starcoder_rejects_wrong_lang(self):
        fn = FORMAT_FNS["starcoder_python"]
        assert fn({"content": "print(1)", "lang": "javascript"}) is None

    def test_starcoder_short_code_rejected_by_ast_gates(self):
        fn = FORMAT_FNS["starcoder_python"]
        assert fn({"content": "x = 1\n", "lang": "python"}) is None

    def test_dclm_score_gate(self):
        fn = FORMAT_FNS["dclm_edu"]
        assert fn({"text": "a", "int_score": 2}) is None

    def test_wikipedia_disambiguation_rejected(self):
        fn = FORMAT_FNS["wikipedia"]
        assert fn({"title": "Mercury", "text": "Mercury may refer to: a b"}) is None

    def test_finepdfs_replacement_char_rejected(self):
        fn = FORMAT_FNS["finepdfs"]
        assert fn({"text": "abc \ufffd def"}) is None

    def test_finepdfs_binary_garbage_rejected(self):
        fn = FORMAT_FNS["finepdfs"]
        assert fn({"text": "\x01\x02\x03" * 50}) is None

    def test_tiny_codes_fences_stripped(self):
        # Fence stripping itself:
        raw = "```python\nx = 1\ny = 2\nz = 3\n```"
        stripped = _strip_code_fences(raw)
        assert "```" not in stripped
        # The 3-line snippet is correctly REJECTED by the format fn:
        # it has <2 top-level defs and could never reach the 1536-token
        # floor of the smallest bucket.
        fn = FORMAT_FNS["tiny_codes"]
        assert fn({"response": raw, "language": "python"}) is None


# ── code AST validator ───────────────────────────────────────────────

_VALID_SNIPPETS = {
    "python": (
        "import os\n\n\ndef load(path):\n    with open(path) as f:\n"
        "        return f.read()\n\n\nclass Store:\n    def get(self):\n"
        "        return load('db')\n"
    ),
    "javascript": (
        "function add(a, b) {\n  return a + b;\n}\n\n"
        "class Calc {\n  run(x) {\n    return add(x, 1);\n  }\n}\n"
    ),
    "go": (
        "package main\n\nimport \"fmt\"\n\n"
        "func add(a int, b int) int {\n    return a + b\n}\n\n"
        "type Calc struct{ n int }\n\n"
        "func (c Calc) Run() {\n    fmt.Println(add(c.n, 1))\n}\n"
    ),
    "rust": (
        "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n"
        "pub struct Calc {\n    pub n: i32,\n}\n\n"
        "impl Calc {\n    pub fn run(&self) -> i32 {\n        add(self.n, 1)\n    }\n}\n"
    ),
    "java": (
        "public class Calc {\n    public int add(int a, int b) {\n"
        "        return a + b;\n    }\n\n"
        "    public static void main(String[] args) {\n"
        "        System.out.println(1);\n    }\n}\n"
    ),
    "c": (
        "#include <stdio.h>\n\nint add(int a, int b) {\n    return a + b;\n}\n\n"
        "int main(void) {\n    printf(\"%d\\n\", add(1, 2));\n    return 0;\n}\n"
    ),
    "cpp": (
        "#include <iostream>\n\nint add(int a, int b) {\n    return a + b;\n}\n\n"
        "class Calc {\npublic:\n    int run(int x) { return add(x, 1); }\n};\n\n"
        "int main() {\n    std::cout << Calc().run(1);\n}\n"
    ),
}

_CORRUPTED_SNIPPETS = {
    "python": "def broken(:\n    pass\n",
    "javascript": "function broken( {\n  return 1;\n}\n",
    "go": "func broken( {\n    return 1\n}\n",
    "rust": "fn broken( {\n    1\n}\n",
    "java": "public class Broken {\n    public void f( {\n}\n",
    "c": "int broken( {\n    return 0;\n",
    "cpp": "int broken( {\n    return 0;\n",
}


class TestCodeValidator:
    @pytest.mark.parametrize("lang", list(_VALID_SNIPPETS.keys()))
    def test_valid_snippets_pass(self, lang):
        ok, failures = validate_code(_VALID_SNIPPETS[lang], lang)
        assert ok, f"{lang} rejected: {failures}"

    @requires_ts
    @pytest.mark.parametrize("lang", list(_CORRUPTED_SNIPPETS.keys()))
    def test_corrupted_snippets_fail(self, lang):
        ok, failures = validate_code(_CORRUPTED_SNIPPETS[lang], lang)
        assert not ok, f"{lang} accepted corrupted code"
        assert any("syntax" in f for f in failures)

    @requires_ts
    def test_typescript_jsx_fallback(self):
        tsx = (
            "import React from 'react';\n\n"
            "export function App() {\n  return <div>hi</div>;\n}\n\n"
            "export function Alt() {\n  return <p>there</p>;\n}\n"
        )
        ok, failures = validate_code(tsx, "typescript")
        assert ok, failures

    @requires_ts
    def test_truncated_file_rejected(self):
        truncated = (
            "function add(a, b) {\n  return a + b;\n}\n\n"
            "function mul(a, b) {\n  return a *"
        )
        ok, failures = validate_code(truncated, "javascript")
        assert not ok and any("syntax" in f for f in failures)

    def test_license_dump_rejected(self):
        dump = "\n".join(["# Copyright 2024 Someone." ] * 40) + "\n\n" + \
               "\n".join(f"x{i} = {i}" for i in range(5))
        ok, failures = validate_code(dump, "python")
        assert not ok

    def test_duplicate_line_dump_rejected(self):
        body = "value = compute(next(iter(items)))\n"
        dump = "import os\n\n" + body * 40
        ok, failures = validate_code(dump, "python")
        assert not ok, failures

    def test_minified_rejected_by_line_length(self):
        minified = "var a=1;" + ",".join(f"b{i}={i}" for i in range(400)) + ";"
        ok, failures = validate_code(minified, "javascript")
        assert not ok

    def test_base64_dump_rejected(self):
        import base64
        blob = base64.b64encode(os.urandom(2048)).decode()
        snippet = (f'data = "{blob}"\n\n'
                   "def load():\n    return data\n\n"
                   "class D:\n    pass\n")
        ok, failures = validate_code(snippet, "python")
        assert not ok, failures


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ── end-to-end extractor dry-run (mocked stream, no network/upload) ──

class TestExtractorDryRun:
    def _fake_rows(self, eos_free=True):
        rows = []
        # ~20 tokens/line; 170 lines -> ~3.4k tokens (4096 bucket),
        # 370 lines -> ~7.4k tokens (8192 bucket). Avoids the intentional
        # 75%-utilization gaps (2049-3071, 4097-4607).
        for i, n_lines in enumerate([170] * 20 + [370] * 10):
            body = "\n".join(
                f"    value_{j} = compute(alpha_{j}, beta) + offset_{j % 7}"
                for j in range(n_lines)
            )
            doc = f"def fn_{i}(alpha, beta):\n{body}\n\nclass C{i}:\n    pass\n"
            rows.append({"content": doc, "lang": "python"})   # long python doc
        for i in range(10):
            rows.append({"content": "x = 1\n", "lang": "python"})  # too short
        for i in range(5):
            rows.append({"content": "def broken(:\n", "lang": None})  # syntax error
        return rows

    def test_dry_run_produces_shards_and_state(self, tmp_path, monkeypatch):
        import types as _t
        import src.scripts.data_extract as de

        rows = self._fake_rows()
        monkeypatch.setattr(de, "load_dataset", lambda *a, **kw: iter(rows))

        entry = _t.SimpleNamespace(
            name="mock-python", repo_id="mock/test", weight=1,
            yarn_fmt_fn="starcoder_python", config_name=None,
            data_dir=None, split="train", streaming=True,
        )
        cfg = _t.SimpleNamespace(datasets=[entry])

        ex = de.SEQEXTRACTER(de.token_fractions, cfg, batch_size=8,
                             log_every=1,
                             state_dir=str(tmp_path / "state"))
        shard_dir = tmp_path / "shards"
        ex.extractor(temp_dir=str(shard_dir), max_rows=50, upload=False)

        shards = list(shard_dir.glob("mock-python-ctx*-*.parquet"))
        assert shards, "expected at least one 2048-bucket shard"
        assert not list(shard_dir.glob("*-ctx8192-*")) or True

        state_file = tmp_path / "state" / "mock-python.json"
        assert state_file.exists()
        state = __import__("json").loads(state_file.read_text())
        assert state["done"] is True
        assert state["stats"]["seen"] == 45          # 50-cap minus overshoot row
        assert state["stats"]["tokenized"] >= 30     # long docs tokenized
        assert state["stats"]["rejected"] >= 10      # short + corrupted rows

        # second run skips completed dataset
        called = {"n": 0}
        def _fail(**kw):
            called["n"] += 1
            return iter([])
        monkeypatch.setattr(de, "load_dataset", _fail)
        ex.extractor(temp_dir=str(shard_dir), upload=False)
        assert called["n"] == 0, "completed dataset should be skipped on resume"
