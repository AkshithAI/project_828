"""Long-sequence (YaRN Phase-3) data extraction pipeline.

Extracts documents that are *naturally* 2k / 4k / 6k / 8k tokens (strictly
validated), buckets them per the token_fractions curriculum, shards to
parquet, and uploads one HF dataset repo per (dataset, bucket).

Architecture (throughput-oriented):

    HF stream (rows)
      ├─ Phase 1  per-row cheap ops:  column extract → clean → quality gates
      │            → code AST validation → CHAR-LENGTH PREFILTER
      │            (kills most rows before any tokenization)
      ├─ Phase 2  batched tokenization: tokenizer.encode_batch
      │            (gigatoken Rust backend / HF fast tokenizer — GIL-free)
      └─ Phase 3  per-row int ops: token count → ctx bucket → budget check

Quality gates are ported from production pipelines (see
code_ast_validator.py): StarCoder2 text filters, OpenCoder RefineCode
structural rules, and per-language AST syntax validation.

Known accepted behaviour:
  * The document that crosses a bucket's token budget is kept (slight
    overshoot) — deliberate, per pipeline owner.
"""

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.configs.model_config import PHASE_3_CONFIG, PhaseConfig
from src.scripts.tokenizer import tokenizer_v1 as tokenizer
from src.scripts.code_ast_validator import validate_code
from src.scripts.dataloader import (
    _clean_html_text,
    _clean_stackexchange_text,
    _fmt_stackexchange_programming_cs,
    _latex_density,
)

# ── Curriculum ────────────────────────────────────────────────────────

token_fractions = {
    350_000_000: {2048: 0.70, 4096: 0.30},
    1_500_000_000: {2048: 0.15, 4096: 0.65, 6144: 0.20},
    3_000_000_000: {2048: 0.10, 4096: 0.25, 6144: 0.55, 8192: 0.10},
    5_500_000_000: {2048: 0.05, 4096: 0.15, 6144: 0.35, 8192: 0.45},
    7_000_000_000: {2048: 0.05, 4096: 0.10, 6144: 0.15, 8192: 0.70},
}

CTX_BUCKETS = (2048, 4096, 6144, 8192)

# ── Tokenization / bucketing ──────────────────────────────────────────

# Bucket ranges: contiguous so no documents are dropped between buckets.
# A doc must reach >= 1536 tokens to enter the smallest bucket.
_BUCKET_RANGES: Tuple[Tuple[int, int, int], ...] = (
    (2048, 1536, 2048),
    (4096, 2049, 4096),
    (6144, 4097, 6144),
    (8192, 6145, 8192),
)

# Char-length prefilter bounds (conservative chars-per-token window).
# A doc below MIN_CHARS cannot reach 1536 tokens even at ~2 chars/token;
# a doc above MAX_CHARS must exceed 8192 tokens even at ~6 chars/token.
MIN_CHARS = 3072
MAX_CHARS = 49_152


def _bucket_from_count(num_tokens: int) -> Optional[int]:
    for ctx, lo, hi in _BUCKET_RANGES:
        if lo <= num_tokens <= hi:
            return ctx
    return None


def _tokenizer(text: str) -> Optional[Tuple[int, int]]:
    """Tokenize one doc and return (num_tokens, ctx_bucket) or None."""
    num_tokens = len(tokenizer.encode(text))
    ctx_bucket = _bucket_from_count(num_tokens)
    if ctx_bucket is None:
        return None
    return num_tokens, ctx_bucket


def _encode_batch(texts: List[str]) -> List[List[int]]:
    """Batched tokenization — releases the GIL on both backends."""
    fn = getattr(tokenizer, "encode_batch", None)
    if fn is not None:
        return fn(texts)
    return tokenizer(texts, add_special_tokens=False)["input_ids"]


# ── Quality gates (shared) ────────────────────────────────────────────

_RE_FENCE = re.compile(r"^```[\w+-]*\n?|```\s*$", re.MULTILINE)


def _strip_code_fences(code: str) -> str:
    return _RE_FENCE.sub("", code).strip()


def _pdf_quality_gate(text: str) -> Optional[str]:
    """PDF-extraction artifact guards."""
    if "\ufffd" in text:                                   # replacement char
        return None
    alnum = sum(ch.isalnum() for ch in text)
    if alnum / max(len(text), 1) < 0.60:                   # binary garbage
        return None
    hyphen_breaks = len(re.findall(r"[a-z]-\n[a-z]", text))
    if hyphen_breaks / max(len(text) / 1000, 1) > 20:      # hyphenation spam
        return None
    return text


def _prose_quality_gate(text: str) -> Optional[str]:
    """Shared prose gates: HTML cleanup + latex-density cap."""
    if _latex_density(text) > 5.0:
        return None
    return text


def _code_gate(text: str, lang: str) -> Optional[str]:
    """AST syntax + structural validation (hard gate)."""
    ok, _failures = validate_code(text, lang)
    return text if ok else None


# ── Format functions (dataloader contract: Dict -> Optional[str]) ────

_STARCODER_LANGS = {
    "starcoder_python": "python", "starcoder_javascript": "javascript",
    "starcoder_java": "java", "starcoder_typescript": "typescript",
    "starcoder_cpp": "cpp", "starcoder_c": "c",
    "starcoder_go": "go", "starcoder_rust": "rust",
}


def _make_starcoder_fn(lang: str) -> Callable[[Dict], Optional[str]]:
    def _fmt(row: Dict[str, Any]) -> Optional[str]:
        content = row.get("content")
        if not content:
            return None
        row_lang = (row.get("lang") or "").lower()
        if row_lang and lang not in row_lang:
            return None
        return _code_gate(content, lang)
    return _fmt


def _fmt_tiny_codes(row: Dict[str, Any]) -> Optional[str]:
    code = row.get("response") or ""
    if code.startswith("```"):
        code = _strip_code_fences(code)
    code_lines = [l for l in code.split("\n")
                  if l.strip() and not l.strip().startswith(("#", "//"))]
    if len(code_lines) < 3:
        return None
    lang = (row.get("language") or "").lower()
    if lang in ("python", "javascript", "java", "go", "rust", "c", "cpp"):
        return _code_gate(code, lang)
    return code  # language unknown/untracked — text gates only


def _fmt_dclm_edu(row: Dict[str, Any]) -> Optional[str]:
    score = row.get("edu_int_score", row.get("edu_score", 0))
    if isinstance(score, float):
        score = int(score)
    if score < 3:
        return None
    text = row.get("text") or ""
    cleaned = _clean_html_text(text)
    return _prose_quality_gate(cleaned)


def _fmt_wikipedia(row: Dict[str, Any]) -> Optional[str]:
    text = row.get("text") or ""
    title = row.get("title") or ""
    if len(text) < 500 or "may refer to:" in text[:200]:
        return None
    doc = f"{title}\n\n{text}" if title else text
    return _prose_quality_gate(doc)


def _fmt_fineweb(row: Dict[str, Any]) -> Optional[str]:
    text = row.get("text") or ""
    if not text:
        return None
    return _prose_quality_gate(_clean_html_text(text))


def _fmt_finepdfs(row: Dict[str, Any]) -> Optional[str]:
    text = row.get("text") or ""
    if not text:
        return None
    return _pdf_quality_gate(_clean_html_text(text))


def _fmt_stackexchange(row: Dict[str, Any]) -> Optional[str]:
    """Adapter: dataloader's cleaner already enforces site whitelist,
    chrome stripping, latex density, min length."""
    return _fmt_stackexchange_programming_cs(row)


def _fmt_default(row: Dict[str, Any]) -> Optional[str]:
    text = row.get("text") or row.get("content") or ""
    if not text:
        return None
    return _prose_quality_gate(_clean_html_text(text))


FORMAT_FNS: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {
    "default": _fmt_default,
    **{name: _make_starcoder_fn(lang) for name, lang in _STARCODER_LANGS.items()},
    "tiny_codes": _fmt_tiny_codes,
    "stackexchange_programming_cs": _fmt_stackexchange,
    "dclm_edu": _fmt_dclm_edu,
    "wikipedia": _fmt_wikipedia,
    "fineweb_dedup": _fmt_fineweb,
    "finepdfs": _fmt_finepdfs,
}


# ── Extractor ─────────────────────────────────────────────────────────

class SEQEXTRACTER:
    def __init__(self,
                 token_fractions: Dict[int, Dict[int, float]],
                 phase_config: PhaseConfig = None,
                 batch_size: int = 256,
                 log_every: int = 50,
                 state_dir: str = "./extraction_state",
        ):
        self.token_fractions = token_fractions
        self.phase_config = phase_config
        self.batch_size = batch_size
        self.log_every = log_every
        self.state_dir = Path(state_dir)
        self.calculate_token_budget()

    def calculate_token_budget(self):
        """Cumulative per-bucket token budgets across all curriculum stages."""
        self.buckets: Dict[int, float] = {ctx: 0.0 for ctx in CTX_BUCKETS}
        previous_tokens = 0
        for end_tokens, ctx_fractions in sorted(self.token_fractions.items()):
            stage_budget = end_tokens - previous_tokens
            for ctx, frac in ctx_fractions.items():
                self.buckets[ctx] = self.buckets.get(ctx, 0.0) + frac * stage_budget
            previous_tokens = end_tokens

    # ── resume state ──────────────────────────────────────────────

    def _state_path(self, ds_name: str) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir / f"{ds_name}.json"

    def _load_state(self, ds_name: str) -> Optional[Dict]:
        p = self._state_path(ds_name)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _save_state(self, ds_name: str, state: Dict) -> None:
        self._state_path(ds_name).write_text(json.dumps(state))

    # ── batch flush ───────────────────────────────────────────────

    def _flush_token_batch(self, texts: List[str], stats: Dict[str, int],
                           num_tok_per_bucket: Dict[int, int],
                           token_budget: Dict[int, float],
                           bucket_buffers: Dict[int, List[str]],
                           bucket_exhausted: Dict[int, bool]) -> None:
        """Tokenize a micro-batch, bucket survivors, enforce budgets."""
        for text, ids in zip(texts, _encode_batch(texts)):
            stats["tokenized"] += 1
            ctx = _bucket_from_count(len(ids))
            if ctx is None or bucket_exhausted[ctx]:
                stats["out_of_range"] += 1
                continue
            num_tok_per_bucket[ctx] += len(ids)
            bucket_buffers[ctx].append(text)
            if num_tok_per_bucket[ctx] >= token_budget[ctx]:
                bucket_exhausted[ctx] = True

    def _extract_single(self, ds_entry, total_weight: float,
                        temp_dir: str, source_split: Optional[str],
                        max_rows: Optional[int], upload: bool,
                        hf_namespace: str) -> None:
        """Extract one dataset. Runs in a subprocess for memory isolation."""
        fmt_fn = FORMAT_FNS.get(ds_entry.yarn_fmt_fn or "default")
        if fmt_fn is None:
            raise KeyError(
                f"yarn_fmt_fn={ds_entry.yarn_fmt_fn!r} has no format "
                f"function; available: {sorted(FORMAT_FNS)}"
            )

        weight_frac = ds_entry.weight / total_weight
        token_budget = {ctx: self.buckets[ctx] * weight_frac
                        for ctx in CTX_BUCKETS}
        num_tok_per_bucket = {ctx: 0 for ctx in CTX_BUCKETS}
        bucket_buffers: Dict[int, List[str]] = {ctx: [] for ctx in CTX_BUCKETS}
        bucket_exhausted = {ctx: False for ctx in CTX_BUCKETS}
        shard_idx_per_bucket = {ctx: 0 for ctx in CTX_BUCKETS}
        schema = pa.schema([("text", pa.string())])

        stats = {"seen": 0, "rejected": 0, "tokenized": 0,
                 "out_of_range": 0, "_i": 0}

        def _write_shard(ctx: int) -> None:
            if not bucket_buffers[ctx]:
                return
            path = os.path.join(
                temp_dir,
                f"{ds_entry.name}-ctx{ctx}-{shard_idx_per_bucket[ctx]:05d}.parquet",
            )
            pq.write_table(
                pa.Table.from_pydict({"text": bucket_buffers[ctx]},
                                     schema=schema),
                path, compression="snappy",
            )
            bucket_buffers[ctx].clear()
            shard_idx_per_bucket[ctx] += 1

        resolved_split = (source_split or ds_entry.split or "train").lower()
        data_stream = load_dataset(
            ds_entry.repo_id,
            name=ds_entry.config_name,
            data_dir=ds_entry.data_dir,
            split=resolved_split,
            streaming=ds_entry.streaming,
        )
        # Shuffle only for datasets with known ordering issues
        # (e.g. stackexchange is sorted alphabetically by site).
        # DO NOT shuffle large-row datasets like dclm-edu — the shuffle
        # buffer pre-fills 1k rows and each web page can be MBs, causing OOM.
        _NEEDS_SHUFFLE = {"stackexchange_programming_cs"}
        if (ds_entry.yarn_fmt_fn in _NEEDS_SHUFFLE
                and hasattr(data_stream, 'shuffle')):
            data_stream = data_stream.shuffle(seed=42, buffer_size=1_000)

        print(f"[extract] {ds_entry.name} (fmt={ds_entry.yarn_fmt_fn}, "
              f"budgets={ {k: f'{v/1e6:.0f}M' for k, v in token_budget.items()} })")
        pending: List[str] = []

        def _flush_pending() -> None:
            if not pending:
                return
            self._flush_token_batch(
                list(pending), stats, num_tok_per_bucket,
                token_budget, bucket_buffers, bucket_exhausted)
            pending.clear()

        for row in data_stream:
            stats["seen"] += 1
            if max_rows is not None and stats["seen"] > max_rows:
                print(f"[dry-run] row cap {max_rows} reached")
                break

            # ── Phase 1: cheap per-row ops ──
            try:
                text = fmt_fn(row)
            except Exception:
                stats["rejected"] += 1
                continue
            if text is None or not (MIN_CHARS <= len(text) <= MAX_CHARS):
                stats["rejected"] += 1
                continue
            pending.append(text)

            # ── Phase 2: batched tokenization ──
            if len(pending) >= self.batch_size:
                _flush_pending()

            if all(bucket_exhausted.values()):
                break

            if stats["seen"] % (self.batch_size * self.log_every) == 0:
                pct = {c: f"{num_tok_per_bucket[c] / max(token_budget[c], 1):.0%}"
                       for c in CTX_BUCKETS}
                print(f"  rows={stats['seen']:,} rej={stats['rejected']:,} "
                      f"tok={stats['tokenized']:,} oor={stats['out_of_range']:,} "
                      f"budget={pct}")

        _flush_pending()

        # ── final shards + budget enforcement on leftovers ──
        for ctx in CTX_BUCKETS:
            if num_tok_per_bucket[ctx] >= token_budget[ctx]:
                bucket_exhausted[ctx] = True
            _write_shard(ctx)

        print(f"[done] {ds_entry.name}: {stats}")
        state = {"done": True, "stats": {
            k: v for k, v in stats.items() if not k.startswith("_")}}
        self._save_state(ds_entry.name, state)

        if upload:
            api = HfApi()
            for ctx in CTX_BUCKETS:
                if shard_idx_per_bucket[ctx] == 0:
                    continue
                target_repo_id = f"{hf_namespace}/{ds_entry.name}-ctx-{ctx}"
                api.create_repo(repo_id=target_repo_id,
                                repo_type="dataset", exist_ok=True)
                print(f"[upload] {shard_idx_per_bucket[ctx]} shards "
                      f"-> {target_repo_id}")
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=target_repo_id,
                    repo_type="dataset",
                    path_in_repo="data",
                    allow_patterns=f"{ds_entry.name}-ctx{ctx}-*.parquet",
                )

    # ── main ──────────────────────────────────────────────────────

    def extractor(self,
                  temp_dir: Optional[str] = None,
                  shard_size: int = 100_000,
                  source_split: Optional[str] = None,
                  max_rows: Optional[int] = None,
                  upload: bool = True,
        ):
        """Run extraction for every dataset in the phase config.

        Each dataset runs in a **subprocess** to guarantee full memory
        reclamation between datasets. On constrained machines (8 GB),
        HF ``load_dataset`` accumulates module-level caches that
        ``gc.collect()`` cannot free.

        Args:
            temp_dir:   shard staging dir (unique tmpdir per run by default).
            shard_size: docs per intermediate shard.
            max_rows:   DEBUG cap on rows per dataset (dry-run).
            upload:     set False for dry runs (shards stay on disk).
        """
        import multiprocessing as mp
        ctx = mp.get_context("fork")

        api = HfApi()
        hf_namespace = api.whoami()["name"] if upload else "dryrun"
        owned_temp = temp_dir is None
        temp_dir = temp_dir or tempfile.mkdtemp(prefix="seq_extract_")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            total_weight = sum(ds.weight for ds in self.phase_config.datasets)
            for ds_entry in self.phase_config.datasets:
                # ── resume: skip fully completed datasets ──
                state = self._load_state(ds_entry.name) or {}
                if state.get("done"):
                    print(f"[skip] {ds_entry.name}: already complete")
                    continue

                # Run in a child process for memory isolation.
                p = ctx.Process(
                    target=self._extract_single,
                    args=(ds_entry, total_weight, temp_dir,
                          source_split, max_rows, upload, hf_namespace),
                )
                p.start()
                p.join()

                if p.exitcode != 0:
                    print(f"[ERROR] {ds_entry.name} exited with code "
                          f"{p.exitcode} — skipping")
                    continue
        finally:
            if owned_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YaRN long-sequence extractor")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="DEBUG row cap per dataset (dry-run)")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--temp-dir", type=str, default=None)
    args = parser.parse_args()

    extractor = SEQEXTRACTER(token_fractions, PHASE_3_CONFIG)
    extractor.extractor(temp_dir=args.temp_dir,
                        max_rows=args.max_rows,
                        upload=not args.no_upload)

