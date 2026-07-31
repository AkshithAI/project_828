from transformers import AutoTokenizer
import gigatoken as gt

# ── Gigatoken-Accelerated Tokenizer ──────────────────────────────────────────
#
# Gigatoken (github.com/marcelroed/gigatoken) provides ~1000x faster BPE
# tokenization via a Rust/SIMD backend. Critically, it RELEASES THE PYTHON
# GIL during encoding, enabling true multi-core CPU parallelism alongside
# the asyncio event loop and GPU kernels.
#
# Two interfaces are exposed:
#
#   tokenizer     – HF-compatible drop-in (.encode, .decode, .eos_token_id,
#                   .vocab_size, .encode_batch).  Used by inference, eval
#                   benchmarks, and any code that needs single-string encoding.
#
#   gt_tokenizer  – Raw gigatoken.Tokenizer for maximum-throughput batch
#                   encoding in the ZeroStallDataLoader pipeline.

# Using pretrained microsoft/phi-2 tokenizer (50k vocab size, code-specialized, 100% Gigatoken Rust SIMD compatible)
_hf_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2", trust_remote_code=True
)

# Initialize gigatoken from HF tokenizer to retain named special tokens (eos_token_id, etc.)
try:
    gt_tokenizer = gt.Tokenizer(_hf_tokenizer)
    tokenizer = gt_tokenizer.as_hf()
    if tokenizer.eos_token_id is None and _hf_tokenizer.eos_token_id is not None:
        tokenizer.eos_token_id = _hf_tokenizer.eos_token_id
    if not hasattr(tokenizer, 'encode_batch'):
        tokenizer.encode_batch = lambda texts: gt_tokenizer._backend.encode_batch(texts)
    print(f"[Tokenizer] Successfully loaded Gigatoken Rust/SIMD backend (microsoft/phi-2, eos_token_id={tokenizer.eos_token_id}).")
except Exception as _e:
    print(f"[Tokenizer] Gigatoken initialization skipped ({_e}); using HF AutoTokenizer.")
    tokenizer = _hf_tokenizer
    if not hasattr(tokenizer, 'encode_batch'):
        tokenizer.encode_batch = lambda texts: tokenizer(texts, add_special_tokens=False)['input_ids']

# Set vocab_size to actual total token count (including added tokens) padded to a multiple of 128 for GPU alignment
_raw_vocab_len = max(len(_hf_tokenizer), getattr(_hf_tokenizer, 'vocab_size', 0))
_padded_vocab_size = ((_raw_vocab_len + 127) // 128) * 128

try:
    type(tokenizer).vocab_size = property(lambda self: _padded_vocab_size)
except Exception:
    tokenizer.vocab_size = _padded_vocab_size
