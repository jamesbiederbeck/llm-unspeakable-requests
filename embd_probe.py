#!/usr/bin/env python3
"""
Probe what embedding dimensions "mean" by pulling the real token embeddings
for a prompt straight out of a GGUF file, perturbing the prompt's own content
tokens' vectors, feeding those modified vectors into llama.cpp via the raw
llama_batch API (bypassing the normal token -> embedding lookup), and
generating a continuation. By default only the prompt's own tokens are
perturbed -- BOS/role-marker/chat-template scaffolding tokens decode normally
(--perturb-all-tokens perturbs everything instead).

Two perturbation modes:
  - axis clamp (default): set component INDEX of every content token to VALUE.
    INDEX can be a single int, a comma-separated list, and/or python-style
    slices (start:stop or start:stop:step) to sweep many dimensions in one
    model load, e.g. "5,10,50:60:2".
  - --diff-mean WORDS_A WORDS_B: add a difference-of-means direction (mean
    embedding of WORDS_A minus WORDS_B) scaled by VALUE, instead of clamping
    a raw axis. Usually more interpretable than a raw index since it isn't
    fighting superposition.

Usage:
    python embd_probe.py --model ./model.gguf "The cat sat on the mat" 42 3.0
    python embd_probe.py --model ./model.gguf --baseline "The cat sat on the mat" 0,5,50:400:50 3.0
    python embd_probe.py --model ./model.gguf --diff-mean " king, prince" " queen, princess" "The king spoke." 0 4.0

Requires: llama-cpp-python, gguf, numpy
    pip install llama-cpp-python gguf numpy
"""
import argparse
import ctypes
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from llama_cpp import Llama, llama_cpp, llama_chat_format

# llama_kv_cache_clear was renamed/restructured into llama_memory_clear(llama_get_memory(ctx), data)
# in the llama.cpp memory-API refactor; support both so this script works against older and newer builds.
if hasattr(llama_cpp, "llama_memory_clear"):
    def _kv_cache_clear(ctx):
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(ctx), True)
else:
    _kv_cache_clear = llama_cpp.llama_kv_cache_clear

# (background, foreground) blessed capability names, cycled per token.
TOKEN_PALETTE = [
    ("on_red", "white"),
    ("on_green", "black"),
    ("on_yellow", "black"),
    ("on_blue", "white"),
    ("on_magenta", "white"),
    ("on_cyan", "black"),
    ("on_bright_red", "black"),
    ("on_bright_green", "black"),
    ("on_bright_blue", "black"),
    ("on_bright_magenta", "black"),
]


def parse_indices(spec: str, n_embd: int) -> list[int]:
    """Parse '5,10,50:60:2' into a sorted, deduped list of dimension indices."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bounds = part.split(":")
            bounds = [int(b) if b else None for b in bounds]
            indices.update(range(*slice(*bounds).indices(n_embd)))
        else:
            indices.add(int(part))
    for idx in indices:
        if not (0 <= idx < n_embd):
            raise SystemExit(f"index {idx} out of range [0, {n_embd})")
    return sorted(indices)


def find_subsequence(haystack, needle):
    """Return (start, end) of the first occurrence of needle in haystack, or None."""
    n = len(needle)
    if n == 0:
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i, i + n
    return None


def warn_if_per_layer_embeddings(reader):
    """Gemma-3n/Gemma-4-family architectures inject an extra per-layer embedding table
    at every transformer layer (not just the input); our raw llama_batch.embd injection
    only supplies token_embd.weight at the input, so it structurally can't reproduce
    what these architectures need -- output degrades even with a true no-op
    (unmodified) injected embedding. Verified empirically: identical to the real
    token-path output on plain-architecture models (e.g. Llama), but garbled on
    gemma-4-E4B even at zero perturbation. Warn rather than silently mislead."""
    if any("per_layer_token_embd" in t.name for t in reader.tensors):
        print(
            "WARNING: this GGUF has per-layer embedding tensors (per_layer_token_embd.weight), "
            "typical of the Gemma-3n/Gemma-4 family. The raw embd-injection technique this script "
            "uses only feeds token_embd.weight at the input and can't reproduce per-layer embedding "
            "injection, so generated output will be degraded/garbled even with zero perturbation. "
            "Perturbation-effect comparisons on this model are not reliable -- use a plain-architecture "
            "model (e.g. Llama) for meaningful results.",
            file=sys.stderr,
        )


def load_token_embeddings(gguf_path: str) -> np.ndarray:
    """Read and dequantize the token embedding matrix straight from the GGUF file.

    Returns a float32 array of shape (n_vocab, n_embd). Doing this ourselves
    (rather than asking llama.cpp for it) is what lets us edit individual
    token vectors before they ever enter the forward pass.
    """
    reader = GGUFReader(gguf_path)
    warn_if_per_layer_embeddings(reader)
    tensor = next((t for t in reader.tensors if t.name == "token_embd.weight"), None)
    if tensor is None:
        candidates = [t.name for t in reader.tensors if "embd" in t.name or "wte" in t.name]
        raise RuntimeError(
            "Couldn't find tensor 'token_embd.weight' in the GGUF. "
            f"Similarly-named tensors present: {candidates}"
        )
    arr = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
    # gguf stores tensors in ne-order (n_embd, n_vocab); the reader already
    # gives us .data reshaped to numpy row-major order, i.e. (n_vocab, n_embd).
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected token_embd.weight shape after dequant: {arr.shape}")
    return arr


def decode_tokens(ctx, token_ids, n_past):
    n = len(token_ids)
    batch = llama_cpp.llama_batch_init(n, 0, 1)
    batch.n_tokens = n
    for i, tid in enumerate(token_ids):
        batch.token[i] = tid
        batch.pos[i] = n_past + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = True
    ret = llama_cpp.llama_decode(ctx, batch)
    llama_cpp.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"llama_decode (tokens) failed: {ret}")
    return n_past + n


def decode_embeddings(ctx, embd_array, n_embd, n_past):
    n = embd_array.shape[0]
    batch = llama_cpp.llama_batch_init(n, n_embd, 1)
    batch.n_tokens = n
    flat = np.ascontiguousarray(embd_array, dtype=np.float32).flatten()
    ctypes.memmove(batch.embd, flat.ctypes.data, flat.nbytes)
    for i in range(n):
        batch.pos[i] = n_past + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = True
    ret = llama_cpp.llama_decode(ctx, batch)
    llama_cpp.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"llama_decode (embd) failed: {ret}")
    return n_past + n


def make_clamp_fn(idx, value):
    def perturb(vecs):
        vecs[:, idx] = value
        return vecs
    return perturb


def make_direction_fn(direction, scale):
    def perturb(vecs):
        return vecs + scale * direction
    return perturb


def decode_perturbed(ctx, embed_table, token_ids, n_embd, perturb_fn, content_range=None):
    """Decode token_ids with `perturb_fn(vecs) -> vecs` applied to the relevant token vectors.

    If content_range=(start, end) is given, only that span is perturbed: the
    prefix and suffix are decoded as normal token batches (untouched embedding
    lookup) and only the content span goes through the raw embd path -- three
    llama_decode calls chained via the persistent KV cache, matching the
    prefix/soft-span/suffix technique. Without a content_range, every token in
    token_ids is perturbed (the whole sequence goes through the embd path).
    """
    if content_range is None:
        vecs = perturb_fn(embed_table[token_ids].copy())
        return decode_embeddings(ctx, vecs, n_embd, 0)

    start, end = content_range
    n_past = 0
    if start > 0:
        n_past = decode_tokens(ctx, token_ids[:start], n_past)
    vecs = perturb_fn(embed_table[token_ids[start:end]].copy())
    n_past = decode_embeddings(ctx, vecs, n_embd, n_past)
    if end < len(token_ids):
        n_past = decode_tokens(ctx, token_ids[end:], n_past)
    return n_past


def get_last_logits(ctx, n_vocab):
    # -1 = logits of the last token processed by the most recent llama_decode call.
    ptr = llama_cpp.llama_get_logits_ith(ctx, -1)
    return np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()


def generate(llm, ctx, n_vocab, n_past, n_predict, eos_id):
    generated = []
    for _ in range(n_predict):
        logits = get_last_logits(ctx, n_vocab)
        tok = int(np.argmax(logits))
        if tok == eos_id:
            break
        generated.append(tok)
        n_past = decode_tokens(ctx, [tok], n_past)
    return llm.detokenize(generated).decode("utf-8", errors="replace")


def print_tokenized(llm, token_ids):
    """Print the prompt with each token highlighted in a different color."""
    from blessed import Terminal

    t = Terminal()
    parts = []
    for i, tid in enumerate(token_ids):
        text = llm.detokenize([tid]).decode("utf-8", errors="replace")
        bg_name, fg_name = TOKEN_PALETTE[i % len(TOKEN_PALETTE)]
        styled = getattr(t, bg_name)(getattr(t, fg_name)(text))
        parts.append(styled)
    print("".join(parts) + t.normal)
    print()
    for i, tid in enumerate(token_ids):
        text = llm.detokenize([tid]).decode("utf-8", errors="replace")
        bg_name, fg_name = TOKEN_PALETTE[i % len(TOKEN_PALETTE)]
        swatch = getattr(t, bg_name)(getattr(t, fg_name)(" " + text + " "))
        print(f"{i:>4}  id={tid:<8} {swatch}")


def get_chat_template(llm):
    """Pull the model's own chat_template out of its GGUF metadata, if it has one."""
    return llm._model.metadata().get("tokenizer.chat_template")


def render_chat_prompt(llm, template, user_content):
    """Render a single user turn through the model's own Jinja2 chat_template."""
    bos_token = llm.detokenize([llm.token_bos()], special=True).decode("utf-8", errors="replace")
    eos_token = llm.detokenize([llm.token_eos()], special=True).decode("utf-8", errors="replace")
    formatter = llama_chat_format.Jinja2ChatFormatter(
        template=template, eos_token=eos_token, bos_token=bos_token, add_generation_prompt=True,
    )
    return formatter(messages=[{"role": "user", "content": user_content}]).prompt


def tokenize_prompt(llm, template, text):
    """Tokenize `text` as a user turn via the model's chat_template when available,
    falling back to a raw completion-style prompt (with BOS) otherwise."""
    if template:
        try:
            rendered = render_chat_prompt(llm, template, text)
            print(f"Rendered chat prompt: {rendered!r}", file=sys.stderr)
            # The template already emits the literal bos_token text; add_bos=False avoids
            # doubling it up, and special=True lets that literal text tokenize back into
            # the real BOS/turn-marker special tokens instead of raw text.
            return llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
        except Exception as e:
            print(f"chat_template rendering failed ({e}); falling back to raw prompt", file=sys.stderr)
    return llm.tokenize(text.encode("utf-8"), add_bos=True)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_cosine(llm, embed_table, word_a, word_b):
    """Compare two words' token embeddings: per-token-pair cosine similarity,
    plus a pooled (mean-of-tokens) comparison for multi-token words."""
    ids_a = llm.tokenize(word_a.encode("utf-8"), add_bos=False, special=False)
    ids_b = llm.tokenize(word_b.encode("utf-8"), add_bos=False, special=False)

    def describe(ids):
        return [(tid, llm.detokenize([tid]).decode("utf-8", errors="replace")) for tid in ids]

    toks_a, toks_b = describe(ids_a), describe(ids_b)
    print(f"{word_a!r} -> {toks_a}")
    print(f"{word_b!r} -> {toks_b}")
    print()

    print("per-token cosine similarity:")
    header = "".join(f"{repr(tb):>16}" for _, tb in toks_b)
    print(" " * 18 + header)
    for tid_a, ta in toks_a:
        row = "".join(f"{cosine(embed_table[tid_a], embed_table[tid_b]):16.4f}" for tid_b, _ in toks_b)
        print(f"{repr(ta):>16}  {row}")

    pooled_a = embed_table[ids_a].mean(axis=0)
    pooled_b = embed_table[ids_b].mean(axis=0)
    print(f"\npooled (mean-of-tokens) cosine similarity: {cosine(pooled_a, pooled_b):.4f}")


def print_top_tokens(llm, embed_table, idx, n):
    """For dimension `idx`, print the n tokens with the highest and lowest values
    along that component -- a cheap first pass at whether a raw dimension clusters
    around any human-interpretable concept. Raw embedding dims are rarely clean
    single-concept axes (superposition), so treat clusters as a hint, not proof."""
    col = embed_table[:, idx]
    order = np.argsort(col)
    top = order[::-1][:n]
    bottom = order[:n]

    def show(ids, label):
        print(f"\n{label} (dim {idx}):")
        for tid in ids:
            piece = llm.detokenize([int(tid)], special=True).decode("utf-8", errors="replace")
            print(f"  {col[tid]:+.4f}  id={tid:<8} {piece!r}")

    show(top, f"top {n}")
    show(bottom, f"bottom {n}")


def pooled_embedding(llm, embed_table, word):
    ids = llm.tokenize(word.encode("utf-8"), add_bos=False, special=False)
    return embed_table[ids].mean(axis=0)


def diff_mean_direction(llm, embed_table, words_a, words_b):
    """Difference-of-means direction: mean embedding of words_a minus words_b
    (e.g. royal-ish words minus neutral ones) -- generally a more meaningful
    perturbation axis than a raw embedding-dimension index, since it's not
    fighting superposition the way a single basis dimension is."""
    vecs_a = np.stack([pooled_embedding(llm, embed_table, w) for w in words_a])
    vecs_b = np.stack([pooled_embedding(llm, embed_table, w) for w in words_b])
    print(f"  group A: {words_a}", file=sys.stderr)
    print(f"  group B: {words_b}", file=sys.stderr)
    return vecs_a.mean(axis=0) - vecs_b.mean(axis=0)


def explain_diff(llm, ctx, n_vocab, template, string_a, string_b, n_predict, eos_id):
    prompt = f'Explain the difference between "{string_a}" and "{string_b}".'
    ids = tokenize_prompt(llm, template, prompt)
    _kv_cache_clear(ctx)
    n_past = decode_tokens(ctx, ids, 0)
    return generate(llm, ctx, n_vocab, n_past, n_predict, eos_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "prompt", type=str, nargs="?", default=None,
        help="prompt text; every token's embedding gets edited (unused with --cosine)",
    )
    ap.add_argument(
        "index", type=str, nargs="?", default=None,
        help="dimension index/indices to overwrite, e.g. '5' or '5,10,50:400:50' (unused with --tokenize/"
        "--cosine/--top-dim; still required as a positional placeholder with --diff-mean, but ignored)",
    )
    ap.add_argument(
        "value", type=float, nargs="?", default=None,
        help="value to write, or scale factor with --diff-mean (unused with --tokenize/--cosine/--top-dim)",
    )
    ap.add_argument("--model", required=True, help="path to the .gguf model file")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--n-predict", type=int, default=100)
    ap.add_argument("--baseline", action="store_true", help="also run the unmodified prompt for comparison")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="after each perturbed run, ask the model (normal, unperturbed) to explain the "
        "difference between the baseline continuation and the perturbed one",
    )
    ap.add_argument("--compare-predict", type=int, default=60, help="tokens to generate for --compare explanations")
    ap.add_argument(
        "--tokenize", action="store_true",
        help="just print the prompt with each token in a different color and exit (no perturbation/generation)",
    )
    ap.add_argument(
        "--raw", action="store_true",
        help="tokenize the prompt as raw completion text (just BOS + prompt) instead of wrapping it "
        "through the model's own chat_template from GGUF metadata",
    )
    ap.add_argument(
        "--cosine", nargs=2, metavar=("WORD_A", "WORD_B"),
        help="print cosine similarity between two words' token embeddings (e.g. --cosine cat tiger) and exit",
    )
    ap.add_argument(
        "--top-dim", type=str, default=None, metavar="INDEX",
        help="for each dimension index/indices (e.g. '5' or '5,10,50:60'), print the top/bottom-N "
        "vocab tokens by value along that component, and exit",
    )
    ap.add_argument("--top-n", type=int, default=20, help="how many tokens to show per end with --top-dim")
    ap.add_argument(
        "--perturb-all-tokens", action="store_true",
        help="perturb every token in the rendered prompt (BOS/role-marker/chat-template scaffolding "
        "included), not just the tokens belonging to the prompt text itself. Off by default: by "
        "default only the prompt's own content tokens go through the raw embd path (prefix/content/"
        "suffix llama_decode calls chained through the KV cache); scaffolding decodes normally.",
    )
    ap.add_argument(
        "--diff-mean", nargs=2, metavar=("WORDS_A", "WORDS_B"),
        help="perturb via a difference-of-means direction instead of a raw axis: mean embedding of "
        "comma-separated WORDS_A minus WORDS_B (e.g. --diff-mean ' king, prince' ' queen, princess'), "
        "added to the content tokens' embeddings scaled by VALUE. INDEX is ignored in this mode.",
    )
    ap.add_argument(
        "--target", type=str, default=None,
        help="narrow perturbation to just this substring of the prompt (e.g. a single instruction "
        "word like 'fix') instead of every content token in the prompt. Must tokenize to a "
        "contiguous span within the prompt's own tokens.",
    )
    args = ap.parse_args()

    standalone_mode = args.tokenize or args.cosine or args.top_dim
    if not standalone_mode:
        if args.diff_mean:
            if args.value is None:
                ap.error("value (the scale to apply to the diff-mean direction) is required with --diff-mean")
        elif args.index is None or args.value is None:
            ap.error("index and value are required unless --tokenize/--cosine/--top-dim/--diff-mean is given")
    if not (args.cosine or args.top_dim) and args.prompt is None:
        ap.error("prompt is required unless --cosine/--top-dim is given")

    # logits_all=True: with logits_all=False the underlying context's output-logits
    # buffer intermittently comes back NULL from llama_get_logits_ith when we drive
    # llama_decode manually via the raw batch API (observed empirically on
    # llama-cpp-python 0.3.5). logits_all=True reserves output space for every
    # position and reliably avoids it, at the cost of a bit more memory.
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, logits_all=True)
    ctx = llm.ctx
    n_vocab = llm.n_vocab()

    if args.cosine or args.top_dim:
        print(f"Loading token embedding matrix from {args.model} ...", file=sys.stderr)
        embed_table = load_token_embeddings(args.model)
        if args.cosine:
            print_cosine(llm, embed_table, args.cosine[0], args.cosine[1])
        if args.top_dim:
            for idx in parse_indices(args.top_dim, embed_table.shape[1]):
                print_top_tokens(llm, embed_table, idx, args.top_n)
        return

    template = None if args.raw else get_chat_template(llm)
    if not args.raw and not template:
        print("No tokenizer.chat_template in GGUF metadata; using raw prompt", file=sys.stderr)
    token_ids = tokenize_prompt(llm, template, args.prompt)
    print(f"Prompt tokens ({len(token_ids)}): {token_ids}", file=sys.stderr)

    if args.tokenize:
        print_tokenized(llm, token_ids)
        return

    print(f"Loading token embedding matrix from {args.model} ...", file=sys.stderr)
    embed_table = load_token_embeddings(args.model)
    n_vocab_gguf, n_embd_gguf = embed_table.shape
    print(f"token_embd.weight: n_vocab={n_vocab_gguf} n_embd={n_embd_gguf}", file=sys.stderr)

    n_embd = llm.n_embd()
    if n_embd != n_embd_gguf:
        raise SystemExit(f"n_embd mismatch: llama.cpp reports {n_embd}, GGUF tensor has {n_embd_gguf}")

    content_range = None
    if not args.perturb_all_tokens:
        content_ids = llm.tokenize(args.prompt.encode("utf-8"), add_bos=False, special=False)
        content_range = find_subsequence(token_ids, content_ids)
        if content_range is None:
            print(
                f"couldn't find content tokens {content_ids} as a contiguous span in the rendered "
                f"prompt tokens {token_ids}; perturbing everything instead",
                file=sys.stderr,
            )
        else:
            print(f"Content span: token_ids[{content_range[0]}:{content_range[1]}]", file=sys.stderr)

            if args.target:
                target_ids = llm.tokenize(args.target.encode("utf-8"), add_bos=False, special=False)
                rel = find_subsequence(content_ids, target_ids)
                if rel is None:
                    print(
                        f"--target {args.target!r}: couldn't find its tokens {target_ids} as a "
                        f"contiguous span within the content tokens {content_ids}; perturbing the "
                        "whole prompt instead",
                        file=sys.stderr,
                    )
                else:
                    content_range = (content_range[0] + rel[0], content_range[0] + rel[1])
                    print(f"Target span: token_ids[{content_range[0]}:{content_range[1]}]", file=sys.stderr)

    if args.diff_mean:
        # Deliberately not .strip()-ing words: a leading space is how BPE tokenizers mark
        # a word-initial token (see --cosine's cat/tiger note), so " king, prince" must
        # split into [" king", " prince"] with the space preserved, not ["king", "prince"].
        words_a = [w for w in args.diff_mean[0].split(",") if w.strip()]
        words_b = [w for w in args.diff_mean[1].split(",") if w.strip()]
        direction = diff_mean_direction(llm, embed_table, words_a, words_b)
        runs = [(f"diff-mean({words_a} - {words_b}) x {args.value}", make_direction_fn(direction, args.value))]
    else:
        indices = parse_indices(args.index, n_embd_gguf)
        runs = [(f"index {idx} set to {args.value}", make_clamp_fn(idx, args.value)) for idx in indices]

    eos_id = llm.token_eos()

    baseline_out = None
    if args.baseline or args.compare:
        _kv_cache_clear(ctx)
        n_past = decode_tokens(ctx, token_ids, 0)
        baseline_out = generate(llm, ctx, n_vocab, n_past, args.n_predict, eos_id)
        if args.baseline:
            print("\n=== baseline (unmodified) ===")
            print(baseline_out)

    for label, perturb_fn in runs:
        _kv_cache_clear(ctx)
        n_past = decode_perturbed(ctx, embed_table, token_ids, n_embd, perturb_fn, content_range)
        out = generate(llm, ctx, n_vocab, n_past, args.n_predict, eos_id)
        print(f"\n=== {label} ===")
        print(out)

        if args.compare:
            explanation = explain_diff(
                llm, ctx, n_vocab, template,
                args.prompt + baseline_out, args.prompt + out, args.compare_predict, eos_id,
            )
            print(f"--- model's explanation of the difference ({label}) ---")
            print(explanation)


if __name__ == "__main__":
    main()
