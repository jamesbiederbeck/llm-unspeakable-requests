# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A tool (`embd_probe.py`) for injecting perturbed token embeddings directly into a running llama.cpp GGUF model via the raw `llama_batch` ctypes API, to study what individual embedding dimensions and word-difference directions do to generation — used as a mechanism for "native instructions": dialing one aspect of a response via a soft vector instead of natural language, then asking the model to explain what a given setting does. See `README.md` for the goal/usage writeup and `embedding-cliffs.md` for findings.

No build system, no test suite, no lint config. This is a two-script research tool plus its own writeup.

## Commands

```
pip install llama-cpp-python gguf numpy blessed
python3 -m py_compile embd_probe.py axis_decompose.py   # syntax check
```

There is no automated test suite. The cheapest way to sanity-check a change against a real model without waiting on generation:

```
python3 embd_probe.py --model MODEL.gguf --cosine " cat" " dog"          # exercises GGUF loading + tokenization, no generation
python3 embd_probe.py --model MODEL.gguf --tokenize "some prompt"        # exercises chat templating + tokenization only
python3 embd_probe.py --model MODEL.gguf --top-dim 0 --top-n 3           # exercises the embedding-table dequant path
```

`llama-cpp-python` must be new enough to know the target model's architecture (built/tested against 0.3.35) or model loading fails with `unknown model architecture`. GGUF files go in `models/`, which is gitignored — not provided in the repo.

## Architecture

**The core trick (`embd_probe.py`):** llama.cpp's public API has no `inputs_embeds` equivalent. A `llama_batch` is all-token or all-embd, never mixed, but the KV cache persists across separate `llama_decode` calls — so a perturbed prompt is decoded in three pieces via `decode_perturbed()`: a normal token batch for everything before the target span (`decode_tokens`), a raw `embd` batch carrying perturbed vectors for the target span only (`decode_embeddings`), and a normal token batch for everything after. Three `llama_decode` calls, one continuous cache. By default the "target span" is the prompt's own content tokens (chat-template scaffolding decodes normally); `--target TEXT` narrows it further to one substring, `--perturb-all-tokens` widens it back to everything.

**Where the vectors come from:** `load_token_embeddings()` reads `token_embd.weight` straight out of the GGUF file via the `gguf` package and dequantizes it by hand (K-quants included) — not obtained through llama.cpp's runtime, which only exposes output-side embeddings. This is what makes individual token vectors editable before the forward pass.

**Two perturbation modes, unified through one interface:** both `main()` branches (axis-clamp vs `--diff-mean`) build a list of `(label, perturb_fn)` pairs — `perturb_fn(vecs) -> vecs` — via `make_clamp_fn`/`make_direction_fn`, then loop the same `decode_perturbed` + `generate` + optional `--compare` (`explain_diff`) call over every run. Adding a third perturbation mode means adding another `perturb_fn` constructor and populating `runs` accordingly, not touching the decode/generate loop.

**Prompting:** `tokenize_prompt()` renders through the model's own `tokenizer.chat_template` GGUF metadata (via `llama_chat_format.Jinja2ChatFormatter`, pulled from `llm._model.metadata()`) rather than tokenizing raw text, with `--raw` to opt out. Special/BOS token strings must be fetched with `detokenize(..., special=True)` — the default suppresses them, which silently breaks template rendering if forgotten.

**Known architecture incompatibility:** Gemma-4/Gemma-3n models use per-layer embeddings (an extra table injected at every transformer layer, not just the input) — `warn_if_per_layer_embeddings()` detects `per_layer_token_embd.weight` in the GGUF and warns that embd-injection results on that model are unreliable even at zero perturbation. This is a structural limitation of the technique, not a bug to fix casually.

**`axis_decompose.py`** is a companion, not a duplicate: it imports `embd_probe` (as `ep`) and reuses its model/embedding loading and `diff_mean_direction()` so tokenization/pooling behavior stays identical, then adds direction-inspection math (`sorted_indices_by_abs`, `cosine`, `project`) that `embd_probe.py` doesn't need for injection itself.

**`METHODOLOGY.md`** documents the actual workflow for using both tools together: how to find a candidate axis (`--top-dim` for raw dims, `axis_decompose.py --axis` to check a `--diff-mean` direction isn't leaking a second sense), then how to characterize it on a target word (confirm the token span first, coarse sweep, bisect the transition/"cliff", check monotonicity at extreme scale). Read it before adding new perturbation-axis experiments.
