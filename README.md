# embd_probe

Probe what token embedding dimensions and word-difference directions actually do inside a running GGUF model — by injecting perturbed vectors directly via llama.cpp's raw `llama_batch` API and asking the model itself to explain the effect.

Not word-analogy arithmetic. The goal is **native instructions**: a soft, continuously-dialable vector that steers one aspect of a response (e.g. "how hacky should this fix be") without spelling it out in words, plus a way to ask the model what a given setting of that dial means. See [`embedding-cliffs.md`](embedding-cliffs.md) for the write-up of what that turned up — short version: raw embedding dimensions are entangled and don't mean much individually, but word-difference directions produce sharp, legible, and surprisingly non-linear effects (a coherence *cliff*, not a smooth dial). [`embedding_cliffs.ipynb`](embedding_cliffs.ipynb) is a runnable version of the same experiments, one cell per use case. [`METHODOLOGY.md`](METHODOLOGY.md) covers the actual workflow for finding and characterizing a perturbation axis — start there before running new experiments.

## How it works

llama.cpp has no `inputs_embeds` equivalent — you drive it through `llama_batch` directly. Token embeddings are read straight out of the GGUF file (via the `gguf` package, dequantizing K-quants by hand) rather than asked of llama.cpp's runtime, which only exposes output-side embeddings. That's what makes individual token vectors editable before they ever enter the forward pass.

A `llama_batch` is all-token or all-embd, never mixed — but the KV cache persists across separate `llama_decode` calls. So a perturbed prompt is decoded in three pieces: a normal token batch for everything before the target word, a raw `embd` batch carrying the perturbed vector(s) for the target word only, and a normal token batch for everything after.

Prompts are rendered through the model's own `tokenizer.chat_template` pulled from GGUF metadata, not fed raw.

## Setup

```
pip install llama-cpp-python gguf numpy blessed
```

`llama-cpp-python` needs to be new enough to know about your model's architecture — this was built and tested against 0.3.35. If a model fails to load with an `unknown model architecture` error, upgrade.

**Known limitation:** Gemma-4 / Gemma-3n models use per-layer embeddings (an extra embedding table injected at every transformer layer, not just the input). This tool only injects at the input, so output is garbled on those architectures even with zero perturbation — `embd_probe.py` detects `per_layer_token_embd.weight` in the GGUF and warns automatically. Plain architectures (Llama, etc.) work fine.

## Usage

```
python embd_probe.py --model ./model.gguf "prompt text" INDEX VALUE
```

By default only the prompt's own content tokens are perturbed (chat-template scaffolding like `<|start_header_id|>` decodes normally). Key flags:

| flag | what it does |
|---|---|
| `INDEX VALUE` | Clamp one or more raw embedding dimensions to a fixed value. `INDEX` accepts lists and slices: `"5,10,50:400:50"`. |
| `--diff-mean A B` | Perturb via a difference-of-means direction instead of a raw axis: mean(embed(A)) − mean(embed(B)), comma-separated word lists, scaled by `VALUE`. Usually more interpretable than a raw index. |
| `--target TEXT` | Narrow perturbation to one substring of the prompt (e.g. a single word) instead of the whole prompt. |
| `--compare` | After generating, ask the model — unperturbed — to explain the difference between the baseline and perturbed continuations. |
| `--baseline` | Also print the unmodified generation for comparison. |
| `--cosine A B` | Per-token-pair and pooled cosine similarity between two words' embeddings. |
| `--top-dim N` | For a dimension, list the vocabulary's highest- and lowest-scoring tokens on that axis. |
| `--tokenize` | Print a prompt with each token colored differently in the terminal. |
| `--raw` | Skip chat templating; tokenize the prompt as plain completion text. |
| `--perturb-all-tokens` | Perturb every rendered token, including chat-template scaffolding, instead of just the prompt's own content. |

Example — dial the "hackiness" of how a model defines the word "Fix":

```
python embd_probe.py --model ./llama-3.1-8b-instruct.gguf \
  --target " Fix" --compare \
  --diff-mean " hack, kludge, quick, patch, hacky" " clean, proper, robust, elegant, correct" \
  "What does the word Fix mean to you? Explain in one sentence." 0 3
```

(`0` is an ignored positional placeholder — `--diff-mean` replaces the axis-clamp `INDEX`, but the CLI still expects three positional args.)

## axis_decompose.py

A companion tool for inspecting a `--diff-mean` direction rather than injecting it: which raw dimensions dominate it, and how it relates to other diff-mean axes (cosine similarity, projection length).

```
python axis_decompose.py --model ./model.gguf \
  --target-a " hack, kludge, quick" " clean, proper, robust" \
  --axis size " big, huge, giant" " tiny, small, little" \
  --axis animal " lion, tiger, wolf" " deer, rabbit, mouse"
```

## Files

- `embd_probe.py` — the main tool.
- `axis_decompose.py` — inspect/decompose a diff-mean direction.
- [`embedding-cliffs.md`](embedding-cliffs.md) — write-up of findings.
- [`embedding_cliffs.ipynb`](embedding_cliffs.ipynb) — the same experiments as a runnable notebook, one cell per use case.
- [`METHODOLOGY.md`](METHODOLOGY.md) — workflow for isolating and characterizing a perturbation axis.
- [`CLAUDE.md`](CLAUDE.md) — repo guidance for Claude Code.
- `models/` — not tracked in git; put your own GGUF files here.
