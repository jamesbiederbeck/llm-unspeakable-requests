# Embedding Cliffs

A tool for perturbing single tokens inside a running llama.cpp model and asking the model itself what changed — plus what it turned up: raw dimensions are entangled, but word-difference directions produce sharp, legible, and surprisingly non-linear effects.

**Tool:** `embd_probe.py`
**Backend:** llama-cpp-python 0.3.35, raw ctypes `llama_batch`
**Models:** Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct

---

## 0. The goal

Not word-analogy arithmetic. The aim is **native instructions**: a soft, continuously-dialable vector that steers one aspect of a response — think "how hacky should this fix be" — without spelling it out in words, plus a way to ask the model what a given setting of that dial actually means.

Every experiment below follows the same loop: pick a direction in embedding space, inject it into exactly one word of a real prompt, generate, then hand both the unperturbed and perturbed outputs back to the model and ask it to explain the difference. The model's own explanation is the readout instrument.

## 1. How the tool works

llama.cpp has no `inputs_embeds` equivalent — you drive it through `llama_batch` directly.

**Getting the vectors.** Token embeddings are read straight out of the GGUF file with the `gguf` package and dequantized by hand (K-quants included) — not asked of llama.cpp's runtime, which only exposes output-side embeddings. This is what makes individual token vectors editable before they ever enter the forward pass.

**Injecting the perturbation.** A `llama_batch` is all-token or all-embd, never mixed — but the KV cache persists across separate `llama_decode` calls. So the prompt is split three ways: a normal token batch for everything before the target word, a raw `embd` batch carrying the perturbed vector(s) for the target word only, and a normal token batch for everything after. Three decode calls, one continuous cache.

> **Default changed mid-project.** Perturbation now targets only the prompt's own content tokens by default — originally every rendered token was perturbed, including chat-template scaffolding (`<|start_header_id|>`, system-preamble date stamps, etc.), which added noise no one asked for. A further `--target TEXT` flag narrows this to a single word or phrase inside the prompt.

**Prompting.** Prompts are rendered through the model's own `tokenizer.chat_template` pulled from GGUF metadata (via `llama_chat_format.Jinja2ChatFormatter`) rather than fed raw — instruct models expect their native turn format, and skipping it measurably degraded output quality.

## 2. Modes

| flag | what it does |
|---|---|
| `INDEX VALUE` | Clamp one or more raw embedding dimensions to a fixed value across the target span. `INDEX` accepts lists and Python-style slices: `"5,10,50:400:50"`. |
| `--diff-mean A B` | Compute mean(embed(A)) − mean(embed(B)) over comma-separated word lists and *add* it, scaled by `VALUE`, instead of clamping an axis. The workhorse of every finding below. |
| `--target TEXT` | Narrow perturbation to one substring of the prompt instead of the whole content span. |
| `--compare` | After generating, ask the model — unperturbed — to explain the difference between the baseline and perturbed continuations. |
| `--cosine A B` | Per-token-pair and pooled cosine similarity between two words' embeddings. |
| `--top-dim N` | For a dimension, list the vocabulary's highest- and lowest-scoring tokens on that axis. |
| `--tokenize` | Print a prompt with each token in a different terminal color, via `blessed`. |

## 3. What broke along the way

Three real bugs, not model behavior — worth recording since they'd bite anyone doing this from scratch.

- **NULL logits.** With `logits_all=False`, `llama_get_logits_ith` intermittently returned a null pointer when driving `llama_decode` manually — non-deterministic, same code, different runs. Fixed by constructing with `logits_all=True` and flagging every position's logits, reading back with `ith=-1`.
- **API rename.** `llama_kv_cache_clear(ctx)` became `llama_memory_clear(llama_get_memory(ctx), data)` in llama.cpp's memory-API refactor. The tool now detects and supports both.
- **Per-layer embeddings break embd-injection entirely.** Gemma-4 / Gemma-3n architectures inject a second embedding table at *every* transformer layer, not just the input. Feeding only `token_embd.weight` through `llama_batch.embd` produces garbled output even with zero perturbation — confirmed by injecting the true, unmodified embedding and still getting broken text. The tool now detects `per_layer_token_embd.weight` in the GGUF and warns rather than silently producing misleading results.

---

## 4. Raw dimensions don't mean anything on their own

Before touching word directions: does any single one of a 2048-dim Llama-3.2-1B embedding correspond to a legible concept?

`--top-dim` across five sampled dimensions (0, 100, 500, 1000, 2000) turned up no clean single-concept clusters — mostly unrelated subword fragments and multilingual tokens, magnitudes all clustered tightly around ±0.07–0.09 with no dimension standing out as unusually high-variance. Two mild, inconclusive leans: dim 500's lowest tokens skewed social/relational (*friendships, parental, masculinity*); dim 1000's lowest skewed toward non-Latin scripts.

Consistent with superposition: individual axes in a dense embedding are entangled combinations of many features, not dedicated detectors. Word-*difference* directions turned out to be the productive unit, not raw axes.

## 5. Cosine similarity sanity check

Before trusting any direction math, confirm the geometry behaves the way it should.

| pair | cosine |
|---|---|
| cat ↔ dog | 0.3402 |
| cat ↔ car | 0.1985 |
| cat ↔ tiger | 0.1453 |

Sensible ordering — closest co-occurring pair (cat/dog) scores highest. One gotcha worth flagging: BPE tokenizes `"tiger"` and `" tiger"` (leading space) as entirely different tokens; only the space-prefixed, word-initial form gives a clean single-token comparison.

## 6. "Fix": dialing hackiness with one word

Direction: mean(*hack, kludge, quick, patch, hacky*) − mean(*clean, proper, robust, elegant, correct*), injected into a single occurrence of the word **Fix**.

First attempt perturbed the whole prompt — a five-line buggy-code description — and blew straight through coherence at any scale strong enough to see an effect: ±5 collapsed into repetition loops or nonsense before any stylistic signal emerged. Narrowing injection to the single word `Fix` (via `--target`) fixed that; the same direction became legible instead of destructive.

| scale | model's answer to "what does Fix mean?" |
|---|---|
| −9 | "The word **authentic** means genuine, sincere, and true..." |
| −6 | "The word **proper** means conforming to..." |
| −3 | "The word **correct** or **proper** means aligning with standards..." |
| 0 | "The word **fix**... refers to making something right, repairing..." |
| 3 | "...fix... often in a **temporary or makeshift manner**" |
| 8 | "...fix... often in a **casual or informal sense**" |
| 10 | model hedges, evades — breaking down |

An asymmetric effect either side of zero: negative scale (toward "clean") flips *which word the model thinks it read* — it names an actual group-B word outright. Positive scale (toward "hack") keeps the word "fix" intact but shifts its *connotation* toward makeshift/informal.

> **Scale 3, asked to explain the difference:**
> "The word 'fix' can have two distinct meanings, with the first referring to making something right or operational again, and the second implying **a temporary or makeshift solution** to a problem."
>
> The model articulates the hack-vs-proper distinction unprompted — the word "hack" never appears in its answer.

### Bigger model, sharper signal

Same experiment, same scale, on Llama-3.1-8B-Instruct: the model names **"hack" itself**, unprompted —

> "The word **"hack"** or "hack into"... is often used to describe **a quick fix or a makeshift solution**"
>
> **Compare step:** "...the first explanation implying a **permanent and thorough solution**, and the second explanation suggesting a **temporary or makeshift solution**."

Where the 1B model shifted connotation without naming the axis, the 8B model surfaces the exact axis word directly, plus a near-textbook compare explanation. Same technique, same scale — model capacity visibly sharpens the readout.

## 7. The dial isn't a dial — it's a cliff

Direction: mean(*lion, bear, tiger*) − mean(*bunny, gazelle, deer*), on the single word **Fish** (8B model).

| scale | result |
|---|---|
| −3 | "deers" / "deer" — confused, doesn't fully resolve |
| −1.5 … 1.5 | clean, stable "fish" — near-identical wording throughout |
| 2.0 | still "fish," but redefined as **carnivorous, sharp-toothed** — connotation drifting toward predator before identity flips |
| 2.25 – 2.5 | breaks down — repetitive, incoherent, neither word |
| 2.75 | clean lock onto **"Tiger"** |
| 3 | clean "Tiger," fully formed definition |

Sketch of the shape (not measured data): coherence stays high and roughly flat across most of the tested range, drops sharply in a narrow transitional band (~2.25–2.5), then recovers to a *different* stable identity. Two basins, one crossing — not a gradual slope.

```
 coherent  "deer"-ish            "fish"                      "tiger"
    ▲         ●───────●───●            ●──●──●         ●──────────●
    │                       ╲         ╱        ╲       ╱
    │                        ╲       ╱          ╲     ╱
    │                         ╲     ╱     ╱garbled╲   ╱
    └────────────────────────────────────zone──────────────────────▶ scale
         −3          −1.5       0    2.0  2.25  2.5   2.75    3
```

The same axis on a fully unrelated single-token word, **Boat**, showed an even sharper version of this: identical, stable "boat" definitions through scale 1.3, then an instant jump straight to "boy" vs "girl" discussion at 1.4 — no transitional garbling at all, just a step function.

> **Not always symmetric, not always resolving.** A big/small axis — mean(*tower, whale*) − mean(*mouse, flea, atom*) — on "Fish" resolved cleanly toward the small group by scale −4/−5 ("...you meant 'flea' or 'mouse' as in a small rodent"), but the positive (big) direction **never resolved** even out to scale 5 — stuck permanently in a garbled near-miss basin ("wander," "Waver," "wahler," all phonetically whale-adjacent but never landing). Likely because *tower* and *whale* are semantically distant from each other, so their mean sits between concepts rather than at one.

## 8. It overrides identity, not connotation

Applying the woman/lady/girl ↔ slur direction to a target word actually related to the axis (*Woman* itself) produced what looked like nuanced connotation shift — literal-biological framing on one side, social-identity framing on the other. Applying the same direction to a fully unrelated word (**Boat**) exposed what's really happening.

> +3 scale, target word "Boat": **"The word 'girl' or 'woman' is often used interchangeably..."** — no trace of "boat" survives.
>
> **Compare step, unprompted:** "...the second one appears to be a fragment and seems to be discussing the word 'girl' or 'woman', not 'Boat'."

The mechanism is **token-identity override**, not connotation blending. What looked like graceful semantic modulation on well-chosen target words was really the override landing on a plausible neighbor because the original word was already close to the injected direction. An unrelated word unmasks it immediately.

## 9. Open questions

- Does injecting at an intermediate transformer layer (residual-stream steering, rather than input-only) produce genuine connotation blending instead of identity override?
- Is cliff width predictable from word-group tightness — does a semantically coherent 5-word cluster always produce a narrower transition band than a 2-word, semantically distant pair like *tower, whale*?
- Whole-sentence perturbation (as opposed to single-word `--target`) is still on the table for the actual "dial one aspect of a whole response" goal — single-word targeting was needed to keep output legible, but the original ambition was sentence- or paragraph-scale steering.

---

*embd_probe.py — local GGUF · raw llama_batch · no training*
