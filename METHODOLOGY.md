# Isolating a dimension/direction of interest and using it to transform a token

Two ways to find an axis worth perturbing, then one procedure to apply it.

## 1. Isolating a dimension of interest

### a) Raw axis (single embedding dimension)
Raw dims are rarely clean single-concept axes (superposition — each dim
usually encodes fragments of many unrelated concepts). Use `--top-dim` as a
cheap first-pass probe:

```
python3 embd_probe.py --model MODEL.gguf --top-dim 1075 --top-n 15
```

This prints the vocab tokens with the highest/lowest values along that raw
dimension. If the top/bottom tokens cluster around a human-interpretable
theme, the dim is a candidate; if they look unrelated, it's fighting
superposition and isn't a clean axis.

You can find *candidate* dims by first building a diff-mean direction (below)
and using `axis_decompose.py` to list which raw dims dominate it by
magnitude:

```
python3 axis_decompose.py --model MODEL.gguf \
  --target-a " WORDS_A" " WORDS_B" --top-n 10
```

This prints the target axis's dims sorted by `|value|` descending (via
`sorted_indices_by_abs`, i.e. `np.argsort(-np.abs(vec))`) — the dims that
move the most when you add this direction.

### b) Diff-mean direction (preferred — not fighting superposition)
Pick two small word groups that isolate the *one* semantic contrast you
want, and take `mean(embed(group_A)) - mean(embed(group_B))`. Keep every
other axis of meaning constant between the groups so the direction only
carries the contrast you care about:

```
--diff-mean " germ, bacteria, pathogen, contagion" " malware, trojan, ransomware, spyware"
```

**Isolation matters.** An axis built from surface-adjacent words (e.g.
`virus, worm, trojan` vs `program, app, script`) mixes senses — it's mostly
"malware" but has some "illness" leakage, which shows up as *non-monotonic*
behavior when you scale it (flips to biological virus at one scale, back to
malware at another, oscillates further out). Check this with
`axis_decompose.py --axis`, which reports cosine similarity between your
target axis and named candidate axes:

```
python3 axis_decompose.py --model MODEL.gguf \
  --target-a " virus, worm, trojan" " program, app, script" \
  --axis germ    " germ, bacteria, pathogen, contagion" " health, medicine, doctor, hospital" \
  --axis malware " malware, spyware, ransomware, hacking" " software, program, app, code"
```

High cosine with "malware", near-zero with "germ" → the axis is basically a
malware axis, not a germ/malware disambiguator. Building the axis directly
from unambiguous word groups (germ words vs malware words, neither
containing "virus" itself) gives a much cleaner, monotonic axis.

## 2. Using the direction to transform a token

Once you have a direction (raw dim or diff-mean), apply it to a specific
token in a real prompt with `--target` so only that word's embedding gets
perturbed, and sweep `value` (the scale) to find where the model's
interpretation flips:

```
PROMPT="what is a virus"
for scale in -3 -1.5 0 1.5 3; do
  python3 embd_probe.py --model MODEL.gguf --n-predict 40 --target " virus" \
    --diff-mean " germ, bacteria, pathogen, contagion" " malware, trojan, ransomware, spyware" \
    "$PROMPT" 0 $scale
done
```

Procedure for characterizing an axis on a target word:
1. **Confirm the target span first** — run with `--n-predict 5` and grep
   `Target span` to make sure the word tokenizes as expected (single vs
   multi-token; multi-token targets behave less cleanly since all their
   tokens get perturbed together).
2. **Coarse sweep** (`-3, -1.5, 0, 1.5, 3`) to see whether there's a flip at
   all, and which side it's on.
3. **Narrow the cliff** by bisecting between the last-stable and
   first-flipped scale (e.g. found `-0.75` still biological, `-1.0` already
   malware for the germ/malware axis on "what is a virus" — a sharp,
   sub-1.0-wide cliff, unlike axes with a wide garbled transition zone).
4. **Check monotonicity** by pushing further past the flip (e.g. `-4, -6,
   -8`). A clean axis stays on the flipped side; a mixed/leaky axis can
   oscillate back (seen with the virus/worm/trojan vs program/app/script
   axis, which returned to the biological sense at `-8` after being
   malware at `-3`/`-4`/`-6`).

A clean, well-isolated axis: sharp single cliff, stays flipped past it, no
garbled/incoherent zone at the transition. A mixed axis: wide garbled
zone at the transition, and/or non-monotonic reversion at extreme scale —
both are signs the two word groups aren't isolating one contrast.
