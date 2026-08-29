#!/usr/bin/env python3
"""Render raw embedding/direction vectors as a red(-)/grey(0)/green(+) heatmap,
one pixel per dimension in native index order -- no PCA, no dimensionality
reduction, no reordering. Each row is one vector; stack multiple rows to
compare directions/tokens side by side at the same color scale.
"""
import argparse

import numpy as np
from PIL import Image

import embd_probe as ep
from axis_decompose import sorted_indices_by_abs


def color_map(vec: np.ndarray, vmax: float) -> np.ndarray:
    """value -> RGB: 0 is grey(128,128,128), -vmax is pure red, +vmax is pure green.
    Values are clamped to [-vmax, vmax] before mapping (no rescaling of outliers)."""
    t = np.clip(vec / vmax, -1.0, 1.0)
    grey = np.array([128, 128, 128], dtype=np.float64)
    red = np.array([255, 40, 40], dtype=np.float64)
    green = np.array([40, 220, 40], dtype=np.float64)
    out = np.empty((len(vec), 3), dtype=np.uint8)
    neg = t < 0
    pos = ~neg
    out[neg] = (grey + (grey - red) * t[neg, None]).astype(np.uint8)   # t negative -> toward red
    out[pos] = (grey + (green - grey) * t[pos, None]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--word", action="append", default=[], metavar="WORD",
                     help="raw pooled token embedding for WORD; repeatable, one row each")
    ap.add_argument("--diff-mean", action="append", nargs=3, metavar=("LABEL", "WORDS_A", "WORDS_B"), default=[],
                     help="a diff-mean direction as a row; repeatable")
    ap.add_argument("--row-height", type=int, default=24, help="pixel height per row")
    ap.add_argument("--vmax", type=float, default=None,
                     help="fixed color scale (value at full red/green saturation); "
                          "default: max |value| across all rows, so rows share one scale")
    ap.add_argument("--diff", action="append", nargs=3, metavar=("LABEL_A", "LABEL_B", "NEW_LABEL"), default=[],
                     help="append a row = row[LABEL_A] - row[LABEL_B] (labels must match earlier "
                          "--word/--diff-mean labels); repeatable")
    ap.add_argument("--sort-by", metavar="LABEL",
                     help="reorder all rows' columns (dims) by descending |value| of row LABEL. "
                          "This permutes column order identically across every row -- it does not "
                          "touch any value, average dims together, or rotate the basis (unlike PCA); "
                          "it just clusters the reference row's largest-magnitude dims on the left "
                          "so correlated/uncorrelated structure across rows is easier to see by eye.")
    ap.add_argument("--out", default="heatmap.png")
    args = ap.parse_args()

    if not args.word and not args.diff_mean:
        ap.error("need at least one --word or --diff-mean")

    llm = ep.Llama(model_path=args.model, n_ctx=512, n_gpu_layers=0, verbose=False, embedding=False)
    embed_table = ep.load_token_embeddings(args.model)
    n_embd = embed_table.shape[1]

    rows = []  # (label, vec)
    for w in args.word:
        rows.append((w, ep.pooled_embedding(llm, embed_table, w)))
    for label, wa, wb in args.diff_mean:
        words_a = [x for x in wa.split(",") if x.strip()]
        words_b = [x for x in wb.split(",") if x.strip()]
        rows.append((label, ep.diff_mean_direction(llm, embed_table, words_a, words_b)))

    by_label = {label: vec for label, vec in rows}
    for label_a, label_b, new_label in args.diff:
        if label_a not in by_label or label_b not in by_label:
            ap.error(f"--diff labels must match an existing row; have {list(by_label)}")
        diff_vec = by_label[label_a] - by_label[label_b]
        rows.append((new_label, diff_vec))
        by_label[new_label] = diff_vec

    col_order = None
    if args.sort_by:
        if args.sort_by not in by_label:
            ap.error(f"--sort-by label not found; have {list(by_label)}")
        col_order = sorted_indices_by_abs(by_label[args.sort_by])
        rows = [(label, vec[col_order]) for label, vec in rows]
        print(f"columns reordered by |value| of row {args.sort_by!r} (permutation only -- no value changed)")

    vmax = args.vmax or max(float(np.max(np.abs(v))) for _, v in rows)
    print(f"width={n_embd}px (1px/dim, {'sorted by |' + args.sort_by + '|' if args.sort_by else 'native index order'}), vmax={vmax:.4f}")

    img = Image.new("RGB", (n_embd, args.row_height * len(rows)))
    pixels = img.load()
    for row_i, (label, vec) in enumerate(rows):
        colors = color_map(vec, vmax)
        y0 = row_i * args.row_height
        for x in range(n_embd):
            r, g, b = colors[x]
            for dy in range(args.row_height):
                pixels[x, y0 + dy] = (int(r), int(g), int(b))
        print(f"row {row_i}: {label!r}  |v|max={float(np.max(np.abs(vec))):.4f}")

    img.save(args.out)
    print(f"wrote {args.out} ({n_embd}x{args.row_height * len(rows)})")


if __name__ == "__main__":
    main()
