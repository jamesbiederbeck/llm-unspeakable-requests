#!/usr/bin/env python3
"""Decompose a diff-mean direction into components along other diff-mean axes,
and inspect which raw embedding dimensions dominate a vector.

Reuses embd_probe.py's model/embedding loading and diff_mean_direction so the
tokenization and pooling behavior stays identical to the perturbation tool.
"""
import argparse
import sys

import numpy as np

import embd_probe as ep


def sorted_indices_by_abs(vec: np.ndarray) -> np.ndarray:
    """Indices of vec sorted by descending absolute value, e.g. [1,0,5,-3] -> [2,3,1,0]."""
    return np.argsort(-np.abs(vec))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def project(vec, onto):
    onto_unit = onto / np.linalg.norm(onto)
    coeff = np.dot(vec, onto_unit)
    return coeff, coeff * onto_unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-a", nargs=2, metavar=("WORDS_A", "WORDS_B"), required=True,
                     help="the axis being decomposed, e.g. ' virus, worm, trojan' ' program, app, script'")
    ap.add_argument("--axis", action="append", nargs=3, metavar=("NAME", "WORDS_A", "WORDS_B"), default=[],
                     help="a candidate axis to project onto; repeatable")
    ap.add_argument("--top-n", type=int, default=15, help="how many top |value| dims to print")
    args = ap.parse_args()

    llm = ep.Llama(model_path=args.model, n_ctx=512, n_gpu_layers=0, verbose=False, embedding=False)
    embed_table = ep.load_token_embeddings(args.model)

    target_words_a = [w for w in args.target_a[0].split(",") if w.strip()]
    target_words_b = [w for w in args.target_a[1].split(",") if w.strip()]
    print(f"target axis: {target_words_a} - {target_words_b}", file=sys.stderr)
    target = ep.diff_mean_direction(llm, embed_table, target_words_a, target_words_b)

    idx = sorted_indices_by_abs(target)
    print(f"\ntop {args.top_n} dims of target axis by |value|:")
    for i in idx[:args.top_n]:
        print(f"  dim {i:<5} {target[i]:+.4f}")

    for name, wa, wb in args.axis:
        words_a = [w for w in wa.split(",") if w.strip()]
        words_b = [w for w in wb.split(",") if w.strip()]
        direction = ep.diff_mean_direction(llm, embed_table, words_a, words_b)
        cos = cosine(target, direction)
        coeff, _ = project(target, direction)
        print(f"\naxis '{name}' ({words_a} - {words_b}):")
        print(f"  cosine(target, axis) = {cos:+.4f}")
        print(f"  target's projection length onto axis = {coeff:+.4f}  (|axis|={np.linalg.norm(direction):.4f}, |target|={np.linalg.norm(target):.4f})")


if __name__ == "__main__":
    main()
