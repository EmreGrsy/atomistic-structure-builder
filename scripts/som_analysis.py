"""Self-organizing-map analysis of the evaluation prompts.

Methodology follows GENIUS (arXiv:2512.06404, Fig. 5): prompts are embedded
with OpenAI text-embedding-3-large (3072-d, unit-normalized); a 10x10
hexagonally-packed SOM is trained for 50,000 iterations in mini-batches of 50
with a Gaussian neighborhood and linearly decreasing learning rate. Map
quality is reported as Quantization Error (mean input-to-BMU distance) and
Topological Error (fraction of inputs whose first and second BMUs are not
grid-adjacent). Outputs U-matrix, BMU hit map, and per-cell category/success
data to data/out/eval/som.json for the documentation page.

    python scripts/som_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID = 10
ITERS = 50_000
BATCH = 50
SEED = 42


def hex_coords(grid: int) -> np.ndarray:
    """2D positions of a pointy-top hexagonally packed grid (odd rows offset)."""
    pts = []
    for j in range(grid):
        for i in range(grid):
            pts.append((i + 0.5 * (j % 2), j * np.sqrt(3.0) / 2.0))
    return np.array(pts)


def embed_prompts(prompts: list[str]) -> np.ndarray:
    from mtagent.llm import get_openai_key
    from openai import OpenAI
    client = OpenAI(api_key=get_openai_key())
    out = client.embeddings.create(model="text-embedding-3-large", input=prompts)
    X = np.array([d.embedding for d in out.data])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def train_som(X: np.ndarray, grid: int = GRID, iters: int = ITERS,
              batch: int = BATCH, seed: int = SEED):
    rng = np.random.default_rng(seed)
    coords = hex_coords(grid)                              # (grid^2, 2)
    W = rng.normal(size=(grid * grid, X.shape[1]))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    grid_d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    sigma0, sigma1 = grid / 2.0, 0.5
    lr0, lr1 = 0.5, 0.01
    x2 = (X ** 2).sum(axis=1)
    for t in range(iters):
        frac = t / iters
        sigma = sigma0 + (sigma1 - sigma0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        sel = rng.integers(0, len(X), size=batch)
        Xb = X[sel]
        # batch SOM step: BMU per sample, Gaussian-weighted pull on all cells
        d2 = x2[sel][:, None] + (W ** 2).sum(axis=1)[None, :] - 2.0 * Xb @ W.T
        bmu = np.argmin(d2, axis=1)
        G = np.exp(-grid_d2[bmu] / (2.0 * sigma * sigma))  # (batch, cells)
        W += (lr / batch) * (G.T @ Xb - G.sum(axis=0)[:, None] * W)
    return W, coords


def som_metrics(X: np.ndarray, W: np.ndarray, coords: np.ndarray):
    d = np.linalg.norm(X[:, None, :] - W[None, :, :], axis=2)   # (n, cells)
    order = np.argsort(d, axis=1)
    bmu, second = order[:, 0], order[:, 1]
    qe = float(d[np.arange(len(X)), bmu].mean())
    grid_d = np.linalg.norm(coords[bmu] - coords[second], axis=1)
    te = float((grid_d > 1.1).mean())                           # not hex-adjacent
    return bmu, qe, te


def u_matrix(W: np.ndarray, coords: np.ndarray) -> np.ndarray:
    dc = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    adj = (dc > 1e-6) & (dc < 1.1)
    dw = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    with np.errstate(invalid="ignore"):
        u = np.where(adj, dw, np.nan)
    return np.nanmean(u, axis=1)


def main() -> None:
    results = [json.loads(ln) for ln in
               (ROOT / "data/out/eval/results.jsonl").read_text().splitlines()]
    prompts = [r["prompt"] for r in results]
    print(f"embedding {len(prompts)} prompts (text-embedding-3-large)…")
    X = embed_prompts(prompts)
    print(f"training {GRID}x{GRID} hex SOM: {ITERS} iterations, batch {BATCH}…")
    W, coords = train_som(X)
    bmu, qe, te = som_metrics(X, W, coords)
    um = u_matrix(W, coords)
    print(f"QE = {qe:.4f}   TE = {te:.4f}")

    cells = []
    for c in range(GRID * GRID):
        idx = np.where(bmu == c)[0]
        cells.append({
            "hits": int(len(idx)),
            "fails": int(sum(1 for i in idx if not results[i]["ok"])),
            "categories": sorted({results[i]["category"] for i in idx}),
        })
    out = {"grid": GRID, "qe": round(qe, 4), "te": round(te, 4),
           "coords": coords.round(4).tolist(),
           "umatrix": [round(float(v), 4) for v in um],
           "cells": cells,
           "method": "GENIUS-style (arXiv:2512.06404): text-embedding-3-large, "
                     f"{GRID}x{GRID} hex SOM, {ITERS} iters, batch {BATCH}, "
                     "Gaussian neighborhood, linear LR decay"}
    path = ROOT / "data/out/eval/som.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
