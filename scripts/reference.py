#!/usr/bin/env python3
"""Generate independent NumPy FP32 reference logits from the pinned ATOME01 model.

Run: uv run --with numpy==2.4.6 python scripts/reference.py
"""
from pathlib import Path
import hashlib
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIM, HEAD, LAYERS, VOCAB, KERNEL, TOP_K = 256, 64, 8, 256, 5, 4
MODEL_SHA256 = "fdf8a6b69eacc5e4834e488759593198e482399887fce2c5b048a599844ae2f5"


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 7

    def floats(self, count):
        result = np.frombuffer(self.data, "<f4", count, self.pos).copy()
        self.pos += count * 4
        return result

    def matrix(self, rows, cols):
        scale = self.floats(1)[0]
        size = (rows * cols + 3) // 4
        packed = np.frombuffer(self.data, np.uint8, size, self.pos)
        self.pos += size
        codes = ((packed[:, None] >> np.arange(0, 8, 2)) & 3).reshape(-1)
        assert not np.any(codes == 2)
        signs = np.where(codes == 3, -1, codes).astype(np.float32)
        return (signs[: rows * cols] * scale).reshape(rows, cols)

    def norm(self):
        return self.floats(DIM), self.floats(DIM)


def norm(x, params):
    gamma, beta = params
    centered = x - x.mean(-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5) * gamma + beta


def softmax(x):
    exp = np.exp(x - x.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def load():
    data = (ROOT / "models/atome_944k.bin").read_bytes()
    assert hashlib.sha256(data).hexdigest() == MODEL_SHA256
    assert data[:7] == b"ATOME01"
    r = Reader(data)
    embed = r.matrix(VOCAB, DIM)
    blocks = []
    for _ in range(LAYERS):
        blocks.append((
            r.norm(), r.matrix(DIM, KERNEL),
            r.floats(DIM), r.floats(DIM), r.floats(DIM),
            r.matrix(HEAD, DIM), r.matrix(HEAD, DIM),
            r.matrix(DIM, DIM), r.matrix(3, DIM),
        ))
    final_norm, unembed = r.norm(), r.matrix(VOCAB, DIM)
    assert r.pos == len(data)
    return embed, blocks, final_norm, unembed


def forward(model, prompt):
    embed, blocks, final_norm, unembed = model
    x = embed[list(prompt)].copy()
    seq = len(prompt)
    for params, conv, a, b, c, wq, wk, wv, router in blocks:
        h = norm(x, params)
        routes = softmax(h @ router.T)
        local = np.zeros_like(h)
        for lag in range(min(KERNEL, seq)):
            local[lag:] += h[:seq - lag] * conv[:, lag]
        state = np.zeros_like(h)
        hidden = np.zeros(DIM, np.float32)
        for t in range(seq):
            hidden = np.tanh(a) * hidden + b * h[t]
            state[t] = c * hidden
        scores = (h @ wq.T) @ (h @ wk.T).T / np.float32(8)
        scores[np.triu_indices(seq, 1)] = -np.inf
        sparse = np.zeros_like(scores)
        for t in range(seq):
            k = min(TOP_K, t + 1)
            indices = np.argsort(-scores[t], kind="stable")[:k]
            sparse[t, indices] = softmax(scores[t, indices])
        attention = sparse @ (h @ wv.T)
        x = x + (routes[:, 0:1] * local + routes[:, 1:2] * state + routes[:, 2:3] * attention)
    return (norm(x[-1:], final_norm) @ unembed.T)[0]


def main():
    model = load()
    output = bytearray()
    prompts = [b"A", b"Once upon a", b"abcdefghijklmnop", bytes([0, 127, 128, 255])]
    generated = b"Once upon a"
    for _ in range(4):
        prompts.append(generated)
        generated += bytes([int(forward(model, generated).argmax())])
    for prompt in prompts:
        logits = forward(model, prompt).astype("<f4")
        output += struct.pack("<B", len(prompt)) + prompt + logits.tobytes()
        print(f"{prompt!r}: next={logits.argmax()} range=[{logits.min():.5f},{logits.max():.5f}]")
    destination = ROOT / "tests/fixtures/reference.bin"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output)
    print(f"Greedy continuation: {generated!r}; wrote {destination}")


if __name__ == "__main__":
    main()
