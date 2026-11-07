#!/usr/bin/env python3
"""Calculate integer evals in Python so that ruby.exe's `evaltest` command can compare against them.

  python make_eval_vectors.py path/to/ruby_small.bin

Reads the weights and Opening_Book.json only. Writes eval_vectors.txt beside this script, one
"<centipawns>;<FEN>" per line. Results follow the formulas documented at the top of train.py.
"""
import json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.dont_write_bytecode = True
import torch
import train


def read_weights(path):
    with open(path, "rb") as fh:
        magic, version, inputs, hidden, hidden2 = struct.unpack("<4sIIII", fh.read(20))
        assert magic == b"RUBY" and version == 1 and inputs == 768, "unexpected weights file"

        def take(dtype, *shape):
            data = np.frombuffer(fh.read(int(np.prod(shape)) * np.dtype(dtype).itemsize), dtype=dtype)
            return data.reshape(shape)

        q = dict(ft_w=take("<i2", inputs, hidden), ft_b=take("<i2", hidden),
                 l1_w=take("<i2", hidden2, 2 * hidden), l1_b=take("<i4", hidden2),
                 l2_w=take("<i2", hidden2, hidden2), l2_b=take("<i4", hidden2),
                 out_w=take("<i2", 1, hidden2), out_b=take("<i4", 1))
        assert not fh.read(1), "extra bytes at the end of the weights file"
    return q


def main():
    q = read_weights(sys.argv[1])
    with open(os.path.join(HERE, "..", "..", "Opening_Book.json")) as fh:
        fens = list(json.load(fh))
    step = max(len(fens) // 1000, 1)
    fens = fens[::step][:1000]
    fens += ["8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",        # endgames and lopsided positions
             "4k3/6KP/8/8/8/8/7p/8 w - - 0 1",
             "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
             "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10"]
    boards = torch.tensor([train.fen_to_board(f) for f in fens], dtype=torch.uint8)
    cps = train.quantised_cp(q, boards).numpy()
    out = os.path.join(HERE, "eval_vectors.txt")
    with open(out, "w") as fh:
        for cp, fen in zip(cps, fens):
            fh.write(f"{int(round(float(cp)))};{fen}\n")
    print(f"Wrote {len(fens)} positions to {out}")


if __name__ == "__main__":
    main()
