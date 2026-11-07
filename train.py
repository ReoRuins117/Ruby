#!/usr/bin/env python3
"""
train.py - train Ruby's small NNUE evaluation network from parsed/dataset.

  python train.py                  # train (30 passes) and export weights for the C engine
  python train.py --resume         # continue from the last finished pass in --out
  python train.py --export-only    # re-export the best checkpoint in --out

Needs numpy and torch. Stop any time with Ctrl+C; finished passes are kept.

Network (mover-relative input, two perspectives sharing one feature layer):

  768 inputs -> 256 (mover)    \
                                 concat 512 -> 32 -> 32 -> 1
  768 inputs -> 256 (opponent) /

Input feature = square * 12 + piece_code - 1, where square is 0..63 (0 = a8 from that side's view)
and piece_code is 1-6 for that side's P N B R Q K and 7-12 for the other side's. The output is in
units of 400 centipawns. Every activation is a clipped ReLU, which caps values to 0..1 so that the
trained network converts cleanly to integers for the C engine.

Export file (little-endian):
  header   char[4] "RUBY", u32 version, u32 inputs (768), u32 hidden (256), u32 hidden2 (32)
  ft_w     i16[768][hidden]         scale 255
  ft_b     i16[hidden]              scale 255
  l1_w     i16[hidden2][2*hidden]   scale 256
  l1_b     i32[hidden2]             scale 255*256, plus 128 so that >> 8 rounds to nearest
  l2_w     i16[hidden2][hidden2]    scale 256
  l2_b     i32[hidden2]             scale 255*256, plus 128 so that >> 8 rounds to nearest
  out_w    i16[hidden2]             scale 1024
  out_b    i32                      scale 255*1024

Integer inference in C:
  acc[i]  = ft_b[i] + sum of ft_w[f][i] over the active features f, for each perspective
  a       = clamp(acc, 0, 255), mover's values first, then the opponent's
  h1      = clamp((l1_w * a  + l1_b) >> 8, 0, 255)
  h2      = clamp((l2_w * h1 + l2_b) >> 8, 0, 255)
  cp      = (out_w * h2 + out_b) * 400 / (1024 * 255)
Every value above fits in 16 bits and each sum in 32 bits, so AVX2's madd instruction can compute them.
"""
import argparse, csv, glob, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "parsed", "dataset")
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "RubyTraining", "small_nnue")

CP_SCALE = 400.0              # 400 centipawns = 1.0 of network output
NUM_FEATURES = 768
EMPTY = NUM_FEATURES          # padding index: empty squares add nothing
HIDDEN2 = 32
Q_FT = 255                    # integer scale of clipped activations and the feature layer
Q_W = 256                     # integer scale of the hidden layers' weights
SHIFT = 8                     # dividing by Q_W in the C engine: x >> SHIFT
Q_OUT = 1024                  # integer scale of the output weights, finer because they set the final score
MAX_WEIGHT = 127 / 64         # weight limit that keeps the C engine's 32-bit sums from overflowing

SWAP_COLOURS = torch.tensor([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
PIECE_VALUES = torch.tensor([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=torch.float32)
FLIP_RANKS = torch.arange(64) ^ 56   # square as seen by the other side


# ----------------------------------------------------------------------------- model
def feature_ids(boards):
    """boards: uint8 (n, 64) from one side's view -> feature index per square (n, 64)."""
    squares = torch.arange(64, device=boards.device)
    codes = boards.long()
    return torch.where(codes > 0, squares * 12 + codes - 1, torch.full_like(codes, EMPTY))


def perspectives(boards):
    """Return feature ids from the mover's view and from the opponent's view."""
    other = SWAP_COLOURS.to(boards.device)[boards[:, FLIP_RANKS.to(boards.device)].long()]
    return feature_ids(boards), feature_ids(other)


class RubyNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, hidden, mode="sum", padding_idx=EMPTY)
        self.ft_bias = nn.Parameter(torch.full((hidden,), 0.5))
        self.l1 = nn.Linear(2 * hidden, HIDDEN2)
        self.l2 = nn.Linear(HIDDEN2, HIDDEN2)
        self.out = nn.Linear(HIDDEN2, 1)
        with torch.no_grad():
            nn.init.normal_(self.ft.weight, std=0.05)   # small: about 22 of these are summed per side
            self.ft.weight[EMPTY].zero_()
            nn.init.constant_(self.l1.bias, 0.5)        # start inside the clipped ReLU's active range
            nn.init.constant_(self.l2.bias, 0.5)

    def forward(self, boards):
        us, them = perspectives(boards)
        a = torch.cat([self.ft(us), self.ft(them)], 1) + self.ft_bias.repeat(2)
        a = a.clamp(0, 1)
        h = self.l1(a).clamp(0, 1)
        h = self.l2(h).clamp(0, 1)
        return self.out(h).squeeze(1)

    def limit_weights(self):
        """Keep weights small enough for the C engine's integer arithmetic."""
        with torch.no_grad():
            for layer in (self.l1, self.l2, self.out):
                layer.weight.clamp_(-MAX_WEIGHT, MAX_WEIGHT)


def win_prob_loss(pred, target):
    """Squared error in win-probability space: errors at +900 cp matter less than errors near 0."""
    return ((torch.sigmoid(pred) - torch.sigmoid(target)) ** 2).mean()


# ----------------------------------------------------------------------------- data
def load_split(data_dir, split, keep_tactical, limit=None):
    files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npz")))
    if not files:
        sys.exit(f"ERROR: no {split}_*.npz files in {data_dir}. Pass the right folder with --data.")
    boards, cps, seen = [], [], 0
    for f in files:
        with np.load(f) as d:
            b, c = d["boards"], d["cp"]
            seen += len(c)
            if not keep_tactical:
                if "flags" not in d.files:
                    sys.exit(f"ERROR: {f} has no flags column. Use --keep-tactical for this dataset.")
                quiet = (d["flags"] & 1) == 0   # bit 0: next game move was a capture or promotion
                b, c = b[quiet], c[quiet]
        boards.append(b)
        cps.append(c)
        print(f"  loaded {os.path.basename(f)}", flush=True)
        if limit and sum(map(len, cps)) >= limit:
            break
    boards, cps = np.concatenate(boards)[:limit], np.concatenate(cps)[:limit]
    print(f"{split}: {len(cps):,} positions kept from {seen:,} read", flush=True)
    return torch.from_numpy(boards), torch.from_numpy(cps.astype(np.float32))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_cp(model, boards, device, batch=65536):
    model.eval()
    return torch.cat([model(b.to(device)).cpu() for b in boards.split(batch)]) * CP_SCALE


def report(pred, cp):
    decided = cp.abs() > 50
    return dict(loss=win_prob_loss(pred / CP_SCALE, cp / CP_SCALE).item(),
                median_cp_error=(pred - cp).abs().median().item(),
                right_side=((pred > 0) == (cp > 0))[decided].float().mean().item())


def show_report(name, r):
    print(f"{name}: loss {r['loss']:.5f} | typical error {r['median_cp_error']:.0f} cp | "
          f"favours the right side {100 * r['right_side']:.1f}%", flush=True)


def fen_to_board(fen):
    """FEN -> mover-relative uint8 board, the same layout as the dataset."""
    placement, turn = fen.split()[:2]
    black = turn == "b"
    out = [0] * 64
    for row, text in enumerate(placement.split("/")):
        col = 0
        for ch in text:
            if ch.isdigit():
                col += int(ch)
                continue
            kind = "pnbrqk".index(ch.lower()) + 1
            mine = ch.isupper() != black
            out[(7 - row if black else row) * 8 + col] = kind if mine else kind + 6
            col += 1
    return out


def sanity_check(model, device):
    tests = [("start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", "near 0"),
             ("mover up a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", "strongly +"),
             ("mover down a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b", "strongly -"),
             ("mover up a rook", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", "clearly +")]
    boards = torch.tensor([fen_to_board(f) for _, f, _ in tests], dtype=torch.uint8)
    print("Sanity positions:")
    for (name, _, expected), value in zip(tests, predict_cp(model, boards, device).tolist()):
        print(f"  {name:20s} {value:+7.0f} cp   (expect {expected})")


# ----------------------------------------------------------------------------- C export
def quantise(model):
    def int_array(tensor, scale, dtype):
        info = np.iinfo(dtype)
        return np.rint(tensor.detach().cpu().numpy() * scale).clip(info.min, info.max).astype(dtype)

    bias_scale, half = Q_FT * Q_W, Q_W // 2      # adding half turns the later shift into round-to-nearest
    return dict(ft_w=int_array(model.ft.weight[:NUM_FEATURES], Q_FT, np.int16),
                ft_b=int_array(model.ft_bias, Q_FT, np.int16),
                l1_w=int_array(model.l1.weight, Q_W, np.int16),
                l1_b=int_array(model.l1.bias, bias_scale, np.int32) + half,
                l2_w=int_array(model.l2.weight, Q_W, np.int16),
                l2_b=int_array(model.l2.bias, bias_scale, np.int32) + half,
                out_w=int_array(model.out.weight, Q_OUT, np.int16),
                out_b=int_array(model.out.bias, Q_FT * Q_OUT, np.int32))


def write_weights(q, hidden, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<4sIIII", b"RUBY", 1, NUM_FEATURES, hidden, HIDDEN2))
        for name in ("ft_w", "ft_b", "l1_w", "l1_b", "l2_w", "l2_b", "out_w", "out_b"):
            fh.write(q[name].astype(q[name].dtype.newbyteorder("<")).tobytes())
    os.replace(tmp, path)


def quantised_cp(q, boards, batch=2048):
    """Run the integer network exactly as the C engine will, to prove the export is faithful."""
    ft = np.vstack([q["ft_w"], np.zeros((1, q["ft_w"].shape[1]), np.int16)]).astype(np.int32)
    w1, w2, w3 = q["l1_w"].astype(np.int32), q["l2_w"].astype(np.int32), q["out_w"].astype(np.int32)
    results = []
    for chunk in boards.split(batch):
        us, them = (ids.numpy() for ids in perspectives(chunk))
        a = np.hstack([np.clip(ft[ids].sum(1) + q["ft_b"], 0, Q_FT) for ids in (us, them)])
        h1 = np.clip((a @ w1.T + q["l1_b"]) >> SHIFT, 0, Q_FT)
        h2 = np.clip((h1 @ w2.T + q["l2_b"]) >> SHIFT, 0, Q_FT)
        results.append((h2 @ w3.T + q["out_b"]).squeeze(1) * CP_SCALE / (Q_OUT * Q_FT))
    return torch.from_numpy(np.concatenate(results).astype(np.float32))


def export(model, val_boards, device, path):
    q = quantise(model)
    write_weights(q, model.hidden, path)
    sample = val_boards[:20000]
    gap = (predict_cp(model, sample, device) - quantised_cp(q, sample)).abs()
    print(f"Exported {path}\nInteger vs float check on {len(sample):,} positions: "
          f"average gap {gap.mean():.2f} cp, largest {gap.max():.1f} cp", flush=True)


# ----------------------------------------------------------------------------- training
def save_checkpoint(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def learning_rate(step, total, base, warmup=500, floor=1e-5):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def train(a, device):
    print("Loading data...", flush=True)
    train_b, train_cp = load_split(a.data, "train", a.keep_tactical, a.limit)
    val_b, val_cp = load_split(a.data, "val", a.keep_tactical, a.limit)
    material = PIECE_VALUES[val_b.long()].sum(1) * 100
    show_report("Piece-count baseline", report(material, val_cp))

    os.makedirs(a.out, exist_ok=True)
    last_path, best_path = os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt")
    model = RubyNet(a.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    first_epoch, step, best = 0, 0, float("inf")
    if a.resume:
        if not os.path.exists(last_path):
            sys.exit(f"ERROR: nothing to resume in {a.out}.")
        ck = torch.load(last_path, map_location=device)
        if ck["hidden"] != a.hidden:
            sys.exit(f"ERROR: checkpoint has --hidden {ck['hidden']}, but this run uses {a.hidden}.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        first_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        print(f"Resuming after pass {first_epoch}.")

    steps_per_epoch = math.ceil(len(train_cp) / a.batch_size)
    total_steps = steps_per_epoch * a.epochs
    log_path = os.path.join(a.out, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["pass", "train_loss", "val_loss", "val_median_cp_error", "val_right_side", "seconds"])

    print(f"Training on {device}: {a.epochs} passes, {steps_per_epoch:,} steps each.", flush=True)
    try:
        for epoch in range(first_epoch, a.epochs):
            model.train()
            start, running = time.time(), 0.0
            for i, idx in enumerate(torch.randperm(len(train_cp)).split(a.batch_size), 1):
                boards = train_b[idx].to(device, non_blocking=True)
                target = train_cp[idx].to(device, non_blocking=True) / CP_SCALE
                for group in opt.param_groups:
                    group["lr"] = learning_rate(step, total_steps, a.lr)
                loss = win_prob_loss(model(boards), target)
                if not torch.isfinite(loss):
                    sys.exit("ERROR: loss became NaN or infinite. Try a lower --lr, or use --resume.")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.limit_weights()
                step += 1
                running += loss.item()
                if i % a.print_every == 0:
                    rate = i * a.batch_size / (time.time() - start)
                    print(f"  pass {epoch + 1} step {i:,}/{steps_per_epoch:,} | loss {running / i:.5f} | "
                          f"{rate:,.0f} positions/s", flush=True)

            result = report(predict_cp(model, val_b, device), val_cp)
            seconds = time.time() - start
            show_report(f"Pass {epoch + 1}/{a.epochs} ({seconds:.0f} s), validation", result)
            improved = result["loss"] < best
            best = min(best, result["loss"])
            state = dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step,
                         best=best, hidden=a.hidden)
            save_checkpoint(last_path, **state)
            if improved:
                save_checkpoint(best_path, **state)
            with open(log_path, "a", newline="") as fh:
                csv.writer(fh).writerow([epoch + 1, running / i, result["loss"], result["median_cp_error"],
                                         result["right_side"], round(seconds)])
    except KeyboardInterrupt:
        print("\nStopped. Finished passes are saved; use --resume to continue.")

    if os.path.exists(best_path):
        finish(a, device, val_b, best_path)


def finish(a, device, val_b, ckpt_path):
    ck = torch.load(ckpt_path, map_location=device)
    model = RubyNet(ck["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    sanity_check(model, device)
    export(model, val_b, device, os.path.join(a.out, "ruby_small.bin"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA, help="folder with train_*.npz and val_*.npz")
    p.add_argument("--out", default=DEFAULT_OUT, help="where checkpoints, the log and weights go")
    p.add_argument("--epochs", type=int, default=30, help="passes through the training data")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--hidden", type=int, default=256, help="feature layer size per side")
    p.add_argument("--keep-tactical", action="store_true", help="keep positions where the next move was a capture")
    p.add_argument("--limit", type=int, default=None, help="use only this many positions (for short tests)")
    p.add_argument("--print-every", type=int, default=200, help="steps between progress lines")
    p.add_argument("--resume", action="store_true", help="continue from last.pt in --out")
    p.add_argument("--export-only", action="store_true", help="re-export best.pt without training")
    p.add_argument("--cpu", action="store_true", help="run without the GPU")
    a = p.parse_args()

    torch.manual_seed(1)
    device = "cpu" if a.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: PyTorch can't see a GPU. Install a CUDA build of torch, or pass --cpu.")
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    if a.export_only:
        val_b, _ = load_split(a.data, "val", a.keep_tactical, a.limit)
        finish(a, device, val_b, os.path.join(a.out, "best.pt"))
    else:
        train(a, device)


if __name__ == "__main__":
    main()
