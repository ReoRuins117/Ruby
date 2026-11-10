#!/usr/bin/env python3
"""
parse_pgn.py - raw PGN -> Stockfish-labelled, deduplicated, mover-relative training data.

Positions are deduplicated BEFORE Stockfish sees them, evaluation runs in parallel (one single-thread
engine per worker, fixed node limit), and results go straight to compact resumable shards.

  python parse_pgn.py bench    --pgn "LI 2015-03.pgn"        # 1) sanity-check + find the best worker count
  python parse_pgn.py parse    --pgn "LI 2015-03.pgn"        # 2) the real run (safe to stop and re-run to resume)
  python parse_pgn.py finalize                                # 3) shuffle into train_*.npz / val_*.npz

Shard / chunk columns (same board layout and hash as prep_data.py, so old and new data can be merged):
  boards uint8 (n,64)  0 = empty, 1-6 = mover's P N B R Q K, 7-12 = opponent's  (mover = side to move)
  cp     int16         Stockfish eval, centipawns, from the mover's view
  res    int8          game result from the mover's view: 1 win, 0 draw, -1 loss (127 = unknown, old data)
  ply    uint8         plies played so far (capped at 255)
  flags  uint8         bit 0: next move in the game is a capture/promotion, bit 1: mover is in check
Needs: numpy, python-chess, a Stockfish binary. Input can be .pgn, .pgn.bz2 or .pgn.zst (zst needs `pip install zstandard`).
"""
import argparse, bz2, collections, glob, io, itertools, json, multiprocessing as mp, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENGINE = "C:/Users/joshu/OneDrive/Documents/Programs/Python/Ruby/Stockfish/stockfish-windows-x86-64-avx2.exe"
DEFAULT_PGN = os.path.join(HERE, "LI 2015-03.pgn")
RES = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
UNKNOWN_RES = 127
FAILED = 32767                     # sentinel for an evaluation that errored (always filtered out)
COLS = ("boards", "cp", "res", "ply", "flags")
IDX = np.arange(64)


# ----------------------------------------------------------------------------- hashing (identical to prep_data.py)
def splitmix64(x):
    """Deterministic 64-bit mixer. Uses only basic numpy arithmetic (never imports numpy.random)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


ZOBRIST = splitmix64(np.arange(64 * 13, dtype=np.uint64) + np.uint64(20240607)).reshape(64, 13)


def encode(board):
    """Mover-relative board as uint8[64]; index = (7 - rank) * 8 + file after mirroring when Black is to move."""
    import chess
    me = board.turn
    flip = me == chess.BLACK
    out = np.zeros(64, np.uint8)
    for sq, p in board.piece_map().items():
        if flip:
            sq ^= 56
        out[(7 - sq // 8) * 8 + sq % 8] = p.piece_type + (0 if p.color == me else 6)
    return out


def pos_hash(codes):
    return int(np.bitwise_xor.reduce(ZOBRIST[IDX, codes]))


# ----------------------------------------------------------------------------- PGN reading
def open_text(path):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            sys.exit("ERROR: .zst input needs `pip install zstandard` (or decompress the file first).")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                                encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_games(path, skip=0):
    """Yield the text of each game; the first `skip` games are passed over cheaply (used to resume)."""
    cur, n = [], 0
    with open_text(path) as f:
        for line in f:
            if line.startswith("[Event "):
                if cur:
                    yield "".join(cur)
                cur = []
                n += 1
            if n > skip:
                cur.append(line)
    if cur:
        yield "".join(cur)


def parse_games(texts, seen, a, st):
    """Replay games, apply the cheap filters, drop positions already in `seen`, return the new ones."""
    import chess, chess.pgn
    codes, hashes, fens, res, ply_l, flags = [], [], [], [], [], []
    for text in texts:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        hd = game.headers
        r = RES.get(hd.get("Result", "*"))
        if r is None or "FEN" in hd or hd.get("Variant", "Standard") != "Standard":
            st["skipped_games"] += 1
            continue
        if a.min_elo:
            try:
                if min(int(hd.get("WhiteElo", 0)), int(hd.get("BlackElo", 0))) < a.min_elo:
                    st["skipped_games"] += 1
                    continue
            except ValueError:
                st["skipped_games"] += 1
                continue
        board = game.board()
        moves = list(game.mainline_moves())
        for i, mv in enumerate(moves):
            board.push(mv)
            ply = i + 1
            if ply <= a.min_ply:
                continue
            check = board.is_check()
            if check and not a.keep_checks:
                continue
            nxt = moves[i + 1] if i + 1 < len(moves) else None
            noisy = nxt is not None and (board.is_capture(nxt) or nxt.promotion is not None)
            if noisy and a.quiet:
                continue
            if board.is_insufficient_material() or not any(board.generate_legal_moves()):
                continue
            st["cand"] += 1
            c = encode(board)
            h = pos_hash(c)
            if h in seen:
                st["dups"] += 1
                continue
            seen.add(h)
            codes.append(c); hashes.append(h); fens.append(board.fen())
            res.append(r if board.turn == chess.WHITE else -r)
            ply_l.append(min(ply, 255)); flags.append((1 if noisy else 0) | (2 if check else 0))
    n = len(fens)
    return {"codes": np.stack(codes) if n else np.zeros((0, 64), np.uint8),
            "h": np.array(hashes, dtype=np.uint64), "fens": fens,
            "res": np.array(res, dtype=np.int8), "ply": np.array(ply_l, dtype=np.uint8),
            "flags": np.array(flags, dtype=np.uint8)}


# ----------------------------------------------------------------------------- evaluation workers
_ENGINE = None
_STATE = {}


def _start_engine():
    import chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(_STATE["path"])
    eng.configure({"Threads": 1, "Hash": _STATE["hash"]})
    return eng


def _quit_engine():
    try:
        _ENGINE.quit()
    except Exception:
        pass


def _init_worker(path, nodes, hash_mb, barrier=None):
    global _ENGINE
    from multiprocessing import util
    _STATE.update(path=path, nodes=nodes, hash=hash_mb)
    _ENGINE = _start_engine()
    util.Finalize(None, _quit_engine, exitpriority=10)
    if barrier is not None:
        barrier.wait()


def eval_chunk(fens):
    """Evaluate FEN strings; returns int32 centipawns from the side to move's view (FAILED on error)."""
    global _ENGINE
    import chess, chess.engine
    out = np.empty(len(fens), np.int32)
    limit = chess.engine.Limit(nodes=_STATE["nodes"])
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for attempt in (0, 1):
            try:
                score = _ENGINE.analyse(board, limit)["score"].pov(board.turn).score(mate_score=32000)
                out[i] = max(-32000, min(32000, int(score)))
                break
            except chess.engine.EngineTerminatedError:
                try:
                    _ENGINE = _start_engine()
                except Exception:
                    out[i] = FAILED
                    break
            except Exception:
                out[i] = FAILED
                break
        else:
            out[i] = FAILED
    return out


def selftest(a):
    """Check the engine path, the sign convention and the board encoding before any long run."""
    import chess, chess.engine
    try:
        eng = chess.engine.SimpleEngine.popen_uci(a.engine)
    except Exception as e:
        sys.exit(f"ERROR: could not start Stockfish at {a.engine}\n  {e}\n  Pass the right path with --engine.")
    eng.configure({"Threads": 1, "Hash": a.hash})
    lim = chess.engine.Limit(nodes=a.nodes)

    def ev(fen):
        b = chess.Board(fen)
        return eng.analyse(b, lim)["score"].pov(b.turn).score(mate_score=32000)

    start = ev(chess.STARTING_FEN)
    w = ev("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")   # Black has no queen, White to move
    b = ev("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")   # same, Black to move
    eng.quit()
    ok = abs(start) < 150 and w > 500 and b < -500
    print(f"selftest: start position {start:+d} cp (expect about 0 to +50)")
    print(f"selftest: mover up a queen {w:+d} cp (expect > +500), mover down a queen {b:+d} cp (expect < -500)")
    bd = chess.Board(); bd.push_san("e4")                              # Black to move
    c0, c1 = encode(chess.Board()), encode(bd)
    enc_ok = c0[60] == 6 and c0[4] == 12 and c1[60] == 6 and c1[4] == 12 and c1[28] == 7 and c1[36] == 0
    print("selftest: board encoding", "OK" if enc_ok else "WRONG")
    if not ok or not enc_ok:
        sys.exit("ERROR: selftest failed - do not run the full parse. Send me this output.")


# ----------------------------------------------------------------------------- shards / progress
def load_hashes(folder, seen):
    n = 0
    for f in sorted(glob.glob(os.path.join(folder, "shard_*.npz"))):
        h = np.load(f)["h"]
        seen.update(h.tolist())
        n += len(h)
    return n


def write_atomic(path, writer, mode="wb"):
    tmp = path + ".tmp"
    with open(tmp, mode) as fh:                       # closed before the rename (required on Windows)
        writer(fh)
    os.replace(tmp, path)


class ShardWriter:
    def __init__(self, out_dir, settings, pgn_info):
        self.dir, self.settings, self.pgn_info = out_dir, settings, pgn_info
        nums = [int(re.search(r"shard_(\d+)\.npz$", f).group(1)) for f in glob.glob(os.path.join(out_dir, "shard_*.npz"))]
        self.next, self.parts, self.n = (max(nums) + 1 if nums else 0), [], 0

    def add(self, part):
        if len(part["cp"]):
            self.parts.append(part); self.n += len(part["cp"])

    def flush(self, games_done):
        if self.parts:
            data = {k: np.concatenate([p[k] for p in self.parts]) for k in COLS + ("h",)}
            path = os.path.join(self.dir, f"shard_{self.next:05d}.npz")
            write_atomic(path, lambda fh: np.savez(fh, **data))
            self.next += 1
            self.parts, self.n = [], 0
        prog = dict(self.pgn_info, games_done=games_done, settings=self.settings)
        write_atomic(os.path.join(self.dir, "progress.json"), lambda fh: json.dump(prog, fh), mode="w")


def fmt_hours(sec):
    return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min"


# ----------------------------------------------------------------------------- commands
def cmd_bench(a):
    if not os.path.exists(a.pgn):
        sys.exit(f"ERROR: PGN not found: {a.pgn}  (pass --pgn)")
    selftest(a)
    st = collections.Counter()
    seen = set()
    t = time.time()
    batch = parse_games(list(itertools.islice(iter_games(a.pgn), a.games)), seen, a, st)
    dt = time.time() - t
    n = len(batch["fens"])
    print(f"\nparsed {a.games} games in {dt:.1f} s ({a.games / dt:.0f} games/s)")
    print(f"candidate positions {st['cand']}, already-seen {st['dups']} ({100 * st['dups'] / max(st['cand'], 1):.0f}%), "
          f"new {n} (~{n / a.games:.0f} per game; duplicate share grows as more games are read)")
    fens = batch["fens"][:a.positions]
    chunks = [fens[i:i + a.chunk] for i in range(0, len(fens), a.chunk)]
    print(f"\nevaluating {len(fens)} positions at {a.nodes} nodes each:")
    best = 0
    for w in [int(x) for x in a.workers_list.split(",")]:
        barrier = mp.Barrier(w + 1)
        with mp.Pool(w, _init_worker, (a.engine, a.nodes, a.hash, barrier)) as pool:
            try:
                barrier.wait(timeout=120)
            except Exception:
                sys.exit("ERROR: workers failed to start their engines.")
            t = time.time()
            pool.map(eval_chunk, chunks)
            rate = len(fens) / (time.time() - t)
        best = max(best, rate)
        print(f"  {w:2d} workers: {rate:7.0f} evals/s  ({1000 * w / rate:.1f} ms per eval per worker)")
    print(f"\nprojected time for {a.target:,} positions at the best rate: {fmt_hours(a.target / best)}")
    print("If 'ms per eval per worker' is far from ~10, change --nodes (more nodes = slower, better labels).")


def cmd_parse(a):
    pgn, out_dir = os.path.abspath(a.pgn), os.path.abspath(a.out_dir)
    if not os.path.exists(pgn):
        sys.exit(f"ERROR: PGN not found: {pgn}  (pass --pgn)")
    selftest(a)
    os.makedirs(out_dir, exist_ok=True)
    settings = dict(nodes=a.nodes, min_ply=a.min_ply, max_eval=a.max_eval, quiet=a.quiet,
                    keep_checks=a.keep_checks, min_elo=a.min_elo)
    pgn_info = dict(pgn=os.path.basename(pgn), pgn_size=os.path.getsize(pgn))
    prog_path, games_done = os.path.join(out_dir, "progress.json"), 0
    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        same = all(prog.get(k) == v for k, v in pgn_info.items()) and prog.get("settings") == settings
        if not same and not a.force:
            sys.exit("ERROR: this folder holds a run with a different PGN or different settings.\n"
                     f"  saved: {prog.get('pgn')} {prog.get('settings')}\n  now:   {pgn_info['pgn']} {settings}\n"
                     "  Use a new --out-dir, or --force to continue anyway.")
        games_done = prog["games_done"]
        print(f"Resuming after {games_done:,} games.")
    seen = set()
    have = load_hashes(out_dir, seen)
    if a.known_hashes:
        k = load_hashes(a.known_hashes, seen)
        print(f"Loaded {k:,} positions from {a.known_hashes} to skip.")
    print(f"{have:,} positions already in {out_dir}")

    writer = ShardWriter(out_dir, settings, pgn_info)
    games = iter_games(pgn, skip=games_done)
    st, pending, submitted, failed, kept = collections.Counter(), collections.deque(), 0, 0, 0
    games_read = last_end = games_done
    exhausted, t0, evaluated = False, time.time(), 0
    pool = mp.Pool(a.workers, _init_worker, (a.engine, a.nodes, a.hash))
    try:
        while True:
            while len(pending) < 2 and not exhausted and have + submitted < a.max_positions:
                texts = list(itertools.islice(games, a.games_per_step))
                if not texts:
                    exhausted = True
                    break
                batch = parse_games(texts, seen, a, st)
                games_read += len(texts)
                fens = batch["fens"]
                submitted += len(fens)
                chunks = [fens[i:i + a.chunk] for i in range(0, len(fens), a.chunk)]
                pending.append((games_read, batch, pool.map_async(eval_chunk, chunks)))
            if not pending:
                break
            end, batch, ar = pending.popleft()
            parts = ar.get()
            cp = np.concatenate(parts) if parts else np.zeros(0, np.int32)
            failed += int((cp == FAILED).sum())
            ok = np.abs(cp) < a.max_eval * 100          # also drops mates and failed evals
            writer.add(dict(boards=batch["codes"][ok], cp=cp[ok].astype(np.int16), res=batch["res"][ok],
                            ply=batch["ply"][ok], flags=batch["flags"][ok], h=batch["h"][ok]))
            kept += int(ok.sum()); evaluated += len(cp); last_end = end
            if writer.n >= a.shard_size:
                writer.flush(last_end)
            el = time.time() - t0
            rate = evaluated / el
            left = max(a.max_positions - have - submitted + sum(len(p[1]["fens"]) for p in pending), 0)
            print(f"games {last_end:,} | kept {have + kept:,} | skipped as duplicates "
                  f"{100 * st['dups'] / max(st['cand'], 1):.0f}% | {rate:.0f} evals/s | "
                  f"{'ETA ' + fmt_hours(left / rate) if not exhausted and rate else ''}"
                  + (f" | FAILED EVALS {failed}" if failed else ""), flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted - saving what has been collected so far...")
    finally:
        if pending:
            pool.terminate()
        else:
            pool.close(); pool.join()
        writer.flush(last_end)
    print(f"Done for now: {have + kept:,} positions in {out_dir}; progress saved at game {last_end:,}.")
    if exhausted:
        print("Reached the end of the PGN.")
    elif have + submitted >= a.max_positions:
        print("Reached --max-positions. Re-run with a larger value to continue from here.")
    else:
        print("Re-run the same command to resume.")


class ChunkWriter:
    def __init__(self, prefix, size, seed):
        self.prefix, self.size, self.seed, self.k, self.n, self.parts = prefix, size, seed, 0, 0, []

    def add(self, cols):
        n = len(cols["cp"])
        if n:
            self.parts.append(cols); self.n += n
        while self.n >= self.size:
            self.flush(self.size)

    def flush(self, m=None):
        if not self.n:
            return
        merged = {k: np.concatenate([p[k] for p in self.parts]) for k in COLS}
        total = len(merged["cp"])
        m = m or total
        keys = splitmix64(np.arange(m, dtype=np.uint64) ^ np.uint64(self.seed + 1000003 * self.k))
        perm = np.argsort(keys, kind="stable")                 # deterministic shuffle
        np.savez(f"{self.prefix}_{self.k:04d}.npz", **{k: v[:m][perm] for k, v in merged.items()})
        self.k += 1
        self.parts, self.n = ([{k: v[m:] for k, v in merged.items()}], total - m) if total > m else ([], 0)


def cmd_finalize(a):
    in_dir = os.path.abspath(a.in_dir)
    out_dir = os.path.abspath(a.out_dir or os.path.join(in_dir, "dataset"))
    srcs = [(f, False) for f in sorted(glob.glob(os.path.join(in_dir, "shard_*.npz")))]
    for d in a.more_dirs:                                     # e.g. parsed_bp from convert_binpack.py (results known)
        srcs += [(f, False) for f in sorted(glob.glob(os.path.join(os.path.abspath(d), "shard_*.npz")))]
    if a.extra_dir:
        srcs += [(f, True) for f in sorted(glob.glob(os.path.join(os.path.abspath(a.extra_dir), "shard_*.npz")))]
    if not any(not e for _, e in srcs):
        sys.exit(f"ERROR: no shard_*.npz files in {in_dir}. Run `parse` first.")
    print(f"{len(srcs)} shards ({sum(e for _, e in srcs)} from --extra-dir)")
    hs = [np.load(f)["h"] for f, _ in srcs]
    sizes = np.array([len(h) for h in hs])
    allh = np.concatenate(hs); del hs
    _, first = np.unique(allh, return_index=True)             # new data comes first, so it wins duplicates
    keep = np.zeros(len(allh), bool); keep[first] = True
    del allh, first
    print(f"{len(keep):,} positions -> {int(keep.sum()):,} unique ({100 * (1 - keep.mean()):.1f}% duplicates removed)")
    os.makedirs(out_dir, exist_ok=True)
    train = ChunkWriter(os.path.join(out_dir, "train"), a.chunk, 1)
    val = ChunkWriter(os.path.join(out_dir, "val"), a.chunk, 2)
    ends = np.cumsum(sizes)
    unknown = 0
    for (f, extra), e, s in zip(srcs, ends, ends - sizes):
        k = keep[s:e]
        if not k.any():
            continue
        d = np.load(f)
        n = int(k.sum())
        cols = dict(boards=d["boards"][k], cp=d["cp"][k])
        if extra:
            cols.update(res=np.full(n, UNKNOWN_RES, np.int8), ply=np.zeros(n, np.uint8), flags=np.zeros(n, np.uint8))
            unknown += n
        else:
            cols.update(res=d["res"][k], ply=d["ply"][k], flags=d["flags"][k])
        no = int(re.search(r"shard_(\d+)\.npz$", f).group(1))
        (val if no % a.val_every == a.val_every // 2 else train).add(cols)
    train.flush(); val.flush()
    print(f"wrote {train.k} train and {val.k} val chunks to {out_dir} ({unknown:,} positions have unknown game result)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(c):
        c.add_argument("--pgn", default=DEFAULT_PGN, help="input PGN (default: 'LI 2015-03.pgn' next to this script)")
        c.add_argument("--engine", default=DEFAULT_ENGINE, help="path to the Stockfish executable")
        c.add_argument("--nodes", type=int, default=15000, help="nodes per evaluation (about 10 ms on a fast core)")
        c.add_argument("--hash", type=int, default=16, help="Stockfish hash MB per worker")
        c.add_argument("--min-ply", type=int, default=10, help="skip the first N plies (your book covers them)")
        c.add_argument("--max-eval", type=float, default=10.0, help="drop |eval| >= this many pawns (incl. mates)")
        c.add_argument("--min-elo", type=int, default=0, help="skip games where either player is below this rating")
        c.add_argument("--quiet", action="store_true", help="skip positions where the next move is a capture/promotion")
        c.add_argument("--keep-checks", action="store_true", help="keep positions where the mover is in check")
        c.add_argument("--chunk", type=int, default=100, help="positions per worker task")

    b = sub.add_parser("bench", help="selftest + speed test for different worker counts")
    common(b)
    b.add_argument("--games", type=int, default=400); b.add_argument("--positions", type=int, default=3000)
    b.add_argument("--workers-list", default="4,8,12,14")
    b.add_argument("--target", type=int, default=15_000_000, help="positions for the time projection")
    b.set_defaults(fn=cmd_bench)

    q = sub.add_parser("parse", help="the real run (resumable)")
    common(q)
    q.add_argument("--out-dir", default=os.path.join(HERE, "parsed"))
    q.add_argument("--workers", type=int, default=12)
    q.add_argument("--max-positions", type=int, default=15_000_000, help="stop after this many evaluated positions")
    q.add_argument("--games-per-step", type=int, default=1000)
    q.add_argument("--shard-size", type=int, default=200_000)
    q.add_argument("--known-hashes", default=None,
                   help="folder of old shard_*.npz (e.g. Normal/converted): skip positions already labelled there")
    q.add_argument("--force", action="store_true", help="resume even if the settings or PGN changed")
    q.set_defaults(fn=cmd_parse)

    f = sub.add_parser("finalize", help="merge shards into shuffled train/val chunks")
    f.add_argument("--in-dir", default=os.path.join(HERE, "parsed")); f.add_argument("--out-dir", default=None)
    f.add_argument("--extra-dir", default=None, help="old shards (Normal/converted) to merge in; their result is 'unknown'")
    f.add_argument("--more-dirs", nargs="*", default=[], help="more shard folders with known results (e.g. parsed_bp)")
    f.add_argument("--chunk", type=int, default=4_000_000)
    f.add_argument("--val-every", type=int, default=50, help="~1 in N shards is held out for validation")
    f.set_defaults(fn=cmd_finalize)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    mp.freeze_support()
    main()
