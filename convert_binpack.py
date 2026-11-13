#!/usr/bin/env python3
"""
convert_binpack.py - Stockfish .binpack training data -> Ruby's shard format (same columns as parse_pgn.py).

  python convert_binpack.py dump      FILE.binpack [--n 20]          # print the first positions as FEN, to eyeball
  python convert_binpack.py check     FILE.binpack [--n 200000]      # decode and verify every move is legal
  python convert_binpack.py calibrate FILE.binpack [--n 3000]        # fit the binpack-score -> Ruby-cp scale with Stockfish
  python convert_binpack.py convert   FILE.binpack --scale S [--max-positions N] [--every K]   # write shards (resumable)

Output shards (in --out-dir, default parsed_bp/) hold boards, cp, res, ply, flags, h exactly like parse_pgn.py's,
with the same filters (skip the first --min-ply plies, positions in check, insufficient material, |cp| >= --max-eval).
Merge them with:  python parse_pgn.py finalize --more-dirs parsed_bp

Format reader: written from the description in Stockfish's docs/binpack.md and from Sopel97's
nnue_data_compress (MIT License, Copyright 2019 Tomasz Sobczyk). See THIRD_PARTY.md.
The data itself (official-stockfish/master-binpacks) is under the Open Database License (ODbL).
"""
import argparse, glob, json, multiprocessing as mp, os, re, sys, time
import numpy as np
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ("boards", "cp", "res", "ply", "flags")
MAX_CHUNK = 100 * 1024 * 1024
POPCOUNT = int.bit_count


# ----------------------------------------------------------------------------- low-level decoding
def u2s(r):
    """Sopel's unsignedToSigned for 16-bit values (sign bit stored in bit 0)."""
    r = ((r << 15) | (r >> 1)) & 0xFFFF
    if r & 0x8000:
        r ^= 0x7FFF
    return r - 0x10000 if r & 0x8000 else r


def used_bits_safe(n):
    return 0 if n == 0 else (n - 1).bit_length()


def nth_bit(bb, n):
    for _ in range(n):
        bb &= bb - 1
    return (bb & -bb).bit_length() - 1


class Bits:
    """MSB-first bit stream over a bytes object, starting at byte `start`."""
    __slots__ = ("d", "p", "start")

    def __init__(self, data, start):
        self.d, self.start, self.p = data, start, start * 8

    def get(self, count):
        if count == 0:
            return 0
        p, d = self.p, self.d
        i = p >> 3
        w = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2]       # chunks are padded with 2 zero bytes
        self.p = p + count
        return (w >> (24 - (p & 7) - count)) & ((1 << count) - 1)

    def vle16(self, block=4):
        v, off, mask = 0, 0, (1 << block) - 1
        while True:
            b = self.get(block + 1)
            v |= (b & mask) << off
            if not b >> block:
                return v
            off += block

    def bytes_read(self):
        return (self.p - self.start * 8 + 7) // 8


QUEEN_RAYS = [chess.BB_DIAG_ATTACKS[s][0] | chess.BB_RANK_ATTACKS[s][0] | chess.BB_FILE_ATTACKS[s][0]
              for s in chess.SQUARES]


def slider_attacked(k, diag, line, occ):
    return bool((chess.BB_DIAG_ATTACKS[k][chess.BB_DIAG_MASKS[k] & occ] & diag)
                or ((chess.BB_RANK_ATTACKS[k][chess.BB_RANK_MASKS[k] & occ]
                     | chess.BB_FILE_ATTACKS[k][chess.BB_FILE_MASKS[k] & occ]) & line))


def ep_after(board, move):
    """En passant square the binpack format records after `move` (board is BEFORE the move), else None.
    Follows Sopel's Position::isEpPossible, but on the board AFTER the double push (pushed pawn on `to`,
    `from` empty). Sopel's original runs it before the move, which wrongly lets a pawn still on `from` block
    a pin; the Stockfish tools that wrote the current data use the real post-move rule (seen in test80 data)."""
    frm, to = move.from_square, move.to_square
    if board.piece_type_at(frm) != chess.PAWN or (frm ^ to) != 16:
        return None
    ep = (frm + to) >> 1
    mover = board.turn
    them = not mover
    attackers = chess.BB_PAWN_ATTACKS[mover][ep] & board.pieces_mask(chess.PAWN, them)
    if not attackers:
        return None
    k = board.king(them)
    b = board.pieces_mask(chess.BISHOP, mover)
    r = board.pieces_mask(chess.ROOK, mover)
    q = board.pieces_mask(chess.QUEEN, mover)
    if not ((b | r | q) & QUEEN_RAYS[k]):
        return ep
    occ0 = board.occupied ^ chess.BB_SQUARES[frm] ^ chess.BB_SQUARES[to]      # after the push
    for sq in chess.scan_forward(attackers):
        occ = ((occ0 ^ chess.BB_SQUARES[sq]) | chess.BB_SQUARES[ep]) ^ chess.BB_SQUARES[to]
        if not slider_attacked(k, b | q, r | q, occ):
            return ep
    return None


def decode_stem(d, o):
    """32-byte stem at offset o -> (board, ep, move, score, ply, result)."""
    occ = int.from_bytes(d[o:o + 8], "big")
    board = chess.Board(None)
    rights, ep, turn = 0, None, chess.WHITE
    k = 0
    for sq in chess.scan_forward(occ):
        nib = (d[o + 8 + (k >> 1)] >> (4 * (k & 1))) & 15
        k += 1
        if nib < 12:
            board._set_piece_at(sq, nib // 2 + 1, nib % 2 == 0)
        elif nib == 12:
            if chess.square_rank(sq) == 3:
                board._set_piece_at(sq, chess.PAWN, chess.WHITE); ep = sq - 8
            else:
                board._set_piece_at(sq, chess.PAWN, chess.BLACK); ep = sq + 8
        elif nib == 13:
            board._set_piece_at(sq, chess.ROOK, chess.WHITE); rights |= chess.BB_SQUARES[sq]
        elif nib == 14:
            board._set_piece_at(sq, chess.ROOK, chess.BLACK); rights |= chess.BB_SQUARES[sq]
        else:
            board._set_piece_at(sq, chess.KING, chess.BLACK); turn = chess.BLACK
    board.turn = turn
    board.castling_rights = rights
    board.ep_square = ep
    m = (d[o + 24] << 8) | d[o + 25]
    move = packed_move(board, m)
    score = u2s((d[o + 26] << 8) | d[o + 27])
    pr = (d[o + 28] << 8) | d[o + 29]
    board.halfmove_clock = (d[o + 30] << 8) | d[o + 31]
    ply = pr & 0x3FFF
    board.fullmove_number = ply // 2 + 1
    return board, ep, move, score, ply, u2s(pr >> 14)


def packed_move(board, m):
    """Sopel's CompressedMove: 2 bits type, 6 from, 6 to, 2 promotion. Castling is king -> rook square."""
    if m == 0:
        return chess.Move.null()
    typ, frm, to = m >> 14, (m >> 8) & 63, (m >> 2) & 63
    if typ == 1:
        return chess.Move(frm, to, chess.KNIGHT + (m & 3))
    if typ == 2:
        return chess.Move(frm, (to & 56) | (6 if to > frm else 2))
    return chess.Move(frm, to)


def next_move(board, ep, bits):
    """Decode one movetext move for `board` (side to move), using the binpack's own ep square."""
    us = board.turn
    ours = board.occupied_co[us]
    occ = board.occupied
    frm = nth_bit(ours, bits.get(used_bits_safe(POPCOUNT(ours))))
    pt = board.piece_type_at(frm)
    if pt == chess.PAWN:
        targets = board.occupied_co[not us] | (chess.BB_SQUARES[ep] if ep is not None else 0)
        dest = chess.BB_PAWN_ATTACKS[us][frm] & targets
        fwd = 8 if us else -8
        f1 = frm + fwd
        if not occ & chess.BB_SQUARES[f1]:
            dest |= chess.BB_SQUARES[f1]
            if chess.square_rank(frm) == (1 if us else 6) and not occ & chess.BB_SQUARES[f1 + fwd]:
                dest |= chess.BB_SQUARES[f1 + fwd]
        n = POPCOUNT(dest)
        if chess.square_rank(frm) == (6 if us else 1):
            mid = bits.get(used_bits_safe(n * 4))
            return chess.Move(frm, nth_bit(dest, mid // 4), chess.KNIGHT + mid % 4)
        return chess.Move(frm, nth_bit(dest, bits.get(used_bits_safe(n))))
    if pt == chess.KING:
        attacks = chess.BB_KING_ATTACKS[frm] & ~ours
        na = POPCOUNT(attacks)
        mine = board.castling_rights & (chess.BB_RANK_1 if us else chess.BB_RANK_8)
        mid = bits.get(used_bits_safe(na + POPCOUNT(mine)))
        if mid >= na:
            long_ = mid == na and bool(mine & (chess.BB_A1 if us else chess.BB_A8))
            return chess.Move(frm, (frm & 56) | (2 if long_ else 6))
        return chess.Move(frm, nth_bit(attacks, mid))
    attacks = board.attacks_mask(frm) & ~ours
    return chess.Move(frm, nth_bit(attacks, bits.get(used_bits_safe(POPCOUNT(attacks)))))


class DecodeError(Exception):
    """The decoded game went off the rails: everything after this point in the chunk is unreliable."""


def iter_chunk(d):
    """Yield (board, move, score, ply, result) for every position in one chunk (board is live: copy if kept)."""
    o, end = 0, len(d) - 2                                   # last 2 bytes are padding
    while o + 34 <= end:
        board, ep, move, score, ply, res = decode_stem(d, o)
        n = (d[o + 32] << 8) | d[o + 33]
        o += 34
        if not board.is_pseudo_legal(move):
            raise DecodeError(f"illegal stem move {move.uci()} in {board.fen()}")
        yield board, move, score, ply, res
        if n:
            bits, last = Bits(d, o), -score
            for _ in range(n):
                nep = ep_after(board, move)
                board.push(move)
                ep = nep
                try:
                    move = next_move(board, ep, bits)
                except (ValueError, IndexError) as e:
                    raise DecodeError(f"cannot decode a move in {board.fen()}: {e}")
                if not board.is_pseudo_legal(move):
                    raise DecodeError(f"decoded illegal move {move.uci()} in {board.fen()}")
                score = last + u2s(bits.vle16())
                last = -score
                ply += 1
                res = -res
                yield board, move, score, ply, res
            o += bits.bytes_read()


def chunk_offsets(path):
    """(offset, size) of each complete BINP chunk; a truncated last chunk (partial download) is ignored."""
    out, total = [], os.path.getsize(path)
    with open(path, "rb") as f:
        o = 0
        while o + 8 <= total:
            f.seek(o)
            h = f.read(8)
            if h[:4] != b"BINP":
                sys.exit(f"ERROR: no BINP header at byte {o}. Not a binpack file (a .zst must be decompressed first).")
            size = int.from_bytes(h[4:], "little")
            if size > MAX_CHUNK:
                sys.exit(f"ERROR: chunk at byte {o} claims {size} bytes. Malformed file?")
            if o + 8 + size > total:
                break
            out.append((o + 8, size))
            o += 8 + size
    return out


def read_chunk(path, off, size):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(size) + b"\0\0"


# ----------------------------------------------------------------------------- Ruby encoding
def encode(board):
    """Mover-relative board, identical to parse_pgn.encode."""
    me = board.turn
    flip = 56 if me == chess.BLACK else 0
    out = np.zeros(64, np.uint8)
    for sq, p in board.piece_map().items():
        s = sq ^ flip
        out[(7 - (s >> 3)) * 8 + (s & 7)] = p.piece_type + (0 if p.color == me else 6)
    return out


def hashes(codes):
    from parse_pgn import ZOBRIST
    return np.bitwise_xor.reduce(ZOBRIST[np.arange(64), codes.astype(np.intp)], axis=1)


def wanted(board, move, score, ply, a):
    # Stockfish's filtered packs store 32002 (VALUE_NONE) for skipped positions and 0 for positions whose move is
    # a capture/promotion or that are in check, so those scores are placeholders, not evaluations. Drop them all.
    if ply < a.min_ply or abs(score) >= 32000 or board.is_check():
        return False
    if move.promotion is not None or board.is_capture(move):
        return False
    return not board.is_insufficient_material()


def convert_chunk(job):
    path, off, size, a = job
    codes, cp, res, ply_l, flags = [], [], [], [], []
    k = 0
    lim = a.max_eval * 100
    try:
        for board, move, score, ply, r in iter_chunk(read_chunk(path, off, size)):
            if not wanted(board, move, score, ply, a):
                continue
            k += 1
            if k % a.every:
                continue
            c = int(round(score * a.scale))
            if abs(c) >= lim:
                continue
            noisy = board.is_capture(move) or move.promotion is not None
            codes.append(encode(board)); cp.append(c); res.append(r)
            ply_l.append(min(ply, 255)); flags.append(1 if noisy else 0)
    except DecodeError as e:
        # A wrong move may already have been accepted a little before the check caught it, so drop the chunk.
        return dict(bad=f"chunk at byte {off}: {e}", **{k: np.zeros((0, 64) if k == "boards" else 0, t)
                    for k, t in (("boards", np.uint8), ("cp", np.int16), ("res", np.int8), ("ply", np.uint8),
                                 ("flags", np.uint8), ("h", np.uint64))})
    n = len(cp)
    boards = np.stack(codes) if n else np.zeros((0, 64), np.uint8)
    return dict(boards=boards, cp=np.array(cp, np.int16), res=np.array(res, np.int8),
                ply=np.array(ply_l, np.uint8), flags=np.array(flags, np.uint8),
                h=hashes(boards) if n else np.zeros(0, np.uint64))


# ----------------------------------------------------------------------------- commands
def first_chunks(path, k=1):
    offs = chunk_offsets(path)
    if not offs:
        sys.exit("ERROR: the file holds no complete chunk.")
    return [read_chunk(path, o, s) for o, s in offs[:k]], offs


def cmd_dump(a):
    (d,), offs = first_chunks(a.file)
    print(f"{len(offs)} complete chunks, first is {offs[0][1]:,} bytes")
    for i, (board, move, score, ply, res) in enumerate(iter_chunk(d)):
        if i >= a.n:
            break
        print(f"{board.fen():72s} {move.uci():6s} score {score:+6d} ply {ply:3d} result {res:+d}")


def cmd_check(a):
    offs = chunk_offsets(a.file)
    print(f"{len(offs):,} complete chunks ({sum(s for _, s in offs) / 1e9:.2f} GB)")
    n = bad = chains = 0
    t = time.time()
    for off, size in offs:
        prev_ply = None
        for board, move, score, ply, res in iter_chunk(read_chunk(a.file, off, size)):
            if prev_ply is None or ply != prev_ply + 1:
                chains += 1
            prev_ply = ply
            n += 1
            if not board.is_legal(move) or not board.is_valid():
                bad += 1
                if bad <= 5:
                    print(f"ILLEGAL: {board.fen()} {move.uci()} (position {n:,})")
            if n >= a.n:
                break
        if n >= a.n:
            break
    dt = time.time() - t
    print(f"{n:,} positions in {chains:,} chains, {bad} illegal moves or invalid boards, "
          f"{n / dt:,.0f} positions/s on one core")
    if bad:
        sys.exit("ERROR: the decoder is wrong somewhere. Do not convert.")


def cmd_calibrate(a):
    import chess.engine
    from parse_pgn import DEFAULT_ENGINE
    offs = chunk_offsets(a.file)
    step = max(1, len(offs) // 50)
    pos = []
    for off, size in offs[::step]:                          # spread the sample over the file
        got = 0
        for board, move, score, ply, res in iter_chunk(read_chunk(a.file, off, size)):
            if wanted(board, move, score, ply, a) and abs(score) < 2000 and (ply % 7 == 0):
                pos.append((board.fen(), score)); got += 1
                if got >= a.n // 50 + 1:
                    break
        if len(pos) >= a.n:
            break
    print(f"evaluating {len(pos)} positions with Stockfish at {a.nodes} nodes (your parser's setting)...")
    eng = chess.engine.SimpleEngine.popen_uci(a.engine or DEFAULT_ENGINE)
    eng.configure({"Threads": 1, "Hash": 16})
    xs, ys = [], []
    for fen, s in pos:
        b = chess.Board(fen)
        cp = eng.analyse(b, chess.engine.Limit(nodes=a.nodes))["score"].pov(b.turn).score(mate_score=32000)
        if abs(cp) < 1000:
            xs.append(s); ys.append(cp)
    eng.quit()
    x, y = np.array(xs, float), np.array(ys, float)
    scale = float((x * y).sum() / (x * x).sum())
    r = float(np.corrcoef(x, y)[0, 1])
    med = float(np.median(np.abs(y - scale * x)))
    print(f"{len(x)} usable pairs. Best fit: Ruby cp = {scale:.3f} x binpack score "
          f"(correlation {r:.3f}, median difference {med:.0f} cp)")
    for lo, hi in ((0, 100), (100, 300), (300, 1000)):
        m = (np.abs(y) >= lo) & (np.abs(y) < hi)
        if m.sum() > 20:
            print(f"  |cp| {lo}-{hi}: {m.sum()} positions, ratio {float((x[m] * y[m]).sum() / (x[m] ** 2).sum()):.3f}")
    print(f"\nUse:  python convert_binpack.py convert \"{a.file}\" --scale {scale:.3f}")


def cmd_convert(a):
    path = os.path.abspath(a.file)
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    settings = dict(file=os.path.basename(path), scale=a.scale, every=a.every, min_ply=a.min_ply, max_eval=a.max_eval)
    prog_path = os.path.join(out_dir, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"files": {}, "kept": 0}
    st = prog["files"].get(settings["file"], {"chunks_done": 0, "settings": settings})
    if st["settings"] != settings and not a.force:
        sys.exit(f"ERROR: {settings['file']} was started with other settings: {st['settings']}. Use --force.")
    offs = chunk_offsets(path)
    done = st["chunks_done"]
    print(f"{len(offs):,} complete chunks, {done:,} already converted, {prog['kept']:,} positions in {out_dir}")
    nums = [int(re.search(r"shard_(\d+)\.npz$", f).group(1)) for f in glob.glob(os.path.join(out_dir, "shard_*.npz"))]
    nxt = max(nums) + 1 if nums else 0
    parts, n_parts, t0, conv = [], 0, time.time(), 0

    def flush(chunks_done):
        nonlocal parts, n_parts, nxt
        if parts:
            data = {k: np.concatenate([p[k] for p in parts]) for k in COLS + ("h",)}
            tmp = os.path.join(out_dir, f"shard_{nxt:05d}.npz.tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, **data)
            os.replace(tmp, tmp[:-4])
            nxt += 1
            prog["kept"] += len(data["cp"])
            parts, n_parts = [], 0
        st["chunks_done"] = chunks_done
        prog["files"][settings["file"]] = st
        with open(prog_path + ".tmp", "w") as fh:
            json.dump(prog, fh)
        os.replace(prog_path + ".tmp", prog_path)

    # --chunk-step K converts every K-th chunk, which spreads a smaller sample over the whole file.
    picked = list(range(done, len(offs), a.chunk_step))
    jobs = [(path, *offs[k], a) for k in picked]
    i = done
    with mp.Pool(a.workers) as pool:
        try:
            for k, part in zip(picked, pool.imap(convert_chunk, jobs)):
                i = k + 1
                if "bad" in part:
                    st["bad_chunks"] = st.get("bad_chunks", 0) + 1
                    print(f"SKIPPED a chunk that failed to decode ({st['bad_chunks']} so far): {part.pop('bad')}",
                          flush=True)
                parts.append(part); n_parts += len(part["cp"]); conv += len(part["cp"])
                if n_parts >= a.shard_size:
                    flush(i)
                if (k // a.chunk_step) % 20 == 0:
                    rate = conv / (time.time() - t0)
                    print(f"chunks {i:,}/{len(offs):,} | kept {prog['kept'] + n_parts:,} | {rate:,.0f} positions/s",
                          flush=True)
                if prog["kept"] + n_parts >= a.max_positions:
                    print("Reached --max-positions.")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted - saving...")
            pool.terminate()
        flush(i)
    print(f"Done for now: {prog['kept']:,} positions in {out_dir}, {i:,}/{len(offs):,} chunks of this file.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(c):
        c.add_argument("file")
        c.add_argument("--min-ply", type=int, default=10)
        c.add_argument("--max-eval", type=float, default=10.0, help="drop |cp| >= this many pawns")

    c = sub.add_parser("dump"); common(c); c.add_argument("--n", type=int, default=20); c.set_defaults(fn=cmd_dump)
    c = sub.add_parser("check"); common(c); c.add_argument("--n", type=int, default=200_000); c.set_defaults(fn=cmd_check)
    c = sub.add_parser("calibrate"); common(c)
    c.add_argument("--n", type=int, default=3000); c.add_argument("--nodes", type=int, default=15000)
    c.add_argument("--engine", default=None); c.set_defaults(fn=cmd_calibrate)
    c = sub.add_parser("convert"); common(c)
    c.add_argument("--scale", type=float, required=True, help="from `calibrate`")
    c.add_argument("--every", type=int, default=1, help="keep 1 in K positions (spreads data over more games)")
    c.add_argument("--max-positions", type=int, default=100_000_000)
    c.add_argument("--chunk-step", type=int, default=1, help="convert every K-th chunk only")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--shard-size", type=int, default=1_000_000)
    c.add_argument("--out-dir", default=os.path.join(HERE, "parsed_bp"))
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_convert)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    mp.freeze_support()
    main()
