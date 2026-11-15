#!/usr/bin/env python3
"""fast_sprt.py - test a Ruby build against another with fastchess, which is much faster than
elo_match.py (it is written in C++ and runs games with less overhead).

  python fast_sprt.py ../new_ruby.exe ../ruby.exe                     # SPRT [0, 5] at 8+0.08
  python fast_sprt.py ../new_ruby.exe ../ruby.exe --tc 2+0.02 --elo1 3
  python fast_sprt.py ../ruby.exe ../ruby.exe --option RfpMargin=90   # same build, one setting changed
  python fast_sprt.py ../new_ruby.exe ../ruby.exe --games 2000 --no-sprt

Each opening is played twice with colours swapped, and results are counted in game pairs. Ruby's
own book and the Lichess tablebase are turned off. Needs fastchess (github.com/Disservin/fastchess):
on PATH, at ~/RubyData/tools/fastchess.exe, or given with --fastchess. Openings default to
~/RubyData/books/UHO_Lichess_4852_v1.epd (from github.com/official-stockfish/books, CC0); the
balanced 8moves_v3.pgn works too. Stop with Ctrl+C; fastchess prints the result so far.
"""
import argparse
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
DEFAULT_FASTCHESS = shutil.which("fastchess") or os.path.join(HOME, "RubyData", "tools", "fastchess.exe")
DEFAULT_OPENINGS = os.path.join(HOME, "RubyData", "books", "UHO_Lichess_4852_v1.epd")
HERE = os.path.dirname(os.path.abspath(__file__))


def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    p = argparse.ArgumentParser(description="SPRT two Ruby builds against each other with fastchess.")
    p.add_argument("new", help="the build being tested")
    p.add_argument("base", help="the build it is compared with")
    p.add_argument("--tc", default="8+0.08", help="time control, seconds+increment (default 8+0.08)")
    p.add_argument("--concurrency", type=int, default=3, help="games at once (default 3)")
    p.add_argument("--games", type=int, default=20000, help="most games to play (default 20000)")
    p.add_argument("--elo0", type=float, default=0)
    p.add_argument("--elo1", type=float, default=5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--no-sprt", action="store_true", help="play --games games and report Elo only")
    p.add_argument("--openings", default=DEFAULT_OPENINGS, help=".epd or .pgn opening file")
    p.add_argument("--hash", type=int, default=16)
    p.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                   help="UCI option for the new build only (repeatable), e.g. a tunable search number")
    p.add_argument("--base-option", action="append", default=[], metavar="NAME=VALUE",
                   help="UCI option for the base build only (repeatable)")
    p.add_argument("--fastchess", default=DEFAULT_FASTCHESS)
    p.add_argument("--pgn", default=None, help="save games here (default: results/<new>_vs_<base>.pgn)")
    a = p.parse_args()

    for path, what in ((a.fastchess, "fastchess"), (a.openings, "opening file"), (a.new, "new build"),
                       (a.base, "base build")):
        if not os.path.exists(path):
            sys.exit(f"No {what} at {path}")
    new, base = os.path.abspath(a.new), os.path.abspath(a.base)
    new_name, base_name = name_of(new), name_of(base)
    if new_name == base_name:
        new_name, base_name = new_name + "_new", base_name + "_base"
    if a.pgn is None:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        a.pgn = os.path.join(HERE, "results", f"{new_name}_vs_{base_name}.pgn")

    fmt = "pgn" if a.openings.lower().endswith(".pgn") else "epd"
    cmd = [a.fastchess,
           "-engine", f"cmd={new}", f"name={new_name}", *[f"option.{o}" for o in a.option],
           "-engine", f"cmd={base}", f"name={base_name}", *[f"option.{o}" for o in a.base_option],
           "-each", f"tc={a.tc}", "option.OwnBook=false", "option.Tablebase=false", f"option.Hash={a.hash}",
           "-openings", f"file={a.openings}", f"format={fmt}", "order=random",
           "-games", "2", "-rounds", str((a.games + 1) // 2), "-repeat",
           "-concurrency", str(a.concurrency),
           "-recover", "-ratinginterval", "50",
           "-pgnout", f"file={a.pgn}"]
    if not a.no_sprt:
        cmd += ["-sprt", f"elo0={a.elo0}", f"elo1={a.elo1}", f"alpha={a.alpha}", f"beta={a.beta}",
                "model=logistic"]   # plain Elo, as elo_match.py uses
    print(" ".join(cmd), flush=True)
    # fastchess prints a line for every game start and finish; keep only the results and warnings.
    # "Bestmove does not match" is harmless: when time runs out partway through a depth, Ruby plays
    # the better move that depth already found, without printing a line for the unfinished depth.
    skip = ("Started game", "Finished game", "Moves;", "Info;", "Position;", "Warning; Bestmove does not match")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        try:
            for line in proc.stdout:
                if not line.startswith(skip):
                    print(line, end="", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
