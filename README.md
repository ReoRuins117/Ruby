# Ruby

A chess engine with a small NNUE evaluation network trained in Python (PyTorch) and a search
written in C++. It speaks UCI, so it works in chess GUIs and with lichess-bot.

## Files

| Path | What it is |
|---|---|
| `train.py` | Trains the network from shard folders and exports `ruby_small.bin` |
| `convert_binpack.py` | Converts Stockfish `.binpack` training data into shards for `train.py` |
| `Opening_Book.json` | Opening book, keyed by FEN |
| `engine/` | The C++ engine: board, move generation, search, NNUE evaluation, book, tablebase |
| `engine/ruby_small.bin` | Trained weights |
| `engine/build.bat` | Builds `ruby.exe` |
| `engine/tests/elo_match.py` | Matches against Stockfish, and `--play` to play Ruby yourself |
| `engine/tests/fast_sprt.py` | Tests one build against another with fastchess |
| `engine/tests/make_eval_vectors.py` | Makes `eval_vectors.txt` to check the C++ eval matches Python |

The training data is not included.

## Network

The current network (v3) has 8 king buckets: each side's own king position picks one of 8 sets
of 768 inputs (piece and square), 6144 in all. The board is mirrored left-right when that king is
on files e-h. They feed a 512-wide feature layer, used once from the side to move's view and once
from the opponent's. The two halves are joined (1024) and go through 32 -> 32 -> 1, with clipped
ReLU throughout. The output is in units of 400 centipawns. The weights are stored as integers;
the file format (version 1 without buckets, version 2 with) is documented at the top of
`train.py`. The engine reads both versions.

v3 was trained on Stockfish training data (Leela games rescored by Stockfish) and beat the
previous 256-wide network by about 26 Elo in an SPRT at 8+0.08. The current file is v3 fine-tuned
for 6 more passes at a lower learning rate, which beat v3 by about 19 Elo in the same test.

## Search

Iterative deepening with aspiration windows and principal variation search, run on several
threads that share a lock-free transposition table (Lazy SMP). The table also remembers which
positions have been on a principal variation, and those are reduced less.

- Pruning: razoring, reverse futility pruning, null-move pruning, ProbCut, futility and late move
  pruning, history pruning and SEE pruning of quiet moves and captures.
- Extensions and reductions: check extensions, singular extensions (with double extensions and
  multi-cut), late move reductions guided by history, and internal iterative reductions.
- Move ordering: hash move, SEE-split captures with capture history, killers, countermoves, and
  quiet history with continuation history from one and two plies back. Cutoffs that win by a
  wide margin update the history more.
- Quiescence search of captures, with delta pruning and SEE.

Ruby picks opening moves from `Opening_Book.json` and looks up endgames of 7 pieces or fewer in
the Lichess Syzygy tablebase API (`tablebase.lichess.ovh`) when it has enough time.

## Building

Needs g++ with C++17 (for example MinGW-w64 / WinLibs). From `engine/`:

```
build.bat
```

`-march=native` tunes the build for the CPU it is built on, so rebuild on each computer.

## Training

Needs Python with numpy and torch, and a GPU for reasonable speed.

```
python convert_binpack.py calibrate path/to/file.binpack
python convert_binpack.py convert path/to/file.binpack --scale <from calibrate> --out-dir path/to/shards
python train.py --buckets 8 --hidden 512 --wdl 0.25 --data path/to/shards --data more/shards
```

v3 used those settings, with `--pass-size 130000000 --epochs 24`. It was then fine-tuned with
`--init <v3 run>/best.pt --lr 3e-4 --epochs 6` (same other settings), which starts from a finished
run's weights with a fresh optimizer and schedule. `--data` can be given several
times; data is streamed from disk, so it can be bigger than memory, and folders are re-scanned
before each pass so a running conversion's new shards are picked up. Checkpoints and the exported
weights go to `--out` (default `~/RubyTraining/small_nnue`). See `python train.py --help` for all
options.

## Testing

`ruby.exe` accepts these commands besides UCI:

| Command | Checks |
|---|---|
| `perfttest` | Move generation against known perft counts |
| `evaltest` | The C++ eval against `tests/eval_vectors.txt` |
| `updatetest` | Move-by-move eval updates against evals calculated from scratch |
| `evalspeed [noeval]` | Times evals along a fixed random walk of moves, to compare networks |
| `booktest` | Opening book lookups |
| `tbtest` | Lichess tablebase lookups |
| `bench` | Search speed |
| `tunables` | Lists the tunable search numbers (name, int, value, min, max, step, rate) for SPSA |

Matches against Stockfish (needs python-chess and a Stockfish executable):

```
python engine/tests/elo_match.py --elo 2500 --games 100
python engine/tests/elo_match.py --play
```

To test a change, play the new build against the old one with an SPRT, which stops as soon as the
result is clear (here: is the change worth at least 5 Elo, or 0 or less?):

```
python engine/tests/elo_match.py --ruby C:/path/new_ruby.exe --vs-ruby C:/path/ruby.exe --tc 10+0.1 --sprt --elo0 0 --elo1 5
```

`--openings` takes openings from an `.epd` or `.pgn` file instead of `Opening_Book.json`, which only
has a few hundred distinct lines. The test sets from
[official-stockfish/books](https://github.com/official-stockfish/books) (CC0) work well:
`UHO_Lichess_4852_v1.epd` (unbalanced starting positions, so fewer draws) or `8moves_v3.pgn`
(balanced).

`fast_sprt.py` runs the same kind of test with [fastchess](https://github.com/Disservin/fastchess),
whose low overhead keeps short time controls such as 2+0.02 free of time losses, so tests can run
several times faster than at 8+0.08. Build fastchess from source (`mingw32-make`, then link with
`-static`) and put it on PATH or at `~/RubyData/tools/fastchess.exe`; openings default to
`~/RubyData/books/UHO_Lichess_4852_v1.epd`:

```
python engine/tests/fast_sprt.py engine/new_ruby.exe engine/ruby.exe --tc 2+0.02
python engine/tests/fast_sprt.py engine/ruby.exe engine/ruby.exe --option RfpMargin=90
```

fastchess may warn "Bestmove does not match beginning of last PV". That is expected: when time
runs out partway through a depth, Ruby plays the better move that depth has already found.

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 16 | Transposition table size in MB |
| `OwnBook` | true | Use `Opening_Book.json` |
| `Tablebase` | true | Use the Lichess tablebase API |
| `EvalFile` | `ruby_small.bin` next to `ruby.exe` | Weights file |
| `BookFile` | `Opening_Book.json` one folder above `ruby.exe` | Book file |
| `Level` | `uncle_sam` | Playing strength (see below) |
| `QsFutility`, `RfpMargin`, `LmrDiv`, ... | (tuned values) | The search's margins and reductions, for tuning with SPSA; `tunables` lists them all with their ranges |

## Strength levels

| Level | `elo_match.py` flag | Rough strength (Stockfish's scale) |
|---|---|---|
| `blunderbuss` | `--blunderbuss` | about 1000 |
| `furious_chicken` | `--furious_chicken` | about 1300 |
| `squidinator` | `--squidinator` | about 1900 |
| `uncle_sam` | `--uncle-sam` (default) | full strength |

The weaker levels score every move with a shallow search and pick one at random, favouring better
moves, and use less of the opening book. For example, `python engine/tests/elo_match.py --play --squidinator`.

## License

MIT. See `LICENSE`.
