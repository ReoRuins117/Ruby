# Ruby

A chess engine with a small NNUE evaluation network trained in Python (PyTorch) and a search
written in C++. It speaks UCI, so it works in chess GUIs and with lichess-bot.

## Strength

Measured at 60+1 against Stockfish 18 with its strength limiter on (Stockfish's own scale, not a
human rating):

| Opponent | Result | Score |
|---|---|---|
| Stockfish 2500 | +88 =9 -3 | 92.5% |
| Stockfish 2900 | +29 =52 -19 | 55.0% |
| Stockfish, full strength (1 thread) | +0 =2 -48 | 2.0% |

That puts Ruby at about 2935 on Stockfish's limited-strength scale.

## Files

| Path | What it is |
|---|---|
| `train.py` | Trains the network from `parsed/dataset` and exports `ruby_small.bin` |
| `Opening_Book.json` | Opening book, keyed by FEN |
| `engine/` | The C++ engine: board, move generation, search, NNUE evaluation, book, tablebase |
| `engine/ruby_small.bin` | Trained weights |
| `engine/build.bat` | Builds `ruby.exe` |
| `engine/tests/elo_match.py` | Matches against Stockfish, and `--play` to play Ruby yourself |
| `engine/tests/make_eval_vectors.py` | Makes `eval_vectors.txt` to check the C++ eval matches Python |

The training data is not included.

## Network

768 inputs (piece and square) feed a 256-wide feature layer, used once from the side to move's
view and once from the opponent's. The two halves are joined (512) and go through 32 -> 32 -> 1,
with clipped ReLU throughout. The output is in units of 400 centipawns. The weights are stored as
integers; the file format is documented at the top of `train.py`.

## Search

Iterative deepening with aspiration windows, principal variation search, transposition table,
null-move pruning, reverse futility pruning, late move reductions, killer and history move
ordering, MVV-LVA captures, quiescence search and check extensions.

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
python train.py
```

Checkpoints and the exported `ruby_small.bin` go to `~/RubyTraining/small_nnue`. See
`python train.py --help` for options.

## Testing

`ruby.exe` accepts these commands besides UCI:

| Command | Checks |
|---|---|
| `perfttest` | Move generation against known perft counts |
| `evaltest` | The C++ eval against `tests/eval_vectors.txt` |
| `updatetest` | Move-by-move eval updates against evals calculated from scratch |
| `booktest` | Opening book lookups |
| `tbtest` | Lichess tablebase lookups |
| `bench` | Search speed |

Matches against Stockfish (needs python-chess and a Stockfish executable):

```
python engine/tests/elo_match.py --elo 2500 --games 100
python engine/tests/elo_match.py --play
```

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 16 | Transposition table size in MB |
| `OwnBook` | true | Use `Opening_Book.json` |
| `Tablebase` | true | Use the Lichess tablebase API |
| `EvalFile` | `ruby_small.bin` next to `ruby.exe` | Weights file |
| `BookFile` | `Opening_Book.json` one folder above `ruby.exe` | Book file |
| `Level` | `uncle_sam` | Playing strength (see below) |

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
