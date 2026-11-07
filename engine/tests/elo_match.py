#!/usr/bin/env python3
"""
elo_match.py - estimate Ruby's strength by playing Stockfish limited to a chosen Elo.

  python elo_match.py --elo 1800 --games 200
  python elo_match.py --elo 2000 --games 200
  python elo_match.py --full --games 50         # no strength limit on Stockfish

Stop with Ctrl+C at any time; the result so far is still printed.

  python elo_match.py --play                     # play Ruby yourself, in the terminal
  python elo_match.py --play --color white --think 10
  python elo_match.py --play --squidinator       # play a weaker Ruby

Ruby's strength levels, weakest first: --blunderbuss, --furious_chicken, --squidinator and
--uncle-sam (full strength, the default). They work in --play mode and in matches.

In --play mode Ruby uses its own opening book and the Lichess tablebase, and spends --think seconds
per move. Your clock isn't timed.

Each opening is played twice with colours swapped, so neither side gets a luckier set of positions.
Openings come from Opening_Book.json. Ruby's own book is turned off because the shared opening
replaces it. Its tablebase is off unless you pass --tablebase, because a long match would send
Lichess a lot of requests. Needs python-chess.
"""
import argparse, datetime, math, os, queue, random, threading, time
import chess, chess.engine, chess.pgn
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(ENGINE_DIR)
DEFAULT_RUBY = os.path.join(ENGINE_DIR, "ruby.exe")
DEFAULT_STOCKFISH = os.path.join(ROOT, "Stockfish", "stockfish-windows-x86-64-avx2.exe")
DEFAULT_BOOK = os.path.join(ROOT, "Opening_Book.json")


def make_openings(book, count, plies, rng):
    """Random book lines, favouring moves that were played often."""
    openings, seen = [], set()
    while len(openings) < count:
        board = chess.Board()
        for _ in range(plies):
            entry = book.get(board.fen())
            if not entry:
                break
            total = sum(entry.values())
            common = [(m, n) for m, n in entry.items() if n >= 0.10 * total]
            if total < 10 or not common:
                break
            board.push_uci(rng.choices([m for m, _ in common], [n for _, n in common])[0])
        key = " ".join(m.uci() for m in board.move_stack)
        if key not in seen or len(seen) > 50000:
            seen.add(key)
            openings.append(list(board.move_stack))
    return openings


def parse_tc(text):
    base, _, inc = text.partition("+")
    return float(base), float(inc or 0)


def start_engines(a):
    ruby = chess.engine.SimpleEngine.popen_uci(a.ruby)
    ruby.configure({"OwnBook": False, "Tablebase": a.tablebase, "Hash": a.hash, "Level": a.level})
    stockfish = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    limit = {} if a.full else {"UCI_LimitStrength": True, "UCI_Elo": a.elo}
    stockfish.configure({**limit, "Threads": a.sf_threads, "Hash": a.sf_hash or a.hash})
    return ruby, stockfish


def stop_engines(*engines):
    for engine in engines:
        try:
            engine.quit()
        except Exception:
            try:
                engine.close()
            except Exception:
                pass


def play_game(a, ruby, stockfish, opening, ruby_white):
    """Returns (result from Ruby's view: 1, 0.5 or 0; how it ended; the PGN game)."""
    base, inc = parse_tc(a.tc)
    board = chess.Board()
    for move in opening:
        board.push(move)
    clock = {chess.WHITE: base, chess.BLACK: base}
    players = {chess.WHITE: ruby if ruby_white else stockfish, chess.BLACK: stockfish if ruby_white else ruby}
    game_id = object()
    reason, winner = None, None

    while reason is None:
        outcome = board.outcome(claim_draw=True)
        if outcome:
            reason, winner = outcome.termination.name.lower(), outcome.winner
            break
        mover = board.turn
        limit = chess.engine.Limit(white_clock=clock[chess.WHITE], black_clock=clock[chess.BLACK],
                                   white_inc=inc, black_inc=inc)
        started = time.perf_counter()
        move = players[mover].play(board, limit, game=game_id).move
        clock[mover] -= time.perf_counter() - started
        if clock[mover] <= 0:
            reason, winner = "lost_on_time", not mover
            break
        clock[mover] += inc
        board.push(move)

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = a.ruby_name if ruby_white else f"Stockfish {a.label}"
    game.headers["Black"] = f"Stockfish {a.label}" if ruby_white else a.ruby_name
    game.headers["Result"] = "1/2-1/2" if winner is None else ("1-0" if winner == chess.WHITE else "0-1")
    game.headers["TimeControl"] = a.tc
    game.headers["Termination"] = reason
    ruby_colour = chess.WHITE if ruby_white else chess.BLACK
    score = 0.5 if winner is None else float(winner == ruby_colour)
    return score, reason, game


class Tally:
    def __init__(self, a):
        self.a, self.wins, self.draws, self.losses = a, 0, 0, 0
        self.reasons, self.lock = {}, threading.Lock()

    @property
    def games(self):
        return self.wins + self.draws + self.losses

    def add(self, score, reason, game, pgn_path):
        with self.lock:
            self.wins += score == 1
            self.draws += score == 0.5
            self.losses += score == 0
            label = reason if not (score == 0 and reason == "lost_on_time") else "RUBY_lost_on_time"
            self.reasons[label] = self.reasons.get(label, 0) + 1
            with open(pgn_path, "a", encoding="utf-8") as fh:
                print(game, file=fh, end="\n\n")
            print(f"game {self.games}/{self.a.games}: {'win' if score == 1 else 'draw' if score else 'loss'} "
                  f"({reason}) | +{self.wins} ={self.draws} -{self.losses}", flush=True)

    def summary(self):
        n = self.games
        if not n:
            return "No games finished."
        score = (self.wins + 0.5 * self.draws) / n
        lines = [f"\n{self.a.ruby_name} vs Stockfish {self.a.label}: +{self.wins} ={self.draws} -{self.losses} "
                 f"in {n} games, scoring {100 * score:.1f}%"]
        if self.wins == n or self.losses == n:
            side = "above" if self.wins == n else "below"
            lines.append(f"A clean sweep can't give an estimate: Ruby is well {side} this level."
                         + ("" if self.a.full else " Try another --elo."))
        elif score in (0.0, 1.0):
            lines.append("No estimate available.")
        else:
            elo = -400 * math.log10(1 / score - 1)
            variance = (self.wins * (1 - score) ** 2 + self.draws * (0.5 - score) ** 2 + self.losses * score ** 2) / n
            sd = math.sqrt(variance / n)
            low, high = max(score - 1.96 * sd, 0.001), min(score + 1.96 * sd, 0.999)
            to_elo = lambda s: -400 * math.log10(1 / s - 1)
            lines.append(f"Estimated difference {elo:+.0f} Elo (95% range {to_elo(low):+.0f} to {to_elo(high):+.0f})")
            if not self.a.full:
                lines.append(f"Estimated Ruby strength at this time control: about {self.a.elo + elo:.0f} on Stockfish's scale")
        lines.append(f"How games ended: {self.reasons}")
        return "\n".join(lines)


def worker(a, jobs, tally, pgn_path, errors, stopping):
    ruby, stockfish = start_engines(a)
    try:
        while not stopping.is_set():
            try:
                opening, ruby_white = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                score, reason, game = play_game(a, ruby, stockfish, opening, ruby_white)
                tally.add(score, reason, game, pgn_path)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError) as e:
                errors.append(repr(e))                         # a crash or illegal move: report it, carry on
                print(f"ENGINE ERROR, game not counted: {e!r}", flush=True)
                stop_engines(ruby, stockfish)
                ruby, stockfish = start_engines(a)
    finally:
        stop_engines(ruby, stockfish)


def show_board(board, human):
    """Print the board with your side at the bottom. White is uppercase, Black lowercase."""
    ranks = range(7, -1, -1) if human == chess.WHITE else range(8)
    files = range(8) if human == chess.WHITE else range(7, -1, -1)
    print()
    for rank in ranks:
        cells = (board.piece_at(chess.square(file, rank)) for file in files)
        print(f"{rank + 1}  " + " ".join(piece.symbol() if piece else "." for piece in cells))
    print("   " + " ".join(chess.FILE_NAMES[file] for file in files) + "\n")


def read_human_move(board, text):
    """Accepts SAN (Nf3) or UCI (g1f3). Returns None if the text isn't a legal move."""
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            pass
    return None


PLAY_HELP = ("Enter moves like Nf3, e4, O-O or g1f3. Other commands:\n"
             "  moves   list your legal moves\n"
             "  undo    take back your last move\n"
             "  resign  give up this game\n"
             "  quit    stop without a result\n"
             "  help    show this list")


def play_human(a):
    human = {"white": chess.WHITE, "black": chess.BLACK}.get(a.color, random.choice([chess.WHITE, chess.BLACK]))
    ruby = chess.engine.SimpleEngine.popen_uci(a.ruby)
    ruby.configure({"OwnBook": True, "Tablebase": True, "Hash": max(a.hash, 64), "Level": a.level})
    board, game_id, result, reason = chess.Board(), object(), "*", "unfinished"
    print(f"You are {'White' if human == chess.WHITE else 'Black'}. {a.ruby_name} thinks for up to {a.think:g} s per move.\n{PLAY_HELP}")
    try:
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome:
                result, reason = outcome.result(), outcome.termination.name.lower()
                break
            if board.turn != human:
                played = ruby.play(board, chess.engine.Limit(time=a.think), info=chess.engine.INFO_SCORE, game=game_id)
                score = played.info.get("score")
                note = (f"  (Ruby's eval: {score.pov(not human)})" if score
                        else "" if a.level != "uncle_sam" else "  (book or tablebase move)")
                print(f"\nRuby plays {board.san(played.move)}{note}")
                board.push(played.move)
                continue

            show_board(board, human)
            text = input("Your move: ").strip()
            command = text.lower()
            if command == "quit":
                break
            if command == "resign":
                result, reason = ("0-1" if human == chess.WHITE else "1-0"), "resignation"
                break
            if command == "help":
                print(PLAY_HELP)
            elif command == "moves":
                print(" ".join(sorted(board.san(m) for m in board.legal_moves)))
            elif command == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                else:
                    print("Nothing to undo yet.")
            elif text:
                move = read_human_move(board, text)
                if move is None:
                    print(f"'{text}' isn't a legal move here. Type moves to list them.")
                else:
                    board.push(move)
    except (KeyboardInterrupt, EOFError):
        print("\nGame stopped.")
    finally:
        stop_engines(ruby)

    show_board(board, human)
    print(f"Result: {result} ({reason})")
    if board.move_stack:
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "You" if human == chess.WHITE else a.ruby_name
        game.headers["Black"] = a.ruby_name if human == chess.WHITE else "You"
        game.headers["Result"] = result
        game.headers["Termination"] = reason
        if a.pgn is None:
            os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            a.pgn = os.path.join(HERE, "results", f"you_vs_{a.ruby_name.lower().replace(' ', '_')}_{stamp}.pgn")
        with open(a.pgn, "a", encoding="utf-8") as fh:
            print(game, file=fh, end="\n\n")
        print(f"Game saved to {a.pgn}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--play", action="store_true", help="play Ruby yourself instead of running a match")
    p.add_argument("--color", choices=["white", "black", "random"], default="random", help="your side in --play")
    p.add_argument("--think", type=float, default=5.0, help="Ruby's seconds per move in --play")
    levels = p.add_mutually_exclusive_group()
    for flag, name, text in [("--blunderbuss", "blunderbuss", "weakest"),
                             ("--furious_chicken", "furious_chicken", "second weakest"),
                             ("--squidinator", "squidinator", "second strongest"),
                             ("--uncle-sam", "uncle_sam", "full strength (default)")]:
        levels.add_argument(flag, dest="level", action="store_const", const=name, help=f"Ruby's {text} level")
    p.set_defaults(level="uncle_sam")
    p.add_argument("--elo", type=int, default=1800, help="Stockfish's limited strength (1320 or higher)")
    p.add_argument("--full", action="store_true", help="play Stockfish at full strength, ignoring --elo")
    p.add_argument("--games", type=int, default=200, help="total games, rounded up to an even number")
    p.add_argument("--tc", default="60+1", help="time control as seconds+increment")
    p.add_argument("--concurrency", type=int, default=max((os.cpu_count() or 2) // 2, 1), help="games at once")
    p.add_argument("--opening-plies", type=int, default=8, help="book moves played before the engines take over")
    p.add_argument("--hash", type=int, default=16, help="hash MB per engine")
    p.add_argument("--sf-threads", type=int, default=1, help="Stockfish threads per game")
    p.add_argument("--sf-hash", type=int, default=0, help="Stockfish hash MB (default: same as --hash)")
    p.add_argument("--tablebase", action="store_true", help="let Ruby use the Lichess tablebase")
    p.add_argument("--seed", type=int, default=1, help="choose a different value for different openings")
    p.add_argument("--ruby", default=DEFAULT_RUBY)
    p.add_argument("--stockfish", default=DEFAULT_STOCKFISH)
    p.add_argument("--book", default=DEFAULT_BOOK)
    p.add_argument("--pgn", default=None, help="where to save games (default: a new file in tests/results)")
    a = p.parse_args()
    a.ruby_name = "Ruby" if a.level == "uncle_sam" else f"Ruby {a.level}"
    a.label = "full" if a.full else str(a.elo)

    if a.play:
        play_human(a)
        return

    with open(a.book) as fh:
        book = json.load(fh)
    pairs = (a.games + 1) // 2
    a.games = pairs * 2
    openings = make_openings(book, pairs, a.opening_plies, random.Random(a.seed))
    del book
    jobs = queue.Queue()
    for opening in openings:
        jobs.put((opening, True))
        jobs.put((opening, False))

    if a.pgn is None:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        a.pgn = os.path.join(HERE, "results", f"{a.ruby_name.lower().replace(' ', '_')}_vs_sf{a.label}_{stamp}.pgn")

    print(f"{a.games} games at {a.tc} against Stockfish {a.label} ({a.sf_threads} threads, {a.sf_hash or a.hash} MB hash), {a.concurrency} at once.\nGames save to {a.pgn}", flush=True)
    tally, errors, stopping = Tally(a), [], threading.Event()
    threads = [threading.Thread(target=worker, args=(a, jobs, tally, a.pgn, errors, stopping), daemon=True)
               for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping after the current games. Press Ctrl+C again to quit now.")
        stopping.set()
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    print(tally.summary())
    if errors:
        print(f"{len(errors)} ENGINE ERRORS, first: {errors[0]}")


if __name__ == "__main__":
    main()
