// main.cpp - Ruby's UCI interface, plus commands for testing the engine.
//
// Test commands (type these after starting ruby.exe):
//   perfttest            check move generation against known correct counts
//   evaltest <file>      compare evals with the values Python calculated
//   updatetest           check move-by-move eval updates against evals calculated from scratch
//   booktest             show what the opening book plays from the start position
//   tbtest               ask the Lichess tablebase about a sample endgame
//   bench                search some positions and report speed
//   eval, d              show the current position's eval or FEN
#include "board.h"
#include "book.h"
#include "nnue.h"
#include "search.h"
#include "syzygy.h"
#include "tablebase.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

static Board board;
static std::thread worker;
static bool useBook = true, useTablebase = true;
static int64_t moveOverhead = 50;                   // spare milliseconds per move for GUI and network delays
static std::string syzygyPath = "<empty>";

// Strength levels, weakest first. The last plays at full strength; the others score every move with
// a shallow search and pick one at random, favouring better moves (see search::pickWeak).
struct Level { const char* name; int depth, temperature, bookPlies; };
static const Level LEVELS[] = {
    {"blunderbuss",     2,  90,   6},          // about 1000 on Stockfish's scale
    {"furious_chicken", 3,  55,   8},          // aiming for about 1300
    {"squidinator",     5,  25,  12},          // aiming for about 1800
    {"uncle_sam",       0,   0, 999},
};
static const int NUM_LEVELS = 4;
static int level = NUM_LEVELS - 1;
static std::string evalFile, bookFile;

static std::string exeFolder() {
#ifdef _WIN32
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string s(path);
    return s.substr(0, s.find_last_of("\\/") + 1);
#else
    return "./";
#endif
}

static void waitForSearch() {
    if (worker.joinable()) worker.join();
}

static void loadNetwork() {
    if (nnue::load(evalFile)) std::cout << "info string loaded network " << evalFile << std::endl;
    else std::cout << "info string ERROR could not load network " << evalFile << std::endl;
}

// ---------------------------------------------------------------- UCI commands
static void setPosition(std::istringstream& in) {
    std::string word;
    in >> word;
    if (word == "startpos") {
        board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        in >> word;                                 // "moves", if present
    } else if (word == "fen") {
        std::string fen;
        while (in >> word && word != "moves") fen += word + " ";
        if (!board.setFen(fen)) std::cout << "info string ERROR bad FEN" << std::endl;
    }
    while (in >> word) {
        Move m;
        if (!board.parseUci(word, m)) {
            std::cout << "info string ERROR illegal move " << word << std::endl;
            break;
        }
        board.make(m);
    }
}

static void go(std::istringstream& in) {
    search::Limits limits;
    int64_t time[2] = {-1, -1}, increment[2] = {0, 0}, moveTime = -1;
    int movesToGo = 0;
    bool infinite = false, ponder = false;
    std::string word;
    while (in >> word) {
        if (word == "ponder") ponder = true;
        if (word == "wtime") in >> time[WHITE];
        else if (word == "btime") in >> time[BLACK];
        else if (word == "winc") in >> increment[WHITE];
        else if (word == "binc") in >> increment[BLACK];
        else if (word == "movestogo") in >> movesToGo;
        else if (word == "movetime") in >> moveTime;
        else if (word == "depth") in >> limits.depth;
        else if (word == "nodes") in >> limits.nodes;
        else if (word == "infinite") infinite = true;
    }
    limits.depth = std::clamp(limits.depth, 1, search::MAX_PLY - 1);

    // Time: aim for about 1/35 of the clock plus most of the increment. The search stretches or
    // shrinks that by how settled it is, but never past the hard limit, which keeps a reserve.
    int64_t remaining = time[board.side], inc = increment[board.side];
    if (moveTime >= 0) {
        limits.hardMs = std::max<int64_t>(moveTime - moveOverhead, 1);   // a fixed time: use all of it
    } else if (remaining >= 0) {
        int64_t usable = std::max<int64_t>(remaining - moveOverhead, 1);
        int mtg = movesToGo > 0 ? std::min(movesToGo, 50) : 35;
        int64_t base = usable / mtg + inc * 3 / 4;
        if (usable < 3 * inc) base = usable / 4;           // nearly flagging: the increment is all we have
        limits.softMs = std::max<int64_t>(std::min(base, usable / 3), 1);
        limits.hardMs = std::max<int64_t>(std::min(base * 4, usable / (movesToGo == 1 ? 2 : 4)), limits.softMs);
    }

    search::stopRequested = false;
    search::pondering = ponder;
    worker = std::thread([limits, infinite, remaining, moveTime] {
        Move choice = NO_MOVE;
        const Level& lv = LEVELS[level];
        bool full = level == NUM_LEVELS - 1;
        int ply = (board.fullmove - 1) * 2 + board.side;
        // Book and tablebase are skipped for "go infinite", which means the user wants analysis.
        if (!infinite) {
            if (useBook && ply < lv.bookPlies && book::pick(board, choice)) {
                std::cout << "info string book move" << std::endl;
            } else if (!full) {
                choice = search::pickWeak(board, lv.depth, lv.temperature);
            } else {
                std::string category;
                Move tbMove;
                if (board.pieceCount() <= syzygy::largest() && syzygy::probeRoot(board, tbMove, category)) {
                    choice = tbMove;
                    std::cout << "info string syzygy says " << category << std::endl;
                } else if (useTablebase && (moveTime >= 0 ? moveTime : remaining) > 10000 && board.pieceCount() <= 7
                           && tablebase::probe(board, tbMove, category)) {
                    choice = tbMove;
                    std::cout << "info string online tablebase says " << category << std::endl;
                }
            }
        }
        Move ponderMove = NO_MOVE;
        if (choice == NO_MOVE) {
            search::Result r = search::think(board, limits, true);
            choice = r.best;
            ponderMove = r.ponder;
        }
        // UCI: while pondering or analysing, the answer waits for "ponderhit" or "stop".
        while ((infinite || search::pondering) && !search::stopRequested)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::string line = "bestmove " + (choice == NO_MOVE ? std::string("0000") : moveToUci(choice));
        if (ponderMove != NO_MOVE) line += " ponder " + moveToUci(ponderMove);
        std::cout << line << std::endl;
    });
}

static void setOption(std::istringstream& in) {
    std::string word, name, value;
    in >> word;                                     // "name"
    while (in >> word && word != "value") name += (name.empty() ? "" : " ") + word;
    while (in >> word) value += (value.empty() ? "" : " ") + word;
    if (name == "Hash") search::setHashMb(std::atoi(value.c_str()));
    else if (name == "Threads") search::setThreads(std::atoi(value.c_str()));
    else if (name == "MoveOverhead") moveOverhead = std::max(0, std::atoi(value.c_str()));
    else if (name == "Ponder") {}                   // nothing to set: "go ponder" says when to ponder
    else if (name == "SyzygyPath") {
        syzygyPath = value;
        if (syzygy::init(value)) std::cout << "info string syzygy tables up to " << syzygy::largest() << " pieces" << std::endl;
        else if (value != "<empty>" && !value.empty()) std::cout << "info string ERROR no syzygy tables in " << value << std::endl;
    }
    else if (name == "OwnBook") useBook = value == "true";
    else if (name == "Tablebase") useTablebase = value == "true";
    else if (name == "EvalFile") { evalFile = value; loadNetwork(); }
    else if (name == "BookFile") { bookFile = value; book::load(bookFile); }
    else if (name == "Level") {
        for (int i = 0; i < NUM_LEVELS; i++)
            if (value == LEVELS[i].name) level = i;
    }
}

// ---------------------------------------------------------------- test commands
static void perftTest() {
    struct Case { const char* fen; int depth; uint64_t nodes; };
    static const Case cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
    };
    int failed = 0;
    for (const Case& c : cases) {
        Board test;
        test.setFen(c.fen);
        uint64_t hashBefore = test.hash;
        uint64_t got = test.perft(c.depth);
        bool ok = got == c.nodes && test.hash == hashBefore && test.fen() == c.fen;
        failed += !ok;
        std::cout << (ok ? "PASS " : "FAIL ") << "depth " << c.depth << ": got " << got
                  << ", expected " << c.nodes << "  " << c.fen << std::endl;
    }
    std::cout << (failed ? "perfttest FAILED" : "perfttest passed") << std::endl;
}

static void evalTest(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cout << "Can't open " << path << std::endl; return; }
    std::string line;
    int total = 0, failed = 0, worst = 0;
    while (std::getline(in, line)) {
        size_t split = line.find(';');
        if (split == std::string::npos) continue;
        Board test;
        test.setFen(line.substr(split + 1));
        int expected = std::atoi(line.c_str()), got = nnue::evaluate(test);
        int gap = std::abs(expected - got);
        worst = std::max(worst, gap);
        total++;
        if (gap > 1 && ++failed <= 5)                // Python rounds; C++ truncates; 1 cp is expected
            std::cout << "FAIL got " << got << ", expected " << expected << ": " << line.substr(split + 1) << std::endl;
    }
    std::cout << "evaltest " << (failed || !total ? "FAILED" : "passed") << ": " << total
              << " positions, largest gap " << worst << " cp" << std::endl;
}

// Makes and unmakes random moves (and some null moves), evaluating only now and then so that
// updates are chained across several moves, and compares each eval with a fresh board's.
static void updateTest() {
    static const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    };
    std::mt19937 rng(12345);
    int checks = 0, failed = 0;
    for (const char* fen : fens) {
        Board test;
        test.setFen(fen);
        int made = 0;
        std::vector<bool> wasNull;
        for (int step = 0; step < 40000; step++) {
            MoveList list;
            test.legalMoves(list);
            bool goBack = made > 0 && (list.size == 0 || made >= 30 || rng() % 3 == 0);
            if (goBack) {
                if (wasNull.back()) test.unmakeNull(); else test.unmake();
                wasNull.pop_back();
                made--;
            } else if (!test.inCheck() && rng() % 10 == 0) {
                test.makeNull();
                wasNull.push_back(true);
                made++;
            } else if (list.size) {
                test.make(list.moves[rng() % unsigned(list.size)]);
                wasNull.push_back(false);
                made++;
            }
            if (rng() % 4 == 0) {
                Board fresh;
                fresh.setFen(test.fen());
                int got = nnue::evaluate(test), expected = nnue::evaluate(fresh);
                checks++;
                if (got != expected && ++failed <= 5)
                    std::cout << "FAIL got " << got << ", expected " << expected << ": " << test.fen() << std::endl;
            }
        }
    }
    std::cout << "updatetest " << (failed ? "FAILED" : "passed") << ": " << checks << " positions checked" << std::endl;
}

static void bench() {
    static const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    };
    uint64_t nodes = 0;
    auto start = std::chrono::steady_clock::now();
    for (const char* fen : fens) {
        Board test;
        test.setFen(fen);
        search::Limits limits;
        limits.depth = 9;
        search::stopRequested = false;
        search::Result r = search::think(test, limits, false);
        nodes += r.nodes;
        std::cout << "depth " << r.depth << " best " << moveToUci(r.best) << " score " << r.score
                  << " nodes " << r.nodes << "  " << fen << std::endl;
    }
    int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << nodes << " nodes in " << ms << " ms = " << nodes * 1000 / uint64_t(std::max<int64_t>(ms, 1)) << " nodes/s" << std::endl;
}

static void bookTest() {
    book::load(bookFile);
    std::cout << book::size() << " book positions" << std::endl;
    Board test;
    for (int i = 0; i < 8; i++) {
        Move m;
        if (book::pick(test, m)) std::cout << moveToUci(m) << " ";
        else std::cout << "(out of book)";
    }
    std::cout << std::endl;
}

static void tablebaseTest() {
    Board test;
    test.setFen("4k3/6KP/8/8/8/8/7p/8 w - - 0 1");
    Move m;
    std::string category;
    if (tablebase::probe(test, m, category)) std::cout << moveToUci(m) << " (" << category << "), expected h7h8q (win)" << std::endl;
    else std::cout << "No tablebase answer (offline?)" << std::endl;
}

// ---------------------------------------------------------------- main loop
int main() {
    std::string folder = exeFolder();
    evalFile = folder + "ruby_small.bin";
    bookFile = folder + "..\\Opening_Book.json";
    nnue::load(evalFile);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "uci") {
            std::cout << "id name Ruby\nid author Joshua\n"
                      << "option name Hash type spin default 16 min 1 max 16384\n"
                      << "option name Threads type spin default 1 min 1 max 64\n"
                      << "option name Ponder type check default false\n"
                      << "option name MoveOverhead type spin default 50 min 0 max 5000\n"
                      << "option name SyzygyPath type string default <empty>\n"
                      << "option name OwnBook type check default true\n"
                      << "option name Tablebase type check default true\n"
                      << "option name EvalFile type string default " << evalFile << "\n"
                      << "option name BookFile type string default " << bookFile << "\n"
                      << "option name Level type combo default uncle_sam"
                      << " var blunderbuss var furious_chicken var squidinator var uncle_sam\n"
                      << "uciok" << std::endl;
        } else if (cmd == "isready") {
            waitForSearch();
            if (!nnue::loaded()) loadNetwork();
            if (useBook && !book::size()) book::load(bookFile);
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            waitForSearch();
            search::clearHash();
        } else if (cmd == "position") {
            waitForSearch();
            setPosition(in);
        } else if (cmd == "go") {
            waitForSearch();
            if (!nnue::loaded()) { std::cout << "info string ERROR no network loaded" << std::endl; continue; }
            if (useBook && !book::size()) book::load(bookFile);
            go(in);
        } else if (cmd == "ponderhit") {
            search::ponderHit();
        } else if (cmd == "stop") {
            search::stopRequested = true;
            search::pondering = false;
            waitForSearch();
        } else if (cmd == "setoption") {
            waitForSearch();
            setOption(in);
        } else if (cmd == "quit") {
            search::stopRequested = true;
            search::pondering = false;
            break;
        } else if (cmd == "perfttest") { perftTest();
        } else if (cmd == "evaltest") {
            std::string path = folder + "tests\\eval_vectors.txt";
            in >> path;
            evalTest(path);
        } else if (cmd == "updatetest") { updateTest();
        } else if (cmd == "bench") { bench();
        } else if (cmd == "booktest") { bookTest();
        } else if (cmd == "tbtest") { tablebaseTest();
        } else if (cmd == "eval") { std::cout << nnue::evaluate(board) << " cp" << std::endl;
        } else if (cmd == "d") { std::cout << board.fen() << std::endl;
        }
    }
    waitForSearch();
    return 0;
}
