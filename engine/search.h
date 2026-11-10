// search.h - iterative deepening alpha-beta search, on one or more threads sharing a hash table.
#pragma once
#include "board.h"
#include <atomic>
#include <cstdint>

namespace search {

const int MATE = 30000;
const int MATE_BOUND = 29000;    // scores beyond this mean a forced mate was found
const int TB_WIN = 28000;        // tablebase wins score TB_WIN - ply
const int TB_BOUND = 27000;      // scores beyond this are mates or tablebase results
const int MAX_PLY = 128;

struct Limits {
    int depth = MAX_PLY - 1;     // stop after this depth
    int64_t softMs = -1;         // aim to stop around here; adjusted by how settled the search is (-1 = no limit)
    int64_t hardMs = -1;         // abandon the search at this point (-1 = no limit)
    uint64_t nodes = 0;          // stop after this many nodes (0 = no limit)
};

struct Result {
    Move best = NO_MOVE;
    Move ponder = NO_MOVE;       // the reply we expect, for pondering
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
};

extern std::atomic<bool> stopRequested;
// While true, the search ignores its time limits (UCI "go ponder"); clear it with ponderHit().
extern std::atomic<bool> pondering;
void ponderHit();                // the opponent played the expected move: time limits start counting now

void setHashMb(int megabytes);
void clearHash();
void setThreads(int count);
int threads();
// Prints UCI "info" lines as it goes when verbose is true.
Result think(Board& board, const Limits& limits, bool verbose);

// For the weaker levels: scores every legal move with a shallow search, then picks one at random,
// favouring better moves. A higher temperature (in centipawns) makes worse moves more likely.
Move pickWeak(Board& board, int depth, int temperature);

}  // namespace search
