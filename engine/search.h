// search.h - iterative deepening alpha-beta search.
#pragma once
#include "board.h"
#include <atomic>
#include <cstdint>

namespace search {

const int MATE = 30000;
const int MATE_BOUND = 29000;    // scores beyond this mean a forced mate was found
const int MAX_PLY = 128;

struct Limits {
    int depth = MAX_PLY - 1;     // stop after this depth
    int64_t softMs = -1;         // don't start a new depth after this long (-1 = no limit)
    int64_t hardMs = -1;         // abandon the search at this point (-1 = no limit)
    uint64_t nodes = 0;          // stop after this many nodes (0 = no limit)
};

struct Result {
    Move best = NO_MOVE;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
};

extern std::atomic<bool> stopRequested;

void setHashMb(int megabytes);
void clearHash();
// Prints UCI "info" lines as it goes when verbose is true.
Result think(Board& board, const Limits& limits, bool verbose);

// For the weaker levels: scores every legal move with a shallow search, then picks one at random,
// favouring better moves. A higher temperature (in centipawns) makes worse moves more likely.
Move pickWeak(Board& board, int depth, int temperature);

}  // namespace search
