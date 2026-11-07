// tablebase.h - asks the Lichess online tablebase for the best endgame move.
#pragma once
#include "board.h"
#include <string>

namespace tablebase {
// Looks up the position when it has 7 pieces or fewer and no castling rights.
// Returns false, so the engine searches normally, when offline or when the answer is unknown.
bool probe(Board& board, Move& out, std::string& category);
}
