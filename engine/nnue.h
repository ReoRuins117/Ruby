// nnue.h - loads ruby_small.bin and evaluates positions with integer maths.
// The file format and formulas are documented at the top of train.py.
#pragma once
#include "board.h"
#include <string>

namespace nnue {
bool load(const std::string& path);    // returns false if the file is missing or the wrong size
bool loaded();
int evaluate(const Board& board);      // centipawns from the side to move's view
}
