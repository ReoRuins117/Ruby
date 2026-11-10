// syzygy.h - local Syzygy endgame tablebases (set the folder with the SyzygyPath option).
#pragma once
#include "board.h"
#include <string>

namespace syzygy {
bool init(const std::string& path);      // folders separated by ';' on Windows; false if none found
int largest();                           // most pieces covered by the loaded tables, 0 if none
// Win/draw/loss for the side to move: 0 loss, 1 loss saved by the 50-move rule, 2 draw,
// 3 win spoiled by the 50-move rule, 4 win. -1 when it can't answer (only right after a capture
// or pawn move, without castling rights, with few enough pieces).
int probeWdl(const Board& board);
// The move that keeps the best result while respecting the 50-move rule.
bool probeRoot(Board& board, Move& out, std::string& category);
}
