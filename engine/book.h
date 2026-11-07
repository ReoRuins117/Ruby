// book.h - opening book read from Opening_Book.json.
#pragma once
#include "board.h"
#include <string>

namespace book {
bool load(const std::string& path);   // safe to call again: only the first success does any work
size_t size();
// Picks a book move, weighted by how often it was played. Returns false when out of book.
bool pick(Board& board, Move& out);
}
