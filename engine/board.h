// board.h - chess position, move generation, make/unmake.
// Squares: a1 = 0, b1 = 1, ... h8 = 63. file = square & 7, rank = square >> 3.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum { WHITE = 0, BLACK = 1 };
enum { EMPTY = 0, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK };
enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8 };

inline int typeOf(int piece) { return piece ? (piece - 1) % 6 + 1 : 0; }
inline int colorOf(int piece) { return piece >= BP ? BLACK : WHITE; }
inline int makePiece(int color, int type) { return type + color * 6; }

// A move packs from (6 bits), to (6 bits) and promotion piece type (rest; 0 = none).
using Move = uint16_t;
const Move NO_MOVE = 0;
inline Move makeMove(int from, int to, int promo = 0) { return Move(from | (to << 6) | (promo << 12)); }
inline int fromSq(Move m) { return m & 63; }
inline int toSq(Move m) { return (m >> 6) & 63; }
inline int promoOf(Move m) { return m >> 12; }
std::string moveToUci(Move m);

struct MoveList {
    Move moves[256];
    int size = 0;
    void add(Move m) { moves[size++] = m; }
};

struct Undo {
    uint64_t hash;
    Move move;
    int8_t captured, castling, ep;
    int16_t halfmove;
};

class Board {
public:
    int8_t squares[64];
    int side;
    int castling;
    int ep;              // en passant target square, or -1
    int halfmove;
    int fullmove;
    int king[2];
    uint64_t hash;

    Board();
    bool setFen(const std::string& fen);
    // Full FEN. En passant is shown only when a legal capture exists (as in the opening book).
    std::string fen(bool withCounters = true);

    void make(Move m);
    void unmake();
    void makeNull();
    void unmakeNull();

    bool attacked(int square, int bySide) const;
    bool inCheck() const { return attacked(king[side], side ^ 1); }
    bool sideJustMovedInCheck() const { return attacked(king[side ^ 1], side); }

    // Pseudo-legal moves: may leave the king in check, so test after make().
    void generate(MoveList& list, bool capturesOnly = false) const;
    void legalMoves(MoveList& list);
    bool parseUci(const std::string& text, Move& out);

    bool isRepetition() const;
    bool insufficientMaterial() const;
    bool hasNonPawnMaterial(int color) const;
    int pieceCount() const;

    uint64_t perft(int depth);

private:
    std::vector<Undo> history;
    void put(int square, int piece);
    void remove(int square);
    void addPawnMove(MoveList& list, int from, int to) const;
};
