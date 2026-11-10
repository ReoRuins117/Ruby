// board.h - chess position, move generation, make/unmake.
// Squares: a1 = 0, b1 = 1, ... h8 = 63. file = square & 7, rank = square >> 3.
// Each piece kind has a bitboard (bit n set = a piece on square n), and squares[] mirrors them
// so "what is on this square?" stays a single lookup.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum { WHITE = 0, BLACK = 1 };
enum { EMPTY = 0, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK };
enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8 };

using Bitboard = uint64_t;

inline int typeOf(int piece) { return piece ? (piece - 1) % 6 + 1 : 0; }
inline int colorOf(int piece) { return piece >= BP ? BLACK : WHITE; }
inline int makePiece(int color, int type) { return type + color * 6; }

inline Bitboard bit(int square) { return Bitboard(1) << square; }
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }
inline int popLsb(Bitboard& b) { int s = lsb(b); b &= b - 1; return s; }
inline int popCount(Bitboard b) { return __builtin_popcountll(b); }

// Attack lookups (filled in when the program starts).
namespace attacks {
extern Bitboard pawn[2][64], knight[64], king[64];
Bitboard bishop(int square, Bitboard occupied);
Bitboard rook(int square, Bitboard occupied);
inline Bitboard queen(int square, Bitboard occupied) { return bishop(square, occupied) | rook(square, occupied); }
}

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

// The network's first layer for one position, from both sides' views. Each move adds and removes
// at most two pieces, so a position's values are its parent's plus those few changes. They are
// worked out only when the position is evaluated (see nnue.cpp).
const int ACC_MAX = 512;
struct Accumulator {
    alignas(32) int16_t values[2][ACC_MAX];
    bool computed;
    uint8_t added, removed;                   // how many pieces this move added and removed
    uint8_t addPiece[2], addSquare[2], removePiece[2], removeSquare[2];
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
    Bitboard pieces[13];                      // one per piece code; pieces[EMPTY] is unused
    Bitboard colors[2];
    int side;
    int castling;
    int ep;              // en passant target square, or -1
    int halfmove;
    int fullmove;
    int king[2];
    uint64_t hash;

    // acc[accTop] belongs to the current position, acc[accTop - 1] to the one before, and so on.
    mutable std::vector<Accumulator> acc;
    int accTop = 0;

    Board();
    bool setFen(const std::string& fen);
    // Full FEN. En passant is shown only when a legal capture exists (as in the opening book).
    std::string fen(bool withCounters = true);

    Bitboard occupied() const { return colors[WHITE] | colors[BLACK]; }
    Bitboard byType(int color, int type) const { return pieces[makePiece(color, type)]; }

    void make(Move m);
    void unmake();
    void makeNull();
    void unmakeNull();

    bool attacked(int square, int bySide) const;
    Bitboard attackersTo(int square, Bitboard occupied) const;   // both colours
    bool inCheck() const { return attacked(king[side], side ^ 1); }
    bool sideJustMovedInCheck() const { return attacked(king[side ^ 1], side); }

    // Pseudo-legal moves: may leave the king in check, so test after make().
    // capturesOnly gives captures and all promotions.
    void generate(MoveList& list, bool capturesOnly = false) const;
    void legalMoves(MoveList& list);
    bool parseUci(const std::string& text, Move& out);
    bool isCapture(Move m) const { return squares[toSq(m)] != EMPTY || (toSq(m) == ep && typeOf(squares[fromSq(m)]) == PAWN); }
    // Static exchange evaluation: does this move win at least `threshold` centipawns once every
    // capture on its destination square has been played out?
    bool see(Move m, int threshold) const;

    bool isRepetition() const;
    bool insufficientMaterial() const;
    bool hasNonPawnMaterial(int color) const;
    int pieceCount() const { return popCount(occupied()); }
    Move lastMove() const { return history.empty() ? NO_MOVE : history.back().move; }

    uint64_t perft(int depth);

private:
    std::vector<Undo> history;
    bool tracking = false;                    // record piece changes for the accumulator (inside make)
    void pushAccumulator();
    void put(int square, int piece);
    void remove(int square);
};

extern const int SEE_VALUE[7];
