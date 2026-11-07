// board.cpp - see board.h
#include "board.h"
#include <cstdlib>
#include <cstring>
#include <sstream>

// ---------------------------------------------------------------- lookup tables
static int8_t knightTargets[64][9], kingTargets[64][9];   // each list ends with -1
static int8_t rayTargets[64][8][8];
static int8_t rayLength[64][8];
static int castleMask[64];
static uint64_t zPiece[13][64], zCastle[16], zEp[8], zSide;

static uint64_t nextRandom(uint64_t& x) {                 // splitmix64: fixed seed, same keys every run
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void buildSteps(int8_t table[64][9], const int (*steps)[2]) {
    for (int sq = 0; sq < 64; sq++) {
        int n = 0;
        for (int i = 0; i < 8; i++) {
            int f = (sq & 7) + steps[i][0], r = (sq >> 3) + steps[i][1];
            if (f >= 0 && f < 8 && r >= 0 && r < 8) table[sq][n++] = int8_t(r * 8 + f);
        }
        table[sq][n] = -1;
    }
}

namespace {
struct TableSetup {
    TableSetup() {
        static const int knight[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        static const int kingMoves[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        buildSteps(knightTargets, knight);
        buildSteps(kingTargets, kingMoves);
        for (int sq = 0; sq < 64; sq++)
            for (int d = 0; d < 8; d++) {                 // directions 0-3 are rook lines, 4-7 bishop lines
                int f = sq & 7, r = sq >> 3, n = 0;
                for (;;) {
                    f += kingMoves[d][0];
                    r += kingMoves[d][1];
                    if (f < 0 || f > 7 || r < 0 || r > 7) break;
                    rayTargets[sq][d][n++] = int8_t(r * 8 + f);
                }
                rayLength[sq][d] = int8_t(n);
            }
        for (int& m : castleMask) m = 15;
        castleMask[0] = 15 & ~CASTLE_WQ;                  // a1
        castleMask[7] = 15 & ~CASTLE_WK;                  // h1
        castleMask[4] = 15 & ~(CASTLE_WK | CASTLE_WQ);    // e1
        castleMask[56] = 15 & ~CASTLE_BQ;                 // a8
        castleMask[63] = 15 & ~CASTLE_BK;                 // h8
        castleMask[60] = 15 & ~(CASTLE_BK | CASTLE_BQ);   // e8
        uint64_t seed = 20240607;
        for (auto& row : zPiece) for (auto& z : row) z = nextRandom(seed);
        for (auto& z : zCastle) z = nextRandom(seed);
        for (auto& z : zEp) z = nextRandom(seed);
        zSide = nextRandom(seed);
    }
} tableSetup;
}

std::string moveToUci(Move m) {
    std::string s;
    s += char('a' + (fromSq(m) & 7));
    s += char('1' + (fromSq(m) >> 3));
    s += char('a' + (toSq(m) & 7));
    s += char('1' + (toSq(m) >> 3));
    if (promoOf(m)) s += " nbrq"[promoOf(m) - 1];
    return s;
}

// ---------------------------------------------------------------- setup
Board::Board() { setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

void Board::put(int square, int piece) {
    squares[square] = int8_t(piece);
    hash ^= zPiece[piece][square];
}

void Board::remove(int square) {
    hash ^= zPiece[squares[square]][square];
    squares[square] = EMPTY;
}

bool Board::setFen(const std::string& text) {
    std::istringstream in(text);
    std::string placement, turn, rights = "-", epText = "-";
    int half = 0, full = 1;
    in >> placement >> turn >> rights >> epText >> half >> full;
    if (placement.empty() || (turn != "w" && turn != "b")) return false;

    std::memset(squares, 0, sizeof(squares));
    history.clear();
    hash = 0;
    king[WHITE] = king[BLACK] = -1;
    int file = 0, rank = 7;
    for (char c : placement) {
        if (c == '/') { rank--; file = 0; }
        else if (c >= '1' && c <= '8') file += c - '0';
        else {
            const char* names = "PNBRQKpnbrqk";
            const char* found = std::strchr(names, c);
            if (!found || file > 7 || rank < 0) return false;
            int piece = int(found - names) + 1, square = rank * 8 + file++;
            put(square, piece);
            if (typeOf(piece) == KING) king[colorOf(piece)] = square;
        }
    }
    if (king[WHITE] < 0 || king[BLACK] < 0) return false;

    side = turn == "w" ? WHITE : BLACK;
    if (side == BLACK) hash ^= zSide;
    castling = 0;
    for (char c : rights) {
        if (c == 'K') castling |= CASTLE_WK;
        if (c == 'Q') castling |= CASTLE_WQ;
        if (c == 'k') castling |= CASTLE_BK;
        if (c == 'q') castling |= CASTLE_BQ;
    }
    hash ^= zCastle[castling];
    ep = -1;
    if (epText.size() == 2) {
        ep = (epText[1] - '1') * 8 + (epText[0] - 'a');
        hash ^= zEp[ep & 7];
    }
    halfmove = half;
    fullmove = full;
    return true;
}

std::string Board::fen(bool withCounters) {
    std::string s;
    for (int rank = 7; rank >= 0; rank--) {
        int gap = 0;
        for (int file = 0; file < 8; file++) {
            int piece = squares[rank * 8 + file];
            if (!piece) { gap++; continue; }
            if (gap) { s += char('0' + gap); gap = 0; }
            s += " PNBRQKpnbrqk"[piece];
        }
        if (gap) s += char('0' + gap);
        if (rank) s += '/';
    }
    s += side == WHITE ? " w " : " b ";
    std::string rights;
    if (castling & CASTLE_WK) rights += 'K';
    if (castling & CASTLE_WQ) rights += 'Q';
    if (castling & CASTLE_BK) rights += 'k';
    if (castling & CASTLE_BQ) rights += 'q';
    s += rights.empty() ? "-" : rights;

    bool epLegal = false;
    if (ep >= 0) {
        MoveList list;
        legalMoves(list);
        for (int i = 0; i < list.size; i++)
            if (toSq(list.moves[i]) == ep && typeOf(squares[fromSq(list.moves[i])]) == PAWN) epLegal = true;
    }
    s += ' ';
    if (epLegal) {
        s += char('a' + (ep & 7));
        s += char('1' + (ep >> 3));
    } else {
        s += '-';
    }
    if (withCounters) s += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
    return s;
}

// ---------------------------------------------------------------- make / unmake
void Board::make(Move m) {
    int from = fromSq(m), to = toSq(m), promo = promoOf(m);
    int piece = squares[from], type = typeOf(piece);
    Undo u;
    u.hash = hash;
    u.move = m;
    u.captured = squares[to];
    u.castling = int8_t(castling);
    u.ep = int8_t(ep);
    u.halfmove = int16_t(halfmove);

    if (ep >= 0) hash ^= zEp[ep & 7];
    int oldEp = ep;
    ep = -1;
    halfmove++;
    if (type == PAWN) halfmove = 0;

    int captureSquare = to;
    if (type == PAWN && to == oldEp) {                    // en passant: the pawn sits beside the target
        captureSquare = to + (side == WHITE ? -8 : 8);
        u.captured = squares[captureSquare];
    }
    if (u.captured) {
        remove(captureSquare);
        halfmove = 0;
    }
    remove(from);
    put(to, promo ? makePiece(side, promo) : piece);

    if (type == KING) {
        king[side] = to;
        if (to - from == 2) { int rook = squares[to + 1]; remove(to + 1); put(to - 1, rook); }
        if (from - to == 2) { int rook = squares[to - 2]; remove(to - 2); put(to + 1, rook); }
    } else if (type == PAWN && std::abs(to - from) == 16) {
        ep = (from + to) / 2;
        hash ^= zEp[ep & 7];
    }

    hash ^= zCastle[castling];
    castling &= castleMask[from] & castleMask[to];
    hash ^= zCastle[castling];

    if (side == BLACK) fullmove++;
    side ^= 1;
    hash ^= zSide;
    history.push_back(u);
}

void Board::unmake() {
    Undo u = history.back();
    history.pop_back();
    side ^= 1;
    if (side == BLACK) fullmove--;

    int from = fromSq(u.move), to = toSq(u.move);
    int piece = squares[to];
    if (promoOf(u.move)) piece = makePiece(side, PAWN);
    squares[to] = EMPTY;
    squares[from] = int8_t(piece);
    int type = typeOf(piece);

    if (type == PAWN && to == u.ep) {
        squares[to + (side == WHITE ? -8 : 8)] = u.captured;
    } else {
        squares[to] = u.captured;
    }
    if (type == KING) {
        king[side] = from;
        if (to - from == 2) { squares[to + 1] = squares[to - 1]; squares[to - 1] = EMPTY; }
        if (from - to == 2) { squares[to - 2] = squares[to + 1]; squares[to + 1] = EMPTY; }
    }
    castling = u.castling;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
}

void Board::makeNull() {
    Undo u;
    u.hash = hash;
    u.move = NO_MOVE;
    u.captured = EMPTY;
    u.castling = int8_t(castling);
    u.ep = int8_t(ep);
    u.halfmove = int16_t(halfmove);
    history.push_back(u);
    if (ep >= 0) hash ^= zEp[ep & 7];
    ep = -1;
    halfmove = 0;                                         // repetition checks stop at a null move
    side ^= 1;
    hash ^= zSide;
}

void Board::unmakeNull() {
    Undo u = history.back();
    history.pop_back();
    side ^= 1;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
}

// ---------------------------------------------------------------- attacks and moves
bool Board::attacked(int sq, int by) const {
    int file = sq & 7, rank = sq >> 3;
    if (by == WHITE) {
        if (rank > 0 && file > 0 && squares[sq - 9] == WP) return true;
        if (rank > 0 && file < 7 && squares[sq - 7] == WP) return true;
    } else {
        if (rank < 7 && file > 0 && squares[sq + 7] == BP) return true;
        if (rank < 7 && file < 7 && squares[sq + 9] == BP) return true;
    }
    int theirKnight = makePiece(by, KNIGHT), theirKing = makePiece(by, KING);
    for (const int8_t* t = knightTargets[sq]; *t >= 0; t++) if (squares[*t] == theirKnight) return true;
    for (const int8_t* t = kingTargets[sq]; *t >= 0; t++) if (squares[*t] == theirKing) return true;

    int queen = makePiece(by, QUEEN);
    for (int d = 0; d < 8; d++) {
        int slider = makePiece(by, d < 4 ? ROOK : BISHOP);
        for (int i = 0; i < rayLength[sq][d]; i++) {
            int piece = squares[rayTargets[sq][d][i]];
            if (!piece) continue;
            if (piece == slider || piece == queen) return true;
            break;
        }
    }
    return false;
}

void Board::addPawnMove(MoveList& list, int from, int to) const {
    if ((to >> 3) == 0 || (to >> 3) == 7) {
        list.add(makeMove(from, to, QUEEN));
        list.add(makeMove(from, to, ROOK));
        list.add(makeMove(from, to, BISHOP));
        list.add(makeMove(from, to, KNIGHT));
    } else {
        list.add(makeMove(from, to));
    }
}

void Board::generate(MoveList& list, bool capturesOnly) const {
    int enemy = side ^ 1;
    for (int sq = 0; sq < 64; sq++) {
        int piece = squares[sq];
        if (!piece || colorOf(piece) != side) continue;
        int type = typeOf(piece);

        if (type == PAWN) {
            int forward = side == WHITE ? 8 : -8, file = sq & 7, rank = sq >> 3;
            int one = sq + forward;
            bool promotes = (one >> 3) == 0 || (one >> 3) == 7;
            if (!squares[one] && (!capturesOnly || promotes)) {
                addPawnMove(list, sq, one);
                int two = one + forward;
                if (rank == (side == WHITE ? 1 : 6) && !squares[two]) list.add(makeMove(sq, two));
            }
            for (int delta = -1; delta <= 1; delta += 2) {
                if ((delta < 0 && file == 0) || (delta > 0 && file == 7)) continue;
                int to = one + delta;
                if (to == ep || (squares[to] && colorOf(squares[to]) == enemy)) addPawnMove(list, sq, to);
            }
        } else if (type == KNIGHT || type == KING) {
            for (const int8_t* t = (type == KNIGHT ? knightTargets : kingTargets)[sq]; *t >= 0; t++) {
                int target = squares[*t];
                if (target ? colorOf(target) == enemy : !capturesOnly) list.add(makeMove(sq, *t));
            }
        } else {
            int first = type == BISHOP ? 4 : 0, last = type == ROOK ? 4 : 8;
            for (int d = first; d < last; d++)
                for (int i = 0; i < rayLength[sq][d]; i++) {
                    int to = rayTargets[sq][d][i], target = squares[to];
                    if (target) {
                        if (colorOf(target) == enemy) list.add(makeMove(sq, to));
                        break;
                    }
                    if (!capturesOnly) list.add(makeMove(sq, to));
                }
        }
    }

    if (capturesOnly) return;
    int home = side == WHITE ? 4 : 60;                    // e1 or e8
    int kingSide = side == WHITE ? CASTLE_WK : CASTLE_BK, queenSide = side == WHITE ? CASTLE_WQ : CASTLE_BQ;
    if (king[side] != home || attacked(home, enemy)) return;
    int rook = makePiece(side, ROOK);
    if ((castling & kingSide) && squares[home + 3] == rook && !squares[home + 1] && !squares[home + 2]
        && !attacked(home + 1, enemy) && !attacked(home + 2, enemy))
        list.add(makeMove(home, home + 2));
    if ((castling & queenSide) && squares[home - 4] == rook && !squares[home - 1] && !squares[home - 2]
        && !squares[home - 3] && !attacked(home - 1, enemy) && !attacked(home - 2, enemy))
        list.add(makeMove(home, home - 2));
}

void Board::legalMoves(MoveList& list) {
    MoveList all;
    generate(all);
    for (int i = 0; i < all.size; i++) {
        make(all.moves[i]);
        if (!sideJustMovedInCheck()) list.add(all.moves[i]);
        unmake();
    }
}

bool Board::parseUci(const std::string& text, Move& out) {
    MoveList list;
    legalMoves(list);
    for (int i = 0; i < list.size; i++)
        if (moveToUci(list.moves[i]) == text) { out = list.moves[i]; return true; }
    return false;
}

// ---------------------------------------------------------------- draws and counts
bool Board::isRepetition() const {
    int n = int(history.size());
    int reach = n < halfmove ? n : halfmove;              // nothing before the last pawn move or capture can repeat
    for (int back = 2; back <= reach; back += 2)
        if (history[n - back].hash == hash) return true;
    return false;
}

bool Board::insufficientMaterial() const {
    int minors = 0;
    for (int sq = 0; sq < 64; sq++) {
        int type = typeOf(squares[sq]);
        if (type == PAWN || type == ROOK || type == QUEEN) return false;
        if (type == KNIGHT || type == BISHOP) minors++;
    }
    return minors <= 1;
}

bool Board::hasNonPawnMaterial(int color) const {
    for (int sq = 0; sq < 64; sq++) {
        int piece = squares[sq];
        if (piece && colorOf(piece) == color && typeOf(piece) != PAWN && typeOf(piece) != KING) return true;
    }
    return false;
}

int Board::pieceCount() const {
    int n = 0;
    for (int sq = 0; sq < 64; sq++) n += squares[sq] != EMPTY;
    return n;
}

uint64_t Board::perft(int depth) {
    if (depth == 0) return 1;
    MoveList list;
    generate(list);
    uint64_t total = 0;
    for (int i = 0; i < list.size; i++) {
        make(list.moves[i]);
        if (!sideJustMovedInCheck()) total += perft(depth - 1);
        unmake();
    }
    return total;
}
