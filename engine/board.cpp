// board.cpp - see board.h
#include "board.h"
#include <cstdlib>
#include <cstring>
#include <sstream>

const int SEE_VALUE[7] = {0, 100, 300, 300, 500, 900, 0};

// ---------------------------------------------------------------- lookup tables
namespace attacks {
Bitboard pawn[2][64], knight[64], king[64];

// "Magic" lookups for sliding pieces: the pieces on a slider's lines are multiplied by a number
// chosen so that every arrangement that matters lands in its own slot of a table of attacks.
struct Magic {
    Bitboard mask, magic;
    Bitboard* table;
    int shift;
    unsigned index(Bitboard occupied) const { return unsigned(((occupied & mask) * magic) >> shift); }
};
static Magic bishopMagic[64], rookMagic[64];
static Bitboard bishopTable[0x1480], rookTable[0x19000];

Bitboard bishop(int square, Bitboard occupied) { const Magic& m = bishopMagic[square]; return m.table[m.index(occupied)]; }
Bitboard rook(int square, Bitboard occupied) { const Magic& m = rookMagic[square]; return m.table[m.index(occupied)]; }
}

static int castleMask[64];
static uint64_t zPiece[13][64], zCastle[16], zEp[8], zSide;
static const Bitboard RANK_1 = 0xFFULL;
static const Bitboard FILE_A = 0x0101010101010101ULL;

static uint64_t nextRandom(uint64_t& x) {                 // splitmix64: fixed seed, same keys every run
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Attacks along the given directions, stopping at (and including) the first piece in each.
static Bitboard slowSlider(int square, Bitboard occupied, const int (*dirs)[2]) {
    Bitboard result = 0;
    for (int d = 0; d < 4; d++) {
        int f = square & 7, r = square >> 3;
        for (;;) {
            f += dirs[d][0];
            r += dirs[d][1];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            result |= bit(r * 8 + f);
            if (occupied & bit(r * 8 + f)) break;
        }
    }
    return result;
}

static void initMagics(attacks::Magic* magics, Bitboard* table, const int (*dirs)[2], uint64_t& seed) {
    static Bitboard occupancy[4096], reference[4096];
    static int epoch[4096], attempt = 0;                  // shared by both calls, so stale marks never match
    Bitboard* next = table;
    for (int sq = 0; sq < 64; sq++) {
        // Edge squares never change what a slider can reach, so they're left out of the mask.
        Bitboard edges = ((RANK_1 | (RANK_1 << 56)) & ~(RANK_1 << ((sq >> 3) * 8)))
                       | ((FILE_A | (FILE_A << 7)) & ~(FILE_A << (sq & 7)));
        attacks::Magic& m = magics[sq];
        m.mask = slowSlider(sq, 0, dirs) & ~edges;
        m.shift = 64 - popCount(m.mask);
        m.table = next;
        int size = 0;
        Bitboard b = 0;
        do {                                               // every subset of the mask
            occupancy[size] = b;
            reference[size++] = slowSlider(sq, b, dirs);
            b = (b - m.mask) & m.mask;
        } while (b);
        next += size;
        for (int i = 0; i < size;) {
            do m.magic = nextRandom(seed) & nextRandom(seed) & nextRandom(seed);
            while (popCount((m.mask * m.magic) >> 56) < 6);
            attempt++;
            for (i = 0; i < size; i++) {
                unsigned idx = m.index(occupancy[i]);
                if (epoch[idx] < attempt) { epoch[idx] = attempt; m.table[idx] = reference[i]; }
                else if (m.table[idx] != reference[i]) break;
            }
        }
    }
}

namespace {
struct TableSetup {
    TableSetup() {
        static const int knightSteps[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        static const int kingSteps[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        static const int rookDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        static const int bishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (int sq = 0; sq < 64; sq++) {
            int f = sq & 7, r = sq >> 3;
            for (int i = 0; i < 8; i++) {
                int nf = f + knightSteps[i][0], nr = r + knightSteps[i][1];
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) attacks::knight[sq] |= bit(nr * 8 + nf);
                nf = f + kingSteps[i][0], nr = r + kingSteps[i][1];
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) attacks::king[sq] |= bit(nr * 8 + nf);
            }
            if (r < 7 && f > 0) attacks::pawn[WHITE][sq] |= bit(sq + 7);
            if (r < 7 && f < 7) attacks::pawn[WHITE][sq] |= bit(sq + 9);
            if (r > 0 && f > 0) attacks::pawn[BLACK][sq] |= bit(sq - 9);
            if (r > 0 && f < 7) attacks::pawn[BLACK][sq] |= bit(sq - 7);
        }
        uint64_t magicSeed = 728;
        initMagics(attacks::bishopMagic, attacks::bishopTable, bishopDirs, magicSeed);
        initMagics(attacks::rookMagic, attacks::rookTable, rookDirs, magicSeed);

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
    if (tracking) {
        Accumulator& a = acc[size_t(accTop)];
        a.addPiece[a.added] = uint8_t(piece);
        a.addSquare[a.added++] = uint8_t(square);
    }
    squares[square] = int8_t(piece);
    pieces[piece] |= bit(square);
    colors[colorOf(piece)] |= bit(square);
    hash ^= zPiece[piece][square];
}

void Board::remove(int square) {
    int piece = squares[square];
    if (tracking) {
        Accumulator& a = acc[size_t(accTop)];
        a.removePiece[a.removed] = uint8_t(piece);
        a.removeSquare[a.removed++] = uint8_t(square);
    }
    hash ^= zPiece[piece][square];
    pieces[piece] &= ~bit(square);
    colors[colorOf(piece)] &= ~bit(square);
    squares[square] = EMPTY;
}

// A new, not yet calculated accumulator for the position after a move (or a null move).
void Board::pushAccumulator() {
    accTop++;
    if (size_t(accTop) >= acc.size()) acc.resize(acc.size() * 2);
    Accumulator& a = acc[size_t(accTop)];
    a.computed = false;
    a.added = a.removed = 0;
}

bool Board::setFen(const std::string& text) {
    std::istringstream in(text);
    std::string placement, turn, rights = "-", epText = "-";
    int half = 0, full = 1;
    in >> placement >> turn >> rights >> epText >> half >> full;
    if (placement.empty() || (turn != "w" && turn != "b")) return false;

    std::memset(squares, 0, sizeof(squares));
    std::memset(pieces, 0, sizeof(pieces));
    colors[WHITE] = colors[BLACK] = 0;
    history.clear();
    if (acc.empty()) acc.resize(256);
    accTop = 0;
    acc[0].computed = false;                              // worked out from scratch when first needed
    acc[0].added = acc[0].removed = 0;
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
    pushAccumulator();
    tracking = true;
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
    }
    tracking = false;
    if (type == PAWN && std::abs(to - from) == 16) {
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
    accTop--;
    side ^= 1;
    if (side == BLACK) fullmove--;

    // Undoing doesn't need the hash or accumulator bookkeeping, so set squares directly.
    auto lift = [this](int sq) {
        int p = squares[sq];
        pieces[p] &= ~bit(sq);
        colors[colorOf(p)] &= ~bit(sq);
        squares[sq] = EMPTY;
        return p;
    };
    auto drop = [this](int sq, int p) {
        squares[sq] = int8_t(p);
        pieces[p] |= bit(sq);
        colors[colorOf(p)] |= bit(sq);
    };

    int from = fromSq(u.move), to = toSq(u.move);
    int piece = lift(to);
    if (promoOf(u.move)) piece = makePiece(side, PAWN);
    drop(from, piece);
    int type = typeOf(piece);

    if (u.captured) {
        if (type == PAWN && to == u.ep) drop(to + (side == WHITE ? -8 : 8), u.captured);
        else drop(to, u.captured);
    }
    if (type == KING) {
        king[side] = from;
        if (to - from == 2) drop(to + 1, lift(to - 1));
        if (from - to == 2) drop(to - 2, lift(to + 1));
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
    pushAccumulator();                                    // same pieces, so nothing to add or remove
    if (ep >= 0) hash ^= zEp[ep & 7];
    ep = -1;
    halfmove = 0;                                         // repetition checks stop at a null move
    side ^= 1;
    hash ^= zSide;
}

void Board::unmakeNull() {
    Undo u = history.back();
    history.pop_back();
    accTop--;
    side ^= 1;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
}

// ---------------------------------------------------------------- attacks and moves
bool Board::attacked(int sq, int by) const {
    Bitboard occ = occupied();
    return (attacks::pawn[by ^ 1][sq] & pieces[makePiece(by, PAWN)])
        || (attacks::knight[sq] & pieces[makePiece(by, KNIGHT)])
        || (attacks::king[sq] & pieces[makePiece(by, KING)])
        || (attacks::bishop(sq, occ) & (pieces[makePiece(by, BISHOP)] | pieces[makePiece(by, QUEEN)]))
        || (attacks::rook(sq, occ) & (pieces[makePiece(by, ROOK)] | pieces[makePiece(by, QUEEN)]));
}

Bitboard Board::attackersTo(int sq, Bitboard occ) const {
    Bitboard diagonal = pieces[WB] | pieces[BB] | pieces[WQ] | pieces[BQ];
    Bitboard straight = pieces[WR] | pieces[BR] | pieces[WQ] | pieces[BQ];
    return (attacks::pawn[BLACK][sq] & pieces[WP]) | (attacks::pawn[WHITE][sq] & pieces[BP])
         | (attacks::knight[sq] & (pieces[WN] | pieces[BN]))
         | (attacks::king[sq] & (pieces[WK] | pieces[BK]))
         | (attacks::bishop(sq, occ) & diagonal) | (attacks::rook(sq, occ) & straight);
}

// Plays out the exchange on the destination square, each side recapturing with its cheapest
// piece, and stops as soon as one side would rather not continue.
bool Board::see(Move m, int threshold) const {
    int from = fromSq(m), to = toSq(m);
    if (promoOf(m) || (to == ep && typeOf(squares[from]) == PAWN)) return threshold <= 0;
    int swap = SEE_VALUE[typeOf(squares[to])] - threshold;
    if (swap < 0) return false;
    swap = SEE_VALUE[typeOf(squares[from])] - swap;
    if (swap <= 0) return true;

    Bitboard occ = occupied() ^ bit(from) ^ bit(to);
    Bitboard attackers = attackersTo(to, occ);
    Bitboard diagonal = pieces[WB] | pieces[BB] | pieces[WQ] | pieces[BQ];
    Bitboard straight = pieces[WR] | pieces[BR] | pieces[WQ] | pieces[BQ];
    int stm = side, result = 1;
    for (;;) {
        stm ^= 1;
        attackers &= occ;
        Bitboard mine = attackers & colors[stm];
        if (!mine) break;
        result ^= 1;
        Bitboard b;
        if ((b = mine & pieces[makePiece(stm, PAWN)])) {
            if ((swap = SEE_VALUE[PAWN] - swap) < result) break;
            occ ^= b & -b;
            attackers |= attacks::bishop(to, occ) & diagonal;
        } else if ((b = mine & pieces[makePiece(stm, KNIGHT)])) {
            if ((swap = SEE_VALUE[KNIGHT] - swap) < result) break;
            occ ^= b & -b;
        } else if ((b = mine & pieces[makePiece(stm, BISHOP)])) {
            if ((swap = SEE_VALUE[BISHOP] - swap) < result) break;
            occ ^= b & -b;
            attackers |= attacks::bishop(to, occ) & diagonal;
        } else if ((b = mine & pieces[makePiece(stm, ROOK)])) {
            if ((swap = SEE_VALUE[ROOK] - swap) < result) break;
            occ ^= b & -b;
            attackers |= attacks::rook(to, occ) & straight;
        } else if ((b = mine & pieces[makePiece(stm, QUEEN)])) {
            if ((swap = SEE_VALUE[QUEEN] - swap) < result) break;
            occ ^= b & -b;
            attackers |= (attacks::bishop(to, occ) & diagonal) | (attacks::rook(to, occ) & straight);
        } else {                                          // only the king is left
            return (attackers & ~colors[stm]) ? result ^ 1 : result;
        }
    }
    return result;
}

static inline void addPromotions(MoveList& list, int from, int to) {
    list.add(makeMove(from, to, QUEEN));
    list.add(makeMove(from, to, ROOK));
    list.add(makeMove(from, to, BISHOP));
    list.add(makeMove(from, to, KNIGHT));
}

void Board::generate(MoveList& list, bool capturesOnly) const {
    const int us = side, them = side ^ 1;
    const Bitboard occ = occupied(), enemies = colors[them], empty = ~occ;
    const Bitboard targets = capturesOnly ? enemies : ~colors[us];

    // Pawns
    const int up = us == WHITE ? 8 : -8;
    const Bitboard lastRank = us == WHITE ? RANK_1 << 48 : RANK_1 << 8;    // pawns about to promote
    const Bitboard thirdRank = us == WHITE ? RANK_1 << 16 : RANK_1 << 40;
    auto forward = [us](Bitboard b) { return us == WHITE ? b << 8 : b >> 8; };
    Bitboard pawns = byType(us, PAWN);
    Bitboard promoting = pawns & lastRank, normal = pawns & ~lastRank;
    if (!capturesOnly) {
        Bitboard one = forward(normal) & empty;
        Bitboard two = forward(one & thirdRank) & empty;
        while (one) { int to = popLsb(one); list.add(makeMove(to - up, to)); }
        while (two) { int to = popLsb(two); list.add(makeMove(to - 2 * up, to)); }
    }
    for (Bitboard b = normal; b;) {
        int from = popLsb(b);
        Bitboard caps = attacks::pawn[us][from] & enemies;
        if (ep >= 0) caps |= attacks::pawn[us][from] & bit(ep);
        while (caps) list.add(makeMove(from, popLsb(caps)));
    }
    for (Bitboard b = promoting; b;) {
        int from = popLsb(b);
        if (empty & bit(from + up)) addPromotions(list, from, from + up);
        for (Bitboard caps = attacks::pawn[us][from] & enemies; caps;) addPromotions(list, from, popLsb(caps));
    }

    // Pieces
    for (Bitboard b = byType(us, KNIGHT); b;) {
        int from = popLsb(b);
        for (Bitboard t = attacks::knight[from] & targets; t;) list.add(makeMove(from, popLsb(t)));
    }
    for (Bitboard b = byType(us, BISHOP); b;) {
        int from = popLsb(b);
        for (Bitboard t = attacks::bishop(from, occ) & targets; t;) list.add(makeMove(from, popLsb(t)));
    }
    for (Bitboard b = byType(us, ROOK); b;) {
        int from = popLsb(b);
        for (Bitboard t = attacks::rook(from, occ) & targets; t;) list.add(makeMove(from, popLsb(t)));
    }
    for (Bitboard b = byType(us, QUEEN); b;) {
        int from = popLsb(b);
        for (Bitboard t = attacks::queen(from, occ) & targets; t;) list.add(makeMove(from, popLsb(t)));
    }
    int k = king[us];
    for (Bitboard t = attacks::king[k] & targets; t;) list.add(makeMove(k, popLsb(t)));

    if (capturesOnly) return;
    int home = us == WHITE ? 4 : 60;                      // e1 or e8
    int kingSide = us == WHITE ? CASTLE_WK : CASTLE_BK, queenSide = us == WHITE ? CASTLE_WQ : CASTLE_BQ;
    if (k != home || !(castling & (kingSide | queenSide)) || attacked(home, them)) return;
    int rook = makePiece(us, ROOK);
    if ((castling & kingSide) && squares[home + 3] == rook && !squares[home + 1] && !squares[home + 2]
        && !attacked(home + 1, them) && !attacked(home + 2, them))
        list.add(makeMove(home, home + 2));
    if ((castling & queenSide) && squares[home - 4] == rook && !squares[home - 1] && !squares[home - 2]
        && !squares[home - 3] && !attacked(home - 1, them) && !attacked(home - 2, them))
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
    for (int back = 4; back <= reach; back += 2)
        if (history[n - back].hash == hash) return true;
    return false;
}

bool Board::insufficientMaterial() const {
    if (pieces[WP] | pieces[BP] | pieces[WR] | pieces[BR] | pieces[WQ] | pieces[BQ]) return false;
    return popCount(pieces[WN] | pieces[BN] | pieces[WB] | pieces[BB]) <= 1;
}

bool Board::hasNonPawnMaterial(int color) const {
    return colors[color] & ~(pieces[makePiece(color, PAWN)] | pieces[makePiece(color, KING)]);
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
