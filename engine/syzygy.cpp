// syzygy.cpp - see syzygy.h. The probing code itself is Fathom (fathom/, MIT licence).
#include "syzygy.h"
extern "C" {
#include "fathom/tbprobe.h"
}
#include <mutex>

namespace syzygy {

static bool ready = false;
static std::mutex initLock;

bool init(const std::string& path) {
    std::lock_guard<std::mutex> lock(initLock);
    ready = false;
    if (path.empty() || path == "<empty>") { tb_free(); return false; }
    if (!tb_init(path.c_str())) return false;
    ready = TB_LARGEST > 0;
    return ready;
}

int largest() { return ready ? int(TB_LARGEST) : 0; }

static inline unsigned epSquare(const Board& b) {
    if (b.ep < 0) return 0;
    // Fathom wants the square only when a pawn can actually take there.
    return (attacks::pawn[b.side ^ 1][b.ep] & b.byType(b.side, PAWN)) ? unsigned(b.ep) : 0;
}

int probeWdl(const Board& b) {
    if (!ready || b.castling || b.halfmove || b.pieceCount() > int(TB_LARGEST)) return -1;
    unsigned r = tb_probe_wdl(b.colors[WHITE], b.colors[BLACK],
                              b.pieces[WK] | b.pieces[BK], b.pieces[WQ] | b.pieces[BQ],
                              b.pieces[WR] | b.pieces[BR], b.pieces[WB] | b.pieces[BB],
                              b.pieces[WN] | b.pieces[BN], b.pieces[WP] | b.pieces[BP],
                              0, 0, epSquare(b), b.side == WHITE);
    return r == TB_RESULT_FAILED ? -1 : int(r);
}

bool probeRoot(Board& b, Move& out, std::string& category) {
    if (!ready || b.castling || b.pieceCount() > int(TB_LARGEST)) return false;
    unsigned r = tb_probe_root(b.colors[WHITE], b.colors[BLACK],
                               b.pieces[WK] | b.pieces[BK], b.pieces[WQ] | b.pieces[BQ],
                               b.pieces[WR] | b.pieces[BR], b.pieces[WB] | b.pieces[BB],
                               b.pieces[WN] | b.pieces[BN], b.pieces[WP] | b.pieces[BP],
                               unsigned(b.halfmove), 0, epSquare(b), b.side == WHITE, nullptr);
    if (r == TB_RESULT_FAILED || r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE) return false;
    static const int promo[5] = {0, QUEEN, ROOK, BISHOP, KNIGHT};
    Move m = makeMove(int(TB_GET_FROM(r)), int(TB_GET_TO(r)), promo[TB_GET_PROMOTES(r)]);
    MoveList legal;
    b.legalMoves(legal);
    for (int i = 0; i < legal.size; i++) {
        if (legal.moves[i] != m) continue;
        static const char* names[5] = {"loss", "blessed-loss", "draw", "cursed-win", "win"};
        category = names[TB_GET_WDL(r)];
        out = m;
        return true;
    }
    return false;
}

}  // namespace syzygy
