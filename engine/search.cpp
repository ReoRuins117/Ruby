// search.cpp - see search.h
#include "search.h"
#include "nnue.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace search {

std::atomic<bool> stopRequested{false};

// ---------------------------------------------------------------- transposition table
enum { BOUND_EXACT = 1, BOUND_LOWER = 2, BOUND_UPPER = 3 };

struct TTEntry {
    uint64_t key = 0;
    int16_t score = 0;
    Move move = NO_MOVE;
    int8_t depth = 0;
    uint8_t bound = 0;
};

static std::vector<TTEntry> table(size_t(16) * 1024 * 1024 / sizeof(TTEntry));

void setHashMb(int megabytes) {
    size_t wanted = size_t(std::max(megabytes, 1)) * 1024 * 1024 / sizeof(TTEntry);
    table.assign(wanted, TTEntry());
}

void clearHash() { std::fill(table.begin(), table.end(), TTEntry()); }

// Mate scores count from the root, so they're stored relative to the node instead.
static int scoreToTable(int score, int ply) {
    if (score >= MATE_BOUND) return score + ply;
    if (score <= -MATE_BOUND) return score - ply;
    return score;
}

static int scoreFromTable(int score, int ply) {
    if (score >= MATE_BOUND) return score - ply;
    if (score <= -MATE_BOUND) return score + ply;
    return score;
}

// ---------------------------------------------------------------- search state
static const int PIECE_VALUE[7] = {0, 100, 320, 330, 500, 900, 0};

static Board* pos;
static Limits limit;
static uint64_t nodes;
static bool aborted;
static std::chrono::steady_clock::time_point startTime;
static Move killers[MAX_PLY][2];
static int historyScore[2][64][64];
static Move pvTable[MAX_PLY][MAX_PLY];
static int pvLength[MAX_PLY];

static int64_t elapsedMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
}

static void countNode() {
    nodes++;
    if ((nodes & 2047) == 0) {
        if (stopRequested.load() || (limit.hardMs >= 0 && elapsedMs() >= limit.hardMs)) aborted = true;
    }
    if (limit.nodes && nodes >= limit.nodes) aborted = true;
}

static bool isCapture(Move m) {
    return pos->squares[toSq(m)] != EMPTY
        || (typeOf(pos->squares[fromSq(m)]) == PAWN && toSq(m) == pos->ep);
}

static int capturedValue(Move m) {
    int victim = pos->squares[toSq(m)];
    return victim ? PIECE_VALUE[typeOf(victim)] : PIECE_VALUE[PAWN];
}

// Good moves first: alpha-beta cuts far more of the tree when the best move is tried early.
static void scoreMoves(const MoveList& list, int* scores, Move ttMove, int ply) {
    for (int i = 0; i < list.size; i++) {
        Move m = list.moves[i];
        if (m == ttMove) scores[i] = 10000000;
        else if (promoOf(m) == QUEEN) scores[i] = 9000000;
        else if (isCapture(m)) scores[i] = 8000000 + capturedValue(m) * 10 - PIECE_VALUE[typeOf(pos->squares[fromSq(m)])] / 10;
        else if (promoOf(m)) scores[i] = -1000;                       // underpromotions last
        else if (m == killers[ply][0]) scores[i] = 7000000;
        else if (m == killers[ply][1]) scores[i] = 6999999;
        else scores[i] = historyScore[pos->side][fromSq(m)][toSq(m)];
    }
}

static Move pickNext(MoveList& list, int* scores, int start) {
    int best = start;
    for (int i = start + 1; i < list.size; i++)
        if (scores[i] > scores[best]) best = i;
    std::swap(list.moves[start], list.moves[best]);
    std::swap(scores[start], scores[best]);
    return list.moves[start];
}

// ---------------------------------------------------------------- quiescence search
// The network only saw calm positions, so keep playing captures until the position settles.
static int quiescence(int alpha, int beta, int ply) {
    countNode();
    if (aborted) return 0;
    if (pos->insufficientMaterial() || pos->halfmove >= 100) return 0;
    bool inCheck = pos->inCheck();
    if (ply >= MAX_PLY - 1) return inCheck ? 0 : nnue::evaluate(*pos);

    int standPat = -MATE;
    if (!inCheck) {
        standPat = nnue::evaluate(*pos);
        if (standPat >= beta) return standPat;
        alpha = std::max(alpha, standPat);
    }

    MoveList list;
    pos->generate(list, !inCheck);                                     // in check, every escape must be tried
    int scores[256];
    scoreMoves(list, scores, NO_MOVE, ply);
    int best = standPat, legal = 0;
    for (int i = 0; i < list.size; i++) {
        Move m = pickNext(list, scores, i);
        if (!inCheck) {
            if (promoOf(m) && promoOf(m) != QUEEN) continue;
            int gain = capturedValue(m) + (promoOf(m) ? PIECE_VALUE[QUEEN] : 0);
            if (standPat + gain + 200 < alpha) continue;               // can't get back to alpha
        }
        pos->make(m);
        if (pos->sideJustMovedInCheck()) { pos->unmake(); continue; }
        legal++;
        int score = -quiescence(-beta, -alpha, ply + 1);
        pos->unmake();
        if (aborted) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) alpha = score;
            if (score >= beta) return score;
        }
    }
    if (inCheck && legal == 0) return -MATE + ply;
    return best;
}

// ---------------------------------------------------------------- main search
static int negamax(int depth, int alpha, int beta, int ply, bool allowNull) {
    pvLength[ply] = 0;
    bool isRoot = ply == 0;
    bool isPv = beta - alpha > 1;

    if (!isRoot) {
        if (pos->isRepetition() || pos->halfmove >= 100 || pos->insufficientMaterial()) return 0;
        alpha = std::max(alpha, -MATE + ply);                          // a shorter mate is already known
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }
    bool inCheck = pos->inCheck();
    if (ply >= MAX_PLY - 1) return inCheck ? 0 : nnue::evaluate(*pos);
    if (inCheck) depth++;                                              // the network never saw these positions
    if (depth <= 0) return quiescence(alpha, beta, ply);

    countNode();
    if (aborted) return 0;

    TTEntry& entry = table[pos->hash % table.size()];
    Move ttMove = NO_MOVE;
    if (entry.key == pos->hash) {
        ttMove = entry.move;
        if (!isPv && !isRoot && entry.depth >= depth) {
            int score = scoreFromTable(entry.score, ply);
            if (entry.bound == BOUND_EXACT
                || (entry.bound == BOUND_LOWER && score >= beta)
                || (entry.bound == BOUND_UPPER && score <= alpha))
                return score;
        }
    }

    if (!inCheck && !isPv) {
        int staticEval = nnue::evaluate(*pos);
        // Reverse futility: so far above beta at low depth that the opponent can't catch up.
        if (depth <= 3 && std::abs(beta) < MATE_BOUND && staticEval - 120 * depth >= beta) return staticEval;
        // Null move: if passing still beats beta, a real move almost certainly would too.
        if (allowNull && depth >= 3 && staticEval >= beta && pos->hasNonPawnMaterial(pos->side)) {
            int reduction = 2 + depth / 4;
            pos->makeNull();
            int score = -negamax(depth - 1 - reduction, -beta, -beta + 1, ply + 1, false);
            pos->unmakeNull();
            if (aborted) return 0;
            if (score >= beta) return score >= MATE_BOUND ? beta : score;
        }
    }

    MoveList list;
    pos->generate(list);
    int scores[256];
    scoreMoves(list, scores, ttMove, ply);

    int startAlpha = alpha, best = -MATE, legal = 0;
    Move bestMove = NO_MOVE;
    for (int i = 0; i < list.size; i++) {
        Move m = pickNext(list, scores, i);
        bool quiet = !isCapture(m) && !promoOf(m);
        pos->make(m);
        if (pos->sideJustMovedInCheck()) { pos->unmake(); continue; }
        legal++;

        int score;
        if (legal == 1) {
            score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
        } else {
            // Late, quiet moves are probably bad: try them shallower, and re-search if they surprise us.
            int reduction = 0;
            if (depth >= 3 && legal > 4 && quiet && !inCheck && !pos->inCheck())
                reduction = legal > 12 ? 2 : 1;
            score = -negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && reduction)
                score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta)
                score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
        }
        pos->unmake();
        if (aborted) return 0;

        if (score > best) {
            best = score;
            bestMove = m;
            if (score > alpha) {
                alpha = score;
                pvTable[ply][0] = m;
                std::memcpy(&pvTable[ply][1], pvTable[ply + 1], sizeof(Move) * size_t(pvLength[ply + 1]));
                pvLength[ply] = pvLength[ply + 1] + 1;
            }
            if (score >= beta) {
                if (quiet) {
                    if (killers[ply][0] != m) {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = m;
                    }
                    int& h = historyScore[pos->side][fromSq(m)][toSq(m)];
                    h = std::min(h + depth * depth, 1000000);
                }
                break;
            }
        }
    }

    if (legal == 0) return inCheck ? -MATE + ply : 0;                  // checkmate or stalemate

    TTEntry& slot = table[pos->hash % table.size()];
    slot.key = pos->hash;
    slot.score = int16_t(scoreToTable(best, ply));
    slot.move = bestMove;
    slot.depth = int8_t(std::min(depth, 127));
    slot.bound = uint8_t(best >= beta ? BOUND_LOWER : best > startAlpha ? BOUND_EXACT : BOUND_UPPER);
    return best;
}

// ---------------------------------------------------------------- iterative deepening
static std::string scoreText(int score) {
    if (score >= MATE_BOUND) return "mate " + std::to_string((MATE - score + 1) / 2);
    if (score <= -MATE_BOUND) return "mate -" + std::to_string((MATE + score + 1) / 2);
    return "cp " + std::to_string(score);
}

Result think(Board& board, const Limits& limits, bool verbose) {
    pos = &board;
    limit = limits;
    nodes = 0;
    aborted = false;
    startTime = std::chrono::steady_clock::now();
    std::memset(killers, 0, sizeof(killers));
    std::memset(historyScore, 0, sizeof(historyScore));

    Result result;
    MoveList legal;
    board.legalMoves(legal);
    if (legal.size == 0) return result;
    result.best = legal.moves[0];                                      // always have something to play

    int lastScore = 0;
    for (int depth = 1; depth <= limits.depth; depth++) {
        int window = 30, alpha = -MATE, beta = MATE, score;
        if (depth >= 5) {                                              // expect a score near the last one
            alpha = std::max(lastScore - window, -MATE);
            beta = std::min(lastScore + window, MATE);
        }
        for (;;) {
            score = negamax(depth, alpha, beta, 0, false);
            if (aborted) break;
            if (score > alpha && score < beta) break;
            window *= 3;
            if (score <= alpha) alpha = std::max(score - window, -MATE);
            else beta = std::min(score + window, MATE);
            if (window > 2000) { alpha = -MATE; beta = MATE; }
        }
        // A partly finished depth is only trusted if it found a new best move at the root.
        if (pvLength[0] > 0 && (!aborted || pvTable[0][0] != NO_MOVE)) {
            result.best = pvTable[0][0];
            if (!aborted) {
                result.score = score;
                result.depth = depth;
            }
        }
        if (aborted) break;
        lastScore = score;
        result.nodes = nodes;

        if (verbose) {
            int64_t ms = elapsedMs();
            std::string line = "info depth " + std::to_string(depth) + " score " + scoreText(score)
                + " nodes " + std::to_string(nodes) + " nps " + std::to_string(nodes * 1000 / uint64_t(std::max<int64_t>(ms, 1)))
                + " time " + std::to_string(ms) + " pv";
            for (int i = 0; i < pvLength[0]; i++) line += " " + moveToUci(pvTable[0][i]);
            std::printf("%s\n", line.c_str());
            std::fflush(stdout);
        }
        if (limits.softMs >= 0 && elapsedMs() >= limits.softMs) break;
        if (std::abs(score) >= MATE_BOUND && depth >= (MATE - std::abs(score)) + 2) break;   // the mate is settled
    }
    result.nodes = nodes;
    return result;
}

Move pickWeak(Board& board, int depth, int temperature) {
    static std::mt19937 rng(std::random_device{}());
    pos = &board;
    limit = Limits();
    nodes = 0;
    aborted = false;
    stopRequested = false;
    startTime = std::chrono::steady_clock::now();
    std::memset(killers, 0, sizeof(killers));
    std::memset(historyScore, 0, sizeof(historyScore));

    MoveList legal;
    board.legalMoves(legal);
    if (legal.size == 0) return NO_MOVE;
    int scores[256], best = -MATE;
    for (int i = 0; i < legal.size; i++) {
        board.make(legal.moves[i]);
        scores[i] = std::clamp(-negamax(depth - 1, -MATE, MATE, 1, true), -3000, 3000);
        board.unmake();
        best = std::max(best, scores[i]);
    }
    // Each move's chance is exp(-(how much worse than the best) / temperature).
    double weights[256], total = 0;
    for (int i = 0; i < legal.size; i++) total += weights[i] = std::exp(double(scores[i] - best) / temperature);
    double r = std::uniform_real_distribution<double>(0, total)(rng);
    for (int i = 0; i < legal.size; i++) {
        r -= weights[i];
        if (r <= 0) return legal.moves[i];
    }
    return legal.moves[legal.size - 1];
}

}  // namespace search
