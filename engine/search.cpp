// search.cpp - see search.h
//
// Each thread runs its own iterative deepening search of the same position ("Lazy SMP"). They
// share only the hash table, so what one thread finds steers the others. The first thread
// manages time and prints the results.
#include "search.h"
#include "nnue.h"
#include "syzygy.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace search {

std::atomic<bool> stopRequested{false};
std::atomic<bool> pondering{false};

static std::atomic<bool> stopAll{false};
static std::atomic<int64_t> clockStart{0};            // when the time limits started counting
static Limits limit;

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static int64_t elapsedMs() { return nowMs() - clockStart.load(std::memory_order_relaxed); }

// Our clock starts now, but the time already spent pondering counts for something: credit up to
// half the normal budget, so a long ponder answers quickly while the hard limit still leaves room.
void ponderHit() {
    int64_t now = nowMs(), pondered = now - clockStart.load();
    int64_t credit = limit.softMs > 0 ? std::min(pondered, limit.softMs / 2) : 0;
    clockStart = now - credit;
    pondering = false;
}

// ---------------------------------------------------------------- transposition table
enum { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };
const int EVAL_NONE = 32001;

struct TTData {
    Move move;
    int score, eval, depth, bound, gen;
    bool pv;                                              // the position has been on a principal variation
};

// An entry is stored as two 64-bit words, the first being the key mixed with the second. A slot
// half-written by another thread then simply fails to match, instead of giving a wrong answer.
struct TTSlot {
    std::atomic<uint64_t> check{0}, data{0};
};
struct TTBucket {
    TTSlot slot[2];
};

static std::unique_ptr<TTBucket[]> table;
static size_t tableSize = 0;
static uint8_t generation = 0;

static uint64_t pack(Move m, int score, int eval, int depth, int bound, bool pv) {
    return uint64_t(m) | uint64_t(uint16_t(int16_t(score))) << 16 | uint64_t(uint16_t(int16_t(eval))) << 32
         | uint64_t(uint8_t(depth)) << 48 | uint64_t(bound) << 56 | uint64_t(generation & 31) << 58 | uint64_t(pv) << 63;
}

static TTData unpack(uint64_t d) {
    return {Move(d & 0xFFFF), int16_t(d >> 16), int16_t(d >> 32), int((d >> 48) & 0xFF), int((d >> 56) & 3),
            int((d >> 58) & 31), bool(d >> 63)};
}

static TTBucket& bucketFor(uint64_t key) {
    return table[size_t((unsigned __int128)key * tableSize >> 64)];
}

static bool ttProbe(uint64_t key, TTData& out) {
    for (TTSlot& s : bucketFor(key).slot) {
        uint64_t d = s.data.load(std::memory_order_relaxed), c = s.check.load(std::memory_order_relaxed);
        if ((c ^ d) == key && d) { out = unpack(d); return true; }
    }
    return false;
}

static void ttStore(uint64_t key, Move m, int score, int eval, int depth, int bound, bool pv = false) {
    depth = std::clamp(depth, 0, 255);
    TTBucket& b = bucketFor(key);
    TTSlot* target = nullptr;
    int worst = INT_MAX;
    for (TTSlot& s : b.slot) {
        uint64_t d = s.data.load(std::memory_order_relaxed), c = s.check.load(std::memory_order_relaxed);
        TTData old = unpack(d);
        if ((c ^ d) == key && d) {
            // Keep a much deeper result for this position unless the new one is exact.
            if (bound != BOUND_EXACT && depth + 3 < old.depth && old.gen == (generation & 31)) return;
            if (m == NO_MOVE) m = old.move;
            if (eval == EVAL_NONE) eval = old.eval;
            target = &s;
            break;
        }
        int value = d ? old.depth - 8 * ((generation - old.gen) & 31) : INT_MIN;   // replace old, shallow ones
        if (value < worst) { worst = value; target = &s; }
    }
    uint64_t d = pack(m, score, eval, depth, bound, pv);
    target->data.store(d, std::memory_order_relaxed);
    target->check.store(key ^ d, std::memory_order_relaxed);
}

void setHashMb(int megabytes) {
    tableSize = std::max<size_t>(size_t(std::max(megabytes, 1)) * 1024 * 1024 / sizeof(TTBucket), 1);
    table.reset(new TTBucket[tableSize]);
}

static int hashFull() {                                   // per mille of the table used this search
    int used = 0;
    for (size_t i = 0; i < 500 && i < tableSize; i++)
        for (TTSlot& s : table[i].slot) {
            uint64_t d = s.data.load(std::memory_order_relaxed);
            used += d && unpack(d).gen == (generation & 31);
        }
    return used;
}

// Mate and tablebase scores count from the root, so they're stored relative to the node instead.
static int scoreToTable(int score, int ply) {
    if (score >= TB_BOUND) return score + ply;
    if (score <= -TB_BOUND) return score - ply;
    return score;
}

static int scoreFromTable(int score, int ply) {
    if (score >= TB_BOUND) return score - ply;
    if (score <= -TB_BOUND) return score + ply;
    return score;
}

// ---------------------------------------------------------------- tuning numbers
static int lmrTable[64][64];

namespace {
struct LmrSetup {
    LmrSetup() {
        for (int d = 1; d < 64; d++)
            for (int m = 1; m < 64; m++) lmrTable[d][m] = int(0.8 + std::log(d) * std::log(m) / 2.3);
    }
} lmrSetup;
}

static int statBonus(int depth) { return std::min(170 * depth - 100, 1600); }

// Move statistics drift towards the bonus and saturate around +-16384.
static void gravity(int16_t& entry, int bonus) {
    int v = entry;
    v += bonus - v * std::abs(bonus) / 16384;
    entry = int16_t(std::clamp(v, -16000, 16000));
}

// ---------------------------------------------------------------- worker threads
struct Stack {
    int staticEval = EVAL_NONE;
    Move move = NO_MOVE, excluded = NO_MOVE;
    int piece = 0, to = 0;                                // the move played from this ply
    Move killers[2] = {NO_MOVE, NO_MOVE};
    int doubleExtensions = 0;
    bool ttPv = false;
};

const int SCORE_TT = 1 << 30, SCORE_GOOD_NOISY = 1 << 28, SCORE_KILLER = 1 << 27, SCORE_BAD_NOISY = -(1 << 28);

struct Worker {
    int id = 0;
    Board board;
    std::atomic<uint64_t> nodes{0};
    uint64_t tbHits = 0;
    int selDepth = 0, rootDepth = 0;
    Result result;

    Stack stackBase[MAX_PLY + 10];
    int16_t history[2][64][64];
    int16_t captureHistory[13][64][7];
    Move counterMoves[13][64];
    int16_t contHist[13][64][13][64];                     // [previous piece][its square][piece][square]
    Move pv[MAX_PLY + 1][MAX_PLY + 1];
    int pvLength[MAX_PLY + 1];
    uint64_t rootNodes[64][64];                           // nodes spent below each root move

    Worker() { clearStats(); }

    void clearStats() {
        std::memset(history, 0, sizeof(history));
        std::memset(captureHistory, 0, sizeof(captureHistory));
        std::memset(counterMoves, 0, sizeof(counterMoves));
        std::memset(contHist, 0, sizeof(contHist));
    }

    Stack* ss(int ply) { return &stackBase[ply + 4]; }
    bool stopped() const { return stopAll.load(std::memory_order_relaxed); }

    void countNode() {
        uint64_t n = nodes.load(std::memory_order_relaxed) + 1;
        nodes.store(n, std::memory_order_relaxed);
        if (id == 0 && (n & 1023) == 0) {
            if (stopRequested.load()) stopAll = true;
            else if (!pondering.load() && limit.hardMs >= 0 && elapsedMs() >= limit.hardMs) stopAll = true;
        }
        if (id == 0 && limit.nodes && n >= limit.nodes) stopAll = true;
    }

    int16_t* cont(int ply) { return &contHist[ss(ply)->piece][ss(ply)->to][0][0]; }

    int quietScore(Move m, int piece, int ply) {
        int to = toSq(m);
        return history[board.side][fromSq(m)][to] + cont(ply - 1)[piece * 64 + to] + cont(ply - 2)[piece * 64 + to];
    }

    void updateQuiet(Move m, int piece, int ply, int bonus) {
        int to = toSq(m);
        gravity(history[board.side][fromSq(m)][to], bonus);
        gravity(cont(ply - 1)[piece * 64 + to], bonus);
        gravity(cont(ply - 2)[piece * 64 + to], bonus);
    }

    int capturedType(Move m) const {
        if (toSq(m) == board.ep && typeOf(board.squares[fromSq(m)]) == PAWN) return PAWN;
        return typeOf(board.squares[toSq(m)]);
    }

    // Good moves first: alpha-beta cuts far more of the tree when the best move is tried early.
    void scoreMoves(const MoveList& list, int* scores, Move ttMove, int ply) {
        Stack* s = ss(ply);
        Move counter = counterMoves[ss(ply - 1)->piece][ss(ply - 1)->to];
        for (int i = 0; i < list.size; i++) {
            Move m = list.moves[i];
            int piece = board.squares[fromSq(m)];
            if (m == ttMove) {
                scores[i] = SCORE_TT;
            } else if (board.isCapture(m) || promoOf(m)) {
                if (promoOf(m) && promoOf(m) != QUEEN) { scores[i] = SCORE_BAD_NOISY - 100000; continue; }
                int victim = capturedType(m);
                int value = 10 * SEE_VALUE[victim] + captureHistory[piece][toSq(m)][victim] / 32 + (promoOf(m) ? 8000 : 0);
                scores[i] = (board.see(m, 0) ? SCORE_GOOD_NOISY : SCORE_BAD_NOISY) + value;
            } else if (m == s->killers[0]) {
                scores[i] = SCORE_KILLER;
            } else if (m == s->killers[1]) {
                scores[i] = SCORE_KILLER - 1;
            } else if (m == counter) {
                scores[i] = SCORE_KILLER - 2;
            } else {
                scores[i] = quietScore(m, piece, ply);
            }
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

    void updatePv(int ply, Move m) {
        pv[ply][0] = m;
        std::memcpy(&pv[ply][1], pv[ply + 1], sizeof(Move) * size_t(pvLength[ply + 1]));
        pvLength[ply] = pvLength[ply + 1] + 1;
    }

    void setMove(int ply, Move m, int piece) {
        Stack* s = ss(ply);
        s->move = m;
        s->piece = piece;
        s->to = toSq(m);
    }

    int qsearch(int alpha, int beta, int ply);
    int negamax(int depth, int alpha, int beta, int ply, bool cutNode);
    void iterate(bool verbose);
};

static std::vector<std::unique_ptr<Worker>> workers;

static uint64_t totalNodes() {
    uint64_t n = 0;
    for (auto& w : workers) n += w->nodes.load(std::memory_order_relaxed);
    return n;
}

static uint64_t totalTbHits() {
    uint64_t n = 0;
    for (auto& w : workers) n += w->tbHits;
    return n;
}

void setThreads(int count) {
    count = std::clamp(count, 1, 256);
    while (int(workers.size()) > count) workers.pop_back();
    while (int(workers.size()) < count) {
        workers.emplace_back(new Worker());
        workers.back()->id = int(workers.size()) - 1;
    }
}

int threads() { return std::max(int(workers.size()), 1); }

void clearHash() {
    if (!table) setHashMb(16);
    for (size_t i = 0; i < tableSize; i++)
        for (TTSlot& s : table[i].slot) { s.check = 0; s.data = 0; }
    for (auto& w : workers) w->clearStats();
}

namespace {
struct DefaultSetup {
    DefaultSetup() { setHashMb(16); }                    // threads are made on first use, after board tables exist
} defaultSetup;
}

// ---------------------------------------------------------------- quiescence search
// The network only saw calm positions, so keep playing captures until the position settles.
int Worker::qsearch(int alpha, int beta, int ply) {
    pvLength[ply] = 0;
    countNode();
    if (stopped()) return 0;
    if (board.halfmove >= 100 || board.insufficientMaterial() || board.isRepetition()) return 0;
    bool inCheck = board.inCheck();
    if (ply >= MAX_PLY - 1) return inCheck ? 0 : nnue::evaluate(board);
    bool pvNode = beta - alpha > 1;

    TTData tt;
    bool hit = ttProbe(board.hash, tt);
    int ttScore = hit ? scoreFromTable(tt.score, ply) : 0;
    if (!pvNode && hit && (tt.bound & (ttScore >= beta ? BOUND_LOWER : BOUND_UPPER))) return ttScore;

    int best = -MATE, standPat = -MATE, rawEval = EVAL_NONE;
    if (!inCheck) {
        rawEval = hit && tt.eval != EVAL_NONE ? tt.eval : nnue::evaluate(board);
        standPat = rawEval;
        if (hit && (tt.bound & (ttScore > standPat ? BOUND_LOWER : BOUND_UPPER))) standPat = ttScore;
        if (standPat >= beta) {
            if (!hit) ttStore(board.hash, NO_MOVE, scoreToTable(standPat, ply), rawEval, 0, BOUND_LOWER, pvNode);
            return standPat;
        }
        alpha = std::max(alpha, standPat);
        best = standPat;
    }

    MoveList list;
    board.generate(list, !inCheck);                       // in check, every escape must be tried
    int scores[256];
    scoreMoves(list, scores, hit ? tt.move : NO_MOVE, ply);
    int legal = 0;
    Move bestMove = NO_MOVE;
    for (int i = 0; i < list.size; i++) {
        Move m = pickNext(list, scores, i);
        if (!inCheck) {
            if (promoOf(m) && promoOf(m) != QUEEN) continue;
            if (!promoOf(m)) {
                int futility = standPat + SEE_VALUE[capturedType(m)] + 150;
                if (futility <= alpha) { best = std::max(best, futility); continue; }   // can't get back to alpha
            }
            if (!board.see(m, 0)) continue;                // loses material
        }
        int piece = board.squares[fromSq(m)];
        board.make(m);
        if (board.sideJustMovedInCheck()) { board.unmake(); continue; }
        legal++;
        setMove(ply, m, piece);
        int score = -qsearch(-beta, -alpha, ply + 1);
        board.unmake();
        if (stopped()) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) {
                bestMove = m;
                if (pvNode) updatePv(ply, m);
                if (score >= beta) break;
                alpha = score;
            }
        }
    }
    if (inCheck && legal == 0) return -MATE + ply;
    ttStore(board.hash, bestMove, scoreToTable(best, ply), rawEval, 0, best >= beta ? BOUND_LOWER : BOUND_UPPER,
            pvNode || (hit && tt.pv));
    return best;
}

// ---------------------------------------------------------------- main search
int Worker::negamax(int depth, int alpha, int beta, int ply, bool cutNode) {
    pvLength[ply] = 0;
    const bool root = ply == 0, pvNode = beta - alpha > 1;
    Stack* s = ss(ply);
    bool inCheck = board.inCheck();
    if (inCheck && ply < 2 * rootDepth) depth++;           // the network never saw these positions
    if (depth <= 0) return qsearch(alpha, beta, ply);

    countNode();
    if (stopped()) return 0;
    if (pvNode) selDepth = std::max(selDepth, ply + 1);

    if (!root) {
        if (board.isRepetition() || board.halfmove >= 100 || board.insufficientMaterial()) return 0;
        if (ply >= MAX_PLY - 1) return inCheck ? 0 : nnue::evaluate(board);
        alpha = std::max(alpha, -MATE + ply);              // a shorter mate is already known
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    const Move excluded = s->excluded;
    TTData tt{};
    bool ttHit = !excluded && ttProbe(board.hash, tt);
    int ttScore = ttHit ? scoreFromTable(tt.score, ply) : 0;
    Move ttMove = ttHit ? tt.move : NO_MOVE;
    if (root && pvLength[0] == 0 && result.best && !ttMove) ttMove = result.best;
    bool ttCapture = ttMove && board.isCapture(ttMove);
    // Positions that have been on a principal variation are likely to matter again.
    const bool ttPv = excluded ? s->ttPv : pvNode || (ttHit && tt.pv);
    s->ttPv = ttPv;

    if (!pvNode && ttHit && tt.depth >= depth && (tt.bound & (ttScore >= beta ? BOUND_LOWER : BOUND_UPPER))) {
        if (ttMove && ttScore >= beta && !ttCapture && !promoOf(ttMove))
            updateQuiet(ttMove, board.squares[fromSq(ttMove)], ply, statBonus(depth));
        if (board.halfmove < 90) return ttScore;
    }

    // Endgame tablebases: right after a capture or pawn move, the exact result is known.
    if (!root && !excluded && board.halfmove == 0 && !board.castling && board.pieceCount() <= syzygy::largest()) {
        int wdl = syzygy::probeWdl(board);
        if (wdl >= 0) {
            tbHits++;
            int value = wdl == 4 ? TB_WIN - ply : wdl == 0 ? -TB_WIN + ply : wdl - 2;
            int bound = wdl == 4 ? BOUND_LOWER : wdl == 0 ? BOUND_UPPER : BOUND_EXACT;
            if (bound == BOUND_EXACT || (bound == BOUND_LOWER ? value >= beta : value <= alpha)) {
                ttStore(board.hash, NO_MOVE, scoreToTable(value, ply), EVAL_NONE, depth + 6, bound, ttPv);
                return value;
            }
        }
    }

    // The position's own evaluation, before searching any moves.
    int eval = EVAL_NONE;
    if (inCheck) {
        s->staticEval = EVAL_NONE;
    } else if (excluded) {
        eval = s->staticEval;                              // same position, worked out already
    } else {
        int raw = ttHit && tt.eval != EVAL_NONE ? tt.eval : nnue::evaluate(board);
        s->staticEval = eval = raw;
        if (ttHit && (tt.bound & (ttScore > eval ? BOUND_LOWER : BOUND_UPPER))) eval = ttScore;
        if (!ttHit) ttStore(board.hash, NO_MOVE, 0, raw, 0, BOUND_NONE, ttPv);
    }
    // Is our position better than two plies ago? If so, pruning can be bolder.
    bool improving = false;
    if (!inCheck) {
        if (ss(ply - 2)->staticEval != EVAL_NONE) improving = s->staticEval > ss(ply - 2)->staticEval;
        else if (ss(ply - 4)->staticEval != EVAL_NONE) improving = s->staticEval > ss(ply - 4)->staticEval;
        else improving = true;
    }

    if (!pvNode && !inCheck && !excluded) {
        // Razoring: hopelessly below alpha, so only captures could help.
        if (depth <= 3 && eval + 200 + 250 * depth <= alpha) {
            int v = qsearch(alpha, alpha + 1, ply);
            if (v <= alpha) return v;
        }
        // Reverse futility: so far above beta at low depth that the opponent can't catch up.
        if (depth <= 8 && eval < TB_BOUND && eval - 80 * (depth - improving) >= beta) return eval;
        // Null move: if passing still beats beta, a real move almost certainly would too.
        if (depth >= 3 && eval >= beta && s->staticEval >= beta && ss(ply - 1)->move != NO_MOVE
            && beta > -TB_BOUND && board.hasNonPawnMaterial(board.side)) {
            int R = 4 + depth / 3 + std::min((eval - beta) / 200, 3);
            setMove(ply, NO_MOVE, 0);
            board.makeNull();
            int v = -negamax(depth - R, -beta, -beta + 1, ply + 1, !cutNode);
            board.unmakeNull();
            if (stopped()) return 0;
            if (v >= beta) return v >= TB_BOUND ? beta : v;
        }
        // ProbCut: a capture that beats beta by a wide margin in a shallow search is very likely
        // to beat beta in a full one.
        int probBeta = beta + 200;
        if (depth >= 5 && std::abs(beta) < TB_BOUND && !(ttHit && tt.depth >= depth - 3 && ttScore < probBeta)) {
            MoveList caps;
            board.generate(caps, true);
            int capScores[256];
            scoreMoves(caps, capScores, ttCapture ? ttMove : NO_MOVE, ply);
            for (int i = 0; i < caps.size; i++) {
                Move m = pickNext(caps, capScores, i);
                if (promoOf(m) && promoOf(m) != QUEEN) continue;
                if (!board.see(m, probBeta - s->staticEval)) continue;
                int piece = board.squares[fromSq(m)];
                board.make(m);
                if (board.sideJustMovedInCheck()) { board.unmake(); continue; }
                setMove(ply, m, piece);
                int v = -qsearch(-probBeta, -probBeta + 1, ply + 1);
                if (v >= probBeta) v = -negamax(depth - 4, -probBeta, -probBeta + 1, ply + 1, !cutNode);
                board.unmake();
                if (stopped()) return 0;
                if (v >= probBeta) {
                    ttStore(board.hash, m, scoreToTable(v, ply), s->staticEval, depth - 3, BOUND_LOWER, ttPv);
                    return v;
                }
            }
        }
    }

    // No stored best move to try first: this node is probably less important than it looks.
    if (!excluded && depth >= 4 && ttMove == NO_MOVE && (pvNode || cutNode)) depth--;

    MoveList list;
    board.generate(list);
    int scores[256];
    scoreMoves(list, scores, ttMove, ply);

    Move quietsTried[64], noisyTried[32];
    int quietCount = 0, noisyCount = 0;
    int best = -MATE, moveCount = 0, startAlpha = alpha;
    Move bestMove = NO_MOVE;
    bool skipQuiets = false;
    Move counter = counterMoves[ss(ply - 1)->piece][ss(ply - 1)->to];

    for (int i = 0; i < list.size; i++) {
        Move m = pickNext(list, scores, i);
        if (m == excluded) continue;
        bool capture = board.isCapture(m), quiet = !capture && !promoOf(m);
        if (skipQuiets && quiet) continue;
        int piece = board.squares[fromSq(m)];
        int hist = quiet ? quietScore(m, piece, ply) : 0;
        bool goodNoisy = !quiet && (m == ttMove || scores[i] >= SCORE_GOOD_NOISY);

        // Pruning: skip moves that are very unlikely to matter (never before one move is searched).
        if (!root && best > -TB_BOUND && board.hasNonPawnMaterial(board.side)) {
            int lmrDepth = std::max(depth - 1 - lmrTable[std::min(depth, 63)][std::min(moveCount + 1, 63)], 0);
            if (quiet) {
                if (depth <= 8 && moveCount >= (3 + depth * depth) / (2 - improving)) { skipQuiets = true; continue; }
                if (!inCheck && lmrDepth <= 8 && s->staticEval + 100 + 120 * lmrDepth <= alpha) continue;
                if (lmrDepth <= 4 && hist < -3000 * depth) continue;
                if (!board.see(m, -25 * lmrDepth * lmrDepth)) continue;
            } else if (depth <= 8 && !board.see(m, -90 * depth)) {
                continue;
            }
        }

        // Singular extension: if every other move is clearly worse than the stored best move,
        // that move deserves a deeper look.
        int extension = 0;
        if (!root && m == ttMove && !excluded && depth >= 8 && (tt.bound & BOUND_LOWER) && tt.depth >= depth - 3
            && std::abs(ttScore) < TB_BOUND && ply < 2 * rootDepth) {
            int sBeta = ttScore - 2 * depth;
            s->excluded = m;
            int v = negamax((depth - 1) / 2, sBeta - 1, sBeta, ply, cutNode);
            s->excluded = NO_MOVE;
            if (stopped()) return 0;
            if (v < sBeta) {
                extension = 1;
                if (!pvNode && v < sBeta - 25 && ss(ply - 1)->doubleExtensions < 8) extension = 2;
            } else if (sBeta >= beta) {
                return sBeta;                              // several moves beat beta: cut now
            } else if (ttScore >= beta) {
                extension = -1;
            }
        }
        s->doubleExtensions = ss(ply - 1)->doubleExtensions + (extension == 2);

        board.make(m);
        if (board.sideJustMovedInCheck()) { board.unmake(); continue; }
        moveCount++;
        setMove(ply, m, piece);
        uint64_t nodesBefore = nodes.load(std::memory_order_relaxed);
        bool givesCheck = board.inCheck();
        int newDepth = depth - 1 + extension;

        int score = -MATE;
        if (depth >= 2 && moveCount > 1 + pvNode && (quiet || !goodNoisy)) {
            // Late moves are probably bad: try them shallower, and re-search if they surprise us.
            int r = lmrTable[std::min(depth, 63)][std::min(moveCount, 63)];
            r += !improving;
            r -= pvNode;
            r -= ttPv && !pvNode;                          // was on a principal variation: look closer
            r += cutNode;
            if (m == s->killers[0] || m == s->killers[1] || m == counter) r--;
            if (givesCheck) r--;
            if (ttCapture && quiet) r++;
            if (quiet) r -= hist / 8000;
            int d = std::clamp(newDepth - r, 1, std::max(newDepth, 1));
            score = -negamax(d, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && d < newDepth) score = -negamax(newDepth, -alpha - 1, -alpha, ply + 1, !cutNode);
        } else if (!pvNode || moveCount > 1) {
            score = -negamax(newDepth, -alpha - 1, -alpha, ply + 1, !cutNode);
        }
        if (pvNode && (moveCount == 1 || (score > alpha && (root || score < beta))))
            score = -negamax(newDepth, -beta, -alpha, ply + 1, false);
        board.unmake();
        if (stopped()) return 0;
        if (root && id == 0) rootNodes[fromSq(m)][toSq(m)] += nodes.load(std::memory_order_relaxed) - nodesBefore;

        if (score > best) {
            best = score;
            if (score > alpha) {
                bestMove = m;
                if (pvNode) updatePv(ply, m);
                if (score >= beta) break;
                alpha = score;
            }
        }
        if (quiet && quietCount < 64) quietsTried[quietCount++] = m;
        else if (!quiet && noisyCount < 32) noisyTried[noisyCount++] = m;
    }

    if (moveCount == 0) return excluded ? alpha : inCheck ? -MATE + ply : 0;   // checkmate or stalemate

    // Learn from the cutoff: reward the move that caused it, penalise the ones tried before it.
    if (best >= beta) {
        int bonus = statBonus(depth + (best > beta + 80));    // a convincing cutoff teaches more
        if (!board.isCapture(bestMove) && !promoOf(bestMove)) {
            if (s->killers[0] != bestMove) {
                s->killers[1] = s->killers[0];
                s->killers[0] = bestMove;
            }
            counterMoves[ss(ply - 1)->piece][ss(ply - 1)->to] = bestMove;
            updateQuiet(bestMove, board.squares[fromSq(bestMove)], ply, bonus);
            for (int i = 0; i < quietCount; i++) updateQuiet(quietsTried[i], board.squares[fromSq(quietsTried[i])], ply, -bonus);
        } else {
            gravity(captureHistory[board.squares[fromSq(bestMove)]][toSq(bestMove)][capturedType(bestMove)], bonus);
        }
        for (int i = 0; i < noisyCount; i++) {
            Move m = noisyTried[i];
            gravity(captureHistory[board.squares[fromSq(m)]][toSq(m)][capturedType(m)], -bonus);
        }
    }

    if (!excluded)
        ttStore(board.hash, bestMove, scoreToTable(best, ply), s->staticEval, depth,
                best >= beta ? BOUND_LOWER : best > startAlpha ? BOUND_EXACT : BOUND_UPPER, ttPv);
    return best;
}

// ---------------------------------------------------------------- iterative deepening
static std::string scoreText(int score) {
    if (score >= MATE_BOUND) return "mate " + std::to_string((MATE - score + 1) / 2);
    if (score <= -MATE_BOUND) return "mate -" + std::to_string((MATE + score + 1) / 2);
    return "cp " + std::to_string(score);
}

void Worker::iterate(bool verbose) {
    int lastScore = 0, stability = 0;
    Move lastBest = NO_MOVE;
    for (int depth = 1; depth <= limit.depth; depth++) {
        rootDepth = depth;
        selDepth = 0;
        int window = 20, alpha = -MATE, beta = MATE, score = 0;
        if (depth >= 4) {                                  // expect a score near the last one
            alpha = std::max(lastScore - window, -MATE);
            beta = std::min(lastScore + window, MATE);
        }
        for (;;) {
            score = negamax(depth, alpha, beta, 0, false);
            if (stopped()) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(score - window, -MATE);
            } else if (score >= beta) {
                beta = std::min(score + window, MATE);
            } else {
                break;
            }
            window += window / 2;
            if (window > 1000) { alpha = -MATE; beta = MATE; }
        }

        // A partly finished depth is only trusted if it found a best move at the root.
        if (pvLength[0] > 0) {
            result.best = pv[0][0];
            result.ponder = pvLength[0] > 1 ? pv[0][1] : NO_MOVE;
            if (!stopped()) {
                result.score = score;
                result.depth = depth;
            }
        }
        if (stopped()) break;
        if (id != 0) continue;

        if (verbose) {
            int64_t ms = std::max<int64_t>(elapsedMs(), 1);
            uint64_t n = totalNodes();
            std::string line = "info depth " + std::to_string(depth) + " seldepth " + std::to_string(selDepth)
                + " score " + scoreText(score) + " nodes " + std::to_string(n)
                + " nps " + std::to_string(n * 1000 / uint64_t(ms)) + " hashfull " + std::to_string(hashFull())
                + " tbhits " + std::to_string(totalTbHits()) + " time " + std::to_string(ms) + " pv";
            for (int i = 0; i < pvLength[0]; i++) line += " " + moveToUci(pv[0][i]);
            std::printf("%s\n", line.c_str());
            std::fflush(stdout);
        }

        // Time: stop sooner when the best move has been stable and took most of the effort, and
        // think longer when it keeps changing or the score is falling.
        Move bestNow = result.best;
        stability = bestNow == lastBest ? std::min(stability + 1, 10) : 0;
        if (!pondering.load() && limit.softMs >= 0) {
            static const double STABILITY[6] = {2.0, 1.4, 1.1, 0.9, 0.8, 0.75};
            uint64_t mine = std::max<uint64_t>(nodes.load(std::memory_order_relaxed), 1);
            double share = double(rootNodes[fromSq(bestNow)][toSq(bestNow)]) / double(mine);
            double effort = std::clamp(2.0 - 1.5 * share, 0.5, 1.6);
            double falling = depth > 1 ? std::clamp(1.0 + (lastScore - score) / 120.0, 0.85, 1.6) : 1.0;
            double scale = STABILITY[std::min(stability, 5)] * effort * falling;
            if (elapsedMs() >= int64_t(double(limit.softMs) * scale)) break;
        }
        if (!pondering.load() && limit.softMs >= 0 && std::abs(score) >= MATE_BOUND && depth >= (MATE - std::abs(score)) + 4) break;
        lastScore = score;
        lastBest = bestNow;
    }
}

static void prepare(Worker& w, const Board& board) {
    w.board = board;
    w.nodes = 0;
    w.tbHits = 0;
    w.result = Result();
    for (Stack& s : w.stackBase) s = Stack();
    std::memset(w.rootNodes, 0, sizeof(w.rootNodes));
    std::memset(w.pvLength, 0, sizeof(w.pvLength));
}

Result think(Board& board, const Limits& limits, bool verbose) {
    if (workers.empty()) setThreads(1);
    limit = limits;
    stopAll = false;
    generation++;
    clockStart = nowMs();

    Result result;
    MoveList legal;
    board.legalMoves(legal);
    if (legal.size == 0) return result;

    for (auto& w : workers) prepare(*w, board);
    std::vector<std::thread> helpers;
    for (size_t i = 1; i < workers.size(); i++) helpers.emplace_back([i] { workers[i]->iterate(false); });
    workers[0]->iterate(verbose);
    stopAll = true;
    for (auto& t : helpers) t.join();

    result = workers[0]->result;
    result.nodes = totalNodes();
    if (result.best == NO_MOVE) result.best = legal.moves[0];   // always have something to play

    // The reply to ponder on: from the principal variation, or failing that the hash table.
    if (result.ponder == NO_MOVE) {
        board.make(result.best);
        TTData tt;
        MoveList replies;
        board.legalMoves(replies);
        if (ttProbe(board.hash, tt))
            for (int i = 0; i < replies.size; i++)
                if (replies.moves[i] == tt.move) result.ponder = tt.move;
        board.unmake();
    }
    return result;
}

Move pickWeak(Board& board, int depth, int temperature) {
    static std::mt19937 rng(std::random_device{}());
    if (workers.empty()) setThreads(1);
    limit = Limits();
    stopAll = false;
    stopRequested = false;
    clockStart = nowMs();
    Worker& w = *workers[0];
    prepare(w, board);
    w.rootDepth = depth;

    MoveList legal;
    w.board.legalMoves(legal);
    if (legal.size == 0) return NO_MOVE;
    int scores[256], best = -MATE;
    for (int i = 0; i < legal.size; i++) {
        Move m = legal.moves[i];
        w.setMove(0, m, w.board.squares[fromSq(m)]);
        w.board.make(m);
        scores[i] = std::clamp(-w.negamax(depth - 1, -MATE, MATE, 1, false), -3000, 3000);
        w.board.unmake();
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
