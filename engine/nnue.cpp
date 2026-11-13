// nnue.cpp - see nnue.h
// The first layer is updated move by move (see Accumulator in board.h); the later layers are small
// and are recalculated for each position, 16 numbers at a time with AVX2.
//
// Version 2 files have 8 king buckets: each view's own king picks which copy of the 768 inputs is
// used, and mirrors the board left-right when it is on files e-h (see train.py). When a king move
// changes its side's bucket or mirroring, that view is recalculated from scratch.
#include "nnue.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace nnue {

static const int NUM_FEATURES = 768;          // inputs per king bucket
static const int NUM_BUCKETS = 8;              // in version 2 files
static int hidden = 0, hidden2 = 0, buckets = 1;

// King bucket by the king's square from its own side's view (index 0 = a8, 56 = a1), as in train.py.
static const uint8_t KING_BUCKET[64] = {
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    4, 4, 5, 5, 5, 5, 4, 4,
    0, 1, 2, 3, 3, 2, 1, 0,
};
static std::vector<int16_t> ftWeights, ftBias, l1Weights, l2Weights, outWeights;
static std::vector<int32_t> l1Bias, l2Bias;
static int32_t outBias = 0;

template <class T>
static bool readArray(std::ifstream& in, std::vector<T>& v, size_t count) {
    v.resize(count);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(count * sizeof(T)));
    return bool(in);
}

bool load(const std::string& path) {
    hidden = 0;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4];
    uint32_t version, inputs, h, h2;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&inputs), 4);
    in.read(reinterpret_cast<char*>(&h), 4);
    in.read(reinterpret_cast<char*>(&h2), 4);
    if (!in || std::memcmp(magic, "RUBY", 4) || h == 0 || h > ACC_MAX || h % 16 || h2 == 0 || h2 > 64 || h2 % 16)
        return false;
    if (!((version == 1 && inputs == NUM_FEATURES) || (version == 2 && inputs == NUM_FEATURES * NUM_BUCKETS)))
        return false;
    bool ok = readArray(in, ftWeights, size_t(inputs) * h)
           && readArray(in, ftBias, h)
           && readArray(in, l1Weights, size_t(h2) * 2 * h)
           && readArray(in, l1Bias, h2)
           && readArray(in, l2Weights, size_t(h2) * h2)
           && readArray(in, l2Bias, h2)
           && readArray(in, outWeights, h2);
    in.read(reinterpret_cast<char*>(&outBias), 4);
    if (!ok || !in) return false;
    in.peek();
    if (!in.eof()) return false;                 // extra bytes mean this isn't the format we expect
    hidden = int(h);
    hidden2 = int(h2);
    buckets = int(inputs) / NUM_FEATURES;
    return true;
}

bool loaded() { return hidden > 0; }

// ---------------------------------------------------------------- first layer
// The bucket and mirroring a view uses when its king stands on kingSquare: bucket * 2 + mirrored.
static inline int kingKey(int kingSquare, int view) {
    if (buckets == 1) return 0;
    int index = view == WHITE ? kingSquare ^ 56 : kingSquare;
    return KING_BUCKET[index] * 2 + ((index & 7) >= 4);
}

// Feature numbering matches the dataset: square index 0 is a8 from that side's view.
static inline const int16_t* featureWeights(int piece, int square, int view, int key) {
    int index = view == WHITE ? square ^ 56 : square;
    if (key & 1) index ^= 7;                                   // mirrored left-right
    int code = typeOf(piece) + (colorOf(piece) == view ? 0 : 6);
    return &ftWeights[size_t((key >> 1) * NUM_FEATURES + index * 12 + code - 1) * size_t(hidden)];
}

// out = base - the rows in sub + the rows in add, 16 values at a time with AVX2. The weights only
// ever add up to values that fit in 16 bits, so the wrapping arithmetic gives the exact sum.
static void addRows(int16_t* out, const int16_t* base, const int16_t* const* sub, int subs,
                    const int16_t* const* add, int adds) {
#ifdef __AVX2__
    for (int i = 0; i < hidden; i += 16) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i));
        for (int k = 0; k < subs; k++)
            x = _mm256_sub_epi16(x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub[k] + i)));
        for (int k = 0; k < adds; k++)
            x = _mm256_add_epi16(x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add[k] + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), x);
    }
#else
    for (int i = 0; i < hidden; i++) {
        int x = base[i];
        for (int k = 0; k < subs; k++) x -= sub[k][i];
        for (int k = 0; k < adds; k++) x += add[k][i];
        out[i] = int16_t(x);
    }
#endif
}

static void refresh(const Board& board, Accumulator& a, int view) {
    int key = kingKey(board.king[view], view);
    const int16_t* rows[32];
    int n = 0;
    for (Bitboard b = board.occupied(); b; ) {
        int sq = popLsb(b);
        rows[n++] = featureWeights(board.squares[sq], sq, view, key);
    }
    addRows(a.values[view], ftBias.data(), nullptr, 0, rows, n);
    a.kingKey[view] = uint8_t(key);
    a.computed[view] = true;
}

// The child's values are the parent's with the moved pieces taken off and put back on.
static void update(const Accumulator& parent, Accumulator& a, int view) {
    int key = parent.kingKey[view];
    const int16_t* sub[2];
    const int16_t* add[2];
    for (int k = 0; k < a.removed; k++) sub[k] = featureWeights(a.removePiece[k], a.removeSquare[k], view, key);
    for (int k = 0; k < a.added; k++) add[k] = featureWeights(a.addPiece[k], a.addSquare[k], view, key);
    addRows(a.values[view], parent.values[view], sub, a.removed, add, a.added);
    a.kingKey[view] = uint8_t(key);
    a.computed[view] = true;
}

// Did the move that led to a move this view's king into another bucket or mirroring?
static bool kingKeyChanged(const Accumulator& a, int view) {
    if (buckets == 1) return false;
    int king = makePiece(view, KING), from = -1, to = -1;
    for (int k = 0; k < a.removed; k++) if (a.removePiece[k] == king) from = a.removeSquare[k];
    for (int k = 0; k < a.added; k++) if (a.addPiece[k] == king) to = a.addSquare[k];
    return from >= 0 && to >= 0 && kingKey(from, view) != kingKey(to, view);
}

// Brings the current position's accumulator up to date, view by view: from the nearest calculated
// one before it if no king move in between changed that view's bucket, otherwise from scratch.
static const Accumulator& current(const Board& board) {
    Accumulator* stack = board.acc.data();
    int top = board.accTop;
    for (int view = 0; view < 2; view++) {
        if (stack[top].computed[view]) continue;
        int base = top;
        bool fromScratch = false;
        while (!stack[base].computed[view]) {
            if (base == 0 || kingKeyChanged(stack[base], view)) { fromScratch = true; break; }
            base--;
        }
        if (fromScratch) refresh(board, stack[top], view);
        else for (int i = base + 1; i <= top; i++) update(stack[i - 1], stack[i], view);
    }
    return stack[top];
}

// ---------------------------------------------------------------- later layers
// Sum of a[i] * b[i]. With AVX2, 16 pairs at a time; the integer result is identical either way.
static inline int32_t dot(const int16_t* a, const int16_t* b, int n) {
#ifdef __AVX2__
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 16) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, y));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += int32_t(a[i]) * b[i];
    return sum;
#endif
}

static inline void clipTo255(const int16_t* in, int16_t* out, int n) {
#ifdef __AVX2__
    const __m256i zero = _mm256_setzero_si256(), top = _mm256_set1_epi16(255);
    for (int i = 0; i < n; i += 16) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        x = _mm256_min_epi16(_mm256_max_epi16(x, zero), top);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), x);
    }
#else
    for (int i = 0; i < n; i++) out[i] = int16_t(std::clamp<int>(in[i], 0, 255));
#endif
}

int evaluate(const Board& board) {
    const Accumulator& a = current(board);
    alignas(32) int16_t input[2 * ACC_MAX];           // mover first, then opponent
    clipTo255(a.values[board.side], input, hidden);
    clipTo255(a.values[board.side ^ 1], input + hidden, hidden);

    alignas(32) int16_t h1[64], h2[64];
    for (int j = 0; j < hidden2; j++) {
        int32_t sum = l1Bias[j] + dot(&l1Weights[size_t(j) * 2 * size_t(hidden)], input, 2 * hidden);
        h1[j] = int16_t(std::clamp(sum >> 8, 0, 255));
    }
    for (int j = 0; j < hidden2; j++) {
        int32_t sum = l2Bias[j] + dot(&l2Weights[size_t(j) * size_t(hidden2)], h1, hidden2);
        h2[j] = int16_t(std::clamp(sum >> 8, 0, 255));
    }
    int64_t output = outBias;
    for (int i = 0; i < hidden2; i++) output += int64_t(outWeights[i]) * h2[i];
    return int(output * 400 / (1024 * 255));
}

}  // namespace nnue
