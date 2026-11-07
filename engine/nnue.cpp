// nnue.cpp - see nnue.h
// The first layer is updated move by move (see Accumulator in board.h); the later layers are small
// and are recalculated for each position, 16 numbers at a time with AVX2.
#include "nnue.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace nnue {

static const int NUM_FEATURES = 768;
static int hidden = 0, hidden2 = 0;
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
    if (!in || std::memcmp(magic, "RUBY", 4) || version != 1 || inputs != NUM_FEATURES || h == 0 || h > ACC_MAX || h % 16 || h2 == 0 || h2 > 64 || h2 % 16)
        return false;
    bool ok = readArray(in, ftWeights, size_t(NUM_FEATURES) * h)
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
    return true;
}

bool loaded() { return hidden > 0; }

// ---------------------------------------------------------------- first layer
// Feature numbering matches the dataset: square index 0 is a8 from that side's view.
static inline const int16_t* featureWeights(int piece, int square, int view) {
    int index = view == WHITE ? square ^ 56 : square;
    int code = typeOf(piece) + (colorOf(piece) == view ? 0 : 6);
    return &ftWeights[size_t(index * 12 + code - 1) * size_t(hidden)];
}

static void refresh(const Board& board, Accumulator& a) {
    for (int view = 0; view < 2; view++) {
        int16_t* v = a.values[view];
        for (int i = 0; i < hidden; i++) v[i] = ftBias[i];
        for (int sq = 0; sq < 64; sq++) {
            int piece = board.squares[sq];
            if (!piece) continue;
            const int16_t* w = featureWeights(piece, sq, view);
            for (int i = 0; i < hidden; i++) v[i] = int16_t(v[i] + w[i]);
        }
    }
    a.computed = true;
}

// The child's values are the parent's with the moved pieces taken off and put back on.
static void update(const Accumulator& parent, Accumulator& a) {
    for (int view = 0; view < 2; view++) {
        const int16_t* from = parent.values[view];
        int16_t* v = a.values[view];
        std::memcpy(v, from, sizeof(int16_t) * size_t(hidden));
        for (int k = 0; k < a.removed; k++) {
            const int16_t* w = featureWeights(a.removePiece[k], a.removeSquare[k], view);
            for (int i = 0; i < hidden; i++) v[i] = int16_t(v[i] - w[i]);
        }
        for (int k = 0; k < a.added; k++) {
            const int16_t* w = featureWeights(a.addPiece[k], a.addSquare[k], view);
            for (int i = 0; i < hidden; i++) v[i] = int16_t(v[i] + w[i]);
        }
    }
    a.computed = true;
}

// Brings the current position's accumulator up to date, starting from the nearest calculated
// one before it, or from scratch if there is none.
static const Accumulator& current(const Board& board) {
    Accumulator* stack = board.acc.data();
    int top = board.accTop;
    if (stack[top].computed) return stack[top];
    int base = top;
    while (base > 0 && !stack[base].computed) base--;
    if (!stack[base].computed) {
        refresh(board, stack[top]);
    } else {
        for (int i = base + 1; i <= top; i++) update(stack[i - 1], stack[i]);
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
    for (int i = 0; i < n; i++) out[i] = int16_t(std::clamp<int>(in[i], 0, 255));
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
