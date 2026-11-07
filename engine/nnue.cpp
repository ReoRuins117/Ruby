// nnue.cpp - see nnue.h
// This recalculates the whole network for each position. That is simple and easy to check;
// incremental updates after each move are a later speed-up.
#include "nnue.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace nnue {

static const int NUM_FEATURES = 768;
static const int MAX_HIDDEN = 1024;
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
    if (!in || std::memcmp(magic, "RUBY", 4) || version != 1 || inputs != NUM_FEATURES || h == 0 || h > MAX_HIDDEN)
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

int evaluate(const Board& board) {
    alignas(64) int32_t accumulator[2][MAX_HIDDEN];
    for (int view = 0; view < 2; view++)
        for (int i = 0; i < hidden; i++) accumulator[view][i] = ftBias[i];

    // Square numbering matches the dataset: index 0 is a8 from that side's view.
    for (int sq = 0; sq < 64; sq++) {
        int piece = board.squares[sq];
        if (!piece) continue;
        for (int view = 0; view < 2; view++) {
            int index = view == WHITE ? sq ^ 56 : sq;
            int code = typeOf(piece) + (colorOf(piece) == view ? 0 : 6);
            const int16_t* weights = &ftWeights[size_t(index * 12 + code - 1) * hidden];
            for (int i = 0; i < hidden; i++) accumulator[view][i] += weights[i];
        }
    }

    alignas(64) int32_t input[2 * MAX_HIDDEN];   // mover first, then opponent
    for (int i = 0; i < hidden; i++) {
        input[i] = std::clamp(accumulator[board.side][i], 0, 255);
        input[hidden + i] = std::clamp(accumulator[board.side ^ 1][i], 0, 255);
    }

    int32_t h1[64], h2[64];
    for (int j = 0; j < hidden2; j++) {
        const int16_t* weights = &l1Weights[size_t(j) * 2 * hidden];
        int32_t sum = l1Bias[j];
        for (int i = 0; i < 2 * hidden; i++) sum += weights[i] * input[i];
        h1[j] = std::clamp(sum >> 8, 0, 255);
    }
    for (int j = 0; j < hidden2; j++) {
        const int16_t* weights = &l2Weights[size_t(j) * hidden2];
        int32_t sum = l2Bias[j];
        for (int i = 0; i < hidden2; i++) sum += weights[i] * h1[i];
        h2[j] = std::clamp(sum >> 8, 0, 255);
    }
    int64_t output = outBias;
    for (int i = 0; i < hidden2; i++) output += int64_t(outWeights[i]) * h2[i];
    return int(output * 400 / (1024 * 255));
}

}  // namespace nnue
