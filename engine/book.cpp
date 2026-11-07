// book.cpp - see book.h
#include "book.h"
#include <cctype>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace book {

static const int MIN_GAMES = 10;          // ignore positions seen fewer times than this
static const double MIN_SHARE = 0.10;     // ignore moves played in under 10% of games from a position

using Entry = std::map<std::string, int>;
static std::unordered_map<std::string, Entry> positions;

// Book keys contain move counters, but the same position can arise at different move numbers,
// so only the first four FEN fields are compared.
static std::string shortKey(const std::string& fen) {
    size_t pos = 0;
    for (int spaces = 0; spaces < 4 && pos != std::string::npos; spaces++) {
        pos = fen.find(' ', pos);
        if (pos != std::string::npos && spaces < 3) pos++;
    }
    return pos == std::string::npos ? fen : fen.substr(0, pos);
}

static void skipSpace(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
}

static bool readString(const std::string& s, size_t& i, std::string& out) {
    skipSpace(s, i);
    if (i >= s.size() || s[i] != '"') return false;
    size_t end = s.find('"', i + 1);
    if (end == std::string::npos) return false;
    out = s.substr(i + 1, end - i - 1);
    i = end + 1;
    return true;
}

static bool expect(const std::string& s, size_t& i, char c) {
    skipSpace(s, i);
    if (i >= s.size() || s[i] != c) return false;
    i++;
    return true;
}

bool load(const std::string& path) {
    if (!positions.empty()) return true;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    size_t i = 0;
    if (!expect(text, i, '{')) return false;
    std::string fen, move;
    while (readString(text, i, fen)) {
        if (!expect(text, i, ':') || !expect(text, i, '{')) break;
        Entry& entry = positions[shortKey(fen)];
        while (readString(text, i, move)) {
            if (!expect(text, i, ':')) break;
            skipSpace(text, i);
            int count = 0;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])))
                count = count * 10 + (text[i++] - '0');
            entry[move] += count;
            if (!expect(text, i, ',')) break;
        }
        if (!expect(text, i, '}')) break;
        if (!expect(text, i, ',')) break;
    }
    return !positions.empty();
}

size_t size() { return positions.size(); }

bool pick(Board& board, Move& out) {
    auto found = positions.find(shortKey(board.fen(false)));
    if (found == positions.end()) return false;
    int total = 0;
    for (auto& item : found->second) total += item.second;
    if (total < MIN_GAMES) return false;

    std::vector<Move> moves;
    std::vector<int> weights;
    for (auto& item : found->second) {
        Move m;
        if (item.second >= MIN_SHARE * total && board.parseUci(item.first, m)) {
            moves.push_back(m);
            weights.push_back(item.second);
        }
    }
    if (moves.empty()) return false;
    static std::mt19937 random{std::random_device{}()};
    std::discrete_distribution<int> choose(weights.begin(), weights.end());
    out = moves[choose(random)];
    return true;
}

}  // namespace book
