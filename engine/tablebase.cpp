// tablebase.cpp - see tablebase.h
// Uses WinHTTP, which is built into Windows, so no extra library is needed.
#include "tablebase.h"
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>

static std::string httpGet(const std::wstring& path) {
    std::string body;
    HINTERNET session = WinHttpOpen(L"Ruby-chess-engine", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) return body;
    WinHttpSetTimeouts(session, 1000, 1000, 1000, 1500);   // never let a slow network eat much clock
    HINTERNET connection = WinHttpConnect(session, L"tablebase.lichess.ovh", INTERNET_DEFAULT_HTTPS_PORT, 0);
    HINTERNET request = connection ? WinHttpOpenRequest(connection, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                                        WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE) : nullptr;
    DWORD status = 0, statusSize = sizeof(status);
    if (request && WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
        && WinHttpReceiveResponse(request, nullptr)
        && WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize, WINHTTP_NO_HEADER_INDEX)
        && status == 200) {
        char buffer[4096];
        DWORD got = 0;
        while (WinHttpReadData(request, buffer, sizeof(buffer), &got) && got) body.append(buffer, got);
    }
    if (request) WinHttpCloseHandle(request);
    if (connection) WinHttpCloseHandle(connection);
    WinHttpCloseHandle(session);
    return body;
}
#else
static std::string httpGet(const std::wstring&) { return ""; }
#endif

static std::string valueAfter(const std::string& text, const std::string& label, size_t from, size_t& end) {
    size_t start = text.find(label, from);
    if (start == std::string::npos) return "";
    start += label.size();
    end = text.find('"', start);
    return end == std::string::npos ? "" : text.substr(start, end - start);
}

namespace tablebase {

bool probe(Board& board, Move& out, std::string& category) {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point retryAfter = Clock::now();
    if (board.pieceCount() > 7 || board.castling || Clock::now() < retryAfter) return false;

    std::wstring path = L"/standard?fen=";
    for (char c : board.fen()) path += c == ' ' ? L'_' : wchar_t(c);
    std::string reply = httpGet(path);
    if (reply.empty()) {
        retryAfter = Clock::now() + std::chrono::seconds(30);   // offline or rate limited: don't keep waiting
        return false;
    }

    // The reply lists the position's result, then the legal moves with the best first.
    // A move's own "category" is from the opponent's side, so it is deliberately not used here.
    size_t movesAt = reply.find("\"moves\":[");
    if (movesAt == std::string::npos) return false;
    size_t end = 0;
    category = valueAfter(reply.substr(0, movesAt), "\"category\":\"", 0, end);
    if (category.empty() || category == "unknown") return false;
    return board.parseUci(valueAfter(reply, "\"uci\":\"", movesAt, end), out);
}

}  // namespace tablebase
