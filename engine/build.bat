@echo off
rem Builds ruby.exe. Run this on each computer you use, because -march=native tunes for that CPU.
cd /d "%~dp0"
g++ -std=c++17 -O3 -march=native -flto -static -pthread -Wall -Wextra main.cpp board.cpp nnue.cpp book.cpp tablebase.cpp search.cpp -o ruby.exe -lwinhttp
if errorlevel 1 (
    echo Build FAILED.
    exit /b 1
)
echo Built ruby.exe
