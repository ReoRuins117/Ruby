@echo off
rem Builds ruby.exe. Run this on each computer you use, because -march=native tunes for that CPU.
cd /d "%~dp0"
gcc -std=gnu11 -Ifathom -O3 -march=native -flto -w -c fathom\tbprobe.c -o fathom\tbprobe.o
if errorlevel 1 goto failed
g++ -std=c++17 -Ifathom -O3 -march=native -flto -static -pthread -Wall -Wextra main.cpp board.cpp nnue.cpp book.cpp tablebase.cpp syzygy.cpp search.cpp fathom\tbprobe.o -o %1ruby.exe -lwinhttp
if errorlevel 1 goto failed
echo Built %1ruby.exe
exit /b 0
:failed
echo Build FAILED.
exit /b 1
