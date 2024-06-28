@echo off

mkdir debug
pushd debug

set SRC=../src/*.c

clang-cl %SRC% kernel32.lib /link /SUBSYSTEM:CONSOLE /OUT:test.exe

popd
