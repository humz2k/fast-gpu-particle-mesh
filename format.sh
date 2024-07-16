#!/usr/bin/env bash

clang-format -i src/**.cpp
clang-format -i src/**.cu
clang-format -i src/**.hpp
clang-format -i drivers/**.cpp
clang-format -i include/**.cpp