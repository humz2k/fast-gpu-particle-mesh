#!/usr/bin/env bash

clang-format -i src/**.cpp
clang-format -i src/cuda/**.cu
clang-format -i src/**.hpp
clang-format -i drivers/**.cpp
clang-format -i drivers/**.hpp
clang-format -i include/**.hpp