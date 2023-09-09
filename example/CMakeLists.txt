cmake_minimum_required(VERSION 3.14)

project(vecmulsum LANGUAGES CXX)

#update_repo()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
#-mavx512f -mavx512bw

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(..)
include_directories(../headers)

add_executable(vecmulsum
  ../headers/log.h ../sources/log.cpp
  ../headers/consts.h
  main.cpp
  ../headers/errors.h
  ../headers/baseconv.h ../sources/baseconv.cpp
  ../headers/simpleconv.h ../sources/simpleconv.cpp
  ../headers/avxconv.h ../sources/avxconv.cpp
  ../headers/avx512conv.h ../sources/avx512conv.cpp
  ../headers/sseconv.h ../sources/sseconv.cpp
  ../headers/utils.h ../sources/utils.cpp
  newargs.h newargs.cpp
  mulsum.h mulsum.cpp
)
