cmake_minimum_required(VERSION 3.14)

project(vecmulsum LANGUAGES CXX)

#update_repo()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
#-mavx512f -mavx512bw

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(..)

add_executable(vecmulsum
  ../log.h ../log.cpp
  ../consts.h
  main.cpp
  ../errors.h
  ../baseconv.h ../baseconv.cpp
  ../simpleconv.h ../simpleconv.cpp
  ../avxconv.cpp ../avxconv.h
  ../avx512conv.cpp ../avx512conv.h
  ../sseconv.h ../sseconv.cpp
  ../utils.h ../utils.cpp
  newargs.h newargs.cpp
  mulsum.h mulsum.cpp
)
