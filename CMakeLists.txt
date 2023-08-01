cmake_minimum_required(VERSION 3.14)

project(convbench LANGUAGES CXX)

#update_repo()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
#-mavx512f -mavx512bw


if(NOT EXISTS "${PROJECT_SOURCE_DIR}/3rdparty/json/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-D__EXCEPTIONS)  # lohmann_json: raise exceptions instead of exit

set(HEADERS
    "headers/args.h"
    "headers/avx512conv.h"
    "headers/avxconv.h"
    "headers/baseconv.h"
    "headers/consts.h"
    "headers/errors.h"
    "headers/jsonargs.h"
    "headers/log.h"
    "headers/simpleconv.h"
    "headers/sseconv.h"
    "headers/utils.h"
)

set(SOURCES
    "sources/args.cpp"
    "sources/avx512conv.cpp"
    "sources/avxconv.cpp"
    "sources/baseconv.cpp"
    "sources/jsonargs.cpp"
    "sources/log.cpp"
    "sources/main.cpp"
    "sources/simpleconv.cpp"
    "sources/sseconv.cpp"
    "sources/utils.cpp"
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(3rdparty/json EXCLUDE_FROM_ALL)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
