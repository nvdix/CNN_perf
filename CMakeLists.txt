cmake_minimum_required(VERSION 3.14)

project(convbench LANGUAGES CXX)

#update_repo()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
#-mavx512f -mavx512bw


if(NOT EXISTS "${PROJECT_SOURCE_DIR}/3rdparty/json/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-D__EXCEPTIONS)  # lohmann_json: raise exceptions instead of exit

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(3rdparty/json EXCLUDE_FROM_ALL)

add_executable(convbench
  log.h log.cpp
  jsonargs.h jsonargs.cpp
  consts.h
  main.cpp
  errors.h
  baseconv.h baseconv.cpp
  simpleconv.h simpleconv.cpp
  avxconv.cpp avxconv.h
  avx512conv.cpp avx512conv.h
  sseconv.h sseconv.cpp
  utils.h utils.cpp
  args.h args.cpp
)

target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
