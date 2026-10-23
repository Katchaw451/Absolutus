cmake_minimum_required(VERSION 3.16)
project(Absolutus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-O3 -Wall -Wextra)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  add_compile_options(-march=native)
  message(STATUS "Native architecture optimizations enabled")
endif()

file(GLOB_RECURSE ABSOLUTUS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(absolutus ${ABSOLUTUS_SOURCES})
target_include_directories(absolutus PRIVATE include)

enable_testing()
