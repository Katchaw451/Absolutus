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

# Static library with all implementations except main()
file(GLOB_RECURSE ABSOLUTUS_LIB_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM ABSOLUTUS_LIB_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(absolutus_lib STATIC ${ABSOLUTUS_LIB_SOURCES})
target_include_directories(absolutus_lib PUBLIC include)

# CLI executable
add_executable(absolutus src/main.cpp)
target_link_libraries(absolutus PRIVATE absolutus_lib)

# Unit tests
enable_testing()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE absolutus_lib)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Integration tests (exit code 0 = pass)
add_test(NAME exp_chsh     COMMAND absolutus --experiment chsh    --trials 1000 --seed 42)
add_test(NAME exp_ghz_10   COMMAND absolutus --experiment ghz     --qubits 10)
add_test(NAME exp_ghz_20   COMMAND absolutus --experiment ghz     --qubits 20)
add_test(NAME exp_vqe      COMMAND absolutus --experiment vqe     --iters 200 --lr 0.1)
add_test(NAME exp_xor      COMMAND absolutus --experiment train_xor    --epochs 2000 --lr 0.1 --momentum 0.9 --seed 42)
add_test(NAME exp_qxor     COMMAND absolutus --experiment train_qxor   --epochs 2000 --lr 0.1 --momentum 0.9 --params 8 --seed 42)
add_test(NAME exp_hybrid   COMMAND absolutus --experiment train_hybrid --epochs 2000 --lr 0.1 --momentum 0.9 --qubits 2 --params 4 --seed 42)

add_test(NAME exp_train_lm COMMAND absolutus --experiment train_lm --epochs 300 --lr 0.05 --seed 42)
