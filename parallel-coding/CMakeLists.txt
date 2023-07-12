cmake_minimum_required(VERSION 3.25)

project(parallel_coding)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
add_executable(parallel_coding main.cpp huffman.h huffman.cpp runlength.h runlength.cpp)
target_link_libraries(parallel_coding PUBLIC MPI::MPI_CXX)

configure_file(alphabet.txt alphabet.txt COPYONLY)
