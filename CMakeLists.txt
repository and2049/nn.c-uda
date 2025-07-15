cmake_minimum_required(VERSION 3.30)
project(nn_c_uda C)

set(CMAKE_C_STANDARD 11)

add_library(nn_c_uda STATIC nn_lib.c)
