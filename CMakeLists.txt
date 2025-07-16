cmake_minimum_required(VERSION 3.30)
project(nn_c_uda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

add_executable(nn_c_uda
        src/main.c
        src/matrix.c
        src/matrix.h
        src/nn.c
        src/nn.h
        src/cuda_kernels.cu
        src/cuda_kernels.h
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(nn_c_uda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)



if(MSVC)
    target_compile_options(nn_c_uda PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
    )
    target_compile_options(nn_c_uda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Xcompiler=/EHsc>
    )
else()
    target_link_libraries(nn_c_uda PRIVATE m)
    target_compile_options(nn_c_uda PRIVATE -Wall -Wextra -Wpedantic -g)
endif()