cmake_minimum_required(VERSION 3.18)
project(nn_c_uda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(LIB_SOURCES
        src/matrix.c
        src/matrix.h
        src/nn.c
        src/nn.h
        src/cuda_kernels.cu
        src/cuda_kernels.h
)

add_executable(nn_c_uda_benchmark src/main.c ${LIB_SOURCES})
target_link_libraries(nn_c_uda_benchmark PRIVATE ${CUDA_LIBRARIES})

add_executable(nn_c_uda_xor src/xor.c ${LIB_SOURCES})
target_link_libraries(nn_c_uda_xor PRIVATE ${CUDA_LIBRARIES})


find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
foreach(TARGET nn_c_uda_benchmark nn_c_uda_xor)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    if(MSVC)
        target_compile_options(${TARGET} PRIVATE
                $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Xcompiler=/EHsc>
        )
    else()
        target_link_libraries(${TARGET} PRIVATE m)
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -g)
    endif()
endforeach()
