cmake_minimum_required(VERSION 3.18)
project(drone_race LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CONDA_PREFIX $ENV{CONDA_PREFIX})

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

add_executable(drone_race
    src/main.cpp
    src/engine.cpp
    src/config.cpp
    src/mppi_controller.cu
    src/kernels.cu
    src/protocol.cpp
)

target_compile_options(drone_race PRIVATE 
    -Wall -Wextra -Wshadow
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -march=native>
)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(drone_race PRIVATE 
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g -G --source-in-ptx -Xcompiler=-Wall,-Wextra>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 -lineinfo>
    )
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(drone_race PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--threads 4>
    )
endif()

target_compile_features(drone_race PRIVATE cxx_std_20)
target_compile_definitions(drone_race PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
)
target_include_directories(drone_race PUBLIC 
    include
    # $ENV{CONDA_PREFIX}/include
    ${ZMQ_INCLUDE_DIRS} 
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(drone_race PRIVATE ${ZMQ_LIBRARY_DIRS})

target_link_libraries(drone_race PRIVATE
    CUDA::curand 
    CUDA::cudart
    ${ZMQ_LIBRARIES}
)

set_target_properties(drone_race PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "native"
)
