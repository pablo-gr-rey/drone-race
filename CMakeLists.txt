cmake_minimum_required(VERSION 3.18)
project(drone_race LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

# set(ENV{PKG_CONFIG_PATH} "~/.local/lib/pkgconfig")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wshadow -std=c++20)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>)
endif()


find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

add_executable(drone_race
    src/main.cpp
    src/engine.cpp
    src/controllers.cpp
    src/config.cpp
    src/mppi_controller.cu
    src/kernels.cu
    src/protocol.cpp
)

target_compile_features(drone_race PRIVATE cxx_std_20)

target_include_directories(drone_race PRIVATE include proto ${ZMQ_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_link_directories(drone_race PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(drone_race PRIVATE
    CUDA::curand CUDA::cudart
    ${ZMQ_LIBRARIES}
)
set_target_properties(drone_race PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "70;80;86;89;90"
)
