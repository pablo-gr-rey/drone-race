#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#define INLINE __forceinline__
#else
#define HD
#define INLINE inline
#endif

#ifdef DEBUG
#define CUDA_CHECK(call)                                                                                                                             \
    do                                                                                                                                               \
    {                                                                                                                                                \
        cudaError_t err = (call);                                                                                                                    \
        if (err != cudaSuccess)                                                                                                                      \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err));   \
    } while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

#include <cuda/std/array>
