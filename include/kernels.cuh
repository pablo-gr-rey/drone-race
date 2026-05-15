#pragma once

#include "config.h"
#include "controllers.h"

#include <curand_kernel.h>
#include <cstddef>

__global__ void initRNGKernel(curandState*, unsigned long long, int);
__global__ void generateNoiseKernel(float*, curandState*, float, int, int, int);
__global__ void fullRolloutKernel(
    int controlAgent,
    // const DeviceEnvironmentConfig envConfig,
    const EnvironmentConfig envConfig,
    const MPPIConfig mc,
    const float* __restrict__ initPos,
    const float* __restrict__ initVel,
    const float* __restrict__ initS,
    const int* __restrict__ initLaps,
    const int* __restrict__ initGates,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    float* __restrict__ finalPos,
    float* __restrict__ finalVel,
    float* __restrict__ finalS,
    int* __restrict__ finalLaps,
    int* __restrict__ finalGates,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts,
    int nTP, int N);

void minReduceCUB(const float* __restrict__ d_costs, float* __restrict__ d_minCost, int N, void* __restrict__ d_temp_storage, size_t temp_storage_bytes);

// Weighted average of noise
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernel(
    const float* __restrict__ costs,
    const float* __restrict__ noise,    // (nTimesteps, nSamples, dim)
    float* __restrict__ nominalAction,  // (nTimesteps, dim)
    float  minCost,
    float  invTemperature,
    int    nSamples,
    int    nTimesteps,
    int    dim);
