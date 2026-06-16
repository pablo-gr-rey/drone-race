#pragma once

#include "config.h"
#include "environment.h"
#include "controllers.h"

#include <curand_kernel.h>
#include <cstddef>

__global__ void initRNGKernel(curandState* states, unsigned long long seed, int n);

__global__ void generateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int nTimesteps, int N);

__global__ void fullRolloutKernel(
    int controlAgent,
    // const DeviceEnvironmentConfig envConfig,
    const EnvironmentConfig envConfig,
    const MPPIConfig mc,
    SimState initState,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    int* __restrict__ branchUsed,
    int* __restrict__ branchTime,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts);

__global__ void buildMaskedCostsKernel(
    const float* __restrict__ costs,       // (nModels+1, N)
    const int* __restrict__ branchUsed,    // (nModels, N)
    const int* __restrict__ branchTime,    // (nModels, N)
    const float* __restrict__ belief,      // (nModels)
    float* __restrict__ maskedCosts,       // ((nModels+1) * T, N)
    int N, int T);

void minReduceCUB(const float* __restrict__ d_in, float* __restrict__ d_out, int N, int nRows, void* __restrict__ d_temp_storage, size_t temp_storage_bytes);

// Weighted average of noise
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernel(
    const float* __restrict__ costs,
    const float* __restrict__ noise,    // (nTimesteps, nSamples, dim)
    float* __restrict__ nominalAction,  // (nTimesteps, dim)
    const float* __restrict__  minCost,
    float  invTemperature,
    int    nSamples,
    int    nTimesteps,
    float* __restrict__ nu);

__global__ void weightedAverageKernelUnified(
    const float* __restrict__ costs,       // (nModels+1, N)
    const float* __restrict__ minCosts,    // (nModels+1)
    const float* __restrict__ noise,       // (nModels+1, T, N, dim)
    const int* __restrict__ branchUsed,    // (nModels, N)
    const int* __restrict__ branchTime,    // (nModels, N)
    const float* __restrict__ belief,      // (nModels)
    float* __restrict__ nominal,           // (nModels+1, T, dim)
    float invTemp,
    int N, int T,
    float* __restrict__ nu                  // (nModels+1)
);

// Clamp nominal actions to maxAccel (avoids them drifting to high-magnitude areas from which it's difficult to recover)
__global__ void clampNominalKernel(
    float* nominal,
    float maxAccel,
    int T);

// Sample opponent models & dyamics ; if failed, atomicAdd 1 to failCount
__global__ void verifyNominalFailureKernel(
    int controlAgent,
    int nVerif,
    EnvironmentConfig envConfig,
    MPPIConfig mc,
    int nTimesteps,
    // const float* __restrict__ initPos,
    // const float* __restrict__ initVel,
    // const float* __restrict__ initS,
    // const int* __restrict__ initLaps,
    // const int* __restrict__ initGates,
    SimState initState,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,     // (nModels+1, T, dim)
    const float* __restrict__ trackPts,
    curandState* __restrict__ rngStates,
    unsigned int* __restrict__ failCount);
