#pragma once

#include "config.h"
#include "controllers.h"

#include <curand_kernel.h>
#include <cstddef>

__global__ void initRNGKernel(curandState* states, unsigned long long seed, int n);
__global__ void generateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int nTimesteps, int N, int dim, int nModels);
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
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
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
    int nModels,
    int    nSamples,
    int    nTimesteps,
    int    dim,
    float* __restrict__ nu);

// Clamp nominal actions to maxAccel (avoids them drifting to high-magnitude areas from which it's difficult to recover)
__global__ void clampNominalKernel(
    float* nominal,
    float maxAccel,
    int nModels,
    int T,
    int dim);
