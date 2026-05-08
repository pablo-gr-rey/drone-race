#pragma once

#include "config.h"
#include "device_config.cuh"

#include <curand_kernel.h>
#include <cstddef>

__global__ void initRNGKernel(curandState*, unsigned long long, int);
__global__ void generateNoiseKernel(float*, curandState*, float, int, int, int);
__global__ void fullRolloutKernel(
    int controlAgent,
    const DeviceEnvironmentConfig cfg,
    const DeviceMPPIConfig mc,
    OpponentModelType oppModel,
    const PIDConfig oppPid,
    const float* __restrict__ initPhys,
    const float* __restrict__ initS,
    const float* __restrict__ initLaps,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    float* __restrict__ finalPhys,
    float* __restrict__ finalS,
    float* __restrict__ finalLaps,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts,
    int nTP, int N);

void minReduceCUB(const float* __restrict__ d_costs, float* __restrict__ d_minCost, int N, void* __restrict__ d_temp_storage, size_t temp_storage_bytes);

// ── Weighted average of noise ────────────────────────────────────────
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
