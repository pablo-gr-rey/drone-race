#pragma once

#include "config.h"
#include "device_config.cuh"
#include "state.h"
#include "track.cuh"
#include "costs.cuh"
#include "opponent_models.cuh"
#include <curand_kernel.h>

// Single kernel: each thread rolls out one full sample trajectory.
__global__ void fullRolloutKernel(
    int controlAgent,
    const DeviceEnvironmentConfig cfg,
    const DeviceMPPIConfig mc,
    OpponentModelType oppModel,
    const PIDConfig oppPid,
    const float* __restrict__ initPhys,      // (physDim,) — shared by all
    const float* __restrict__ initS,         // (nAgents,)
    const float* __restrict__ initLaps,      // (nAgents,)
    const float* __restrict__ nominal,       // (T, dim)
    const float* __restrict__ noise,         // (T, N, dim)
    float* __restrict__ totalCosts,          // (N,)
    // for terminal cost, we need the final state; write it out:
    float* __restrict__ finalPhys,           // (N, physDim)
    float* __restrict__ finalS,              // (N, nAgents)
    float* __restrict__ finalLaps,           // (N, nAgents)
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts,
    int nTP, int N);
