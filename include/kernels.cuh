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
    float* __restrict__ costsTrue,
    int* __restrict__ branchUsed,
    int* __restrict__ branchTime,
    curandState* __restrict__ rngStates);

__global__ void aggregateBranchCostsKernel(
    const float* __restrict__ costsTrue,   // (N_TRUE_MODELS, N)
    const float* __restrict__ belief,      // (N_TRUE_MODELS)
    float* __restrict__ costs,             // (N_BRANCH_PLANS, N)
    int N);

// for each (branchPlan, tLocal), loop over samples, and compute min of costs[branchPlan, s] for eligible samples
__global__ void computeMaskedMinCostsKernel(
    const float* __restrict__ costs,       // (N_BRANCH_PLANS, N)
    const int* __restrict__ branchUsed,    // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
    const int* __restrict__ branchTime,    // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
    const float* __restrict__ belief,      // (N_TRUE_MODELS)
    float* __restrict__ minCosts,          // (N_BRANCH_PLANS, T)
    int N,
    int T);

// Weighted average of noise
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernelUnified(
    const float* __restrict__ costs,       // (nBranchPlans, N)
    const float* __restrict__ minCosts,    // (nBranchPlans, T)
    const float* __restrict__ noise,       // (nBranchPlans, T, N, dim)
    const int* __restrict__ branchUsed,    // (N_TRUE_MODELS, N, nModelFactors)
    const int* __restrict__ branchTime,    // (N_TRUE_MODELS, N, nModelFactors)
    const float* __restrict__ belief,      // (N_TRUE_MODELS)
    float* __restrict__ nominal,           // (nBranchPlans, T, dim)
    float invTemp,
    int N,
    int T,
    float* __restrict__ nu);               // (nBranchPlans)

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
    SimState initState,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,     // (nModels+1, T, dim)
    curandState* __restrict__ rngStates,
    unsigned int* __restrict__ failCount);
