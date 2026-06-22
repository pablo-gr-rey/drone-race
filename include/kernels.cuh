#pragma once

#include "config.h"
#include "environment.h"
#include "controllers.h"

#include <curand_kernel.h>
#include <cstddef>

__global__ void initRNGKernel(curandState* states, unsigned long long seed, int n);

__global__ void generateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int M, int N);

__global__ void fullRolloutKernel(
    int controlAgent,
    // const DeviceEnvironmentConfig envConfig,
    const EnvironmentConfig envConfig,
    const MPPIConfig mc,
    SimState initState,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    const float* __restrict__ B,
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
    float* __restrict__ minCosts,          // (N_BRANCH_PLANS, T or M)
    const MPPIConfig mppiConfig);

// Weighted average of noise
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernelUnified(
    const float* __restrict__ costs,       // (nBranchPlans, N)
    const float* __restrict__ minCosts,    // (nBranchPlans, T or M)
    const float* __restrict__ noise,       // (nBranchPlans, T or M, N, dim)
    const int* __restrict__ branchUsed,    // (nTrueModels, N, nModelFactors)
    const int* __restrict__ branchTime,    // (nTrueModels, N, nModelFactors)
    const float* __restrict__ belief,      // (nTrueModels)
    float* __restrict__ nominal,           // (nBranchPlans, T or M, dim)
    const MPPIConfig mppiConfig,
    float* __restrict__ nu);                // (nBranchPlans)

// interpolate splineNominal into nominal, one thread per (branchplan, t, dim)
__global__ void interpolateSplineKernel(
    const float* __restrict__ splineNominal,    // (nBranchPlans, M, dim)
    float* __restrict__ nominal,            // (nBranchPlans, T, dim)
    const float* __restrict__ B,          // (T, M),
    int T,
    int M
);

// Clamp nominal actions to maxAccel (avoids them drifting to high-magnitude areas from which it's difficult to recover)
__global__ void clampNominalKernel(
    float* __restrict__ nominal,
    float maxAccel,
    int T);

// Shift splineNominal into newSplineNominal (such that newSplineNominal[i] = interpolate(splineNominal)(tau_i + 1), i.e. simulate shifting by one timestep). one thread per (m, dim). ASSUMES tau[m-1] + 1 < T
__global__ void shiftSplineKernel(
    const float* __restrict__ splineNominal,    // (nBranchPlans, M, dim)
    float* __restrict__ newSplineNominal,    // (nBranchPlans, M, dim)
    const float* __restrict__ B,          // (T, M),
    const MPPIConfig mppiConfig,
    int branchIdx
);

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
