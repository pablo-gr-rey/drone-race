#pragma once

#include "config.h"

#include <curand_kernel.h>

__global__ void initRNGKernel(curandState* states, unsigned long long seed, int n);

__global__ void generateNoiseKernel(float* noise, curandState* rng, float stddev, int M, int N);

__global__ void fullRolloutKernel(const EnvironmentConfig envConfig, const MPPIConfig mc, SimState initState, const float* __restrict__ initBelief,
                                  const float* __restrict__ nominal, const float* __restrict__ noise, const float* __restrict__ B,
                                  float* __restrict__ costsTrue, int* __restrict__ branchUsed, int* __restrict__ branchTime,
                                  curandState* __restrict__ rngStates);

__global__ void aggregateBranchCostsKernel(const float* __restrict__ costsTrue, // (N_TRUE_MODELS, N)
                                           const float* __restrict__ belief,    // (N_TRUE_MODELS)
                                           float* __restrict__ costs,           // (N_BRANCH_PLANS, N)
                                           int N);

// for each (branchPlan, tLocal), loop over samples, and compute min of costs[branchPlan, s] for eligible samples
__global__ void computeMaskedMinCostsKernel(const float* __restrict__ costs,    // (N_BRANCH_PLANS, N)
                                            const int* __restrict__ branchUsed, // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
                                            const int* __restrict__ branchTime, // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
                                            const float* __restrict__ belief,   // (N_TRUE_MODELS)
                                            float* __restrict__ minCosts,       // (N_BRANCH_PLANS, T or M)
                                            const MPPIConfig mppiConfig);

// Weighted average of noise
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernelUnified(const float* __restrict__ costs,    // (nBranchPlans, N)
                                             const float* __restrict__ minCosts, // (nBranchPlans, T or M)
                                             const float* __restrict__ noise,    // (nBranchPlans, T or M, N, dim)
                                             const int* __restrict__ branchUsed, // (nTrueModels, N, nModelFactors)
                                             const int* __restrict__ branchTime, // (nTrueModels, N, nModelFactors)
                                             const float* __restrict__ belief,   // (nTrueModels)
                                             float* __restrict__ nominal,        // (nBranchPlans, T or M, dim)
                                             const MPPIConfig mppiConfig,
                                             float* __restrict__ nu); // (nBranchPlans)

// interpolate splineNominal into nominal, one thread per (branchplan, t, dim)
__global__ void interpolateSplineKernel(const float* __restrict__ splineNominal, // (nBranchPlans, M, dim)
                                        float* __restrict__ nominal,             // (nBranchPlans, T, dim)
                                        const float* __restrict__ B,             // (T, M),
                                        int T, int M);

// Clamp nominal actions to maxAccel (avoids them drifting to high-magnitude areas from which it's difficult to recover)
__global__ void clampNominalKernel(float* __restrict__ nominal, float maxAccel, int T);

// Shift splineNominal into newSplineNominal (such that newSplineNominal[i] = interpolate(splineNominal)(tau_i + 1), i.e. simulate
// shifting by one timestep). one thread per (m, dim). ASSUMES tau[m-1] + 1 < T
__global__ void shiftSplineKernel(const float* __restrict__ splineNominal, // (nBranchPlans, M, dim)
                                  float* __restrict__ newSplineNominal,    // (nBranchPlans, M, dim)
                                  const float* __restrict__ B,             // (T, M),
                                  const MPPIConfig mppiConfig, int branchIdx);

// Sample opponent models & dyamics ; if failed, atomicAdd 1 to failCount
__global__ void verifyNominalFailureKernel(EnvironmentConfig envConfig, MPPIConfig mc, SimState initState, const float* __restrict__ initBelief,
                                           const float* __restrict__ nominal, // (nModels+1, T, dim)
                                           curandState* __restrict__ rngStates,
                                           unsigned int* __restrict__ failCount // (T)
);

__global__ void PRMPPIgenerateNoiseKernel(float* noise, curandState* rng, float stddev, int T, int N);

// Sample values of theta according to belief, one thread per p
__global__ void PRMPPIsampleThetaValues(const float* __restrict__ belief,    // (nTrueModels)
                                        int* __restrict__ thetas,            // (P)
                                        curandState* __restrict__ theta_rng, // (P)
                                        int P);

// Full rollout for PRMPPI, one thread per (rob/nom, sample, theta)
__global__ void PRMPPIfullRolloutKernel(const EnvironmentConfig envConfig, const PRMPPIConfig mppiConfig, SimState initState,
                                        const float* __restrict__ nom_nominal, // (T, dim)
                                        const float* __restrict__ rob_nominal, // (T, dim)
                                        const float* __restrict__ noise,       // (T, N, dim)
                                        float* __restrict__ cost_nom,          // (P, N, 2)
                                        float* __restrict__ cost_rob,          // (P, N, 2)
                                        const int* __restrict__ thetas,        // (P)
                                        curandState* __restrict__ rngStates    // (P, N)
);

// For each sample s, compute expCost := avg(cost[p, s, 0]) and safeCost := max(cost[p, s, 1]); stores cost[0, s, 0] := expCost +
// weight * (1 if safeCost < 0), cost[0, s, 1] = safeCost. 2*N threads (1st part for nom, 2nd part for rob)
__global__ void PRMPPIcostAvgKernel(float* __restrict__ cost_nom, // (P, N, 2)
                                    float* __restrict__ cost_rob, // (P, N, 2)
                                    int N, int P, float safetyWeight);

// Compute min costs (3 blocks: nom_full, rob_full, rob_safe). blk*sizeof(float) shared memory
__global__ void PRMPPIcomputeMinCostsKernel(const float* __restrict__ cost_nom, // (P, N) (only first N are considered)
                                            const float* __restrict__ cost_rob, // (P, N) (only first N are considered)
                                            float* __restrict__ minCosts,       // 3
                                            int N);

// Compute weights, and update nominals (nom_nominal + cost_nom[0, s, 0] with minCosts[0] -> cand1; rob_nominal + cost_rob[0, s,
// 0] with mincosts[1] -> cand2; rob_nominal + cost_rob[0, s, 1] with mincosts[2] -> new_rob_nominal), 3*T*DIM blocks. also writes
// into nu. 2*blk*sizeof(float) shared memory
__global__ void PRMPPIWeightedAverageKernel(const float* __restrict__ nom_nominal, // (T, dim)
                                            const float* __restrict__ rob_nominal, // (T, dim)
                                            const float* __restrict__ noise,       // (T, N, dim)
                                            const float* __restrict__ cost_nom,    // (P, N, 2) (only first (N, 0/1) are considered)
                                            const float* __restrict__ cost_rob,    // (P, N, 2) (only first (N, 1) are considered)
                                            const float* __restrict__ minCosts,    // 3
                                            float* __restrict__ cand1_nominal,     // (T, dim)
                                            float* __restrict__ cand2_nominal,     // (T, dim)
                                            float* __restrict__ new_rob_nominal,   // (T, DIM)
                                            int N, int T, float invTempNomFull, float invTempRobFull, float invTempRobSafe,
                                            float* __restrict__ nu // 3
);

// Compute full cost for the 2 candidates nominals and all models, 2 * P threads (using the rng of the first 2 rollouts)
__global__ void PRMPPIcomputeCandidateCostKernel(EnvironmentConfig envConfig, PRMPPIConfig mppiConfig, SimState initState,
                                                 const float* __restrict__ cand1_nominal, // (T, dim)
                                                 const float* __restrict__ cand2_nominal, // (T, dim)
                                                 const int* __restrict__ thetas,          // (P)
                                                 float* __restrict__ candCosts,           // (2, P)
                                                 curandState* __restrict__ rngStates      // (2*P at least)
);

// Compute safe cost for the nominal and all models, P threads (using the rng of the first P rollouts). writes into candCosts[0:P]
__global__ void PRMPPIcomputeSafeCostKernel(EnvironmentConfig envConfig, PRMPPIConfig mppiConfig, SimState initState,
                                            const float* __restrict__ nom_nominal, // (T, dim)
                                            const int* __restrict__ thetas,        // (P)
                                            float* __restrict__ candCosts,         // (P at least)
                                            curandState* __restrict__ rngStates    // (P at least)
);
