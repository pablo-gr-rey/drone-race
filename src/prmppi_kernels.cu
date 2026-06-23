#include "config.h"
#include "state.h"
#include "costs.cuh"
#include "kernels.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>

__global__ void PRMPPIgenerateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int T, int N)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    curandState local = rng[s];
    for (int t = 0; t < T; t++)
        for (int d = 0; d < DIM; d++)
            noise[(t * N + s) * DIM + d] = curand_normal(&local) * stddev;

    rng[s] = local;
}

// Sample values of theta according to belief, one thread per p
__global__ void PRMPPIsampleThetaValues(
    const float* __restrict__ belief,     // (nTrueModels)
    int* __restrict__ thetas,     // (P)
    curandState* __restrict__ theta_rng,  // (P)
    int P
)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P)
        return;

    curandState local = theta_rng[p];

    float u = curand_uniform(&local);

    float accum = 0.0f;
    int theta = N_TRUE_MODELS - 1;

    for (int i = 0; i < N_TRUE_MODELS; i++)
    {
        accum += belief[i];
        if (u <= accum)
        {
            theta = i;
            break;
        }
    }

    thetas[p] = theta;
    theta_rng[p] = local;
}

// Full rollout for PRMPPI, one thread per (rob/nom, sample, theta)
__global__ void PRMPPIfullRolloutKernel(
    int agent,
    const EnvironmentConfig envConfig,
    const PRMPPIConfig mppiConfig,
    SimState initState,
    const float* __restrict__ nom_nominal,    // (T, dim)
    const float* __restrict__ rob_nominal,    // (T, dim)
    const float* __restrict__ noise,      // (T, N, dim)
    float* __restrict__ cost_nom,    // (P, N, 2)
    float* __restrict__ cost_rob,    // (P, N, 2)
    const int* __restrict__ thetas,       // (P)
    curandState* __restrict__ rngStates         // (P, N)
)
{
    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int P = mppiConfig.P;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total = N * P;   // replace by passed P if preferred
    if (tid >= 2 * total)
        return;

    bool robust = tid >= total;

    int idx = robust ? tid - total : tid;

    int p = idx / N;
    int s = idx % N;

    SimState state = initState;
    BranchState branchState;

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_TRUE_MODELS * DIM];
    float egoAction[DIM];

    curandState rng = rngStates[idx];
    DeviceRNG drng{ &rng };

    const float* nominal = robust ? rob_nominal : nom_nominal;

    int theta = thetas[p];

    float runningCost = 0.0f;
    float safeCost = INFINITY;
    float decay = 1.0f;

    bool stop = false;

    for (int t = 0; t < T && !stop; t++)
    {
        for (int d = 0; d < DIM; d++)
            egoAction[d] = nominal[t * DIM + d] + noise[(t * N + s) * DIM + d];

        bool branched = false;

        TerminalType term = environmentStep<false, false>(
            t,
            agent,
            theta,
            envConfig,
            egoAction,
            true,
            state,
            branchState,
            branched,
            0.0f,
            actions,
            nomPidAction,
            drng,
            agent,
            mppiConfig.gateTraversalMargin);

        stop = (term != TERM_NONE);

        runningCost += decay * PRMPPIstateCost(agent, state, t, envConfig, mppiConfig);
        safeCost = min(safeCost, PRMPPIsafetyCost(agent, state, t, envConfig, mppiConfig));

        decay *= 0.9f;
    }

    runningCost += PRMPPIfinalCost(agent, state, envConfig, mppiConfig);

    float* out = robust ? cost_rob : cost_nom;

    out[(p * N + s) * 2 + 0] = runningCost;
    out[(p * N + s) * 2 + 1] = safeCost;

    rngStates[idx] = rng;
}

// For each sample s, compute expCost := avg(cost[p, s, 0]) and safeCost := min(cost[p, s, 1]); stores cost[0, s, 0] := expCost + weight * (1 if safeCost < 0), cost[0, s, 1] = safeCost. 2*N threads (1st part for nom, 2nd part for rob)
__global__ void PRMPPIcostAvgKernel(
    float* __restrict__ cost_nom,   // (P, N, 2)
    float* __restrict__ cost_rob,   // (P, N, 2)
    int N,
    int P,
    float safetyWeight
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2 * N)
        return;

    bool useNom = idx < N;

    int s = idx % N;

    float avg = 0.0f;
    float minSafe = INFINITY;

    float* cost = useNom ? cost_nom : cost_rob;

    for (int p = 0; p < P; p++)
    {
        int off = (p * N + s) * 2;

        avg += cost[off];
        minSafe = fminf(minSafe, cost[off + 1]);
    }

    avg /= (float) P;

    int out = s * 2;

    cost[out] = avg + ((minSafe < 0.0f) ? safetyWeight : 0.0f);
    cost[out + 1] = minSafe;
}

// Compute min costs (3 blocks: nom_full, rob_full, rob_safe). blk*sizeof(float) shared memory
__global__ void PRMPPIcomputeMinCostsKernel(
    const float* __restrict__ cost_nom,    // (P, N, 2) (only first N are considered)
    const float* __restrict__ cost_rob,    // (P, N, 2) (only first N are considered)
    float* __restrict__ minCosts,         // 3
    int N)
{
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int which = blockIdx.x;

    // Grid-stride loop to handle N > blockDim.x
    float val = INFINITY;
    const float* cost_arr = which == 0 ? cost_nom : (which == 1 ? cost_rob : cost_rob + 1);

    for (int i = tid; i < N; i += blockDim.x)
        val = fminf(val, cost_arr[i * 2]);

    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s[tid] = fminf(s[tid], s[tid + stride]);

        __syncthreads();
    }

    if (tid == 0)
        minCosts[which] = s[0];
}

// Compute weights, and update nominals. 3*T*DIM blocks. 2*blk*sizeof(float) shared memory
__global__ void PRMPPIWeightedAverageKernel(
    const float* __restrict__ nom_nominal,    // (T, dim)
    float* __restrict__ rob_nominal,      // (T, dim)
    const float* __restrict__ noise,      // (N, T, dim)
    const float* __restrict__ cost_nom,      // (P, N, 2) (only first (N, 0/1) are considered)
    const float* __restrict__ cost_rob,      // (P, N, 2) (only first (N, 0/1) are considered)
    const float* __restrict__ minCosts,   // 3
    float* __restrict__ cand1_nominal,    // (T, dim)
    float* __restrict__ cand2_nominal,    // (T, dim)
    int N,
    int T,
    float invTempNomFull,
    float invTempRobFull,
    float invTempRobSafe,
    float* __restrict__ nu      // 3
)
{
    extern __shared__ float shared[];

    float* s_num = shared;
    float* s_den = shared + blockDim.x;

    const int tid = threadIdx.x;

    const int which = blockIdx.x / (T * DIM);      // 0,1,2
    const int rem = blockIdx.x % (T * DIM);
    const int t = rem / DIM;
    const int d = rem % DIM;

    const float* nominal = (which == 0) ? nom_nominal : rob_nominal;
    const float* cost_arr = which == 0 ? cost_nom : (which == 1 ? cost_rob : cost_rob + 1);
    float* cand_nominal = (which == 0) ? cand1_nominal : (which == 1 ? cand2_nominal : rob_nominal);

    float invTemp = which == 0 ? invTempNomFull : (which == 1 ? invTempRobFull : invTempRobSafe);

    float minCost = minCosts[which];

    // Grid-stride loop to accumulate over all N samples
    float num = 0.0f;
    float den = 0.0f;

    for (int i = tid; i < N; i += blockDim.x)
    {
        float cost = cost_arr[2 * i];

        float w = __expf(-invTemp * (cost - minCost));
        float u = nominal[t * DIM + d] + noise[(t * N + i) * DIM + d];

        num += w * u;
        den += w;
    }

    s_num[tid] = num;
    s_den[tid] = den;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_num[tid] += s_num[tid + stride];
            s_den[tid] += s_den[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float value;

        if (s_den[0] > 1e-30f)
            value = s_num[0] / s_den[0];
        else
            value = nominal[t * DIM + d];

        cand_nominal[t * DIM + d] = value;

        if (t == 0 && d == 0)
            nu[which] = s_den[0];
    }
}

// Compute full cost for the 2 candidates nominals and all models, 2 * P threads (using the rng of the first 2 rollouts)
__global__ void PRMPPIcomputeCandidateCostKernel(
    int agent,
    EnvironmentConfig envConfig,
    PRMPPIConfig mppiConfig,
    SimState initState,
    const float* __restrict__ cand1_nominal,      // (T, dim)
    const float* __restrict__ cand2_nominal,      // (T, dim)
    const int* __restrict__ thetas,     // (P)
    float* __restrict__ candCosts,        // (2, P)
    curandState* __restrict__ rngStates         // (2*P at least)
)
{
    int P = mppiConfig.P;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= 2 * P)
        return;

    bool second = (tid >= P);
    int p = second ? tid - P : tid;

    const float* nominal = second ? cand2_nominal : cand1_nominal;

    SimState state = initState;
    BranchState branchState;

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_TRUE_MODELS * DIM];
    float egoAction[DIM];

    curandState rng = rngStates[tid];
    DeviceRNG drng{ &rng };

    float cost = 0.0f;
    float safeCost = INFINITY;
    float decay = 1.0f;

    bool stop = false;

    for (int t = 0; t < mppiConfig.nTimesteps && !stop; ++t)
    {
        for (int d = 0; d < DIM; ++d)
            egoAction[d] = nominal[t * DIM + d];

        bool branched = false;

        TerminalType term = environmentStep<false, false>(
            t,
            agent,
            thetas[p],
            envConfig,
            egoAction,
            true,
            state,
            branchState,
            branched,
            0.0f,
            actions,
            nomPidAction,
            drng,
            agent,
            mppiConfig.gateTraversalMargin);

        stop = (term != TERM_NONE);

        cost += decay * PRMPPIstateCost(agent, state, t, envConfig, mppiConfig);

        decay *= 0.9f;

        safeCost = min(safeCost, PRMPPIsafetyCost(agent, state, t, envConfig, mppiConfig));
    }

    cost += PRMPPIfinalCost(agent, state, envConfig, mppiConfig);
    if (safeCost < 0.0f)
        cost += mppiConfig.safetyWeight;

    candCosts[second * P + p] = cost;

    rngStates[tid] = rng;
}

// Compute safe cost for the nominal and all models, P threads (using the rng of the first P rollouts). writes into candCosts[0:P]
__global__ void PRMPPIcomputeSafeCostKernel(
    int agent,
    EnvironmentConfig envConfig,
    PRMPPIConfig mppiConfig,
    SimState initState,
    const float* __restrict__ nom_nominal,    // (T, dim)
    const int* __restrict__ thetas,     // (P)
    float* __restrict__ candCosts,        // (P at least)
    curandState* __restrict__ rngStates     // (P at least)
)
{
    int P = mppiConfig.P;
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= P)
        return;

    SimState state = initState;
    BranchState branchState;

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_TRUE_MODELS * DIM];
    float egoAction[DIM];

    curandState rng = rngStates[p];
    DeviceRNG drng{ &rng };

    bool stop = false;
    float safeCost = INFINITY;

    for (int t = 0; t < mppiConfig.nTimesteps && !stop; ++t)
    {
        for (int d = 0; d < DIM; ++d)
            egoAction[d] = nom_nominal[t * DIM + d];

        bool branched = false;

        TerminalType term = environmentStep<false, false>(
            t,
            agent,
            thetas[p],
            envConfig,
            egoAction,
            true,
            state,
            branchState,
            branched,
            0.0f,
            actions,
            nomPidAction,
            drng,
            agent,
            mppiConfig.gateTraversalMargin);

        safeCost = min(safeCost, PRMPPIsafetyCost(agent, state, t, envConfig, mppiConfig));

        stop = (term != TERM_NONE);
    }

    candCosts[p] = safeCost;

    rngStates[p] = rng;
}
