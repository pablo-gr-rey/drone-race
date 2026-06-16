#include "config.h"
#include "state.h"
#include "costs.cuh"
#include "kernels.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>

// RNG init
__global__ void initRNGKernel(curandState* states, unsigned long long seed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        // printf("initializing RNG %d\n", i);
        curand_init(seed, i, 0, &states[i]);
    }
}

// Noise generation
__global__ void generateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int nTimesteps, int N)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    // printf("Accessing RNG %d\n", s);
    curandState local = rng[s];
    for (int theta = 0; theta <= N_MODELS; theta++)      // generate for nominal + 1 for each model
        for (int t = 0; t < nTimesteps; t++)
            for (int d = 0; d < DIM; d++)
                noise[((theta * nTimesteps + t) * N + s) * DIM + d] = curand_normal(&local) * stddev;

    rng[s] = local;
}

// Fused rollout step
__global__ void fullRolloutKernel(
    int controlAgent,
    const EnvironmentConfig envConfig,
    const MPPIConfig mc,
    SimState initState,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    int* __restrict__ branchesUsed,
    int* __restrict__ branchesTime,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts)
{
    int N = mc.nSamples;

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    SimState state;
    BranchState branchState;

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_MODELS * DIM];
    float egoAction[DIM];

    curandState rng = rngStates[s];
    DeviceRNG drng{ &rng };

    totalCosts[s] = 0.0f;

    int initPredTheta = findConfident(initBelief, mc.minConfidence);

    // initialize values
    for (int theta = 0; theta < N_MODELS; theta++)
    {
        totalCosts[(theta + 1) * mc.nSamples + s] = INFINITY;
        branchesUsed[theta * mc.nSamples + s] = -1;
        branchesTime[theta * mc.nSamples + s] = mc.nTimesteps;
    }

    // find out if we are already committed
    // if so, we only consider that plan (switch to normal MPPI); otherwise, cost at the end could incur slight perturbations for the other branches

    for (int theta = 0; theta < N_MODELS; theta++)    // theta is the model the opponent is actually following
    {
        if (initPredTheta != -1 && theta != initPredTheta)
            continue;

        state = initState;
        initBranchState(branchState, initBelief, mc.minConfidence);

        float cost = 0.0f;
        bool stop = false;
        float decay = 1.0f;

        // ── Main rollout loop ────────────────────────────────────────────
        for (int t = 0; t < mc.nTimesteps && !stop; t++)
        {
            for (int d = 0; d < DIM; d++)
            {
                float nom = nominal[((branchState.predTheta + 1) * mc.nTimesteps + t - branchState.branchingTime) * DIM + d];
                float noisef = noise[(((branchState.predTheta + 1) * mc.nTimesteps + t - branchState.branchingTime) * mc.nSamples + s) * DIM + d];
                egoAction[d] = nom + noisef;
            }

            TerminalType term = simulateContingentStep(
                t,
                controlAgent,
                theta,
                envConfig,
                mc,
                trackPts,
                egoAction,
                true,              // applyPidNoise
                state,
                branchState,
                actions,
                nomPidAction,
                drng,
                controlAgent,
                mc.gateTraversalMargin);

            stop = (term != TERM_NONE);

            cost += stateCost(controlAgent, state.pos, state.vel, state.laps, state.gates, t, envConfig, mc) * decay;
            decay *= 0.9f;
        }

        // Terminal cost
        cost += finalCost(controlAgent, state.pos, state.vel, state.laps, state.gates, envConfig, mc);

        // actual cost is dependent on the probability that the opponent is actually following theta, ie. initBelief[theta], unless we are already committed
        if (initPredTheta == -1)
        {
            branchesTime[theta * mc.nSamples + s] = (branchState.predTheta == -1 ? mc.nTimesteps : branchState.branchingTime);
            totalCosts[s] += initBelief[theta] * cost;
        }
        else
        {
            branchesTime[theta * mc.nSamples + s] = 0;
            totalCosts[s] = cost;
        }

        branchesUsed[theta * mc.nSamples + s] = branchState.predTheta;
        totalCosts[(theta + 1) * mc.nSamples + s] = cost;
    }

    rngStates[s] = rng;
}

// mask costs as explained in controller.h
__global__ void buildMaskedCostsKernel(
    const float* __restrict__ costs,       // (nModels+1, N)
    const int* __restrict__ branchUsed,    // (nModels, N)
    const int* __restrict__ branchTime,    // (nModels, N)
    const float* __restrict__ belief,      // (nModels)
    float* __restrict__ maskedCosts,       // ((nModels+1) * T, N)
    int N, int T)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (N_MODELS + 1) * T * N;
    if (idx >= total)
        return;

    int s = idx % N;
    int tmp = idx / N;
    int tLocal = tmp % T;
    int branchIdx = tmp / T;   // 0 = generic, k+1 = specialized k

    bool eligible = false;


    if (branchIdx == 0)
    {
        float coeff = 0.0f;
        for (int theta = 0; theta < N_MODELS; theta++)
        {
            int tb = branchTime[theta * N + s];
            if (tLocal < tb)
                coeff += belief[theta];
        }
        eligible = (coeff > 0.0f);
    }
    else
    {
        int k = branchIdx - 1;
        int bu = branchUsed[k * N + s];
        int tb = branchTime[k * N + s];
        eligible = (bu == k && tb < T && tb + tLocal < T);
    }

    maskedCosts[idx] = eligible ? costs[branchIdx * N + s] : INFINITY;
}

// cuda reduce min
void minReduceCUB(const float* __restrict__ d_in, float* __restrict__ d_out, int N, int nRows, void* __restrict__ d_temp_storage, size_t temp_storage_bytes)
{
    // since temporary storage size only depends on N (the size of the array to reduce), we can reuse it just fine
    for (int r = 0; r < nRows; r++)
        CUDA_CHECK(cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in + r * N, d_out + r, N));
}

// weighted average
__global__ void weightedAverageKernel(
    const float* __restrict__ costs,
    const float* __restrict__ noise,
    float* __restrict__ nominal,
    const float* __restrict__ minCost, float invTemp,
    int N, int T,
    float* __restrict__ nu)
{
    int mtd = blockIdx.x;
    if (mtd >= (N_MODELS + 1) * T * DIM)
        return;

    int d = mtd % DIM;
    mtd /= DIM;
    int t = mtd % T;
    int theta = mtd / T;

    extern __shared__ float sh[];
    float* s_wn = sh;
    float* s_w = sh + blockDim.x;

    float wSum = 0.0f;
    float wnSum = 0.0f;

    float minC = *minCost;

    for (int s = threadIdx.x; s < N; s += blockDim.x)
    {
        float w = expf(-(costs[s] - minC) / invTemp);
        wSum += w;
        wnSum += w * noise[((theta * T + t) * N + s) * DIM + d];
    }

    s_wn[threadIdx.x] = wnSum;
    s_w[threadIdx.x] = wSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_wn[threadIdx.x] += s_wn[threadIdx.x + stride];
            s_w[threadIdx.x] += s_w[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && s_w[0] > 1e-30f)
        nominal[(theta * T + t) * DIM + d] += s_wn[0] / s_w[0];

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *nu = s_w[0];
    }
}

// we want to add the contribution of nominal + each branch based on its actual contribution to the sample
// for example, if a given branch is never taken in a sample (belief does not get skewed enough), then the noise on that branch should not count towards the corresponding minimal
// likewise, generic noise at time t should only be considered with a coefficient proportional to how much this noise actually influenced the cost (ie. sum of belief[theta] for theta whose branching time is > t)
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
    float* __restrict__ nu)                // nModels: sum of weights for nominal[0], spec[theta, 0]
{
    int btd = blockIdx.x;
    if (btd >= (N_MODELS + 1) * T * DIM)
        return;

    int d = btd % DIM;
    btd /= DIM;
    int tLocal = btd % T;
    int branchIdx = btd / T;   // 0 = generic, k+1 = specialized branch k

    extern __shared__ float sh[];
    float* s_num = sh;
    float* s_den = sh + blockDim.x;

    float num = 0.0f;
    float den = 0.0f;

    float minC = minCosts[branchIdx * T + tLocal];

    for (int s = threadIdx.x; s < N; s += blockDim.x)
    {
        float coeff = 0.0f;

        if (branchIdx == 0)
        {
            // Generic branch at absolute time tLocal: used in each true-model rollout theta if tLocal < branchTime[theta, s]
            for (int theta = 0; theta < N_MODELS; theta++)
            {
                int tb = branchTime[theta * N + s];
                if (tLocal < tb)
                    coeff += belief[theta];
            }
        }
        else
        {
            int k = branchIdx - 1;

            int bu = branchUsed[k * N + s];
            int tb = branchTime[k * N + s];

            // Specialized branch k at local time tLocal: used only if rollout under true model k actually branched to k, and local time is still within horizon
            if (bu == k && tb + tLocal < T)
                coeff = 1.0f;
        }

        if (coeff > 0.0f)
        {
            float cost = costs[branchIdx * N + s];
            float w = expf(-(cost - minC) / invTemp);
            float eps = noise[((branchIdx * T + tLocal) * N + s) * DIM + d];

            num += w * coeff * eps;
            den += w * coeff;
        }
    }

    s_num[threadIdx.x] = num;
    s_den[threadIdx.x] = den;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_num[threadIdx.x] += s_num[threadIdx.x + stride];
            s_den[threadIdx.x] += s_den[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (s_den[0] > 1e-30f)
            nominal[(branchIdx * T + tLocal) * DIM + d] += s_num[0] / s_den[0];

        if (threadIdx.x == 0 && tLocal == 0 && d == 0)
            nu[branchIdx] = s_den[0];
    }
}

// clamp nominals
__global__ void clampNominalKernel(
    float* nominal,
    float maxAccel,
    int T)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nVecs = (N_MODELS + 1) * T;
    if (idx >= nVecs)
        return;

    float sqNorm = 0.0f;
    int base = idx * DIM;
    for (int d = 0; d < DIM; ++d)
    {
        float v = nominal[base + d];
        sqNorm += v * v;
    }

    float maxSq = maxAccel * maxAccel;
    if (sqNorm > maxSq)
    {
        float scale = maxAccel / sqrtf(sqNorm);
        for (int d = 0; d < DIM; ++d)
            nominal[base + d] *= scale;
    }
}

__global__ void verifyNominalFailureKernel(
    int controlAgent,
    int nVerif,
    EnvironmentConfig envConfig,
    MPPIConfig mc,
    int nTimesteps,
    SimState initState,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,     // (nModels+1, T, dim)
    const float* __restrict__ trackPts,
    curandState* __restrict__ rngStates,
    unsigned int* __restrict__ failCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nVerif)
        return;

    curandState rng = rngStates[s];

    // Local rollout state
    SimState state = initState;
    BranchState branchState;
    initBranchState(branchState, initBelief, mc.minConfidence);

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_MODELS * DIM];
    float egoAction[DIM];
    DeviceRNG drng{ &rng };

    // Sample actual opponent model according to initial belief
    int theta = sampleModelFromBelief(branchState.belief, &rng);

    int failed = 0;     // 1 if collision, 2 if outside

    // Dynamics rollout
    for (int t = 0; t < nTimesteps && !failed; t++)
    {

        for (int d = 0; d < DIM; d++)
        {
            int localT = t - branchState.branchingTime;
            int startInd = ((branchState.predTheta + 1) * mc.nTimesteps + localT) * DIM;
            egoAction[d] = nominal[startInd + d];
        }

        TerminalType term = simulateContingentStep(
            t,
            controlAgent,
            theta,
            envConfig,
            mc,
            trackPts,
            egoAction,
            true,          // applyPidNoise in real verification
            state,
            branchState,
            actions,
            nomPidAction,
            drng);

        // Failure checks

        if (term == TERM_EGO_OUTSIDE)
        {
            failed = 2;
            break;
        }
        if (term == TERM_COLLISION)
        {
            failed = 1;
            break;
        }
        if (term == TERM_OPP_OUTSIDE || term == TERM_WIN || term == TERM_OPP_WIN)
            break;
    }

    if (failed)
        atomicAdd(failCount + failed - 1, 1u);

    rngStates[s] = rng;
}
