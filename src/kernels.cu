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
    for (int theta = 0; theta < N_BRANCH_PLANS; theta++)      // generate for nominal + 1 for each model
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
    float* __restrict__ costsTrue,
    int* __restrict__ branchUsed,
    int* __restrict__ branchTime,
    curandState* __restrict__ rngStates)
{
    int N = mc.nSamples;

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    SimState state;
    BranchState branchState;

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_TRUE_MODELS * DIM];
    float egoAction[DIM];

    curandState rng = rngStates[s];
    DeviceRNG drng{ &rng };

    // int initPredTheta = findConfident(initBelief, mc.minConfidence); // now, firstBranchState.predTheta

    BranchState firstBranchState;
    initBranchState(firstBranchState, initBelief, mc.minConfidence, mc.nTimesteps);

    // initialize values
    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
    {
        costsTrue[theta * mc.nSamples + s] = INFINITY;
        for (int k = 0; k < N_MODEL_FACTORS; k++)
        {
            branchUsed[(theta * mc.nSamples + s) * N_MODEL_FACTORS + k] = 0;
            branchTime[(theta * mc.nSamples + s) * N_MODEL_FACTORS + k] = mc.nTimesteps;
        }
    }

    int trueTheta[N_MODEL_FACTORS];

    // find out if we are already committed
    // if so, we only consider that plan (switch to normal MPPI); otherwise, cost at the end could incur slight perturbations for the other branches

    for (int trueThetaFlat = 0; trueThetaFlat < N_TRUE_MODELS; trueThetaFlat++)    // trueTheta is the model the opponent is actually following
    {
        unflattenTrueModelIndex(trueThetaFlat, trueTheta);

        // if we are already committed to a plan which is incompatible with trueTheta, then skip
        // this happen if for any k, firstBranchState.predTheta[k] != 0 and firstBranchState.predTheta[k] != trueTheta[k] + 1
        // otherwise, very high cost on very unlikely models can incur perturbations on the other branches and degrade specialized plan quality

        // if (initPredTheta != -1 && trueTheta != initPredTheta)
        //     continue;

        // bool incompatible = false;
        // for (int k = 0; k < N_MODEL_FACTORS; k++)
        //     if (firstBranchState.predTheta[k] != 0 && firstBranchState.predTheta[k] != trueTheta[k] + 1)
        //     {
        //         incompatible = true;
        //         break;
        //     }
        // if (incompatible)
        //     continue;

        if (!branchCompatibleWithModel(firstBranchState.predTheta, trueTheta))
            continue;

        state = initState;
        branchState = firstBranchState;

        float cost = 0.0f;
        bool stop = false;
        float decay = 1.0f;

        // Main rollout loop
        for (int t = 0; t < mc.nTimesteps && !stop; t++)
        {
            int local_time = t - localBranchTimeOrigin(branchState.predTheta, branchState.branchingTime);
            int flatBranch = flattenBranchIndex(branchState.predTheta);

            for (int d = 0; d < DIM; d++)
            {
                // float nom = nominal[((branchState.predTheta + 1) * mc.nTimesteps + t - branchState.branchingTime) * DIM + d];
                // float noisef = noise[(((branchState.predTheta + 1) * mc.nTimesteps + t - branchState.branchingTime) * mc.nSamples + s) * DIM + d];
                float nom = nominal[(flatBranch * mc.nTimesteps + local_time) * DIM + d];
                float noisef = noise[((flatBranch * mc.nTimesteps + local_time) * mc.nSamples + s) * DIM + d];
                egoAction[d] = nom + noisef;
            }

            TerminalType term = environmentStep(
                t,
                controlAgent,
                trueThetaFlat,
                envConfig,
                mc,
                envConfig.trackPoints,
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

        // actual cost is dependent on the probability that the opponent is actually following trueTheta, ie. initBelief[theta], unless we are already committed
        // if (initPredTheta == -1)
        // {
        //     branchTime[trueTheta * mc.nSamples + s] = (branchState.predTheta == -1 ? mc.nTimesteps : branchState.branchingTime);
        //     costsTrue[s] += initBelief[trueTheta] * cost;
        // }
        // else
        // {
        //     branchTime[trueTheta * mc.nSamples + s] = 0;
        //     costsTrue[s] = cost;
        // }

        // branchUsed[trueTheta * mc.nSamples + s] = branchState.predTheta;
        // costsTrue[(trueTheta + 1) * mc.nSamples + s] = cost;

        costsTrue[trueThetaFlat * mc.nSamples + s] = cost;

        for (int k = 0; k < N_MODEL_FACTORS; k++)
        {
            branchTime[(trueThetaFlat * mc.nSamples + s) * N_MODEL_FACTORS + k] = branchState.branchingTime[k];
            branchUsed[(trueThetaFlat * mc.nSamples + s) * N_MODEL_FACTORS + k] = branchState.predTheta[k];
        }
    }

    rngStates[s] = rng;
}

// compute costs from costsTrue, i.e. for a given branchPlan, costs[branchPlan, s] = expectation of all true costs compatible with that tuple
__global__ void aggregateBranchCostsKernel(
    const float* __restrict__ costsTrue,   // (N_TRUE_MODELS, N)
    const float* __restrict__ belief,      // (N_TRUE_MODELS)
    float* __restrict__ costs,             // (N_BRANCH_PLANS, N)
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_BRANCH_PLANS * N;
    if (idx >= total)
        return;

    int s = idx % N;
    int branchPlan = idx / N;

    int branchTuple[N_MODEL_FACTORS];
    int thetaTuple[N_MODEL_FACTORS];

    unflattenBranchIndex(branchPlan, branchTuple);

    float num = 0.0f;
    float den = 0.0f;

    for (int m = 0; m < N_TRUE_MODELS; m++)
    {
        unflattenTrueModelIndex(m, thetaTuple);

        if (!branchCompatibleWithModel(branchTuple, thetaTuple))
            continue;

        float c = costsTrue[m * N + s];

        // if the branch was skipped in rollout (because it was incompatible), then its cost is INFINITY and we should not consider it here
        if (!isfinite(c))
            continue;

        float w = belief[m];
        num += w * c;
        den += w;
    }

    costs[branchPlan * N + s] = (den > 1e-30f) ? (num / den) : INFINITY;
}

// for each (branchPlan, tLocal), loop over samples, and compute min of costs[branchPlan, s] for eligible samples
__global__ void computeMaskedMinCostsKernel(
    const float* __restrict__ costs,       // (N_BRANCH_PLANS, N)
    const int* __restrict__ branchUsed,    // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
    const int* __restrict__ branchTime,    // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
    const float* __restrict__ belief,      // (N_TRUE_MODELS)
    float* __restrict__ minCosts,          // (N_BRANCH_PLANS, T)
    int N,
    int T)
{
    int bt = blockIdx.x;
    if (bt >= N_BRANCH_PLANS * T)
        return;

    int tLocal = bt % T;
    int branchPlan = bt / T;

    int branchTuple[N_MODEL_FACTORS];
    int thetaTuple[N_MODEL_FACTORS];
    unflattenBranchIndex(branchPlan, branchTuple);

    extern __shared__ float smins[];
    float threadMin = INFINITY;

    for (int s = threadIdx.x; s < N; s += blockDim.x)
    {
        float coeff = 0.0f;

        for (int m = 0; m < N_TRUE_MODELS; m++)
        {
            if (belief[m] <= 0.0f)
                continue;

            unflattenTrueModelIndex(m, thetaTuple);
            if (!branchCompatibleWithModel(branchTuple, thetaTuple))
                continue;

            const int* used = branchUsed + (m * N + s) * N_MODEL_FACTORS;
            const int* bTime = branchTime + (m * N + s) * N_MODEL_FACTORS;

            // int tOrigin = localBranchTimeOrigin(branchTuple, bTime);
            // int tAbs = tOrigin + tLocal;

            // if (tAbs >= T)
            //     continue;

            // bool active = true;
            // for (int k = 0; k < N_MODEL_FACTORS; k++)
            // {
            //     if (branchTuple[k] == 0)
            //     {
            //         // factor k should still be nominal at time tAbs
            //         if (tAbs >= bTime[k])
            //         {
            //             active = false;
            //             break;
            //         }
            //     }
            //     else
            //     {
            //         int requiredModel = branchTuple[k] - 1;
            //         if (used[k] != requiredModel || tAbs < bTime[k])
            //         {
            //             active = false;
            //             break;
            //         }
            //     }
            // }

            // if (active)
            //     coeff += belief[m];

            if (branchActiveAtLocalTime(branchTuple, used, bTime, tLocal, T))
                coeff += belief[m];
        }

        if (coeff > 0.0f)
        {
            float c = costs[branchPlan * N + s];
            if (c < threadMin)
                threadMin = c;
        }
    }

    smins[threadIdx.x] = threadMin;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            smins[threadIdx.x] = fminf(smins[threadIdx.x], smins[threadIdx.x + stride]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        minCosts[branchPlan * T + tLocal] = smins[0];
}

// we want to add the contribution of nominal + each branch based on its actual contribution to the sample
// for example, if a given branch is never taken in a sample (belief does not get skewed enough), then the noise on that branch should not count towards the corresponding minimal (it would just add noise)
// likewise, generic noise at time t should only be considered with a coefficient proportional to how much this noise actually influenced the cost (ie. sum of belief[theta] for theta whose branching time is > t)
// __global__ void weightedAverageKernelUnified(
//     const float* __restrict__ costs,       // (nModels+1, N)
//     const float* __restrict__ minCosts,    // (nModels+1)
//     const float* __restrict__ noise,       // (nModels+1, T, N, dim)
//     const int* __restrict__ branchUsed,    // (nModels, N)
//     const int* __restrict__ branchTime,    // (nModels, N)
//     const float* __restrict__ belief,      // (nModels)
//     float* __restrict__ nominal,           // (nModels+1, T, dim)
//     float invTemp,
//     int N, int T,
//     float* __restrict__ nu)                // nModels: sum of weights for nominal[0], spec[theta, 0]
// {
//     int btd = blockIdx.x;
//     if (btd >= (N_MODELS + 1) * T * DIM)
//         return;

//     int d = btd % DIM;
//     btd /= DIM;
//     int tLocal = btd % T;
//     int branchIdx = btd / T;   // 0 = generic, k+1 = specialized branch k

//     extern __shared__ float sh[];
//     float* s_num = sh;
//     float* s_den = sh + blockDim.x;

//     float num = 0.0f;
//     float den = 0.0f;

//     float minC = minCosts[branchIdx * T + tLocal];

//     for (int s = threadIdx.x; s < N; s += blockDim.x)
//     {
//         float coeff = 0.0f;

//         if (branchIdx == 0)
//         {
//             // Generic branch at absolute time tLocal: used in each true-model rollout theta if tLocal < branchTime[theta, s]
//             for (int theta = 0; theta < N_MODELS; theta++)
//             {
//                 int tb = branchTime[theta * N + s];
//                 if (tLocal < tb)
//                     coeff += belief[theta];
//             }
//         }
//         else
//         {
//             int k = branchIdx - 1;

//             int bu = branchUsed[k * N + s];
//             int tb = branchTime[k * N + s];

//             // Specialized branch k at local time tLocal: used only if rollout under true model k actually branched to k, and local time is still within horizon
//             if (bu == k && tb + tLocal < T)
//                 coeff = 1.0f;
//         }

//         if (coeff > 0.0f)
//         {
//             float cost = costs[branchIdx * N + s];
//             float w = expf(-(cost - minC) / invTemp);
//             float eps = noise[((branchIdx * T + tLocal) * N + s) * DIM + d];

//             num += w * coeff * eps;
//             den += w * coeff;
//         }
//     }

//     s_num[threadIdx.x] = num;
//     s_den[threadIdx.x] = den;
//     __syncthreads();

//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (threadIdx.x < stride)
//         {
//             s_num[threadIdx.x] += s_num[threadIdx.x + stride];
//             s_den[threadIdx.x] += s_den[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x == 0)
//     {
//         if (s_den[0] > 1e-30f)
//             nominal[(branchIdx * T + tLocal) * DIM + d] += s_num[0] / s_den[0];

//         if (threadIdx.x == 0 && tLocal == 0 && d == 0)
//             nu[branchIdx] = s_den[0];
//     }
// }

// Weighted average of noise
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernelUnified(
    const float* __restrict__ costs,       // (nBranchPlans, N)
    const float* __restrict__ minCosts,    // (nBranchPlans, T)
    const float* __restrict__ noise,       // (nBranchPlans, T, N, dim)
    const int* __restrict__ branchUsed,    // (nTrueModels, N, nModelFactors)
    const int* __restrict__ branchTime,    // (nTrueModels, N, nModelFactors)
    const float* __restrict__ belief,      // (nTrueModels)
    float* __restrict__ nominal,           // (nBranchPlans, T, dim)
    float invTemp,
    int N,
    int T,
    float* __restrict__ nu)                // (nBranchPlans)
{
    int btd = blockIdx.x;
    if (btd >= N_BRANCH_PLANS * T * DIM)
        return;

    int d = btd % DIM;
    btd /= DIM;
    int tLocal = btd % T;
    int branchPlan = btd / T;

    int branchTuple[N_MODEL_FACTORS];
    int thetaTuple[N_MODEL_FACTORS];
    unflattenBranchIndex(branchPlan, branchTuple);

    extern __shared__ float sh[];
    float* s_num = sh;
    float* s_den = sh + blockDim.x;

    float num = 0.0f;
    float den = 0.0f;

    float minC = minCosts[branchPlan * T + tLocal];

    // If no eligible sample existed, minC may be INF
    // Then den should remain 0 and update should be skipped
    if (!isinf(minC))
    {
        for (int s = threadIdx.x; s < N; s += blockDim.x)
        {
            float coeff = 0.0f;

            for (int m = 0; m < N_TRUE_MODELS; m++)
            {
                if (belief[m] <= 0.0f)
                    continue;

                unflattenTrueModelIndex(m, thetaTuple);
                if (!branchCompatibleWithModel(branchTuple, thetaTuple))
                    continue;

                const int* used = branchUsed + (m * N + s) * N_MODEL_FACTORS;
                const int* bTime = branchTime + (m * N + s) * N_MODEL_FACTORS;

                // int tOrigin = localBranchTimeOrigin(branchTuple, bTime);
                // int tAbs = tOrigin + tLocal;

                // if (tAbs >= T)
                //     continue;

                // bool active = true;
                // for (int k = 0; k < N_MODEL_FACTORS; k++)
                // {
                //     if (branchTuple[k] == 0)
                //     {
                //         if (tAbs >= bTime[k])
                //         {
                //             active = false;
                //             break;
                //         }
                //     }
                //     else
                //     {
                //         int requiredModel = branchTuple[k] - 1;
                //         if (used[k] != requiredModel || tAbs < bTime[k])
                //         {
                //             active = false;
                //             break;
                //         }
                //     }
                // }

                // if (active)
                //     coeff += belief[m];

                if (branchActiveAtLocalTime(branchTuple, used, bTime, tLocal, T))
                    coeff += belief[m];
            }

            if (coeff > 0.0f)
            {
                float cost = costs[branchPlan * N + s];
                float w = expf(-(cost - minC) / invTemp);
                float eps = noise[((branchPlan * T + tLocal) * N + s) * DIM + d];

                num += w * coeff * eps;
                den += w * coeff;
            }
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
            nominal[(branchPlan * T + tLocal) * DIM + d] += s_num[0] / s_den[0];

        if (tLocal == 0 && d == 0)
            nu[branchPlan] = s_den[0];
    }
}

// clamp nominals
__global__ void clampNominalKernel(
    float* nominal,
    float maxAccel,
    int T)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nVecs = N_BRANCH_PLANS * T;
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
    initBranchState(branchState, initBelief, mc.minConfidence, mc.nTimesteps);

    float actions[N_AGENTS * DIM];
    float nomPidAction[N_TRUE_MODELS * DIM];
    float egoAction[DIM];
    DeviceRNG drng{ &rng };

    // Sample actual opponent model according to initial belief
    int theta = sampleModelFromBelief(branchState.belief, &rng);

    int failed = 0;     // 1 if collision, 2 if outside

    // Dynamics rollout
    for (int t = 0; t < nTimesteps && !failed; t++)
    {
        // int localTime = t - localBranchTimeOrigin(branchState);
        // int startInd = ((branchState.predTheta + 1) * mc.nTimesteps + localT) * DIM;

        int local_time = t - localBranchTimeOrigin(branchState.predTheta, branchState.branchingTime);
        int flatBranch = flattenBranchIndex(branchState.predTheta);

        for (int d = 0; d < DIM; d++)
            egoAction[d] = nominal[(flatBranch * mc.nTimesteps + local_time) * DIM + d];

        TerminalType term = environmentStep(
            t,
            controlAgent,
            theta,
            envConfig,
            mc,
            envConfig.trackPoints,
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
