#include "config.h"
#include "costs.cuh"
#include "kernels.cuh"
#include "state.h"

#include <cub/cub.cuh>
#include <curand_kernel.h>

// RNG init
__global__ void initRNGKernel(curandState* states, unsigned long long seed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        curand_init(seed, i, 0, &states[i]);
}

// Noise generation
__global__ void generateNoiseKernel(float* noise, curandState* rng, float stddev, int M, int N)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    curandState local = rng[s];
    for (int theta = 0; theta < N_BRANCH_PLANS; theta++) // generate for nominal + 1 for each model
        for (int t = 0; t < M; t++)
            for (int d = 0; d < ACTION_DIM; d++)
                noise[((theta * M + t) * N + s) * ACTION_DIM + d] = curand_normal(&local) * stddev;

    rng[s] = local;
}

// Fused rollout step (fills costsTrue, branchUsed, branchTime)
__global__ void fullRolloutKernel(const EnvironmentConfig envConfig, const MPPIConfig mc, SimState initState, const float* __restrict__ initBelief,
                                  const float* __restrict__ nominal, // if USE_SPLINES: (nBranchPlans, M, dim);
                                                                     // otherwise: (nBranchPlans, T, dim)
                                  const float* __restrict__ noise, const float* __restrict__ B, float* __restrict__ costsTrue,
                                  int* __restrict__ branchUsed, int* __restrict__ branchTime, curandState* __restrict__ rngStates)
{
    int N = mc.nSamples;

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    SimState state;
    BranchState branchState;

    float egoAction[ACTION_DIM];
    ScratchEnvBuffer buffer;

    curandState rng = rngStates[s];
    DeviceRNG drng{&rng};

    BranchState firstBranchState;
    initBranchState(firstBranchState, initBelief, mc.minConfidence);

    // initialize values
    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
    {
        costsTrue[theta * mc.nSamples + s] = INFINITY;
        for (int k = 0; k < N_MODEL_FACTORS; k++)
        {
            branchUsed[(theta * mc.nSamples + s) * N_MODEL_FACTORS + k] = firstBranchState.predTheta[k];
            branchTime[(theta * mc.nSamples + s) * N_MODEL_FACTORS + k] = firstBranchState.branchingTime[k];
        }
    }

    int trueTheta[N_MODEL_FACTORS];

    // find out if we are already committed
    // if so, we only consider that plan (switch to normal MPPI); otherwise,
    // cost at the end could incur slight perturbations for the other branches

    for (int trueThetaFlat = 0; trueThetaFlat < N_TRUE_MODELS; trueThetaFlat++) // trueTheta is the model the opponent is actually
                                                                                // following
    {
        unflattenTrueModelIndex(trueThetaFlat, trueTheta);

        // if we are already committed to a plan which is incompatible with
        // trueTheta, then skip this happen if for any k,
        // firstBranchState.predTheta[k] != 0 and firstBranchState.predTheta[k]
        // != trueTheta[k] + 1 otherwise, very high cost on very unlikely models
        // can incur perturbations on the other branches and degrade specialized
        // plan quality

        if (!branchCompatibleWithModel(firstBranchState.predTheta.data(), trueTheta))
            continue;

        state = initState;
        branchState = firstBranchState;

        float cost = 0.0f;
        bool stop = false;
        float decay = 1.0f;

        int tOrigin = 0; // updated when we branch

        // Main rollout loop
        for (int t = 0; t < mc.nTimesteps && !stop; t++)
        {
            // int local_time = t - localBranchTimeOrigin(branchState.predTheta,
            // branchState.branchingTime);
            int local_time = t - tOrigin;

            int flatBranch = flattenBranchIndex(branchState.predTheta.data());

            if (USE_SPLINES)
            {
                const float* B_row = B + local_time * mc.nKnots;
                for (int d = 0; d < ACTION_DIM; d++)
                {
                    float sum = 0.0f;
                    for (int m = 0; m < mc.nKnots; m++)
                    {
                        float nom = nominal[(flatBranch * mc.nKnots + m) * ACTION_DIM + d];
                        float noisef = noise[((flatBranch * mc.nKnots + m) * mc.nSamples + s) * ACTION_DIM + d];
                        sum += B_row[m] * (nom + noisef);
                    }

                    egoAction[d] = sum;
                }
            }
            else
                for (int d = 0; d < ACTION_DIM; d++)
                {
                    // float nom = nominal[((branchState.predTheta + 1) *
                    // mc.nTimesteps + t - branchState.branchingTime) *
                    // ACTION_DIM + d]; float noisef =
                    // noise[(((branchState.predTheta + 1) * mc.nTimesteps + t -
                    // branchState.branchingTime) * mc.nSamples + s) *
                    // ACTION_DIM + d];
                    float nom = nominal[(flatBranch * mc.nTimesteps + local_time) * ACTION_DIM + d];
                    float noisef = noise[((flatBranch * mc.nTimesteps + local_time) * mc.nSamples + s) * ACTION_DIM + d];
                    egoAction[d] = nom + noisef;
                }

            bool branched = false;

            TerminalType term =
                environmentStep<true, true, false>(t, trueThetaFlat, envConfig, egoAction,
                                                   true, // applyPidNoise
                                                   state, branchState, branched, mc.minConfidence, buffer, drng, mc.gateTraversalMargin);

            if (branched)
                tOrigin = t + 1;

            stop = (term != TERM_NONE);

            cost += stateCost(state, t, envConfig, mc, trueThetaFlat) * decay;
            decay *= DECAY;
        }

        // Terminal cost
        cost += finalCost(state, envConfig, mc, trueThetaFlat);

        // actual cost is dependent on the probability that the opponent is
        // actually following trueTheta, ie. initBelief[theta], unless we are
        // already committed

        costsTrue[trueThetaFlat * mc.nSamples + s] = cost;

        for (int k = 0; k < N_MODEL_FACTORS; k++)
        {
            branchTime[(trueThetaFlat * mc.nSamples + s) * N_MODEL_FACTORS + k] = branchState.branchingTime[k];
            branchUsed[(trueThetaFlat * mc.nSamples + s) * N_MODEL_FACTORS + k] = branchState.predTheta[k];
        }
    }

    rngStates[s] = rng;
}

// compute costs from costsTrue, i.e. for a given branchPlan, costs[branchPlan,
// s] = expectation of all true costs compatible with that tuple
__global__ void aggregateBranchCostsKernel(const float* __restrict__ costsTrue, // (N_TRUE_MODELS, N)
                                           const float* __restrict__ belief,    // (N_TRUE_MODELS)
                                           float* __restrict__ costs,           // (N_BRANCH_PLANS, N)
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

        // if the branch was skipped in rollout (because it was incompatible
        // with initial belief), then its cost is INFINITY and we should not
        // consider it here
        if (!isfinite(c))
            continue;

        float w = belief[m];
        num += w * c;
        den += w;
    }

    costs[branchPlan * N + s] = (den > 1e-30f) ? (num / den) : INFINITY;
}

// for each (branchPlan, tLocal/knot), loop over samples, and compute min of
// costs[branchPlan, s] for eligible samples
__global__ void computeMaskedMinCostsKernel(const float* __restrict__ costs,    // (N_BRANCH_PLANS, N)
                                            const int* __restrict__ branchUsed, // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
                                            const int* __restrict__ branchTime, // (N_TRUE_MODELS, N, N_MODEL_FACTORS)
                                            const float* __restrict__ belief,   // (N_TRUE_MODELS)
                                            float* __restrict__ minCosts,       // (N_BRANCH_PLANS, T or M)
                                            const MPPIConfig mppiConfig)
{
    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int M = mppiConfig.nKnots;

    int bt = blockIdx.x;
    if (bt >= N_BRANCH_PLANS * (USE_SPLINES ? M : T))
        return;

    int tLocal = bt % (USE_SPLINES ? M : T);
    int branchPlan = bt / (USE_SPLINES ? M : T);

    int branchTuple[N_MODEL_FACTORS];
    int thetaTuple[N_MODEL_FACTORS];
    unflattenBranchIndex(branchPlan, branchTuple);

    extern __shared__ float smins[];
    float threadMin = INFINITY;

    int initPredTheta[N_MODEL_FACTORS];
    findConfident(belief, mppiConfig.minConfidence, initPredTheta);

    if (branchCompatibleWithInitialModel(branchTuple, initPredTheta))
        for (int s = threadIdx.x; s < N; s += blockDim.x)
        {
            float coeff = 0.0f;

            for (int trueTheta = 0; trueTheta < N_TRUE_MODELS; trueTheta++)
            {
                if (belief[trueTheta] <= 0.0f)
                    continue;

                unflattenTrueModelIndex(trueTheta, thetaTuple);
                if (!branchCompatibleWithModel(branchTuple, thetaTuple))
                    continue;

                const int* samplePredTheta = branchUsed + (trueTheta * N + s) * N_MODEL_FACTORS;
                const int* sampleBranchTime = branchTime + (trueTheta * N + s) * N_MODEL_FACTORS;

                // if (branchActiveAtLocalTime(branchTuple, used, bTime, tLocal,
                // T))
                //     coeff += belief[m];

                if (USE_SPLINES)
                {
                    // for simplicity, we say that a spline point m contributed
                    // if either tau[m-1], tau[m] or tau[m+1] contributed
                    if (splineSampleContributed(branchTuple, tLocal, sampleBranchTime, samplePredTheta, T, M, mppiConfig.knots.data()))
                        coeff += belief[trueTheta];
                }
                else if (sampleContributed(branchTuple, tLocal, sampleBranchTime, samplePredTheta, T))
                    coeff += belief[trueTheta];
            }

            if (coeff > MIN_COEFF_THRESHOLD)
            {
                float c = costs[branchPlan * N + s];
                if (c < threadMin)
                    threadMin = c;
            }
        }

    smins[threadIdx.x] = threadMin;
    __syncthreads();

    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            smins[threadIdx.x] = fminf(smins[threadIdx.x], smins[threadIdx.x + stride]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        minCosts[branchPlan * (USE_SPLINES ? M : T) + tLocal] = smins[0];
}

// Weighted average of noise
// One block per (timestep/tau * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
// should be used if USE_SPLINES is false!
__global__ void weightedAverageKernelUnified(const float* __restrict__ costs,    // (nBranchPlans, N)
                                             const float* __restrict__ minCosts, // (nBranchPlans, T or M)
                                             const float* __restrict__ noise,    // (nBranchPlans, T or M, N, dim)
                                             const int* __restrict__ branchUsed, // (nTrueModels, N, nModelFactors)
                                             const int* __restrict__ branchTime, // (nTrueModels, N, nModelFactors)
                                             const float* __restrict__ belief,   // (nTrueModels)
                                             float* __restrict__ nominal,        // (nBranchPlans, T or M, dim)
                                             const MPPIConfig mppiConfig,
                                             float* __restrict__ nu) // (nBranchPlans)
{
    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int M = mppiConfig.nKnots;

    int btd = blockIdx.x;
    if (btd >= N_BRANCH_PLANS * (USE_SPLINES ? M : T) * ACTION_DIM)
        return;

    int d = btd % ACTION_DIM;
    btd /= ACTION_DIM;
    int tLocal = btd % (USE_SPLINES ? M : T);
    int branchPlan = btd / (USE_SPLINES ? M : T);

    int branchTuple[N_MODEL_FACTORS];
    int thetaTuple[N_MODEL_FACTORS];
    unflattenBranchIndex(branchPlan, branchTuple);

    extern __shared__ float sh[];
    float* s_num = sh;
    float* s_den = sh + blockDim.x;

    float num = 0.0f;
    float den = 0.0f;

    float minC = minCosts[branchPlan * (USE_SPLINES ? M : T) + tLocal];

    // If no eligible sample existed, minC may be INF
    // Then den should remain 0 and update should be skipped
    if (!isinf(minC))
    {
        for (int s = threadIdx.x; s < N; s += blockDim.x)
        {
            float coeff = 0.0f;

            for (int trueTheta = 0; trueTheta < N_TRUE_MODELS; trueTheta++)
            {
                if (belief[trueTheta] <= 0.0f)
                    continue;

                unflattenTrueModelIndex(trueTheta, thetaTuple);
                if (!branchCompatibleWithModel(branchTuple, thetaTuple))
                    continue;

                const int* samplePredTheta = branchUsed + (trueTheta * N + s) * N_MODEL_FACTORS;
                const int* sampleBranchTime = branchTime + (trueTheta * N + s) * N_MODEL_FACTORS;

                if (USE_SPLINES)
                {
                    // for simplicity, we say that a spline point m contributed
                    // if either tau[m-1], tau[m] or tau[m+1] contributed
                    if (splineSampleContributed(branchTuple, tLocal, sampleBranchTime, samplePredTheta, T, M, mppiConfig.knots.data()))
                        coeff += belief[trueTheta];
                }
                else if (sampleContributed(branchTuple, tLocal, sampleBranchTime, samplePredTheta, T))
                    coeff += belief[trueTheta];
            }

            if (coeff > MIN_COEFF_THRESHOLD)
            {
                float cost = costs[branchPlan * N + s];
                float w = expf(-(cost - minC) / mppiConfig.invTemperature);
                float eps = noise[((branchPlan * (USE_SPLINES ? M : T) + tLocal) * N + s) * ACTION_DIM + d];

                num += w * coeff * eps;
                den += w * coeff;
            }
        }
    }

    s_num[threadIdx.x] = num;
    s_den[threadIdx.x] = den;
    __syncthreads();

    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1)
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
            nominal[(branchPlan * (USE_SPLINES ? M : T) + tLocal) * ACTION_DIM + d] += s_num[0] / s_den[0];

        if (tLocal == 0 && d == 0)
            nu[branchPlan] = s_den[0];
    }
}

// interpolate splineNominal into nominal, one thread per (branchplan, t, dim)
__global__ void interpolateSplineKernel(const float* __restrict__ splineNominal, // (nBranchPlans, M, dim)
                                        float* __restrict__ nominal,             // (nBranchPlans, T, dim)
                                        const float* __restrict__ B,             // (T, M),
                                        int T, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nVecs = N_BRANCH_PLANS * T * ACTION_DIM;
    if (idx >= nVecs)
        return;

    int dim = idx % ACTION_DIM;
    idx /= ACTION_DIM;
    int t = idx % T;
    int branch = idx / T;

    float sum = 0.0f;
    for (int m = 0; m < M; m++)
        sum += B[t * M + m] * splineNominal[(branch * M + m) * ACTION_DIM + dim];

    nominal[(branch * T + t) * ACTION_DIM + dim] = sum;
}

// clamp nominals
__global__ void clampNominalKernel(float* __restrict__ nominal, float maxAccel, int T)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nVecs = N_BRANCH_PLANS * T;
    if (idx >= nVecs)
        return;

    float sqNorm = 0.0f;
    int base = idx * ACTION_DIM;
    for (int d = 0; d < ACTION_DIM; ++d)
    {
        float v = nominal[base + d];
        sqNorm += v * v;
    }

    float maxSq = maxAccel * maxAccel;
    if (sqNorm > maxSq)
    {
        float scale = maxAccel / sqrtf(sqNorm);
        for (int d = 0; d < ACTION_DIM; ++d)
            nominal[base + d] *= scale;
    }
}

// Shift splineNominal into newSplineNominal (such that newSplineNominal[i] =
// interpolate(splineNominal)(tau_i + 1), i.e. simulate shifting by one
// timestep). one thread per (m, dim)
__global__ void shiftSplineKernel(const float* __restrict__ splineNominal, // (nBranchPlans, M, dim)
                                  float* __restrict__ newSplineNominal,    // (nBranchPlans, M, dim)
                                  const float* __restrict__ B,             // (T, M),
                                  const MPPIConfig mppiConfig, int branchIdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int M = mppiConfig.nKnots;
    int nVecs = M * ACTION_DIM;
    if (idx >= nVecs)
        return;

    int dim = idx % ACTION_DIM;
    int newm = idx / ACTION_DIM;

    float sum = 0.0f;

    int tEval = mppiConfig.knots[newm] + 1;
    if (tEval >= mppiConfig.nTimesteps)
        tEval = mppiConfig.nTimesteps - 1;

    for (int m = 0; m < M; m++)
        sum += B[tEval * M + m] * splineNominal[(branchIdx * M + m) * ACTION_DIM + dim];

    newSplineNominal[(branchIdx * M + newm) * ACTION_DIM + dim] = sum;
}

__global__ void verifyNominalFailureKernel(EnvironmentConfig envConfig, MPPIConfig mc, SimState initState, const float* __restrict__ initBelief,
                                           const float* __restrict__ nominal, // (nModels+1, T, dim)
                                           curandState* __restrict__ rngStates,
                                           unsigned int* __restrict__ failCount) // (T)
{
    int nVerif = mc.nVerifSamples;
    int nTimesteps = mc.verifHorizon;

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nVerif)
        return;

    curandState rng = rngStates[s];

    // Local rollout state
    SimState state = initState;
    BranchState branchState;
    initBranchState(branchState, initBelief, mc.minConfidence);

    ScratchEnvBuffer buffer;
    float egoAction[ACTION_DIM];
    DeviceRNG drng{&rng};

    // Sample actual opponent model according to initial belief
    int theta = sampleModelFromBelief(branchState.belief.data(), &rng);

    int tOrigin = 0;
    int t = 0;
    bool failed = false;

    // Dynamics rollout
    for (; t < nTimesteps; t++)
    {
        int local_time = t - tOrigin;
        int flatBranch = flattenBranchIndex(branchState.predTheta.data());

        for (int d = 0; d < ACTION_DIM; d++)
            egoAction[d] = nominal[(flatBranch * mc.nTimesteps + local_time) * ACTION_DIM + d];

        bool branched = false;

        TerminalType term = environmentStep<true, true, false>(t, theta, envConfig, egoAction,
                                                               true, // applyPidNoise in real verification
                                                               state, branchState, branched, mc.minConfidence, buffer, drng);

        if (branched)
            tOrigin = t + 1;

        // Failure checks

        if (term == TERM_LOSE)
        {
            failed = true;
            break;
        }
        else if (term == TERM_WIN)
            break;
    }

    if (failed)
        atomicAdd(failCount + t, 1u);

    rngStates[s] = rng;
}
