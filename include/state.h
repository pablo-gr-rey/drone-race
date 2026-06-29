#pragma once

#include "cuda_runtime.h"
#include "curand_kernel.h"

#include <boost/math/distributions/beta.hpp>

// #include <cmath>
#include <math.h>
#include <random>

struct BranchState
{
    float belief[N_TRUE_MODELS];
    int predTheta[N_MODEL_FACTORS];      // 0 if nominal for k, theta+1 if specialized (<=> marginal over theta_k=theta > threshold)
    int branchingTime[N_MODEL_FACTORS];     // -1 if not branched, 0 if initially committed, otherwise t+1 if branched at global time t (i.e. origin of new branch)
};

struct DeviceRNG
{
    curandState* state;
};

struct HostRNG
{
    std::normal_distribution<float>* nd;
    std::mt19937* rng;
};

__device__ INLINE float sampleNormal(DeviceRNG& rng)
{
    return curand_normal(rng.state);
}

__host__ INLINE float sampleNormal(HostRNG& rng)
{
    return (*rng.nd)(*rng.rng);
}

// index helpers

// compute trueTheta in [0, N_TRUE_MODELS-1] from theta tuple (size N_MODEL_FACTORS)
HD INLINE int flattenTrueModelIndex(const int* thetaTuple)
{
    int trueTheta = 0;
    int stride = 1;

    for (int f = N_MODEL_FACTORS - 1; f >= 0; --f)
    {
        trueTheta += thetaTuple[f] * stride;
        stride *= MODEL_SIZE(f);
    }

    return trueTheta;
}

// compute theta tuple (size N_MODEL_FACTORS) from trueTheta in [0, N_TRUE_MODELS-1]
HD INLINE void unflattenTrueModelIndex(int trueTheta, int* thetaTuple)
{
    for (int f = N_MODEL_FACTORS - 1; f >= 0; --f)
    {
        thetaTuple[f] = trueTheta % MODEL_SIZE(f);
        trueTheta /= MODEL_SIZE(f);
    }
}

// compute branchIndex in [0, N_BRANCH_PLANS-1] from branch tuple (size N_MODEL_FACTORS, 0=nominal, i+1=specialized i)
HD INLINE int flattenBranchIndex(const int* branchTuple)
{
    int branch = 0;
    int stride = 1;

    for (int f = N_MODEL_FACTORS - 1; f >= 0; --f)
    {
        branch += branchTuple[f] * stride;
        stride *= BRANCH_SIZE(f);
    }

    return branch;
}

// compute branch tuple (size N_MODEL_FACTORS, 0=nominal, i+1=specialized i) from branchIndex in [0, N_BRANCH_PLANS-1]
HD INLINE void unflattenBranchIndex(int branch, int* branchTuple)
{
    for (int f = N_MODEL_FACTORS - 1; f >= 0; --f)
    {
        branchTuple[f] = branch % BRANCH_SIZE(f);
        branch /= BRANCH_SIZE(f);
    }
}

// return k-th value of trueTheta for k < N_FACTOR_MODELS
HD INLINE int getTrueModelIndex(int trueTheta, int k)
{
    for (int f = N_MODEL_FACTORS - 1; f > k; --f)
        trueTheta /= MODEL_SIZE(f);

    return trueTheta % MODEL_SIZE(k);
}

HD INLINE void computeMarginal(const float* __restrict__ belief, int factor, float* __restrict__ marginal)
{
    for (int theta = 0; theta < MODEL_SIZE(factor); theta++)
        marginal[theta] = 0.0f;

    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
        marginal[getTrueModelIndex(theta, factor)] += belief[theta];
}

// fill predTheta (for k < N_MODEL_FACTORS predTheta[k] = theta+1 if there exists theta such that belief marginalized over theta_k=theta is > threshold, otherwise 0)
HD INLINE void findConfident(const float* __restrict__ belief, float threshold, int* __restrict__ predTheta)
{
    // for (int theta = 0; theta < N_MODELS; theta++)
    //     if (belief[theta] > threshold)
    //         return theta;

    // return -1;

    float marginal[MAX_MODEL_SIZE];

    for (int k = 0; k < N_MODEL_FACTORS; k++)
    {
        predTheta[k] = 0;
        computeMarginal(belief, k, marginal);

        for (int theta = 0; theta < MODEL_SIZE(k); theta++)
            if (marginal[theta] > threshold)
            {
                predTheta[k] = theta + 1;
                break;
            }
    }
}

// return trueTheta sampled from belief
__device__ INLINE int sampleModelFromBelief(const float* __restrict__ belief, curandState* __restrict__ rng)
{
    float u = curand_uniform(rng);
    float cdf = 0.0f;

    for (int k = 0; k < N_TRUE_MODELS; k++)
    {
        cdf += belief[k];
        if (u <= cdf)
            return k;
    }

    return N_TRUE_MODELS - 1;
}

// return lower-bound on p such that P(Binom(n, p) = k) >= alpha
__host__ inline double clopperPearsonLowerBound(uint k, uint n, double alpha)
{
    if (k == 0)
        return 0.0;

    boost::math::beta_distribution<double> dist(k, n - k + 1);
    return boost::math::quantile(dist, alpha);
}

// return upper-bound on p such that P(Binom(n, p) = k) >= alpha
__host__ inline double clopperPearsonUpperBound(uint k, uint n, double alpha)
{
    if (k == n)
        return 1.0;

    boost::math::beta_distribution<double> dist(k + 1, n - k);
    return boost::math::quantile(dist, 1.0 - alpha);
}

// initialize branch state (copy belief, find confident, set branch times = -1 if not committed on parameter and 0 otherwise)
HD INLINE void initBranchState(
    BranchState& branchState,
    const float* __restrict__ initBelief,
    float threshold)
{
    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
        branchState.belief[theta] = initBelief[theta];

    findConfident(branchState.belief, threshold, branchState.predTheta);

    for (int k = 0; k < N_MODEL_FACTORS; k++)
        branchState.branchingTime[k] = branchState.predTheta[k] == 0 ? -1 : 0;
}

// find local time origin for a given branch state ( = max{branchTime[k] | predTheta[k] != 0}, 0 if we are not specialized)
HD INLINE int localBranchTimeOrigin(const int* __restrict__ predTheta, const int* __restrict__ branchingTime)
{
    int max = 0;
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if (predTheta[k] != 0 && branchingTime[k] > max)
            max = branchingTime[k];

    return max;
}

// return whether we are already committed to a plan which is compatible with trueTheta (<=> forall k, predTheta[k] == 0 or predTheta[k] == trueTheta[k] + 1)
HD INLINE bool branchCompatibleWithModel(const int* __restrict__ predTheta, const int* __restrict__ trueTheta)
{
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if (predTheta[k] != 0 && predTheta[k] != trueTheta[k] + 1)
            return false;

    return true;
}

// return whether the considered branch is compatible with the initial pred theta (<=> forall k, initPredTheta[k] == 0 or initPredTheta[k] == branch[k])
HD INLINE bool branchCompatibleWithInitialModel(const int* __restrict__ branch, const int* __restrict__ initPredTheta)
{
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if (initPredTheta[k] != 0 && initPredTheta[k] != branch[k])
            return false;

    return true;
}

// return whether a sample contributed to the given branch, at the given local time
// this happens if: (1) branch is compatible with final sample pred theta (<=> forall k, branch[k] == 0 or branch[k] == samplePredTheta[k])
// (2) we did not switch to a new branch before tLocal (<=> tGlobal <= min{sampleBranchTime[k] | branch[k] == 0 and samplePredTheta[k] != 0} where tGlobal = tLocal + localBranchTimeOrigin = tLocal + max{sampleBranchTime[k] | branch[k] != 0})
// (3) the sample was not truncated before tLocal (<=> tGlobal < T)
HD INLINE bool sampleContributed(const int* __restrict__ branch, int tLocal, const int* __restrict__ sampleBranchTime, const int* __restrict__ samplePredTheta, int T)
{
    int tGlobal = tLocal + localBranchTimeOrigin(branch, sampleBranchTime);

    if (tGlobal >= T)
        return false;

    for (int k = 0; k < N_MODEL_FACTORS; k++)
    {
        if ((branch[k] != 0 && branch[k] != samplePredTheta[k])       // incompatible
            || (branch[k] == 0 && samplePredTheta[k] != 0 && tGlobal >= sampleBranchTime[k]))     // we switch before tGlobal
            return false;
    }

    return true;
}

// return whether a spline knot contributed. for simplicity, we say that this is <=> tau[m-1], tau[m] or tau[m+1] contributed
HD INLINE bool splineSampleContributed(const int* __restrict__ branch, int m, const int* __restrict__ sampleBranchTime, const int* __restrict__ samplePredTheta, int T, int M, const int* __restrict__ knots)
{
    return ((m > 0 && sampleContributed(branch, knots[m - 1], sampleBranchTime, samplePredTheta, T))
        || sampleContributed(branch, knots[m], sampleBranchTime, samplePredTheta, T)
        || (m < M - 1 && sampleContributed(branch, knots[m + 1], sampleBranchTime, samplePredTheta, T)));
}
 