#pragma once

#include "config.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

#include <boost/math/distributions/beta.hpp>

// #include <cmath>
#include <math.h>
#include <random>

struct SimState
{
    float pos[N_AGENTS * DIM];
    float vel[N_AGENTS * DIM];
    float S[N_AGENTS * N_RACELINES];

    int laps[N_AGENTS];
    int gates[N_AGENTS];
};

struct BranchState
{
    float belief[N_TRUE_MODELS];
    int predTheta[N_MODEL_FACTORS];      // 0 if nominal for k, theta+1 if specialized (<=> marginal over theta_k=theta > threshold)
    int branchingTime[N_MODEL_FACTORS];
};

enum TerminalType
{
    TERM_NONE = 0,
    TERM_COLLISION,
    TERM_EGO_OUTSIDE,
    TERM_OPP_OUTSIDE,
    TERM_WIN,
    TERM_OPP_WIN
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

// Euclidean distance between two agents (positions only)
HD INLINE float agentDist(const float* __restrict__ pos, int a, int b)
{
    float s = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float dx = pos[a * DIM + d] - pos[b * DIM + d];
        s += dx * dx;
    }

    return sqrtf(s);
}

// Speed (L2 norm of velocity)
HD INLINE float agentSpeed(const float* __restrict__ speed, int agent)
{
    float s = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float v = speed[agent * DIM + d];
        s += v * v;
    }
    return sqrtf(s);
}

// Advance = currentGate + nGates * laps + scale * (1 - normalizedDistToGate) (it is much better to pass through a gate than to just be close to it) (this is a rough measure, it doesn't include speed for example) (expect pos to be of size d, ie. pos[0] should be position of actual agent)
// 'pos' points to the agent's contiguous position array of length DIM
HD INLINE float getAdvance(const int* laps, const int* currentGates, int agent, const float* __restrict__ pos, const float* __restrict__ gateCenters)
{
    float sqGateDist = 0.0f;    // sq dist between pos and next gate
    float sqConsGateDist = 0.0f;    // sq dist between current gate and next gate
    int nextGate = (currentGates[agent] + 1) % N_GATES;

    float scale = 0.8f;     // 1.0f means reward is continuous when going through a gate; 0.5f for example means that reward will be between 0.0 and 0.5 before the first gate, 1 and 1.5 between 1st and 2nd, etc

    for (int d = 0; d < DIM; d++)
    {
        sqGateDist += (gateCenters[nextGate * DIM + d] - pos[d]) * (gateCenters[nextGate * DIM + d] - pos[d]);
        sqConsGateDist += (gateCenters[nextGate * DIM + d] - gateCenters[currentGates[agent] * DIM + d]) * (gateCenters[nextGate * DIM + d] - gateCenters[currentGates[agent] * DIM + d]);
    }

    return currentGates[agent] + N_GATES * laps[agent] + scale * (1.0f - sqrtf(sqGateDist / sqConsGateDist));
}

// Linear interpolation of pre-sampled centerline
HD INLINE void sampleCenterline(const float* __restrict__ trackPoints, float s, float* __restrict__ out)
{
    s = s - floorf(s);                      // wrap to [0,1)
    float idx_f = s * N_TRACK_SAMPLES;
    int   idx0 = (int) idx_f;
    int   idx1 = (idx0 + 1) % N_TRACK_SAMPLES;
    float t = idx_f - idx0;
    for (int d = 0; d < DIM; d++)
        out[d] = (1.0f - t) * trackPoints[idx0 * DIM + d] + t * trackPoints[idx1 * DIM + d];
}

// Project position onto sampled track
// Returns best s in [0,1]; writes distance into bestDist
// closestOut may be nullptr
HD INLINE float projectOnTrack(const float* __restrict__ trackPoints,
    const float* __restrict__ pos,
    float* __restrict__ closestOut,
    float& bestDist)
{
    float bestS = 0.0f;
    float bestD2 = __FLT_MAX__;
    float sStep = 1.0f / N_TRACK_SAMPLES;
    for (int i = 0; i < N_TRACK_SAMPLES; i++)
    {
        float d2 = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = trackPoints[i * DIM + d] - pos[d];
            d2 += dx * dx;
        }
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestS = i * sStep;
            if (closestOut)
                for (int d = 0; d < DIM; d++)
                    closestOut[d] = trackPoints[i * DIM + d];
        }
    }
    bestDist = sqrtf(bestD2);
    return bestS;
}

// util function to return the distance from pos to the given trackPoints
HD INLINE float sqDistToSample(const float* __restrict__ trackPoints, int iTrack, const float* __restrict__ pos)
{
    float sqdist = 0.f;
    for (int d = 0; d < DIM; d++)
        sqdist += (trackPoints[iTrack * DIM + d] - pos[d]) * (trackPoints[iTrack * DIM + d] - pos[d]);
    return sqdist;
}

// util function to keep going in one direction until local maximum
HD INLINE int findMinAlongDirection(const float* __restrict__ trackPoints, int iTrack, const float* __restrict__ pos, int delta, float baseSqDist, float& bestDist)
{
    // greedy search + margin (should work if the track is not too weird)
    // TODO: maybe fixed window size is better (and more compiler friendly)
    const int margin = 10;

    int bestI = iTrack;
    bestDist = baseSqDist;

    int currentI = (iTrack + delta + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
    float currentDist = sqDistToSample(trackPoints, currentI, pos);

    while (currentDist < bestDist)
    {
        bestI = currentI;
        bestDist = currentDist;

        currentI = (currentI + delta + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
        currentDist = sqDistToSample(trackPoints, currentI, pos);
    }

    // additional margin
    for (int i = 0; i < margin; i++)
    {
        currentDist = sqDistToSample(trackPoints, currentI, pos);
        if (currentDist < bestDist)
        {
            bestI = currentI;
            bestDist = currentDist;
        }
        currentI = (currentI + delta + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
    }

    return bestI;
}

// return the closest S, and writes the closest track point in closestOut and its distance in bestDist
HD INLINE float fastProjectOnTrack(const float* __restrict__ trackPoints,
    const float* __restrict__ pos,
    float* __restrict__ closestOut,
    float& bestDist,
    float prevS = -1.0f   // negative means unknown -> full scan
)
{
#ifdef CHECK_PROJECTION
    float testS, testDist;
    float testClosestOut[MAX_DIM];
    testS = projectOnTrack(trackPoints, pos, testClosestOut, testDist);
#endif

    if (prevS < 0)
        return projectOnTrack(trackPoints, pos, closestOut, bestDist);

    int baseI = (int) (prevS * N_TRACK_SAMPLES + 0.5f) % N_TRACK_SAMPLES;
    float baseDist = sqDistToSample(trackPoints, baseI, pos);

    // find best forwards and backwards distances
    float bestFDist, bestBDist;
    int bestBI = findMinAlongDirection(trackPoints, baseI, pos, -1, baseDist, bestBDist);
    int bestFI = findMinAlongDirection(trackPoints, baseI, pos, +1, baseDist, bestFDist);

    float bestS;

    if (bestFDist < bestBDist)
    {
        bestDist = sqrtf(bestFDist);
        if (closestOut)
            for (int d = 0; d < DIM; d++)
                closestOut[d] = trackPoints[bestFI * DIM + d];

        bestS = ((float) bestFI) / N_TRACK_SAMPLES;
    }
    else
    {
        bestDist = sqrtf(bestBDist);
        if (closestOut)
            for (int d = 0; d < DIM; d++)
                closestOut[d] = trackPoints[bestBI * DIM + d];

        bestS = ((float) bestBI) / N_TRACK_SAMPLES;
    }

#ifdef CHECK_PROJECTION
    // bool wrong = fabs(bestS - testS) > 1e-5 || fabsf(bestDist - testDist) > 1e-5;
    // for (int d = 0; d < DIM && closestOut; d++)
    //     wrong = wrong || abs(closestOut[d] - testClosestOut[d] > 1e-5);

    bool wrong = fabs(bestDist - testDist) > 1e-5f;

    if (wrong)
    {
        printf("WRONG PROJECTION for pos ");
        for (int d = 0; d < DIM; d++)
            printf("%f ", pos[d]);
        printf(" prevS %f\n: computed bestS %f\tbestDist %f\tclosestOut ", prevS, bestS, bestDist);
        for (int d = 0; d < DIM && closestOut; d++)
            printf("%f ", closestOut[d]);
        printf("\n: expected bestS %f\tbestDist %f\tclosestOut ", testS, testDist);
        for (int d = 0; d < DIM; d++)
            printf("%f ", testClosestOut[d]);
        printf("\n");
    }
#endif

    return bestS;
}

// Update belief if oppAction was observed, nomAction is the nominal action for each model (nTrueModels * dim)
HD INLINE void updateBelief(float* __restrict__ belief, const float* __restrict__ oppAction, const float* __restrict__ nomAction, const PIDConfig* __restrict__ params, float maxPIDaccel)
{
    float sum = 0.0f;
    const float sigmaEnv = 0.2f;    // account for clamping + various imperfections

    // opponent is following a normal distribution around nomAction, with given stddev
    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
    {
        // compute sq value of nominal action
        float sqAccel = 0.f;
        for (int d = 0; d < DIM; d++)
            sqAccel += nomAction[theta * DIM + d] * nomAction[theta * DIM + d];

        float scale = 1.0f;
        if (sqAccel > maxPIDaccel * maxPIDaccel)
            scale = maxPIDaccel / sqrtf(sqAccel);

        float sqDist = 0.f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = oppAction[d] - nomAction[theta * DIM + d] * scale;
            sqDist += dx * dx;
        }

        float sigma = sqrtf(sigmaEnv * sigmaEnv + params[theta].actionNoise * params[theta].actionNoise);
        // d-dimensional normal law with diagonal sigma matrix (sigma^2, ...)

        // TODO: prob better to use other pow since dimension is integer & known (maybe even more efficient if dimension is known at compile time)
        belief[theta] *= expf(-0.5f * sqDist / (sigma * sigma)) / (powf(2.0f * M_PIf32, (float) DIM / 2.0f) * powf(sigma, (float) DIM));
        sum += belief[theta];
    }

    // normalize
    if (sum < 1e-20f)
    {
        float uniform = 1.0f / N_TRUE_MODELS;
        for (int theta = 0; theta < N_TRUE_MODELS; ++theta)
            belief[theta] = uniform;
    }
    else
    {
        for (int theta = 0; theta < N_TRUE_MODELS; theta++)
            belief[theta] /= sum;
    }
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

// Boundary distance = distance to closest boundary (<= 0 if outside)
__device__ INLINE float trackBoundaryDist(const EnvironmentConfig& envConfig, const float* __restrict__ pos)
{
    float minDist = INFINITY;
    for (int d = 0; d < DIM; d++)
        minDist = fminf(minDist, fminf(pos[d] - envConfig.arenaMin[d], envConfig.arenaMax[d] - pos[d]));

    for (int iObs = 0; iObs < N_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dist = fmaxf(0.0f, fmaxf(envConfig.obstacles[iObs * 2 * DIM + d] - pos[d], pos[d] - envConfig.obstacles[(iObs * 2 + 1) * DIM + d]));
            sqDist += dist * dist;
        }
        minDist = fminf(minDist, sqrtf(sqDist));
    }

    for (int iObs = 0; iObs < N_ROUND_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = pos[d] - envConfig.roundObsCenters[iObs * DIM + d];
            sqDist += dx * dx;
        }
        minDist = fminf(minDist, sqrtf(sqDist) - envConfig.roundObsRadius[iObs]);
    }

    return minDist;
}

HD INLINE bool isOutside(const EnvironmentConfig& envConfig, const float* __restrict__ pos, float margin)   // margin should be e.g. minDist/2 in the actual dynamics and (minDist * factor) / 2 for MPPI
{
    for (int d = 0; d < DIM; d++)
        if (pos[d] - margin < envConfig.arenaMin[d] || pos[d] + margin > envConfig.arenaMax[d])
            return true;

    for (int iObs = 0; iObs < N_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dist = fmaxf(0.0f, fmaxf(envConfig.obstacles[iObs * 2 * DIM + d] - pos[d], pos[d] - envConfig.obstacles[(iObs * 2 + 1) * DIM + d]));
            sqDist += dist * dist;
        }

        if (sqDist <= margin * margin)
            return true;
    }

    for (int iObs = 0; iObs < N_ROUND_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = pos[d] - envConfig.roundObsCenters[iObs * DIM + d];
            sqDist += dx * dx;
        }
        if (sqDist <= (margin + envConfig.roundObsRadius[iObs]) * (margin + envConfig.roundObsRadius[iObs]))
            return true;
    }

    return false;
}

// if agent == marginAgent, use additional margin (we restrict to passing within gateRadius * margin of the gate center)
HD INLINE void updateGates(const EnvironmentConfig& envConfig, const float* __restrict__ pos, const float* __restrict__ prevPos, float* __restrict__ currentS, int* __restrict__ currentGates, int* __restrict__ nLaps, int marginAgent = -1, float margin = 00.f)
{
    for (int iAgent = 0; iAgent < N_AGENTS; iAgent++)
    {
        float dist;

        if (currentS[iAgent * N_RACELINES] >= 0.0f)
            for (int iRaceline = 0; iRaceline < N_RACELINES; iRaceline++)
            {
                float s = fastProjectOnTrack(envConfig.trackPoints + iRaceline * N_TRACK_SAMPLES * DIM, pos + iAgent * DIM, nullptr, dist, currentS[iAgent * N_RACELINES + iRaceline]);
                currentS[iAgent * N_RACELINES + iRaceline] = s;
            }

        // check if we passed through next gate: compute lambda = dot(vec, center - x_t) / dot(vec, x_{t+1} - x_t)
        int nextGate = (currentGates[iAgent] + 1) % N_GATES;
        float num = 0.f, denom = 0.f;
        for (int d = 0; d < DIM; d++)
        {
            num += envConfig.gateVectors[nextGate * DIM + d] * (envConfig.gateCenters[nextGate * DIM + d] - prevPos[iAgent * DIM + d]);
            denom += envConfig.gateVectors[nextGate * DIM + d] * (pos[iAgent * DIM + d] - prevPos[iAgent * DIM + d]);
        }

        // direction is inside the gate plan: cannot cross
        if (fabsf(denom) < 1e-10f)
            continue;

        float lambda = num / denom;
        // we cross if 0 <= lambda <= 1 and if the projection of the segment (x_t, x_t+1) on the gate plan (ie. (1 - lambda) * x_t + lambda * x_t+1) is at distance <= radius from the center
        // if we want to make sure we cross the gate in the right direction, we have to check num >= 0 (<=> denom > 0). Here, we allows passing through in both directions

        if (lambda < 0.f || lambda > 1.f)
            continue;

        float sqDist = 0.f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = (1.f - lambda) * prevPos[iAgent * DIM + d] + lambda * pos[iAgent * DIM + d] - envConfig.gateCenters[nextGate * DIM + d];
            sqDist += dx * dx;
        }

        float rad = envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate];
        if (iAgent == marginAgent)
            rad *= margin * margin;

        if (sqDist <= rad)
        {
            currentGates[iAgent]++;
            if (currentGates[iAgent] == N_GATES)
            {
                currentGates[iAgent] = 0;
                nLaps[iAgent]++;
            }
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

// initialize branch state (copy belief, find confident, set branch times = defaultBranchTime)
HD INLINE void initBranchState(
    BranchState& branchState,
    const float* initBelief,
    float threshold,
    int defaultBranchTime)
{
    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
        branchState.belief[theta] = initBelief[theta];

    findConfident(branchState.belief, threshold, branchState.predTheta);

    for (int k = 0; k < N_MODEL_FACTORS; k++)
        branchState.branchingTime[k] = branchState.predTheta[k] == 0 ? defaultBranchTime : 0;
}

// find local time origin for a given branch state (max(branchTime[k] for k such that predTheta[k] != 0, 0 if none))
HD INLINE int localBranchTimeOrigin(const int* predTheta, const int* branchingTime)
{
    int max = 0;
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if (predTheta[k] != 0 && branchingTime[k] > max)
            max = branchingTime[k];

    return max;
}

// return whether we are already committed to a plan which is compatible with trueTheta (this happen if for all k, predTheta[k] == 0 or predTheta[k] == trueTheta[k] + 1)
HD INLINE bool branchCompatibleWithModel(const int* predTheta, const int* trueTheta)
{
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if (predTheta[k] != 0 && predTheta[k] != trueTheta[k] + 1)
            return false;

    return true;
}

// Return whether branchTuple is active for this realized path at local time tLocal (branchTuple[k] = 0 if nominal, i+1 if committed to i; used[k] = 0 if in this sample factor k never committed (and branchingTime[k] = T), otherwise 1+theta and branchingTime is the absolute committment time). Return false if tAbs >= T or if factors are inconsistent
HD INLINE bool branchActiveAtLocalTime(
    const int* branchTuple,      // N_MODEL_FACTORS
    const int* used,             // N_MODEL_FACTORS
    const int* branchingTime,    // N_MODEL_FACTORS
    int tLocal,
    int T)
{
    int tOrigin = localBranchTimeOrigin(branchTuple, branchingTime);
    int tAbs = tOrigin + tLocal;

    if (tAbs >= T)
        return false;

    for (int k = 0; k < N_MODEL_FACTORS; k++)
    {
        if ((branchTuple[k] == 0 && tAbs >= branchingTime[k]) || (branchTuple[k] != 0 && (used[k] != branchTuple[k] || tAbs < branchingTime[k])))
            return false;
    }

    return true;

}
HD INLINE bool branchActiveAtAbsoluteTime(
    const int* branchTuple,      // N_MODEL_FACTORS
    const int* used,             // N_MODEL_FACTORS
    const int* branchingTime,    // N_MODEL_FACTORS
    int tAbs,
    int T)
{
    if (tAbs >= T)
        return false;

    for (int k = 0; k < N_MODEL_FACTORS; k++)
    {
        if ((branchTuple[k] == 0 && tAbs >= branchingTime[k]) ||
            (branchTuple[k] != 0 && (used[k] != branchTuple[k] || tAbs < branchingTime[k])))
            return false;
    }

    return true;
}
