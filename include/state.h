#pragma once

#include "config.h"
#include "cuda_runtime.h"

#include <cmath>
#include <algorithm>
#include <cuda/std/limits>


// Euclidean distance between two agents (positions only)
HD inline float agentDist(const float* pos, int a, int b, int dim)
{
    float s = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float dx = pos[a * dim + d] - pos[b * dim + d];
        s += dx * dx;
    }

    return sqrtf(s);
}

// Speed (L2 norm of velocity)
HD inline float agentSpeed(const float* speed, int agent, int dim)
{
    float s = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float v = speed[agent * dim + d];
        s += v * v;
    }
    return sqrtf(s);
}

// Advance = currentGate + nGates * laps + 0.5 * (1 - normalizedDistToGate) (it is much better to pass through a gate than to just be close to it) (this is a rough measure, it doesn't include speed for example)
// `pos` points to the agent's contiguous position array of length `dim`.
HD inline float getAdvance(const int* laps, const int* currentGates, int agent, int nGates, const float* pos, const float* gateCenters, int dim)
{
    float sqGateDist = 0.0f;    // sq dist between pos and next gate
    float sqConsGateDist = 0.0f;    // sq dist between current gate and next gate
    int nextGate = (currentGates[agent] + 1) % nGates;

    float scale = 1.0f;     // 1.0f means reward is continuous when going through a gate; 0.5f for example means that reward will be between 0.25 and 0.75 before the first gate, 1.25 and 1.75 between 1st and 2nd, etc

    for (int d = 0; d < dim; d++)
    {
        sqGateDist += (gateCenters[nextGate * dim + d] - pos[d]) * (gateCenters[nextGate * dim + d] - pos[d]);
        sqConsGateDist += (gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d]) * (gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d]);
    }

    return currentGates[agent] + nGates * laps[agent] + scale * (1 - sqrtf(sqGateDist / sqConsGateDist));
}

// Linear interpolation of pre-sampled centerline
HD inline void sampleCenterline(const float* trackPoints, int nSamples, int dim, float s, float* out)
{
    s = s - floorf(s);                      // wrap to [0,1)
    float idx_f = s * nSamples;
    int   idx0 = (int) idx_f;
    int   idx1 = (idx0 + 1) % nSamples;
    float t = idx_f - idx0;
    for (int d = 0; d < dim; d++)
        out[d] = (1.0f - t) * trackPoints[idx0 * dim + d] + t * trackPoints[idx1 * dim + d];
}

// Project position onto sampled track
// Returns best s in [0,1]; writes distance into bestDist.
// closestOut may be nullptr.
HD inline float projectOnTrack(const float* trackPoints,
    int nSamples, int dim,
    const float* pos,
    float* closestOut,
    float& bestDist)
{
    float bestS = 0.0f;
    float bestD2 = cuda::std::numeric_limits<float>::max();
    float sStep = 1.0f / nSamples;
    for (int i = 0; i < nSamples; i++)
    {
        float d2 = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            float dx = trackPoints[i * dim + d] - pos[d];
            d2 += dx * dx;
        }
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestS = i * sStep;
            if (closestOut)
                for (int d = 0; d < dim; d++)
                    closestOut[d] = trackPoints[i * dim + d];
        }
    }
    bestDist = sqrtf(bestD2);
    return bestS;
}

// util function to return the distance from pos to the given trackPoints
HD inline float sqDistToSample(const float* trackPoints, int iTrack, int dim, const float* pos)
{
    float sqdist = 0.;
    for (int d = 0; d < dim; d++)
        sqdist += (trackPoints[iTrack * dim + d] - pos[d]) * (trackPoints[iTrack * dim + d] - pos[d]);
    return sqdist;
}

// util function to keep going in one direction until local maximum
HD inline int findMinAlongDirection(const float* trackPoints, int nSamples, int iTrack, int dim, const float* pos, int delta, float baseSqDist, float& bestDist)
{
    // greedy search + margin (should work if the track is not too weird)
    const int margin = 10;

    int bestI = iTrack;
    bestDist = baseSqDist;

    int currentI = (iTrack + delta + nSamples) % nSamples;
    float currentDist = sqDistToSample(trackPoints, currentI, dim, pos);

    while (currentDist < bestDist)
    {
        bestI = currentI;
        bestDist = currentDist;

        currentI = (currentI + delta + nSamples) % nSamples;
        currentDist = sqDistToSample(trackPoints, currentI, dim, pos);
    }

    // additional margin
    for (int i = 0; i < margin; i++)
    {
        currentDist = sqDistToSample(trackPoints, currentI, dim, pos);
        if (currentDist < bestDist)
        {
            bestI = currentI;
            bestDist = currentDist;
        }
        currentI = (currentI + delta + nSamples) % nSamples;
    }

    return bestI;
}

// return the closest S, and writes the closest track point in closestOut and its distance in bestDist
HD inline float fastProjectOnTrack(const float* trackPoints,
    int nSamples, int dim,
    const float* pos,
    float* closestOut,
    float& bestDist,
    float prevS = -1.0f   // negative means unknown -> full scan
)
{
#ifdef CHECK_PROJECTION
    float testS, testDist;
    float testClosestOut[MAX_DIM];
    testS = projectOnTrack(trackPoints, nSamples, dim, pos, testClosestOut, testDist);
#endif

    if (prevS < 0)
        return projectOnTrack(trackPoints, nSamples, dim, pos, closestOut, bestDist);

    int baseI = (int) (prevS * nSamples + 0.5) % nSamples;
    float baseDist = sqDistToSample(trackPoints, baseI, dim, pos);

    // find best forwards and backwards distances
    float bestFDist, bestBDist;
    int bestFI = findMinAlongDirection(trackPoints, nSamples, baseI, dim, pos, +1, baseDist, bestFDist);
    int bestBI = findMinAlongDirection(trackPoints, nSamples, baseI, dim, pos, -1, baseDist, bestBDist);

    float bestS;

    if (bestFDist < bestBDist)
    {
        bestDist = sqrtf(bestFDist);
        if (closestOut)
            for (int d = 0; d < dim; d++)
                closestOut[d] = trackPoints[bestFI * dim + d];

        bestS = ((float) bestFI) / nSamples;
    }
    else
    {
        bestDist = sqrtf(bestBDist);
        if (closestOut)
            for (int d = 0; d < dim; d++)
                closestOut[d] = trackPoints[bestBI * dim + d];

        bestS = ((float) bestBI) / nSamples;
    }

#ifdef CHECK_PROJECTION
    // bool wrong = fabs(bestS - testS) > 1e-5 || fabsf(bestDist - testDist) > 1e-5;
    // for (int d = 0; d < dim && closestOut; d++)
    //     wrong = wrong || abs(closestOut[d] - testClosestOut[d] > 1e-5);

    bool wrong = fabs(bestDist - testDist) > 1e-5f;

    if (wrong)
    {
        printf("WRONG PROJECTION for pos ");
        for (int d = 0; d < dim; d++)
            printf("%f ", pos[d]);
        printf(" prevS %f\n: computed bestS %f\tbestDist %f\tclosestOut ", prevS, bestS, bestDist);
        for (int d = 0; d < dim && closestOut; d++)
            printf("%f ", closestOut[d]);
        printf("\n: expected bestS %f\tbestDist %f\tclosestOut ", testS, testDist);
        for (int d = 0; d < dim; d++)
            printf("%f ", testClosestOut[d]);
        printf("\n");
    }
#endif

    return bestS;
}

// Boundary distance = distance to closest boundary
HD inline float trackBoundaryDist(const float* arenaMin, const float* arenaMax, const float* pos, int dim)
{
    float minDist = INFINITY;
    for (int d = 0; d < dim; d++)
        minDist = fminf(minDist, fminf(pos[d] - arenaMin[d], arenaMax[d] - pos[d]));
    return minDist;
}
