#pragma once

#include "config.h"
#include "state.h"
#include <cfloat>

// Linear interpolation of pre-sampled centerline
// trackPoints: (nSamples, dim)  row-major
__device__ inline void sampleCenterline(const float* trackPoints, int nSamples, int dim, float s, float* out)
{
    s = s - floorf(s);                      // wrap to [0,1)
    float idx_f = s * nSamples;
    int   idx0 = (int) idx_f;
    int   idx1 = min(idx0 + 1, nSamples - 1);
    float t = idx_f - idx0;
    for (int d = 0; d < dim; d++)
        out[d] = (1.0f - t) * trackPoints[idx0 * dim + d] + t * trackPoints[idx1 * dim + d];
}

// Project position onto sampled track
// Returns best s in [0,1]; writes distance into bestDist.
// closestOut may be nullptr.
__device__ inline float projectOnTrack(const float* trackPoints,
    int nSamples, int dim,
    const float* pos,
    float* closestOut,
    float& bestDist)
{
    float bestS = 0.0f;
    float bestD2 = FLT_MAX;
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
__device__ inline float sqDistToSample(const float* trackPoints, int iTrack, int dim, const float* pos)
{
    float sqdist = 0.;
    for (int d = 0; d < dim; d++)
        sqdist += (trackPoints[iTrack * dim + d] - pos[d]) * (trackPoints[iTrack * dim + d] - pos[d]);
    return sqdist;
}

// util function to keep going in one direction until local maximum
__device__ inline int findMinAlongDirection(const float* trackPoints, int nSamples, int iTrack, int dim, const float* pos, int delta, float baseSqDist, float& bestDist)
{
    // greedy search + margin (works if the track is locally convex)
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

__device__ inline float fastProjectOnTrack(const float* trackPoints,
    int nSamples, int dim,
    const float* pos,
    float* closestOut,
    float& bestDist,
    float prevS = -1.0f   // negative means unknown -> full scan
)
{
    if (prevS < 0)
        return projectOnTrack(trackPoints, nSamples, dim, pos, closestOut, bestDist);

    int baseI = (int) (prevS * nSamples + 0.5);
    float baseDist = sqDistToSample(trackPoints, baseI, dim, pos);

    // find best forwards and backwards distances
    float bestFDist, bestBDist;
    int bestFI = findMinAlongDirection(trackPoints, nSamples, baseI, dim, pos, +1, baseDist, bestFDist);
    int bestBI = findMinAlongDirection(trackPoints, nSamples, baseI, dim, pos, -1, baseDist, bestBDist);

    if (bestFDist < bestBDist)
    {
        bestDist = sqrt(bestFDist);
        if (closestOut)
            for (int d = 0; d < dim; d++)
                closestOut[d] = trackPoints[bestFI * dim + d];

        return ((float) bestFI) / nSamples;
    }

    bestDist = sqrt(bestBDist);
    if (closestOut)
        for (int d = 0; d < dim; d++)
            closestOut[d] = trackPoints[bestBI * dim + d];

    return ((float) bestBI) / nSamples;
}

// Boundary distance = distance to closest boundary
__device__ inline float trackBoundaryDist(const float* arenaMin, const float* arenaMax, const float* pos, int dim)
{
    float minDist = INFINITY;
    for (int d = 0; d < dim; d++)
        minDist = min(minDist, min(pos[d] - arenaMin[d], arenaMax[d] - pos[d]));
    return minDist;
}

// Boundary distance = trackWidth/2 – dist to centerline
// __device__ inline float fastTrackBoundaryDist(const float* trackPoints, int nSamples, int dim, float trackWidth, const float* pos, float prevS)
// {
//     float dist;
//     fastProjectOnTrack(trackPoints, nSamples, dim, pos, nullptr, dist, prevS);
//     return trackWidth * 0.5f - dist;
// }
