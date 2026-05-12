#pragma once

#include "config.h"
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ── position of agent along axis d ───────────────────────────────────
HD inline float getPos(const float* phys, int agent, int d, int dim)
{
    return phys[agent * dim * 2 + d * 2];
}

// ── velocity of agent along axis d ───────────────────────────────────
HD inline float getVel(const float* phys, int agent, int d, int dim)
{
    return phys[agent * dim * 2 + d * 2 + 1];
}

HD inline void setPos(float* phys, int agent, int d, int dim, float v)
{
    phys[agent * dim * 2 + d * 2] = v;
}

HD inline void setVel(float* phys, int agent, int d, int dim, float v)
{
    phys[agent * dim * 2 + d * 2 + 1] = v;
}

// ── Euclidean distance between two agents (positions only) ───────────
HD inline float agentDist(const float* phys, int a, int b, int dim)
{
    float s = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float dx = getPos(phys, a, d, dim) - getPos(phys, b, d, dim);
        s += dx * dx;
    }
    return sqrtf(s);
}

// ── Speed (L2 norm of velocity) ──────────────────────────────────────
HD inline float agentSpeed(const float* phys, int agent, int dim)
{
    float s = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float v = getVel(phys, agent, d, dim);
        s += v * v;
    }
    return sqrtf(s);
}

// Advance = currentGate + nGates * laps + 0.5 * (1 - normalizedDistToGate) (it is much better to pass through a gate than to just be close to it) (this is a rough measure, it doesn't include speed for example)
HD inline float getAdvance(const int* laps, const int* currentGates, int agent, int nGates, const float* pos, const float* gateCenters, int dim, int posStride = 2)
{
    float sqGateDist = 0.0f;    // sq dist between pos and next gate
    float sqConsGateDist = 0.0f;    // sq dist between current gate and next gate
    int nextGate = (currentGates[agent] + 1) % nGates;

    for (int d = 0; d < dim; d++)
    {
        sqGateDist += (gateCenters[nextGate * dim + d] - pos[d * posStride]) * (gateCenters[nextGate * dim + d] - pos[d * posStride]);
        sqConsGateDist += (gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d]) * (gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d]);
        // printf("\tfor dim d=%d: currentGate=%d, nextGate=%d, dx cons = %f (from %f to %f), current sqConsGateDist=%f\n",
            // d, currentGates[agent], nextGate, gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d], gateCenters[currentGates[agent] * dim + d], gateCenters[nextGate * dim + d], sqConsGateDist);
    }

    // printf("sqConsGateDist %f sqGateDist %f sqrt(...) %f final res %f\n", sqGateDist, sqConsGateDist, sqrtf(sqGateDist / sqConsGateDist), currentGates[agent] + nGates * laps[agent] + 0.5f - 0.5f * sqrtf(sqGateDist / sqConsGateDist));

    return currentGates[agent] + nGates * laps[agent] + 0.5f - 0.5f * sqrtf(sqGateDist / sqConsGateDist);
}
