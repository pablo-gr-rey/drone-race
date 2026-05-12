#pragma once

#include "config.h"
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

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

    for (int d = 0; d < dim; d++)
    {
        sqGateDist += (gateCenters[nextGate * dim + d] - pos[d]) * (gateCenters[nextGate * dim + d] - pos[d]);
        sqConsGateDist += (gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d]) * (gateCenters[nextGate * dim + d] - gateCenters[currentGates[agent] * dim + d]);
    }

    return currentGates[agent] + nGates * laps[agent] + 0.5f - 0.5f * sqrtf(sqGateDist / sqConsGateDist);
}
