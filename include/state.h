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

// ── Advance = S + laps ───────────────────────────────────────────────
HD inline float getAdvance(const float* S, const float* laps, int agent)
{
    return S[agent] + laps[agent];
}