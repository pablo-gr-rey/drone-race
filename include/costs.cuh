#pragma once

#include "config.h"
#include "device_config.cuh"
#include "state.h"
#include "track.cuh"
#include <cmath>

// ── Running cost ─────────────────────────────────────────────────────
__device__ inline float stateCost(
    int agent,
    const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates,
    int timestep,
    const DeviceEnvironmentConfig& envConfig,
    const DeviceMPPIConfig& mppiConfig,
    const float* trackPoints, int nTP)
{
    float cost = 0.0f;
    const float* curPos = pos + agent * envConfig.dim;

    float decay = powf(0.9f, (float) timestep);

    for (int other = 0; other < envConfig.nAgents; other++)
    {
        if (other == agent)
        {
            // own boundary
            float bd = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, curPos, envConfig.dim);

            float danger = mppiConfig.boundaryThresholdFactor * envConfig.minDist;
            if (bd < danger)
                cost += mppiConfig.boundaryCost * (danger - bd) / danger * decay;
            if (bd < 0.0f)
                cost += mppiConfig.outsideCost * decay;
        }
        else
        {
            float dist = agentDist(pos, agent, other, envConfig.dim);
            if (dist < mppiConfig.oppDistThresholdFactor * envConfig.minDist)
                cost += mppiConfig.oppDistWeight / powf(dist / envConfig.minDist, mppiConfig.oppDistPower) * decay;
            if (dist < envConfig.minDist * mppiConfig.collDistFactor)
                cost += mppiConfig.collisionCost * decay;

            // opponent outside: bonus for us
            const float* oPos = pos + other * envConfig.dim;
            float oBd = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, oPos, envConfig.dim);
            if (oBd < 0.0f)
                cost -= mppiConfig.oppOutsideCost * decay;
        }
    }

    // winner check
    if (laps[agent] >= (float) envConfig.nWinLaps)
        cost -= mppiConfig.winCost * decay;

    return cost;
}

// ── Terminal cost ────────────────────────────────────────────────────
__device__ inline float finalCost(
    int agent,
    const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates,
    const DeviceEnvironmentConfig& envConfig,
    const DeviceMPPIConfig& mppiConfig,
    const float* trackPoints, int nTP)
{
    float cost = 0.0f;
    float maxOppAdv = -1e30f;

    for (int a = 0; a < envConfig.nAgents; a++)
    {
        // float advance = getAdvance(S, laps, currentGates, agent, envConfig.nGates);
        float advance = getAdvance(laps, currentGates, a, envConfig.nGates, pos + a * envConfig.dim, envConfig.gateCenters, envConfig.dim);
        // printf("Advance: %f\n", advance);

        if (a == agent)
        {
            // update advance
            // advance = laps[agent] * envConfig.nGates + currentGates[agent];

            // target: track point at s + targetDistance
            float target[MAX_DIM];
            sampleCenterline(trackPoints, nTP, envConfig.dim, S[a] + envConfig.targetDistance, target);

            float diff[MAX_DIM];
            const float* spd = speed + a * envConfig.dim;
            float diffNorm = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
            {
                diff[d] = target[d] - pos[a * envConfig.dim + d];
                diffNorm += diff[d] * diff[d];
            }
            diffNorm = sqrtf(diffNorm) + 1e-8f;

            float dot = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
                dot += spd[d] * (diff[d] / diffNorm);

            cost -= mppiConfig.finalAdvWeight * advance + mppiConfig.finalSpeedWeight * dot;
        }
        else if (advance > maxOppAdv)
            maxOppAdv = advance;
    }

    if (envConfig.nAgents > 1)
        cost += mppiConfig.finalOppAdvWeight * maxOppAdv;

    return cost;
}
