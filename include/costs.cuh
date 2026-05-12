#pragma once

#include "config.h"
#include "device_config.cuh"
#include "state.h"
#include "track.cuh"
#include <cmath>

// ── Running cost ─────────────────────────────────────────────────────
__device__ inline float stateCost(
    int agent,
    const float* phys, const float* S, const int* laps, const int* currentGates,
    int timestep,
    const DeviceEnvironmentConfig& envConfig,
    const DeviceMPPIConfig& mppiConfig,
    const float* trackPoints, int nTP)
{
    float cost = 0.0f;
    float pos[MAX_DIM];
    for (int d = 0; d < envConfig.dim; d++)
        pos[d] = getPos(phys, agent, d, envConfig.dim);

    float decay = powf(0.9f, (float) timestep);

    for (int other = 0; other < envConfig.nAgents; other++)
    {
        if (other == agent)
        {
            // own boundary
            float bd = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, pos, envConfig.dim);

            float danger = mppiConfig.boundaryThresholdFactor * envConfig.minDist;
            if (bd < danger)
                cost += mppiConfig.boundaryCost * (danger - bd) / danger * decay;
            if (bd < 0.0f)
                cost += mppiConfig.outsideCost * decay;
        }
        else
        {
            float dist = agentDist(phys, agent, other, envConfig.dim);
            if (dist < mppiConfig.oppDistThresholdFactor * envConfig.minDist)
                cost += mppiConfig.oppDistWeight / powf(dist / envConfig.minDist, mppiConfig.oppDistPower) * decay;
            if (dist < envConfig.minDist * mppiConfig.collDistFactor)
                cost += mppiConfig.collisionCost * decay;

            // opponent outside: bonus for us
            float oPos[MAX_DIM];
            for (int d = 0; d < envConfig.dim; d++)
                oPos[d] = getPos(phys, other, d, envConfig.dim);
            // float oBd = trackBoundaryDist(trackPoints, nTP, envConfig.dim, envConfig.trackWidth, oPos);
            // float oBd = trackBoundaryDist(trackPoints, nTP, envConfig.dim, envConfig.trackWidth, oPos, S[other]);
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
    const float* phys, const float* S, const int* laps, const int* currentGates,
    const DeviceEnvironmentConfig& envConfig,
    const DeviceMPPIConfig& mppiConfig,
    const float* trackPoints, int nTP)
{
    float cost = 0.0f;
    float maxOppAdv = -1e30f;

    for (int a = 0; a < envConfig.nAgents; a++)
    {
        // float advance = getAdvance(S, laps, currentGates, agent, envConfig.nGates);
        float advance = getAdvance(laps, currentGates, a, envConfig.nGates, phys + a * envConfig.dim * 2, envConfig.gateCenters, envConfig.dim);
        // printf("Advance: %f\n", advance);

        if (a == agent)
        {
            // update advance
            // advance = laps[agent] * envConfig.nGates + currentGates[agent];

            // target: track point at s + targetDistance
            float target[MAX_DIM];
            sampleCenterline(trackPoints, nTP, envConfig.dim, S[a] + envConfig.targetDistance, target);

            float diff[MAX_DIM], speed[MAX_DIM];
            float diffNorm = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
            {
                diff[d] = target[d] - getPos(phys, a, d, envConfig.dim);
                speed[d] = getVel(phys, a, d, envConfig.dim);
                diffNorm += diff[d] * diff[d];
            }
            diffNorm = sqrtf(diffNorm) + 1e-8f;

            float dot = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
                dot += speed[d] * (diff[d] / diffNorm);

            cost -= mppiConfig.finalAdvWeight * advance + mppiConfig.finalSpeedWeight * dot;
        }
        else if (advance > maxOppAdv)
            maxOppAdv = advance;
    }

    if (envConfig.nAgents > 1)
        cost += mppiConfig.finalOppAdvWeight * maxOppAdv;

    return cost;
}
