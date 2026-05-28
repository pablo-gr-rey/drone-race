#pragma once

#include "config.h"
#include "state.h"
#include <cmath>

// ── Running cost ─────────────────────────────────────────────────────
__device__ INLINE float stateCost(
    int agent,
    const float* __restrict__ pos, const float* __restrict__ speed, const int* __restrict__ laps, const int* __restrict__ currentGates,
    int timestep,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig)
{
    float cost = 0.0f;
    const float* curPos = pos + agent * envConfig.dim;

    for (int other = 0; other < envConfig.nAgents; other++)
    {
        if (other == agent)
        {
            // own boundary
            if (mppiConfig.boundaryCost != 0.0f)
            {
                float bd = trackBoundaryDist(envConfig, curPos);

                float danger = mppiConfig.boundaryThresholdFactor * envConfig.minDist;
                if (bd < danger)
                    cost += mppiConfig.boundaryCost * (danger - bd) / danger;
            }

            // for own agent, add a margin to keep it further off the boundary (since otherwise noise can push it a bit)
            if (isOutside(envConfig, curPos, envConfig.minDist * mppiConfig.collDistFactor / 2.0f))
                cost += mppiConfig.outsideCost;
        }
        else
        {
            float dist = agentDist(pos, agent, other, envConfig.dim);
            if (dist < mppiConfig.oppDistThresholdFactor * envConfig.minDist)
                cost += mppiConfig.oppDistWeight / powf(dist / envConfig.minDist, mppiConfig.oppDistPower);

            if (dist < envConfig.minDist * mppiConfig.collDistFactor)
                cost += mppiConfig.collisionCost;

            // opponent outside: bonus for us
            // float oBd = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, oPos, envConfig.dim);
            // if (oBd < 0.0f)
            if (mppiConfig.oppOutsideCost != 0.0f && isOutside(envConfig, pos + other * envConfig.dim, envConfig.minDist / 2.0f))
                cost -= mppiConfig.oppOutsideCost;
        }
    }

    // winner check
    if (laps[agent] >= (float) envConfig.nWinLaps)
        cost -= mppiConfig.winCost;

    return cost;
}

// ── Terminal cost ────────────────────────────────────────────────────
__device__ INLINE float finalCost(
    int agent,
    const float* __restrict__ pos, const float* __restrict__ speed, const int* __restrict__ laps, const int* __restrict__ currentGates,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig)
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
            cost -= mppiConfig.finalAdvWeight * advance;

            if (mppiConfig.finalSpeedWeight != 0.0f)
            {
                // target: track point at s + targetDistance
                // float target[MAX_DIM];
                // sampleCenterline(trackPoints, nTP, envConfig.dim, currentS[a] + envConfig.targetDistance, target);

                // float diff[MAX_DIM];
                // const float* spd = speed + a * envConfig.dim;
                // float diffNorm = 0.0f;
                // for (int d = 0; d < envConfig.dim; d++)
                // {
                //     diff[d] = target[d] - pos[a * envConfig.dim + d];
                //     diffNorm += diff[d] * diff[d];
                // }
                // diffNorm = sqrtf(diffNorm) + 1e-8f;

                // float dot = 0.0f;
                // for (int d = 0; d < envConfig.dim; d++)
                //     dot += spd[d] * (diff[d] / diffNorm);

                // target direction is nextGate - pos
                int nextGate = (currentGates[agent] + 1) % envConfig.nGates;

                float target[MAX_DIM];
                float sqNorm = 0.0f;
                for (int d = 0; d < envConfig.dim; d++)
                {
                    target[d] = envConfig.gateCenters[nextGate * envConfig.dim + d] - pos[a * envConfig.dim + d];
                    sqNorm += target[d] * target[d];
                }

                float norm = sqrtf(sqNorm) + 1e-5;

                float dot = 0.0f;
                for (int d = 0; d < envConfig.dim; d++)
                    dot += speed[agent * envConfig.dim + d] * target[d] / norm;

                cost -= mppiConfig.finalSpeedWeight * dot;
            }
        }
        else if (advance > maxOppAdv)
            maxOppAdv = advance;
    }

    if (envConfig.nAgents > 1)
        cost += mppiConfig.finalOppAdvWeight * maxOppAdv;

    return cost;
}
