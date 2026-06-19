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
    const float* curPos = pos + agent * DIM;

    for (int other = 0; other < N_AGENTS; other++)
    {
        if (other == agent)
        {
            // own boundary
            if (mppiConfig.boundaryCost != 0.0f)
            {
                float bd = trackBoundaryDist(envConfig, curPos);

                float danger = mppiConfig.boundaryThresholdFactor * envConfig.minDist;
                if (bd < danger)
                    // cost += mppiConfig.boundaryCost * (danger - bd) / danger;
                    cost += mppiConfig.boundaryCost / powf(bd / envConfig.minDist, 2.0f);
            }

            // for own agent, add a margin to keep it further off the boundary (since otherwise noise can push it a bit)
            if (isOutside(envConfig, curPos, envConfig.minDist * mppiConfig.collDistFactor / 2.0f))
                cost += mppiConfig.outsideCost;
        }
        else
        {
            float dist = agentDist(pos, agent, other);
            if (dist < mppiConfig.oppDistThresholdFactor * envConfig.minDist)
                cost += mppiConfig.oppDistWeight / powf(dist / envConfig.minDist, mppiConfig.oppDistPower);

            if (dist < envConfig.minDist * mppiConfig.collDistFactor)
                cost += mppiConfig.collisionCost;

            // opponent outside: bonus for us
            // float oBd = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, oPos);
            // if (oBd < 0.0f)
            if (mppiConfig.oppOutsideCost != 0.0f && isOutside(envConfig, pos + other * DIM, envConfig.minDist / 2.0f))
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

    for (int a = 0; a < N_AGENTS; a++)
    {
        // float advance = getAdvance(S, laps, currentGates, agent);
        float advance = getAdvance(laps, currentGates, a, pos + a * DIM, envConfig.gateCenters);
        // printf("Advance: %f\n", advance);

        if (a == agent)
        {
            cost -= mppiConfig.finalAdvWeight * advance;

            if (mppiConfig.finalSpeedWeight != 0.0f)
            {
                // target direction is nextGate - pos
                int nextGate = (currentGates[agent] + 1) % N_GATES;

                float target[DIM];
                float sqNorm = 0.0f;
                for (int d = 0; d < DIM; d++)
                {
                    target[d] = envConfig.gateCenters[nextGate * DIM + d] - pos[a * DIM + d];
                    sqNorm += target[d] * target[d];
                }

                float norm = sqrtf(sqNorm) + 1e-5;

                float dot = 0.0f;
                for (int d = 0; d < DIM; d++)
                    dot += speed[agent * DIM + d] * target[d] / norm;

                cost -= mppiConfig.finalSpeedWeight * dot;
            }
        }
        else if (advance > maxOppAdv)
            maxOppAdv = advance;
    }

    if (N_AGENTS > 1)
        cost += mppiConfig.finalOppAdvWeight * maxOppAdv;

    return cost;
}
