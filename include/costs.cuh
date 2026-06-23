#pragma once

#include "config.h"
#include "state.h"
#include <cmath>

// Running cost
__device__ INLINE float stateCost(
    int agent,
    const SimState& state,
    int timestep,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig)
{
    float cost = 0.0f;
    const float* curPos = state.pos + agent * DIM;

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
            float dist = agentDist(state.pos, agent, other);
            if (dist < mppiConfig.oppDistThresholdFactor * envConfig.minDist)
                cost += mppiConfig.oppDistWeight / powf(dist / envConfig.minDist, mppiConfig.oppDistPower);

            if (dist < envConfig.minDist * mppiConfig.collDistFactor)
                cost += mppiConfig.collisionCost;

            // opponent outside: bonus for us
            // float oBd = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, oPos);
            // if (oBd < 0.0f)
            if (mppiConfig.oppOutsideCost != 0.0f && isOutside(envConfig, state.pos + other * DIM, envConfig.minDist / 2.0f))
                cost -= mppiConfig.oppOutsideCost;
        }
    }

    // winner check
    if (state.laps[agent] >= (float) envConfig.nWinLaps)
        cost -= mppiConfig.winCost;

    return cost;
}

// Terminal cost
__device__ INLINE float finalCost(
    int agent,
    const SimState& state,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig)
{
    float cost = 0.0f;
    float maxOppAdv = -1e30f;

    for (int a = 0; a < N_AGENTS; a++)
    {
        // float advance = getAdvance(S, laps, currentGates, agent);
        float advance = getAdvance(state.laps, state.gates, a, state.pos + a * DIM, envConfig.gateCenters);
        // printf("Advance: %f\n", advance);

        if (a == agent)
        {
            cost -= mppiConfig.finalAdvWeight * advance;

            if (mppiConfig.finalSpeedWeight != 0.0f)
            {
                // target direction is nextGate - pos
                int nextGate = (state.gates[agent] + 1) % N_GATES;

                float target[DIM];
                float sqNorm = 0.0f;
                for (int d = 0; d < DIM; d++)
                {
                    target[d] = envConfig.gateCenters[nextGate * DIM + d] - state.pos[a * DIM + d];
                    sqNorm += target[d] * target[d];
                }

                float norm = sqrtf(sqNorm) + 1e-5;

                float dot = 0.0f;
                for (int d = 0; d < DIM; d++)
                    dot += state.vel[agent * DIM + d] * target[d] / norm;

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

// Safety cost, > 0 iff state is safe (min of distance to nearest obstacles and to other agents, if any)
__device__ INLINE float safetyCost(
    int agent,
    const SimState& state,
    int timestep,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig)
{
    float cost = trackBoundaryDist(envConfig, state.pos + agent * DIM);

    if (N_AGENTS > 1)
        for (int other = 0; other < N_AGENTS; other++)
            if (other != agent)
            {
                float dist = agentDist(state.pos, agent, other);
                if (cost < dist)
                    cost = dist;
            }

    return cost;
}
