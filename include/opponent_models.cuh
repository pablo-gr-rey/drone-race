#pragma once

#include "config.h"
#include "device_config.cuh"
#include "state.h"
#include "track.cuh"

__device__ inline void predictOpponent(
    OpponentModelType model,
    int opp,
    const float* phys, const float* S, const float* /*laps*/,
    const DeviceEnvironmentConfig& cfg,
    const PIDConfig& pid,
    const float* trackPoints, int nTP,
    float* outAction)
{
    switch (model)
    {
    case OpponentModelType::Dummy:
        for (int d = 0; d < cfg.dim; d++) outAction[d] = 0.0f;
        break;

    case OpponentModelType::PID: {
        float target[MAX_DIM];
        sampleCenterline(trackPoints, nTP, cfg.dim,
            S[opp] + cfg.targetDistance, target);

        float pos[MAX_DIM], vel[MAX_DIM];
        for (int d = 0; d < cfg.dim; d++)
        {
            pos[d] = getPos(phys, opp, d, cfg.dim);
            vel[d] = getVel(phys, opp, d, cfg.dim);
        }

        // PD (no integral in rollout prediction)
        for (int d = 0; d < cfg.dim; d++)
            outAction[d] = pid.kp * (target[d] - pos[d]) + pid.kd * (-vel[d]);

        // repulsion
        for (int other = 0; other < cfg.nAgents; other++)
        {
            if (other == opp) continue;
            float diff[MAX_DIM];
            float dist2 = 0.0f;
            for (int d = 0; d < cfg.dim; d++)
            {
                diff[d] = getPos(phys, other, d, cfg.dim) - pos[d];
                dist2 += diff[d] * diff[d];
            }
            float dist = sqrtf(dist2) + 1e-8f;
            if (dist < pid.repulsionDistFact * cfg.minDist)
            {
                float scale = pid.repulsionFactor
                    / powf(dist, pid.repulsionPower + 1.0f);
                for (int d = 0; d < cfg.dim; d++)
                    outAction[d] -= scale * diff[d];
            }
        }
        break;
    }
    }
}
