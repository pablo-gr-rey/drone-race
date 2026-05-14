#pragma once

#include "config.h"
#include "device_config.cuh"
#include "state.h"
#include "track.cuh"

__device__ inline void predictOpponent(
    OpponentModelType model,
    int opp,
    const float* pos, const float* vel, const float* S, const int* /*laps*/, const int* currentGates,
    const DeviceEnvironmentConfig& envConfig,
    const PIDConfig& pid,
    const float* trackPoints, int nTP,
    float* outAction)
{
    switch (model)
    {
    case OpponentModelType::Dummy:
        for (int d = 0; d < envConfig.dim; d++)
            outAction[d] = 0.0f;
        break;

    case OpponentModelType::PID: {
        // should be the same as PIDController::getControl in controllers.cpp!
        // we follow target at a distance envConfig.targetDistance on our actual raceline index (raceline always contain nSamples points and are concatenated, with raceline 0 being the centerline)
        float target[MAX_DIM];

        if (pid.racelineIndex >= 0)
            sampleCenterline(trackPoints + envConfig.nTrackSamples * pid.racelineIndex * envConfig.dim, nTP, envConfig.dim, S[opp] + envConfig.targetDistance, target);
        else
        {
            int nextGate = (currentGates[opp] + 1) % envConfig.nGates;
            for (int d = 0; d < envConfig.dim; d++)
                target[d] = envConfig.gateCenters[nextGate * envConfig.dim + d];
        }

        const float* curPos = pos + opp * envConfig.dim;
        const float* curVel = vel + opp * envConfig.dim;

        float sqError = 0.0f;
        for (int d = 0; d < envConfig.dim; d++)
        {
            float e = target[d] - curPos[d];
            sqError += e * e;
        }

        // Match CPU logic
        float invDist = rsqrtf(sqError + 1e-5f);

        float vParallelMag = 0.0f;
        for (int d = 0; d < envConfig.dim; d++)
        {
            float dirD = (target[d] - curPos[d]) * invDist;
            vParallelMag += curVel[d] * dirD;
        }

        for (int d = 0; d < envConfig.dim; d++)
        {
            float error = (target[d] - curPos[d]) * invDist;
            float latVelError = curVel[d] - vParallelMag * error;
            outAction[d] = pid.kp * error + pid.kd * (-latVelError);
        }

        // PD (no integral)
        // for (int d = 0; d < envConfig.dim; d++)
        //     outAction[d] = pid.kp * (target[d] - pos[d]) + pid.kd * (-vel[d]);

        // repulsion
        for (int other = 0; other < envConfig.nAgents; other++)
        {
            if (other == opp) continue;
            float diff[MAX_DIM];
            float dist2 = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
            {
                diff[d] = pos[other * envConfig.dim + d] - curPos[d];
                dist2 += diff[d] * diff[d];
            }
            float dist = sqrtf(dist2) + 1e-8f;
            if (dist < pid.repulsionDistFact * envConfig.minDist)
            {
                float scale = pid.repulsionFactor / powf(dist, pid.repulsionPower);
                for (int d = 0; d < envConfig.dim; d++)
                    outAction[d] -= scale * diff[d];
            }
        }
        break;
    }
    }
}
