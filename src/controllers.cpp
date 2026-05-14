#include "controllers.h"
#include "engine.h"
#include "state.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ═════════════════════════════════════════════════════════════════════
// Dummy
// ═════════════════════════════════════════════════════════════════════
DummyController::DummyController(const EnvironmentConfig& c)
{
    name = "dummy"; envConfig = &c;
}

void DummyController::getControl(int /*agent*/, const float* /*pos*/, const float* /*speed*/,
    const float* /*S*/, const int* /*laps*/, const int* /* currentGates */,
    float* outAction)
{
    for (int d = 0; d < envConfig->dim; d++)
        outAction[d] = 0.0f;
}

// ═════════════════════════════════════════════════════════════════════
// PID
// ═════════════════════════════════════════════════════════════════════
PIDController::PIDController(const EnvironmentConfig& c, const PIDConfig& p) : params(p)
{
    name = "PID";
    envConfig = &c;
}

void PIDController::getControl(int agent, const float* pos, const float* speed, const float* S, const int* /* laps */, const int* /* currentGates */, float* outAction)
{
    if (!engine) throw std::runtime_error("PID: engine not set");

    int dim = envConfig->dim;

    std::vector<float> target = engine->getTarget(agent, S, params.racelineIndex);

    const float* curPos = pos + agent * dim;
    const float* velArr = speed + agent * dim;

    float sqError = 0.0f;
    for (int d = 0; d < dim; d++)
        sqError += (target[d] - curPos[d]) * (target[d] - curPos[d]);

    // renormalize error so that it always has norm 1
    float invDist = 1 / sqrt(sqError + 1e-5);

    // compute lateral speed error
    float vParallelMag = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float dirD = (target[d] - curPos[d]) * invDist;
        vParallelMag += velArr[d] * dirD;
    }

    // we only correct the lateral velocity (if we're going in a straight line, we shouldn't brake to be able to reach full speed)
    for (int d = 0; d < dim; d++)
    {
        float error = (target[d] - curPos[d]) * invDist;
        float latVelError = velArr[d] - vParallelMag * error;
        outAction[d] = params.kp * error + params.kd * (-latVelError);
    }

    // for (int d = 0; d < dim; d++)
    // {
    //     float error = target[d] - pos[d];
    //     outAction[d] = params.kp * error + params.kd * (-vel[d]);
    // }

    // repulsion
    for (int other = 0; other < envConfig->nAgents; other++)
    {
        if (other == agent)
            continue;

        float diff[MAX_DIM];
        float dist2 = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            diff[d] = pos[other * dim + d] - curPos[d];
            dist2 += diff[d] * diff[d];
        }

        float dist = std::sqrt(dist2) + 1e-8f;
        if (dist < params.repulsionDistFact * envConfig->minDist)
        {
            float scale = params.repulsionFactor / std::pow(dist, params.repulsionPower);
            for (int d = 0; d < dim; d++)
                outAction[d] -= scale * diff[d];
        }
    }
}
