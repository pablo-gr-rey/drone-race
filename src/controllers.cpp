#include "controllers.h"
#include "engine.h"
#include "state.h"
#include "dynamics.h"
#include <cmath>
#include <algorithm>

// ═════════════════════════════════════════════════════════════════════
// Dummy
// ═════════════════════════════════════════════════════════════════════
DummyController::DummyController(const EnvironmentConfig& c)
{
    name = "dummy"; envConfig = &c;
}

void DummyController::getControl(int /*agent*/, const float* /*phys*/,
    const float* /*S*/, const float* /*laps*/,
    float* outAction)
{
    for (int d = 0; d < envConfig->dim; d++) outAction[d] = 0.0f;
}

// ═════════════════════════════════════════════════════════════════════
// PID
// ═════════════════════════════════════════════════════════════════════
PIDController::PIDController(const EnvironmentConfig& c, const PIDConfig& p) : params(p)
{
    name = "PID";
    envConfig = &c;
}

void PIDController::getControl(int agent, const float* phys, const float* S, const float*, float* outAction)       // unused parameter is laps
{
    if (!engine) throw std::runtime_error("PID: engine not set");

    int dim = envConfig->dim;

    std::vector<float> target = engine->getTarget(agent, S, params.racelineIndex);

    float pos[MAX_DIM], vel[MAX_DIM];
    for (int d = 0; d < dim; d++)
    {
        pos[d] = getPos(phys, agent, d, dim);
        vel[d] = getVel(phys, agent, d, dim);
    }

    for (int d = 0; d < dim; d++)
    {
        float error = target[d] - pos[d];
        outAction[d] = params.kp * error + params.kd * (-vel[d]);
    }

    // repulsion
    for (int other = 0; other < envConfig->nAgents; other++)
    {
        if (other == agent) continue;
        float diff[MAX_DIM]; float dist2 = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            diff[d] = getPos(phys, other, d, dim) - pos[d];
            dist2 += diff[d] * diff[d];
        }
        float dist = std::sqrt(dist2) + 1e-8f;
        if (dist < params.repulsionDistFact * envConfig->minDist)
        {
            float scale = params.repulsionFactor
                / std::pow(dist, params.repulsionPower + 1.0f);
            for (int d = 0; d < dim; d++)
                outAction[d] -= scale * diff[d];
        }
    }
}
