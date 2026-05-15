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
    float* outAction, std::normal_distribution<float>& /* nd */, std::mt19937& /* rng */)
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

void PIDController::getControl(int agent, const float* pos, const float* speed, const float* S, const int* /* laps */, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng)
{
    if (!engine)
        throw std::runtime_error("PID: engine not set");

    computePIDAction(agent, pos, speed, S, currentGates, *envConfig, params, engine->trackPoints.data(), outAction);

    for (int d = 0; d < envConfig->dim; d++)
        outAction[d] += nd(rng) * params.actionNoise;
}
