#include "controllers.h"
#include "engine.h"
#include "state.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

DummyController::DummyController(const EnvironmentConfig& c)
{
    name = "dummy";
    envConfig = &c;
}

void DummyController::getControl(
    int /*agent*/,
    const SimState& /*state*/,
    float* outAction,
    std::normal_distribution<float>& /*nd*/,
    std::mt19937& /*rng*/,
    std::optional<std::vector<float>> /*pastAction*/,
    std::optional<SimState> /*pastState*/)
{
    for (int d = 0; d < DIM; d++)
        outAction[d] = 0.0f;
}

PIDController::PIDController(const EnvironmentConfig& c, const PIDConfig& p)
    : params(p)
{
    name = "PID";
    envConfig = &c;
}

void PIDController::getControl(
    int agent,
    const SimState& state,
    float* outAction,
    std::normal_distribution<float>& nd,
    std::mt19937& rng,
    std::optional<std::vector<float>> /*pastAction*/,
    std::optional<SimState> /*pastState*/)
{
    if (!engine)
        throw std::runtime_error("PID: engine not set");

    computePIDAction(agent, state, *envConfig, params, engine->trackPoints.data(), outAction);

    for (int d = 0; d < DIM; d++)
        outAction[d] += nd(rng) * params.actionNoise;
}

void MPPIController::computeCertifiedLoss()
{
    uint totFailOld = failCountOld[0] + failCountOld[1];
    uint totFailNew = failCountNew[0] + failCountNew[1];

    int n = verifConfig.nVerifSamples;

    double eps1 = clopperPearsonUpperBound(totFailNew, n, verifConfig.beta / 2.0);
    double eps2 = clopperPearsonLowerBound(totFailOld, n, verifConfig.beta / 2.0);

    certifiedLoss = eps1 - eps2;

    std::cout << std::fixed << std::setprecision(5)
        << "Certified loss: " << certifiedLoss
        << " eps1 " << eps1
        << " eps2 " << eps2
        << " (totFailOld totFailNew n "
        << totFailOld << " " << totFailNew << " " << n << ")\n";
}

void MPPIController::computeEpsilon()
{
    uint totFail = failCount[0] + failCount[1];
    int n = verifConfig.nVerifSamples;

    epsilonPartial = clopperPearsonUpperBound(totFail, n, verifConfig.beta);
    epsilon = epsilonPartial + verifConfig.horizon * verifConfig.maxEps;

    std::cout << std::fixed << std::setprecision(5)
        << "For fail count " << totFail
        << " among " << n
        << " found epsilon = " << epsilon
        << " partial " << epsilonPartial
        << std::endl;
}
