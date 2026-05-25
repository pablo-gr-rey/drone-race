#include "controllers.h"
#include "engine.h"
#include "state.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/distributions/binomial.hpp>

// ═════════════════════════════════════════════════════════════════════
// Dummy
// ═════════════════════════════════════════════════════════════════════
DummyController::DummyController(const EnvironmentConfig& c)
{
    name = "dummy"; envConfig = &c;
}

void DummyController::getControl(int /*agent*/, const float* /*pos*/, const float* /*speed*/,
    const float* /*S*/, const int* /*laps*/, const int* /* currentGates */,
    float* outAction, std::normal_distribution<float>& /* nd */, std::mt19937& /* rng */,
    std::optional<std::vector<float>> /* pastAction */, std::optional<std::vector<float>> /* pastPos */, std::optional<std::vector<float>> /* pastVel */, std::optional<std::vector<float>> /* pastS */)
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

void PIDController::getControl(int agent, const float* pos, const float* speed, const float* S, const int* /* laps */, const int* /* currentGates */, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng,
    std::optional<std::vector<float>> /* pastAction */, std::optional<std::vector<float>> /* pastPos */, std::optional<std::vector<float>> /* pastVel */, std::optional<std::vector<float>> /* pastS */)
{
    if (!engine)
        throw std::runtime_error("PID: engine not set");

    computePIDAction(agent, pos, speed, S, *envConfig, params, engine->trackPoints.data(), outAction);

    for (int d = 0; d < envConfig->dim; d++)
        outAction[d] += nd(rng) * params.actionNoise;
}



void MPPIController::computeEpsilon()
{
    // find the smallest element of K >= number of failures
    uint totFail = failCount[0] + failCount[1];

    auto k_it = std::lower_bound(verifConfig.K.begin(), verifConfig.K.end(), totFail);
    if (k_it == verifConfig.K.end())        // totFail is greater than all elements of K
    {
        std::cout << "Total fail count " << totFail << " is greater than all elements of verifConfig.K. Cannot compute epsilon" << std::endl;
        epsilon = 1.0f;
        return;
    }

    int k = *k_it;
    double beta_eff = (double) verifConfig.beta / (double) verifConfig.K.size();

    auto f = [this, k, beta_eff](double eps)
        {
            // double sum = 0.0;
            // for (int i = 0; i <= k; i++)
            //     sum += boost::math::binomial_coefficient<double>(this->verifConfig.nVerifSamples, i) * std::pow(eps, (double) i) * std::pow(1.0 - eps, (double) (this->verifConfig.nVerifSamples - i));

            // return sum - beta_eff;

            // we can directly compute the probability P(X <= k) if X follows a binomial distribution of parameters (nSamples, eps)
            boost::math::binomial_distribution<double> dist(this->verifConfig.nVerifSamples, eps);

            double p_sum = boost::math::cdf(dist, k);
            return p_sum - beta_eff;
        };

    double result = 1.0;
    try
    {
        boost::math::tools::eps_tolerance<double> tol(32);  // 32-bits precision
        uintmax_t nIter = 100;

        result = boost::math::tools::toms748_solve(f, 0.0, 1.0, tol, nIter).first;
    }
    catch (const std::exception& e)
    {
        std::cout << "Failed to solve root for epsilon (error: " << e.what() << ", aborting verification" << std::endl;
        // in this case, we just use default result for epsilon (1.0)
    }

    epsilon = result;
    std::cout << "For fail count " << totFail << " among " << verifConfig.nVerifSamples << " found epsilon = " << epsilon << " (smallest valid valid k is " << k << ")" << std::endl;
}
