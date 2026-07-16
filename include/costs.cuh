#pragma once

#include "config.h"
#include <cmath>

// Running cost
HD INLINE float stateCost(const SimState& state, int /* timestep */, const EnvironmentConfig& envConfig, const MPPIConfig& mppiConfig, int trueTheta)
{
    float cost = 0.0f;

    float bd = Env::trackBoundaryDist<false>(state, envConfig, trueTheta);

    float danger = mppiConfig.boundaryThresholdFactor * envConfig.droneRadius;
    if (bd < danger)
        cost += mppiConfig.boundaryCost * powf(bd / envConfig.droneRadius, 2.0f);

    if (bd < envConfig.droneRadius * mppiConfig.collDistFactor)
        cost += mppiConfig.outsideCost;

    if (Env::isOppWinner(state, envConfig))
        cost += mppiConfig.oppWinCost;

    // winner check
    if (Env::isWinner(state, envConfig))
        cost -= mppiConfig.winCost;

    return cost;
}

// Terminal cost
HD INLINE float finalCost(const SimState& state, const EnvironmentConfig& envConfig, const MPPIConfig& mppiConfig, int /* trueTheta */)
{
    float cost = -mppiConfig.finalAdvWeight * Env::getAdvance(state, envConfig);

    if (mppiConfig.finalOppAdvWeight != 0.0f)
        cost += mppiConfig.finalOppAdvWeight * Env::getOppAdvance(state, envConfig);

    return cost;
}

// for PRMPPI

// Running cost
HD INLINE float PRMPPIstateCost(const SimState& state, int /* timestep */, const EnvironmentConfig& envConfig, const PRMPPIConfig& mppiConfig,
                                int trueTheta)
{
    float cost = 0.0f;

    float bd = Env::trackBoundaryDist(state, envConfig, trueTheta);

    float danger = mppiConfig.boundaryThresholdFactor * envConfig.droneRadius;
    if (bd < danger)
        cost += mppiConfig.boundaryCost * powf(bd / envConfig.droneRadius, 2.0f);

    // winner check
    if (Env::isWinner(state, envConfig))
        cost -= mppiConfig.winCost;

    if (Env::isOppWinner(state, envConfig))
        cost += mppiConfig.oppWinCost;

    return cost;
}

// Terminal cost
HD INLINE float PRMPPIfinalCost(const SimState& state, const EnvironmentConfig& envConfig, const PRMPPIConfig& mppiConfig, int /* trueTheta */)
{
    float cost = -mppiConfig.finalAdvWeight * Env::getAdvance(state, envConfig);

    if (mppiConfig.finalOppAdvWeight != 0.0f)
        cost += mppiConfig.finalOppAdvWeight * Env::getOppAdvance(state, envConfig);

    return cost;
}

// Safety cost, < 0 iff state is safe (negative of min of distance to nearest obstacles and to other agents, if any, minus
// minSafeDist)
HD INLINE float PRMPPIsafetyCost(const SimState& state, int /* timestep */, const EnvironmentConfig& envConfig, const PRMPPIConfig& mppiConfig,
                                 int trueTheta)
{
    // if (Env::isOutside(state, envConfig, envConfig.droneRadius * mppiConfig.collDistFactor, trueTheta))
    //     return 1.0f;

    // here, we do not want to lose if opponent wins (we still want to avoid boundaries, and keep off track)
    return envConfig.droneRadius * mppiConfig.collDistFactor - Env::trackBoundaryDist<false>(state, envConfig, trueTheta);
}
