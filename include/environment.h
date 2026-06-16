#pragma once

#include <random>

#include "config.h"
#include "state.h"
#include "controllers.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"


// simulate environment step
// TODO: merge it with new function (to fully encapsulate dynamics?)
template <typename RNG>
HD INLINE TerminalType applyEnvironmentDynamics(
    int controlAgent,                   // only relevant for terminal stop reason
    const EnvironmentConfig& envConfig,
    const float* trackPoints,
    SimState& state,                      // modified in-place
    float* actions,                    // (nAgents, dim), modified in-place by clamp/noise
    RNG& rng,                           // should be DeviceRNG or HostRNG
    int gateMarginAgent = -1,           // if one agent should have reduced margin, change this (typically to make sure MPPI passes inside the gate)
    float gateMargin = 0.0f)
{
    float prevPos[N_AGENTS * DIM];
    for (int i = 0; i < N_AGENTS * DIM; i++)
        prevPos[i] = state.pos[i];

    // clamp + env action noise
    for (int a = 0; a < N_AGENTS; a++)
    {
        float sqNorm = 0.0f;
        for (int d = 0; d < DIM; d++)
            sqNorm += actions[a * DIM + d] * actions[a * DIM + d];

        float factor = 1.0f;
        float maxSq = envConfig.maxAccel[a] * envConfig.maxAccel[a];
        if (sqNorm > maxSq)
            factor = envConfig.maxAccel[a] / sqrtf(sqNorm);

        for (int d = 0; d < DIM; d++)
        {
            float& v = actions[a * DIM + d];
            v *= factor;

            if (envConfig.actionNoiseLevel != 0.0f)
                v += sampleNormal(rng) * envConfig.actionNoiseLevel;
        }
    }

    // integrate position
    for (int a = 0; a < N_AGENTS; a++)
        for (int d = 0; d < DIM; d++)
            state.pos[a * DIM + d] += envConfig.dt * state.vel[a * DIM + d];

    // integrate velocity
    for (int a = 0; a < N_AGENTS; a++)
        for (int d = 0; d < DIM; d++)
            state.vel[a * DIM + d] += envConfig.dt * actions[a * DIM + d];

    // cap speed
    for (int a = 0; a < N_AGENTS; a++)
    {
        float spd = agentSpeed(state.vel, a);
        if (spd > envConfig.maxSpeed[a])
        {
            float sc = envConfig.maxSpeed[a] / spd;
            for (int d = 0; d < DIM; d++)
                state.vel[a * DIM + d] *= sc;
        }
    }

    // state noise
    if (envConfig.posNoiseLevel != 0.0f || envConfig.speedNoiseLevel != 0.0f)
    {
        for (int a = 0; a < N_AGENTS; a++)
            for (int d = 0; d < DIM; d++)
            {
                if (envConfig.posNoiseLevel != 0.0f)
                    state.pos[a * DIM + d] += sampleNormal(rng) * envConfig.posNoiseLevel;
                if (envConfig.speedNoiseLevel != 0.0f)
                    state.vel[a * DIM + d] += sampleNormal(rng) * envConfig.speedNoiseLevel;
            }
    }

    updateGates(
        envConfig,
        state.pos,
        prevPos,
        state.S,
        state.gates,
        state.laps,
        trackPoints,
        gateMarginAgent,
        gateMargin);

    // terminal detection
    bool collision = false;
    for (int a1 = 0; a1 < N_AGENTS && !collision; a1++)
        for (int a2 = a1 + 1; a2 < N_AGENTS; a2++)
        {
            float dist2 = 0.0f;
            for (int d = 0; d < DIM; d++)
            {
                float dx = state.pos[a1 * DIM + d] - state.pos[a2 * DIM + d];
                dist2 += dx * dx;
            }
            if (dist2 < envConfig.minDist * envConfig.minDist)
            {
                collision = true;
                break;
            }
        }

    if (collision)
        return TERM_COLLISION;

    if (isOutside(envConfig, state.pos + controlAgent * DIM, envConfig.minDist / 2.0f))
        return TERM_EGO_OUTSIDE;

    int oppAgent = 1 - controlAgent;
    if (N_AGENTS > 1 && isOutside(envConfig, state.pos + oppAgent * DIM, envConfig.minDist / 2.0f))
        return TERM_OPP_OUTSIDE;

    if (state.laps[controlAgent] >= envConfig.nWinLaps)
        return TERM_WIN;

    if (N_AGENTS > 1 && state.laps[oppAgent] >= envConfig.nWinLaps)
        return TERM_OPP_WIN;

    return TERM_NONE;
}

// compute opp. nominal actions, PID noise, env dynamics, belief update and branch update
template <typename RNG>
HD INLINE TerminalType simulateContingentStep(
    int t,
    int controlAgent,
    int trueTheta,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig,
    const float* trackPoints,
    const float* egoAction,          // (dim)
    bool applyPidNoise,
    SimState& state,
    BranchState& branchState,
    float* actions,                  // scratch: (nAgents, dim)
    float* nomPidAction,             // scratch: (nModels, dim)
    RNG& rng,
    int gateMarginAgent = -1,
    float gateMargin = 0.0f)
{
    // build actions
    for (int a = 0; a < N_AGENTS; a++)
    {
        if (a == controlAgent)
            for (int d = 0; d < DIM; d++)
                actions[a * DIM + d] = egoAction[d];
        else
        {
            switch (mppiConfig.oppKind)     // should not be MPPI
            {
            case ControllerKind::CONT_DUMMY:
                for (int d = 0; d < DIM; d++)
                    actions[a * DIM + d] = 0.0f;
                break;

            case ControllerKind::CONT_PID:
                for (int thetaT = 0; thetaT < N_MODELS; thetaT++)
                    computePIDAction(
                        a,
                        state,
                        envConfig,
                        mppiConfig.oppPid[thetaT],
                        trackPoints,
                        nomPidAction + thetaT * DIM);

                for (int d = 0; d < DIM; d++)
                {
                    float u = nomPidAction[trueTheta * DIM + d];
                    if (applyPidNoise && mppiConfig.oppPid[trueTheta].actionNoise != 0.0f)
                        u += mppiConfig.oppPid[trueTheta].actionNoise * sampleNormal(rng);

                    actions[a * DIM + d] = u;
                }
                break;

            case ControllerKind::CONT_MPPI: {}
            }
        }
    }

    TerminalType term = applyEnvironmentDynamics(
        controlAgent,
        envConfig,
        trackPoints,
        state,
        actions,
        rng,
        gateMarginAgent,
        gateMargin);

    // update belief from observed opponent action
    int oppAgent = 1 - controlAgent;
    updateBelief(
        branchState.belief,
        actions + oppAgent * DIM,
        nomPidAction,
        mppiConfig.oppPid,
        envConfig.maxAccel[oppAgent]);

    if (branchState.predTheta == -1)
    {
        int conf = findConfident(branchState.belief, mppiConfig.minConfidence);
        if (conf != -1)
        {
            branchState.predTheta = conf;
            branchState.branchingTime = t + 1;
        }
    }

    return term;
}
