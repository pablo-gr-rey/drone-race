#pragma once

#include <random>

#include "config.h"
#include "state.h"
#include "controllers.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

// Shared PID control function. Does not add noise, since this is different on CPU and GPU.
HD INLINE void computePIDAction(
    int agent,
    const SimState& state,
    const EnvironmentConfig& envConfig,
    const PIDConfig& pid,
    float* outAction)
{
    float target[DIM] = {};

    sampleCenterline(
        envConfig.trackPoints + N_TRACK_SAMPLES * pid.racelineIndex * DIM,
        state.S[agent * N_RACELINES + pid.racelineIndex] + envConfig.targetDistance,
        target);

    const float* curPos = state.pos + agent * DIM;
    const float* curVel = state.vel + agent * DIM;

    float sqError = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float e = target[d] - curPos[d];
        sqError += e * e;
    }

    float invDist = 1.0f / sqrtf(sqError + 1e-5f);

    float vParallelMag = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float dirD = (target[d] - curPos[d]) * invDist;
        vParallelMag += curVel[d] * dirD;
    }

    for (int d = 0; d < DIM; d++)
    {
        float error = (target[d] - curPos[d]) * invDist;
        float latVelError = curVel[d] - vParallelMag * error;
        outAction[d] = pid.kp * error + pid.kd * (-latVelError);
    }

    // repulsion
    if (pid.repulsionDistFact != 0.0f)
        for (int other = 0; other < N_AGENTS; other++)
        {
            if (other == agent)
                continue;

            float diff[DIM];
            float dist2 = 0.0f;
            for (int d = 0; d < DIM; d++)
            {
                diff[d] = state.pos[other * DIM + d] - curPos[d];
                dist2 += diff[d] * diff[d];
            }

            float dist = sqrtf(dist2) + 1e-8f;
            if (dist < pid.repulsionDistFact * envConfig.minDist)
            {
                float scale = pid.repulsionFactor / powf(dist / envConfig.minDist, pid.repulsionPower + 1.0f);
                for (int d = 0; d < DIM; d++)
                    outAction[d] -= scale * diff[d];
            }
        }

    // normalize
    float sqAccel = 0.0f;
    for (int d = 0; d < DIM; d++)
        sqAccel += outAction[d] * outAction[d];

    if (sqAccel > envConfig.maxAccel[agent] * envConfig.maxAccel[agent])
    {
        float fact = envConfig.maxAccel[agent] / sqrtf(sqAccel);
        for (int d = 0; d < DIM; d++)
            outAction[d] *= fact;
    }
}

// compute opp. nominal actions, PID noise, env dynamics, belief update and branch update. if we become specialized, set corresponding branching time to t+1
template <typename RNG>
HD INLINE TerminalType environmentStep(
    int t,
    int controlAgent,
    int trueTheta,
    const EnvironmentConfig& envConfig,
    const MPPIConfig& mppiConfig,
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
    // 1. Copy current pos (for gate update)
    float prevPos[N_AGENTS * DIM];
    for (int i = 0; i < N_AGENTS * DIM; i++)
        prevPos[i] = state.pos[i];

    // 2. Build actions and nominal PID actions (for belief update)
    // TODO: with 2 agents, this should be optimized
    for (int a = 0; a < N_AGENTS; a++)
    {
        if (a == controlAgent)
            for (int d = 0; d < DIM; d++)
                actions[a * DIM + d] = egoAction[d];
        else
        {
            for (int thetaT = 0; thetaT < N_TRUE_MODELS; thetaT++)
                computePIDAction(
                    a,
                    state,
                    envConfig,
                    envConfig.oppPid[thetaT],
                    nomPidAction + thetaT * DIM);

            for (int d = 0; d < DIM; d++)
            {
                float u = nomPidAction[trueTheta * DIM + d];
                if (applyPidNoise && envConfig.oppPid[trueTheta].actionNoise != 0.0f)
                    u += envConfig.oppPid[trueTheta].actionNoise * sampleNormal(rng);

                actions[a * DIM + d] = u;
            }
        }
    }

    // 3. Clamp actions and add env action noise
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

    // 4. Integrate position
    for (int a = 0; a < N_AGENTS; a++)
        for (int d = 0; d < DIM; d++)
            state.pos[a * DIM + d] += envConfig.dt * state.vel[a * DIM + d];

    // 5. Integrate speed
    for (int a = 0; a < N_AGENTS; a++)
        for (int d = 0; d < DIM; d++)
            state.vel[a * DIM + d] += envConfig.dt * actions[a * DIM + d];

    // 6. Cap speed
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

    // 7. State noise (pos + speed)
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

    // 8. Update gates with prev pos 
    updateGates(
        envConfig,
        state.pos,
        prevPos,
        state.S,
        state.gates,
        state.laps,
        gateMarginAgent,
        gateMargin);

    // 9. Update belief & branching time
    int oppAgent = 1 - controlAgent;
    updateBelief(
        branchState.belief,
        actions + oppAgent * DIM,
        nomPidAction,
        envConfig.oppPid,
        envConfig.maxAccel[oppAgent]);

    int newPredTheta[N_MODEL_FACTORS];
    findConfident(branchState.belief, mppiConfig.minConfidence, newPredTheta);

    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if (branchState.predTheta[k] == 0 && newPredTheta[k] != 0)
        {
            branchState.predTheta[k] = newPredTheta[k];
            branchState.branchingTime[k] = t + 1;
        }

    // 10. Check for collisions, outside, or win
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

    if (N_AGENTS > 1 && isOutside(envConfig, state.pos + oppAgent * DIM, envConfig.minDist / 2.0f))
        return TERM_OPP_OUTSIDE;

    if (state.laps[controlAgent] >= envConfig.nWinLaps)
        return TERM_WIN;

    if (N_AGENTS > 1 && state.laps[oppAgent] >= envConfig.nWinLaps)
        return TERM_OPP_WIN;

    return TERM_NONE;
}
