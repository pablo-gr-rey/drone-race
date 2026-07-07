#pragma once

#include <math.h>

#include "env_stratrace_defs.h"
#include "protocol.h"
#include "state.h"

namespace EnvStratRace
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig);

void freeDeviceConfig(EnvironmentConfig& d_config);

void pushSimState(Writer& writer, const SimState& state);

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds);

void printState(const SimState& state);

SimState unpackSimState(const void* buf, size_t len, const EnvironmentConfig& envConfig);

OppConfig unpackOppConfig(Reader& reader);

// return (seed, trueTheta)
std::tuple<EnvironmentConfig, int, int> unpackEnvConfig(const void* buf, size_t len);

HD INLINE float sqNorm(float x, float y)
{
    return x * x + y * y;
}

// return vec rotated on dim (ie. (x, y) -> (-y, x))
HD INLINE float rotateVec(int dim, float x, float y)
{
    return -dim * y + (1 - dim) * x;
}

// return track sample index corresponding to s (wrapped to [0, trackLength], floored)
HD INLINE int getWrappedIndex(float s, float trackLength)
{
    s /= trackLength;
    s = s - floorf(s); // wrap to [0,1)
    return (int)(s * N_TRACK_SAMPLES);
}

// Linear interpolation of array at s wrapped to [0, trackLength]. if fullDim, array has size DIM, otherwise size 1
template <bool fullDim> HD INLINE void sampleTrackArray(const float* __restrict__ arr, float s, float* __restrict__ out, float trackLength)
{
    s /= trackLength;
    s = s - floorf(s); // wrap to [0,1)
    float idx_f = s * N_TRACK_SAMPLES;
    int idx0 = (int)idx_f;
    int idx1 = (idx0 + 1) % N_TRACK_SAMPLES;
    float t = idx_f - idx0;

    if constexpr (fullDim)
        for (int d = 0; d < DIM; d++)
            out[d] = (1.0f - t) * arr[idx0 * DIM + d] + t * arr[idx1 * DIM + d];
    else
        *out = (1.0f - t) * arr[idx0] + t * arr[idx1];
}

HD INLINE void sampleNormalizedSpeed(const float* __restrict__ t_grid, float s, float* __restrict__ out, float trackLength)
{
    sampleTrackArray<true>(t_grid, s, out, trackLength);

    float invNorm = 1.0f / sqNorm(out[0], out[1]);
    out[0] *= invNorm;
    out[1] *= invNorm;
}

// util function to return the distance from pos to the given trackPoints
HD INLINE float sqDistToSample(const float* __restrict__ p_grid, int iTrack, const float* __restrict__ pos)
{
    float sqdist = 0.f;
    for (int d = 0; d < DIM; d++)
        sqdist += (p_grid[iTrack * DIM + d] - pos[d]) * (p_grid[iTrack * DIM + d] - pos[d]);
    return sqdist;
}

// return the closest S, writes the closest track point in closestOut and the distance from pos to it in latDist
template <bool writeClosest>
HD INLINE float fastProjectOnTrack(const EnvironmentConfig& envConfig, const float* __restrict__ pos, float prevS, float* __restrict__ closestOut,
                                   float& latDist)
{
    prevS /= envConfig.trackLength;
    prevS = prevS - floorf(prevS);
    int idxPrev = (int)(prevS * N_TRACK_SAMPLES);

    int bestInd = (idxPrev - WINDOW_SIZE + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
    float bestSqDist = sqDistToSample(envConfig.p_grid, bestInd, pos);

    for (int offset = N_TRACK_SAMPLES - WINDOW_SIZE + 1; offset <= N_TRACK_SAMPLES + WINDOW_SIZE; offset++)
    {
        float sqDist = sqDistToSample(envConfig.p_grid, (idxPrev + offset) % N_TRACK_SAMPLES, pos);
        if (sqDist < bestSqDist)
        {
            bestSqDist = sqDist;
            bestInd = (idxPrev + offset) % N_TRACK_SAMPLES;
        }
    }

    // find out best interpolation between 2 segments

    float bestSegSqDist = INFINITY;
    float bestS = 0.0f;

    for (int seg = -1; seg <= 0; ++seg)
    {
        int a = (bestInd + seg + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
        int b = (a + 1) % N_TRACK_SAMPLES;

        float abx = envConfig.p_grid[2 * b] - envConfig.p_grid[2 * a];
        float aby = envConfig.p_grid[2 * b + 1] - envConfig.p_grid[2 * a + 1];
        float sqABdist = sqNorm(abx, aby);

        float t = 0.0f;
        if (sqABdist > 1e-12)
        {
            t = ((pos[0] - envConfig.p_grid[2 * a]) * abx + (pos[1] - envConfig.p_grid[2 * a + 1]) * aby) / sqABdist;
            t = fmaxf(0.0f, fminf(t, 1.0f));
        }

        float projx = envConfig.p_grid[2 * a] + t * abx;
        float projy = envConfig.p_grid[2 * a + 1] + t * aby;

        float sqDist = sqNorm(pos[0] - projx, pos[1] - projy);
        if (sqDist < bestSegSqDist)
        {
            bestSegSqDist = sqDist;
            bestS = ((float)a + t) / N_TRACK_SAMPLES * envConfig.trackLength;

            if constexpr (writeClosest)
            {
                closestOut[0] = projx;
                closestOut[1] = projy;
            }
        }
    }

    latDist = sqrtf(bestSegSqDist);
    return bestS;
}

// Update belief if oppAction (nOpp * dim) was observed, nomAction is the nominal action for each model (nTrueModels * nOpp * dim), maxOppAccel is
// indexed 1-nOpp included
HD INLINE void updateBelief(float* __restrict__ belief, const float* __restrict__ oppAction, const float* __restrict__ nomAction,
                            const OppConfig* __restrict__ params, const float* __restrict__ maxOppAccel)
{
    float sum = 0.0f;
    const float sigmaEnv = 0.2f; // account for clamping + various imperfections

    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
    {
        float logLikelihood = 0.0f;

        for (int iOpp = 0; iOpp < N_OPP; iOpp++)
        {
            // Nominal action for this model and this opponent
            const float* nom = &nomAction[(theta * N_OPP + iOpp) * DIM];
            const float* obs = &oppAction[iOpp * DIM];

            // Compute squared norm of nominal action
            float sqAccel = 0.0f;
            for (int d = 0; d < DIM; d++)
                sqAccel += nom[d] * nom[d];

            float scale = 1.0f;
            if (sqAccel > maxOppAccel[iOpp + 1] * maxOppAccel[iOpp + 1])
                scale = maxOppAccel[iOpp + 1] / sqrtf(sqAccel);

            // Squared distance between observed and (scaled) nominal action
            float sqDist = 0.0f;
            for (int d = 0; d < DIM; d++)
            {
                float dx = obs[d] - nom[d] * scale;
                sqDist += dx * dx;
            }

            float sqSigma = sigmaEnv * sigmaEnv + params[theta].actionNoise[iOpp] * params[theta].actionNoise[iOpp];

            // Accumulate log-likelihood across opponents
            // log N(x; mu, sigma^2 I) = -0.5 * sqDist / sigma^2 - DIM/2 * log(2*pi) - DIM * log(sigma)
            logLikelihood += -0.5f * sqDist / sqSigma - (float)DIM * 0.5f * logf(2.0f * M_PIf32) - (float)DIM * 0.5f * logf(sqSigma);
        }

        belief[theta] *= expf(logLikelihood);
        sum += belief[theta];
    }

    // normalize
    if (sum < 1e-20f)
    {
        float uniform = 1.0f / N_TRUE_MODELS;
        for (int theta = 0; theta < N_TRUE_MODELS; ++theta)
            belief[theta] = uniform;
    }
    else
    {
        for (int theta = 0; theta < N_TRUE_MODELS; theta++)
            belief[theta] /= sum;
    }
}

// Boundary distance = distance of point to closest boundary or opponent (<= 0 if outside), does not take into account agent's
// radius. return -1.0f if other agent won. assumes state and e_n are correctly updated
HD INLINE float trackBoundaryDist(const SimState& state, const EnvironmentConfig& envConfig, int /* trueTheta */)
{
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        if (state.laps[iOpp] >= envConfig.nWinLaps)
            return -1.0f;

    float minSqDist = INFINITY;

    for (int iOpp = 0; iOpp < N_OPP; iOpp++)
        minSqDist = fminf(minSqDist, sqNorm(state.pos[2 * iOpp] - state.pos[0], state.pos[2 * iOpp + 1] - state.pos[1]));

    // float latDist = INFINITY;
    // fastProjectOnTrack<false>(envConfig, state.pos, state.S[0], nullptr, latDist);

    return fminf(sqrtf(minSqDist), state.latDist[0]);
}

HD INLINE bool isOutside(const SimState& state, const EnvironmentConfig& envConfig, float radius, int trueTheta)
// radius should be e.g. minDist/2 in the actual dynamics and (minDist * factor) / 2 for MPPI
{
    return trackBoundaryDist(state, envConfig, trueTheta) < radius;
}

HD INLINE bool isWinner(const SimState& state, const EnvironmentConfig& envConfig)
{
    // did we complete enough laps?

    if (state.laps[0] >= envConfig.nWinLaps)
        return true;

    // opponent outside?
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        if (fabsf(state.latDist[iOpp]) > envConfig.trackWidth)
            return true;

    return false;
}

// Advance = s + laps * trackLength
HD INLINE float getAnyAdvance(const SimState& state, const EnvironmentConfig& envConfig, int agent)
{
    return state.S[agent] + envConfig.trackLength * state.laps[agent];
}

HD INLINE float getAdvance(const SimState& state, const EnvironmentConfig& envConfig)
{
    return getAnyAdvance(state, envConfig, 0);
}

HD INLINE float getOppAdvance(const SimState& state, const EnvironmentConfig& envConfig)
{
    float maxAdvance = -INFINITY;
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        maxAdvance = fmaxf(maxAdvance, getAnyAdvance(state, envConfig, iOpp));
    return maxAdvance;
}

HD INLINE float getMaxAccel(const EnvironmentConfig& envConfig)
{
    return envConfig.maxAccel[0];
}

// Control function, 1 <= iOpp <= N_OPP. does not apply noise
HD INLINE void computeOpponentAction(int iOpp, const SimState& state, const EnvironmentConfig& envConfig, float* outAction, int trueTheta)
{
    OppConfig config = envConfig.oppConfigs[trueTheta];
    float L = envConfig.trackLength;

    // find closest opponent ahead
    float iAhead = 0;
    float deltaSAhead = state.S[0] - state.S[iOpp];
    if (deltaSAhead < 0.0f)
        deltaSAhead += L; // deltaS (always >= 0) is the distance we would have to travel forwards: s_j - s_i if s_j > s_i, otherwise s_j + L - s_i

    for (int j = 1; j <= N_OPP; j++)
        if (j != iOpp)
        {
            float deltaS = state.S[j] - state.S[iOpp];
            if (deltaS < 0.0f)
                deltaS += L;
            if (deltaS < deltaSAhead)
            {
                deltaSAhead = deltaS;
                iAhead = j;
            }
        }

    // find closest opponent behind
    float iBehind = 0;
    float deltaSBehind = state.S[0] - state.S[iOpp];
    if (deltaSBehind > 0.0f)
        deltaSBehind -= L; // deltaS (always <= 0) is the distance we would have to travel backwards: s_j - s_i if s_j < s_i, otherwise s_j - L - s_i

    for (int j = 1; j <= N_OPP; j++)
        if (j != iOpp)
        {
            float deltaS = state.S[j] - state.S[iOpp];
            if (deltaS > 0.0f)
                deltaS -= L;
            if (deltaS > deltaSBehind)
            {
                deltaSBehind = deltaS;
                iBehind = j;
            }
        }

    // compute desired lateral displacement
    float latDis = 0.0f;

    float tang_i[2];
    sampleNormalizedSpeed(envConfig.t_grid, state.S[iOpp], tang_i, L);

    for (int k = 0; k <= 1; k++)
    {
        int j = k ? iAhead : iBehind;
        float deltaS = k ? deltaSAhead : deltaSBehind;
        float deltaE = state.latDist[j] - state.latDist[iOpp];

        float tang_j[2];
        sampleNormalizedSpeed(envConfig.t_grid, state.S[j], tang_j, L);

        float v_i_tang = state.vel[iOpp * 2] * tang_i[0] + state.vel[iOpp * 2 + 1] * tang_i[1];
        float v_j_tang = state.vel[j * 2] * tang_j[0] + state.vel[j * 2 + 1] * tang_j[1];

        // overtaking term
        latDis -= copysignf(fmaxf(0.0f, (config.s1[iOpp - 1] - fabsf(deltaE)) * expf(-config.s2[iOpp - 1] * deltaS * deltaS)), deltaE);

        // blocking term
        if (v_j_tang > v_i_tang && deltaS < 0.0f && deltaS > -L * 0.25f)
            latDis += deltaE * (1 - expf(-config.s3[iOpp - 1] * (v_j_tang - v_i_tang))) * expf(-config.s2[iOpp - 1] * deltaS * deltaS);
    }

    float maxLatDis = envConfig.trackWidth - envConfig.droneRadius * envConfig.maxOppLatDistFact;
    latDis = fmaxf(-maxLatDis, fminf(maxLatDis, latDis));

    // track target point and speed
    float s_target = state.S[iOpp] + PID_TARGET * sqrtf(sqNorm(tang_i[0], tang_i[1]));

    float p_target[2];
    float tang_target[2];
    float kappa_target;

    sampleTrackArray<true>(envConfig.p_grid, s_target, p_target, L);
    sampleNormalizedSpeed(envConfig.t_grid, s_target, tang_target, L);
    sampleTrackArray<false>(envConfig.kappa_grid, s_target, &kappa_target, L);

    float targetSpeed =
        config.speedScale[iOpp - 1] * fminf(envConfig.maxSpeed[iOpp], sqrtf(envConfig.maxAccel[iOpp] / fmaxf(fabsf(kappa_target), 1e-12f)));

    for (int d = 0; d < DIM; d++)
    {
        outAction[d] = 0.0f;

        // position reference
        float p_ref = p_target[d] + latDis * rotateVec(d, tang_target[0], tang_target[1]);
        outAction[d] += envConfig.kP * (p_ref - state.pos[iOpp * 2 + d]);

        // velocity reference
        float v_ref = tang_target[d] * targetSpeed;
        outAction[d] += envConfig.kV * (v_ref - state.vel[iOpp * 2 + d]);

        // centripetal acceleration
        outAction[d] += targetSpeed * targetSpeed * kappa_target * rotateVec(d, tang_target[0], tang_target[1]);
    }

    // normalize
    float norm = sqNorm(outAction[0], outAction[1]);
    float scale = 1.0f;
    if (norm > envConfig.maxAccel[iOpp] * envConfig.maxAccel[iOpp])
        scale = envConfig.maxAccel[iOpp] / sqrtf(scale);
    outAction[0] /= scale;
    outAction[1] /= scale;
}

// compute opp. nominal actions, opp noise + env noise (only if applyNoise is True), env dynamics, belief update (only if
// shouldUpdateBelief) and branch update (only if considerBranching is true; if we become specialized, set corresponding branching
// time to t+1). does not use trackMargin
template <bool shouldUpdateBelief, bool considerBranching, bool addMargin, typename RNG>
HD INLINE TerminalType environmentStep(int t, int trueTheta, const EnvironmentConfig& envConfig,
                                       const float* __restrict__ egoAction, // (dim)
                                       bool applyNoise, // TODO: this could be a template (but probably doesn't matter if we're inlined anyway)
                                       SimState& state,
                                       BranchState& branchState, // belief is updated in-place if shouldUpdateBelief. branchingTime and branchUsed are
                                                                 // updated if considerBranching is true
                                       bool& branched, // set to true if we branched at this step. must be previously initialized to false. unused if
                                                       // considerBranching is false
                                       float minConfidence, // for branching. unused if considerBranching is false
                                       ScratchEnvBuffer& scratch, RNG& rng, float /* trackMargin */ = 0.0f)
{
    // 1. Build actions and nominal opponent actions (for belief update)
    for (int d = 0; d < DIM; d++)
        scratch.actions[d] = egoAction[d];

    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
    {
        for (int thetaT = 0; thetaT < N_TRUE_MODELS; thetaT++)
            computeOpponentAction(iOpp, state, envConfig, scratch.nomOppActions + (thetaT * N_OPP + iOpp - 1) * DIM, trueTheta);

        for (int d = 0; d < DIM; d++)
        {
            float u = scratch.nomOppActions[trueTheta * DIM + d];
            if (applyNoise && envConfig.oppConfigs[trueTheta].actionNoise[iOpp - 1] != 0.0f)
                u += envConfig.oppConfigs[trueTheta].actionNoise[iOpp - 1] * sampleNormal(rng);

            scratch.actions[iOpp * DIM + d] = u;
        }
    }

    // 2. Clamp actions and add env action noise
    for (int a = 0; a < N_AGENTS; a++)
    {
        float sqNorm = 0.0f;
        for (int d = 0; d < DIM; d++)
            sqNorm += scratch.actions[a * DIM + d] * scratch.actions[a * DIM + d];

        float factor = 1.0f;
        float maxSq = envConfig.maxAccel[a] * envConfig.maxAccel[a];
        if (sqNorm > maxSq)
            factor = envConfig.maxAccel[a] / sqrtf(sqNorm);

        for (int d = 0; d < DIM; d++)
        {
            float& v = scratch.actions[a * DIM + d];
            v *= factor;

            if (envConfig.actionNoiseLevel != 0.0f && applyNoise)
                v += sampleNormal(rng) * envConfig.actionNoiseLevel;
        }
    }

    // 3. Integrate position
    for (int a = 0; a < N_AGENTS; a++)
        for (int d = 0; d < DIM; d++)
            state.pos[a * DIM + d] += envConfig.dt * state.vel[a * DIM + d];

    // 4. Update S (to warm-start projection at next step)
    for (int a = 0; a < N_AGENTS; a++)
    {
        float tang[2];
        sampleNormalizedSpeed(envConfig.t_grid, state.S[a], tang, envConfig.trackLength);

        float kappa;
        sampleTrackArray<false>(envConfig.kappa_grid, state.S[a], &kappa, envConfig.trackLength);

        float metric = 1.0f - kappa * state.latDist[a];
        if (fabsf(metric) < 1e-12f)
            metric = copysignf(1e-12f, metric);

        // ds/dt = v_tang / (1 - kappa * latDis)
        float v_tang = state.vel[2 * a] * tang[0] + state.vel[2 * a + 1] * tang[1];
        state.S[a] += envConfig.dt * v_tang / metric;
    }

    // 5. Integrate speed
    for (int a = 0; a < N_AGENTS; a++)
        for (int d = 0; d < DIM; d++)
            state.vel[a * DIM + d] += envConfig.dt * scratch.actions[a * DIM + d];

    // 6. Cap speed
    for (int a = 0; a < N_AGENTS; a++)
    {
        float sqSpd = sqNorm(state.vel[2 * a], state.vel[2 * a + 1]);
        if (sqSpd > envConfig.maxSpeed[a] * envConfig.maxSpeed[a])
        {
            float sc = envConfig.maxSpeed[a] / sqrtf(sqSpd);
            for (int d = 0; d < DIM; d++)
                state.vel[a * DIM + d] *= sc;
        }
    }

    // 7. State noise (pos + speed)
    if ((envConfig.posNoiseLevel != 0.0f || envConfig.speedNoiseLevel != 0.0f) && applyNoise)
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

    // 7.5. Update S & latDisp
    for (int a = 0; a < N_AGENTS; a++)
        state.S[a] = fastProjectOnTrack<false>(envConfig, state.pos + a * DIM, state.S[a], nullptr, state.latDist[a]);

    // 8. Update belief & branching time
    if constexpr (shouldUpdateBelief)
        updateBelief(branchState.belief, scratch.actions + DIM, scratch.nomOppActions, envConfig.oppConfigs, envConfig.maxAccel);

    if constexpr (considerBranching)
    {
        int newPredTheta[N_MODEL_FACTORS];
        findConfident(branchState.belief, minConfidence, newPredTheta);

        for (int k = 0; k < N_MODEL_FACTORS; k++)
            if (branchState.predTheta[k] == 0 && newPredTheta[k] != 0)
            {
                branchState.predTheta[k] = newPredTheta[k];
                branchState.branchingTime[k] = t + 1;
                branched = true;
            }
    }

    if (isOutside(state, envConfig, envConfig.droneRadius, trueTheta))
        return TERM_LOSE;

    if (isWinner(state, envConfig))
        return TERM_WIN;

    return TERM_NONE;
}
} // namespace EnvStratRace
