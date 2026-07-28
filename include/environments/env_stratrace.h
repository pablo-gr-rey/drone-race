#pragma once

#include <math.h>
#include <tuple>
#include <vector>

#include "env_stratrace_defs.h"
#include "protocol.h"
#include "state.h"

namespace EnvStratRace
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig);

void freeDeviceConfig(EnvironmentConfig& d_config);

void pushSimState(Writer& writer, const SimState& state);

void pushAddInfo(Writer& writer, const SimState& state, const SimState& prevState, const EnvironmentConfig& envConfig, int trueTheta);

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds);

void printState(const SimState& state);

SimState unpackSimState(const void* buf, size_t len, const EnvironmentConfig& envConfig);

// to use with ROS. returns new state. prevState should be the previous state, not the engine new step (which already
// contains the laps update, for example)
SimState unpackSimStateRaw(const void* buf, size_t len, const EnvironmentConfig& envConfig, const SimState& prevState);

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
    return dim ? x : -y;
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

    float invNorm = 1.0f / sqrtf(sqNorm(out[0], out[1]));
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

// return the closest S, writes the closest track point in closestOut and the signed distance from pos to it in latDist
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

    float closestX = 0.0f;
    float closestY = 0.0f;

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

            closestX = projx;
            closestY = projy;
        }
    }

    // use correct sign for lat dist (following the same convention: positive if in normal direction, ie. left of tangential vector)
    // we use sign of cross-product between normal (=-tangy, tangx) and (pos - closestOut)
    float tang[2];
    sampleTrackArray<true>(envConfig.t_grid, bestS, tang, envConfig.trackLength);
    float cross = -tang[1] * (pos[0] - closestX) + tang[0] * (pos[1] - closestY);

    latDist = copysignf(sqrtf(bestSegSqDist), cross);

    if constexpr (writeClosest)
    {
        closestOut[0] = closestX;
        closestOut[1] = closestY;
    }

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
template <bool loseOnOppWin = true> HD INLINE float trackBoundaryDist(const SimState& state, const EnvironmentConfig& envConfig, int /* trueTheta */)
{
    if constexpr (loseOnOppWin)
    {
        for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
            if (state.laps[iOpp] >= envConfig.nWinLaps)
                return -1.0f;
    }

    float minSqDist = INFINITY;

    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        minSqDist = fminf(minSqDist, sqNorm(state.pos[2 * iOpp] - state.pos[0], state.pos[2 * iOpp + 1] - state.pos[1]));

    // float latDist = INFINITY;
    // fastProjectOnTrack<false>(envConfig, state.pos, state.S[0], nullptr, latDist);

    return fminf(sqrtf(minSqDist) - envConfig.droneRadius, envConfig.trackWidth - fabsf(state.latDist[0]));
}

template <bool loseOnOppWin = true> HD INLINE bool isOutside(const SimState& state, const EnvironmentConfig& envConfig, float radius, int trueTheta)
// radius should be e.g. minDist/2 in the actual dynamics and (minDist * factor) / 2 for MPPI
{
    return trackBoundaryDist<loseOnOppWin>(state, envConfig, trueTheta) < radius;
}

HD INLINE bool isWinner(const SimState& state, const EnvironmentConfig& envConfig)
{
    // did we complete enough laps?

    if (state.laps[0] >= envConfig.nWinLaps)
        return true;

    // opponent outside?
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        if (fabsf(state.latDist[iOpp]) > envConfig.trackWidth - envConfig.droneRadius)
            return true;

    return false;
}

HD INLINE bool isOppWinner(const SimState& state, const EnvironmentConfig& envConfig)
{
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        if (state.laps[iOpp] >= envConfig.nWinLaps)
            return true;

    return false;
}

// Advance = s / trackLength + laps
HD INLINE float getAnyAdvance(const SimState& state, const EnvironmentConfig& envConfig, int agent)
{
    return state.S[agent] / envConfig.trackLength + state.laps[agent];
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

// TODO: this could take tang_i instead of recomputing it
// return ds/dt for given agent (equal to tangspeed / (1 - curvature * latDist))
HD INLINE float computeSdot(int iAgent, const SimState& state, const EnvironmentConfig& envConfig)
{
    float tang_i[2];
    sampleNormalizedSpeed(envConfig.t_grid, state.S[iAgent], tang_i, envConfig.trackLength);

    // track target point and speed
    float kappa_current;
    sampleTrackArray<false>(envConfig.kappa_grid, state.S[iAgent], &kappa_current, envConfig.trackLength);
    float metric = 1 - kappa_current * state.latDist[iAgent];
    if (fabsf(metric) < 1e-12f)
        metric = copysignf(1e-12f, metric);

    return (tang_i[0] * state.vel[2 * iAgent] + tang_i[1] * state.vel[2 * iAgent + 1]) / metric;
}

// Compute target lateral displacement and target longitudinal speed goal for drone iOpp (1 <= iOpp <= N_OPP)
template <bool printInfo = false>
HD INLINE cuda::std::pair<float, float> computeOppTargetLatDist(int iOpp, const SimState& state, const EnvironmentConfig& envConfig, int trueTheta)
{
    OppConfig config = envConfig.oppConfigs[trueTheta];
    float L = envConfig.trackLength;

    // TODO: why not all opponents?

    // find closest opponent ahead
    int iAhead = 0;
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
    int iBehind = 0;
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

    // float v_i_tang = state.vel[iOpp * 2] * tang_i[0] + state.vel[iOpp * 2 + 1] * tang_i[1];
    float sdot_i = computeSdot(iOpp, state, envConfig);

    float baseSpeed = config.speedScale[iOpp - 1] * envConfig.maxSpeed[iOpp];
    float speedGoal = baseSpeed;

    for (int k = 0; k <= 1; k++)
    {
        int j = k ? iAhead : iBehind;
        float deltaS = k ? deltaSAhead : deltaSBehind;
        float deltaE = state.latDist[j] - state.latDist[iOpp];

        float tang_j[2];
        sampleNormalizedSpeed(envConfig.t_grid, state.S[j], tang_j, envConfig.trackLength);

        // float v_j_tang = state.vel[j * 2] * tang_j[0] + state.vel[j * 2 + 1] * tang_j[1];
        float sdot_j = computeSdot(j, state, envConfig);

        // TODO: hardcoded. we block if sdot_j - sdot_i > -blockTolerance (even if they are going slightly slower, we block in order to avoid being
        // vulnerable to opponents faking low speeds to make us go back to centerline)
        float blockTolerance = envConfig.maxSpeed[j] * 0.3f;

        // bool shouldBlock = (deltaS < 0.0f) && (deltaS > -envConfig.trackLength * 0.25f) && (v_j_tang > v_i_tang);
        bool shouldBlock = (deltaS < 0.0f) && (deltaS > -envConfig.trackLength * 0.25f) && (sdot_j + blockTolerance > sdot_i);

        // we do not consider deltaS: if we are on front and faster, we still need overtaking behavior to make sure we do not crash
        bool shouldOvertake = (sdot_i > sdot_j) && (fabsf(deltaE) < config.s1[iOpp - 1]);

        float factor = expf(-config.s2[iOpp - 1] * deltaS * deltaS);

        float blockTerm = 0.0f, overtakeTerm = 0.0f;

        // blocking term
        if (shouldBlock)
        {
            // blockTerm = deltaE * (1 - expf(-config.s3[iOpp - 1] * (v_j_tang - v_i_tang))) * factor;      // using sdot is more interesting than
            // tangential speed blockTerm = deltaE * (1 - expf(-config.s3[iOpp - 1] * (sdot_j - sdot_i))) * factor;          // since this is our
            // target and not a force, we should simply aim for e_j
            blockTerm = state.latDist[j] * (1 - expf(-config.s3[iOpp - 1] * (sdot_j + blockTolerance - sdot_i))) * factor;
            latDis += blockTerm;

            if constexpr (printInfo)
                printf("\tblocking term: k=%d: j=%d, latDist[j]=%f\ts3 contrib: %f\tfactor: %f\t total %f\n", k, j, state.latDist[j],
                       (1 - expf(-config.s3[iOpp - 1] * (sdot_j + blockTolerance - sdot_i))), factor, blockTerm);
        }

        // overtaking term
        if (shouldOvertake)
        {
            // overtakeTerm = -copysignf(fmaxf(0.0f, config.s1[iOpp - 1] - fabsf(deltaE)), deltaE) * factor;  // same: since this is a target
            // and not a force, we aim for e_j +- s1 at the side
            // since we are in the if, we know that |e_j - e_i| < s1. we aim for e_j +- s1, whichever is closest
            overtakeTerm = (state.latDist[j] - copysignf(config.s1[iOpp - 1], deltaE)) * factor;
            latDis += overtakeTerm;

            float oppLongSpeed = state.vel[2 * j] * tang_j[0] + state.vel[2 * j + 1] * tang_j[1];

            if (deltaS < 3 * envConfig.droneRadius)
                speedGoal = fminf(speedGoal, 0.7f * oppLongSpeed);
            else if (deltaS < 6 * envConfig.droneRadius && config.s1[iOpp - 1] > 1e-6)
            {
                float latFactor = fabsf(deltaE) / config.s1[iOpp - 1]; // 0 if we're close, 1 if we're far enough to overtake
                speedGoal = fminf(speedGoal, latFactor * oppLongSpeed + (1.0f - latFactor) * baseSpeed);
            }
        }

        // if constexpr (printInfo)
        // {
        //     printf("k=%d: j=%d, deltaS=%f\tshouldBlock=%d shouldOvertake=%d: blocking = %f\tovertaking = %f\tsdot_i %f\tsdot_j %f\n", k, j, deltaS,
        //            shouldBlock, shouldOvertake, blockTerm, overtakeTerm, sdot_i, sdot_j);
        //     // printf("%f %f %f %d %f\n", state.latDist[j], (1 - expf(-config.s3[iOpp - 1] * (sdot_j - sdot_i))), factor, iOpp, config.s3[iOpp -
        //     // 1]);
        // }
    }

    float maxLatDis = envConfig.trackWidth - envConfig.droneRadius * envConfig.maxOppLatDistFact;

    latDis = fmaxf(-maxLatDis, fminf(maxLatDis, latDis));

    if constexpr (printInfo)
        printf("Target lat dist for drone %d:\t%f\tspeed goal: %f\t(base speed %f)\n", iOpp, latDis, speedGoal, baseSpeed);

    return {latDis, speedGoal};
}

// Control function, 1 <= iOpp <= N_OPP. does not apply noise
/*
HD INLINE void computeOpponentAction(int iOpp, const SimState& state, const EnvironmentConfig& envConfig, float* outAction, int trueTheta)
{
    auto [latDis, speedGoal] = computeOppTargetLatDist(iOpp, state, envConfig, trueTheta);
    float latDisPrev = state.latDist[iOpp];
    float latDis_dot = (latDis - latDisPrev) / envConfig.dt;

    float L = envConfig.trackLength;

    float tang_i[2];
    sampleNormalizedSpeed(envConfig.t_grid, state.S[iOpp], tang_i, L);

    float s_dot = computeSdot(iOpp, state, envConfig);
    s_dot = fmaxf(s_dot, envConfig.maxSpeed[iOpp] * 0.5f);
    float s_target = state.S[iOpp] + PID_TARGET * envConfig.dt * s_dot;

    float p_target[2];
    float tang_target[2];
    float kappa_target;

    float tang_now[2];

    sampleTrackArray<true>(envConfig.p_grid, s_target, p_target, L);
    sampleNormalizedSpeed(envConfig.t_grid, s_target, tang_target, L);
    sampleTrackArray<false>(envConfig.kappa_grid, s_target, &kappa_target, L);
    sampleNormalizedSpeed(envConfig.t_grid, state.S[iOpp], tang_now, envConfig.trackLength);

    float metric = 1.0f - kappa_target * latDis;
    if (fabsf(metric) < 1e-6f)
        metric = copysignf(1e-6f, metric);
    float kappa_eff = kappa_target / metric;

    // Speed along the offset path (not centerline speed)
    // float speedScale = envConfig.oppConfigs[trueTheta].speedScale[iOpp - 1];
    float maxCurvatureSpeed = sqrtf(envConfig.maxAccel[iOpp] / fmaxf(fabsf(kappa_eff), 1e-12f));
    // float pathSpeed = speedScale * fminf(envConfig.maxSpeed[iOpp], maxCurvatureSpeed);
    float pathSpeed = fminf(speedGoal, maxCurvatureSpeed);

    // Position reference
    float p_ref[2];
    p_ref[0] = p_target[0] + latDis * rotateVec(0, tang_target[0], tang_target[1]);
    p_ref[1] = p_target[1] + latDis * rotateVec(1, tang_target[0], tang_target[1]);

    // Velocity reference
    // Tangential component: pathSpeed along offset path tangent (same direction as t_hat)
    // Lateral component: latDis_dot along normal (to track changing lateral target)
    float v_ref[2];
    v_ref[0] = pathSpeed * tang_target[0] + latDis_dot * rotateVec(0, tang_target[0], tang_target[1]);
    v_ref[1] = pathSpeed * tang_target[1] + latDis_dot * rotateVec(1, tang_target[0], tang_target[1]);

    // Feedforward: centripetal acceleration for the OFFSET path
    float ff[2];
    ff[0] = pathSpeed * pathSpeed * kappa_eff * rotateVec(0, tang_target[0], tang_target[1]);
    ff[1] = pathSpeed * pathSpeed * kappa_eff * rotateVec(1, tang_target[0], tang_target[1]);

    // PD + Feedforward
    for (int d = 0; d < DIM; d++)
        outAction[d] = ff[d] + envConfig.kP * (p_ref[d] - state.pos[iOpp * 2 + d]) + envConfig.kV * (v_ref[d] - state.vel[iOpp * 2 + d]);

    float normS = sqNorm(outAction[0], outAction[1]);
    float scale = 1.0f;
    if (normS > envConfig.maxAccel[iOpp] * envConfig.maxAccel[iOpp])
    {
        scale = envConfig.maxAccel[iOpp] / sqrtf(normS);
        outAction[0] *= scale;
        outAction[1] *= scale;
    }
}
*/

HD INLINE void computeOpponentAction(int iOpp, const SimState& state, const EnvironmentConfig& envConfig, float* outAction, int trueTheta)
{
    auto [latDis_target, speedGoal] = computeOppTargetLatDist(iOpp, state, envConfig, trueTheta);

    float L = envConfig.trackLength;
    float dt = envConfig.dt;
    float a_max = envConfig.maxAccel[iOpp];
    float v_max = envConfig.maxSpeed[iOpp];

    // current state
    float s_cur = state.S[iOpp];
    float e_cur = state.latDist[iOpp];

    // sample track geometry at current position
    float tang_cur[2], norm_cur[2];
    float kappa_cur;
    sampleNormalizedSpeed(envConfig.t_grid, s_cur, tang_cur, L);
    sampleTrackArray<false>(envConfig.kappa_grid, s_cur, &kappa_cur, L);

    norm_cur[0] = rotateVec(0, tang_cur[0], tang_cur[1]);
    norm_cur[1] = rotateVec(1, tang_cur[0], tang_cur[1]);

    float vx = state.vel[iOpp * 2 + 0];
    float vy = state.vel[iOpp * 2 + 1];

    float metric = 1.0f - kappa_cur * e_cur;
    if (fabsf(metric) < 1e-6f)
        metric = copysignf(1e-6f, metric);

    float s_dot = (tang_cur[0] * vx + tang_cur[1] * vy) / metric;
    float e_dot = norm_cur[0] * vx + norm_cur[1] * vy;

    // use curvature slightly ahead to compensate for delay
    float tau_lookahead = 3.0f * dt;
    float s_ahead = s_cur + s_dot * tau_lookahead;
    float kappa_ahead;
    sampleTrackArray<false>(envConfig.kappa_grid, s_ahead, &kappa_ahead, L);

    // effective curvature at offset path
    float metric_ahead = 1.0f - kappa_ahead * e_cur;
    if (fabsf(metric_ahead) < 1e-6f)
        metric_ahead = copysignf(1e-6f, metric_ahead);
    float kappa_eff = kappa_ahead / metric_ahead;

    // lateral control: critically-damped (slightly overdamped) PD + feedforward
    float omega_n = envConfig.kP;
    float xi = envConfig.xi; // damping ratio (slightly overdamped to make sure to avoid overshoot even with delay)

    float delta_e = e_cur - latDis_target;

    // TODO: hardcoded
    if (fabsf(delta_e) < envConfig.droneRadius * envConfig.minLatReactionFactor)
        delta_e = 0.0f;

    float a_lat = -omega_n * omega_n * delta_e - 2.0f * xi * omega_n * e_dot + kappa_eff * s_dot * s_dot;

    float a_lat_clamped = fminf(fabsf(a_lat), a_max * envConfig.maxLatAccBudget);
    a_lat_clamped = copysignf(a_lat_clamped, a_lat);

    // longitudinal control: P on speed with curvature speed limit
    float maxCurvatureSpeed = sqrtf(a_max / fmaxf(fabsf(kappa_eff), 1e-6f));
    float v_target = fminf(speedGoal, fminf(v_max, maxCurvatureSpeed));

    float k_lon = envConfig.kV;
    float a_lon_desired = k_lon * (v_target - s_dot);

    float a_lon_budget = sqrtf(fmaxf(a_max * a_max - a_lat_clamped * a_lat_clamped, 0.0f));

    float a_lon = fminf(fabsf(a_lon_desired), a_lon_budget);
    a_lon = copysignf(a_lon, a_lon_desired);

    outAction[0] = a_lon * tang_cur[0] + a_lat_clamped * norm_cur[0];
    outAction[1] = a_lon * tang_cur[1] + a_lat_clamped * norm_cur[1];

    float normSq = outAction[0] * outAction[0] + outAction[1] * outAction[1];
    if (normSq > a_max * a_max)
    {
        float scale = a_max / sqrtf(normSq);
        outAction[0] *= scale;
        outAction[1] *= scale;
    }
}

// compute opp. nominal actions, opp noise + env noise (only if applyNoise is True), env dynamics, belief update (only if
// shouldUpdateBelief) and branch update (only if considerBranching is true; if we become specialized, set corresponding branching
// time to t+1). does not use trackMargin
template <bool shouldUpdateBelief, bool considerBranching, bool addMargin, bool loseOnOppWin = true, bool applyEnvNoise = true, typename RNG>
HD INLINE TerminalType environmentStep(int t, int trueTheta, const EnvironmentConfig& envConfig,
                                       const float* __restrict__ egoAction, // (dim)
                                       bool applyNoise, // TODO: this could be a template (but probably doesn't matter if we're inlined anyway)
                                       SimState& state,
                                       BranchState& branchState, // belief is updated in-place if shouldUpdateBelief. branchingTime and branchUsed
                                                                 // are updated if considerBranching is true
                                       bool& branched, // set to true if we branched at this step. must be previously initialized to false. unused
                                                       // if considerBranching is false
                                       float minConfidence, // for branching. unused if considerBranching is false
                                       ScratchEnvBuffer& scratch, RNG& rng, float /* trackMargin */ = 0.0f)
{
    // 0. Copy initial S (for laps update)
    cuda::std::array<float, N_AGENTS> prevS = state.S;

#ifdef CHEATING
    // TODO: me when I cheat
    float desLatDist[N_OPP];
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        desLatDist[iOpp - 1] = computeOppTargetLatDist(iOpp, state, envConfig, trueTheta);
#endif

    // 1. Build actions and nominal opponent actions (for belief update)
    for (int d = 0; d < DIM; d++)
        scratch.actions[d] = egoAction[d];

    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
    {
        for (int thetaT = 0; thetaT < N_TRUE_MODELS; thetaT++)
            computeOpponentAction(iOpp, state, envConfig, scratch.nomOppActions.data() + (thetaT * N_OPP + iOpp - 1) * DIM, thetaT);

        for (int d = 0; d < DIM; d++)
        {
            float u = scratch.nomOppActions[(trueTheta * N_OPP + iOpp - 1) * DIM + d];
            if (applyNoise && envConfig.oppConfigs[trueTheta].actionNoise[iOpp - 1] != 0.0f)
                u += envConfig.oppConfigs[trueTheta].actionNoise[iOpp - 1] * sampleNormal(rng);

            scratch.actions[iOpp * DIM + d] = u;
        }
    }

    // 2. Clamp actions and add env action noise
    for (int a = 0; a < N_AGENTS; a++)
    {
        float sqActNorm = 0.0f;
        for (int d = 0; d < DIM; d++)
            sqActNorm += scratch.actions[a * DIM + d] * scratch.actions[a * DIM + d];

        float factor = 1.0f;
        float maxSq = envConfig.maxAccel[a] * envConfig.maxAccel[a];
        if (sqActNorm > maxSq)
            factor = envConfig.maxAccel[a] / sqrtf(sqActNorm);

        for (int d = 0; d < DIM; d++)
        {
            float& v = scratch.actions[a * DIM + d];
            v *= factor;

            if constexpr (applyEnvNoise)
                if (envConfig.actionNoiseLevel != 0.0f)
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

        // float metric = 1.0f - kappa * state.latDist[a];
        // if (fabsf(metric) < 1e-12f)
        //     metric = copysignf(1e-12f, metric);

        // // ds/dt = v_tang / (1 - kappa * latDis)
        // float v_tang = state.vel[2 * a] * tang[0] + state.vel[2 * a + 1] * tang[1];
        float sdot = computeSdot(a, state, envConfig);
        state.S[a] += envConfig.dt * sdot;
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
    if constexpr (applyEnvNoise)
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

    // 8. Update S & latDisp
    for (int a = 0; a < N_AGENTS; a++)
        state.S[a] = fastProjectOnTrack<false>(envConfig, state.pos.data() + a * DIM, state.S[a], nullptr, state.latDist[a]);

#ifdef CHEATING
    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
    {
        state.latDist[iOpp] = desLatDist[iOpp - 1];

        float postrack[2];
        float tang[2];
        sampleTrackArray<true>(envConfig.p_grid, state.S[iOpp], postrack, envConfig.trackLength);
        sampleNormalizedSpeed(envConfig.t_grid, state.S[iOpp], tang, envConfig.trackLength);

        state.pos[2 * iOpp] = postrack[0] - tang[1] * desLatDist[iOpp - 1];
        state.pos[2 * iOpp + 1] = postrack[1] + tang[0] * desLatDist[iOpp - 1];
    }
#endif

    // 9. Update laps
    for (int a = 0; a < N_AGENTS; a++)
    {
        if (state.S[a] < prevS[a] - envConfig.trackLength * 0.5f)
            state.laps[a]++;
        if (state.S[a] > prevS[a] + envConfig.trackLength * 0.5f)
            state.laps[a]--;
    }

    // 10. Update belief & branching time
    if constexpr (shouldUpdateBelief)
        updateBelief(branchState.belief.data(), scratch.actions.data() + DIM, scratch.nomOppActions.data(), envConfig.oppConfigs.data(),
                     envConfig.maxAccel.data());

    if constexpr (considerBranching)
    {
        int newPredTheta[N_MODEL_FACTORS];
        findConfident(branchState.belief.data(), minConfidence, newPredTheta);

        for (int k = 0; k < N_MODEL_FACTORS; k++)
            if (branchState.predTheta[k] == 0 && newPredTheta[k] != 0)
            {
                branchState.predTheta[k] = newPredTheta[k];
                branchState.branchingTime[k] = t + 1;
                branched = true;
            }
    }

    // if constexpr (!addMargin && !shouldUpdateBelief && !considerBranching)
    // {
    //     printf("in envStep: %d %f\n", isOutside(state, envConfig, envConfig.droneRadius, trueTheta), trackBoundaryDist(state, envConfig,
    //     trueTheta));
    // }

    if (isOutside<loseOnOppWin>(state, envConfig, envConfig.droneRadius, trueTheta))
        return TERM_LOSE;

    if (isWinner(state, envConfig))
        return TERM_WIN;

    if constexpr (!loseOnOppWin)
    {
        if (isOutside<true>(state, envConfig, envConfig.droneRadius, trueTheta))
            return TERM_OPP_WIN;
    }

    return TERM_NONE;
}
} // namespace EnvStratRace
