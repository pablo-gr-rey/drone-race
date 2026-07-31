#pragma once

#include <math.h>

#include "env_dronerace_defs.h"
#include "protocol.h"
#include "state.h"

namespace EnvDroneRace
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig);

void freeDeviceConfig(EnvironmentConfig& d_config);

void pushSimState(Writer& writer, const SimState& state);

void pushAddInfo(Writer& writer, const SimState& state, const SimState& prevState, const EnvironmentConfig& envConfig, int trueTheta);

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds);

void printState(const SimState& state);

SimState unpackSimState(const void* buf, size_t len, const EnvironmentConfig& envConfig);

PIDConfig unpackPIDConfig(Reader& reader);

SimState unpackSimStateRaw(const void* buf, size_t len, const EnvironmentConfig& envConfig, const SimState& prevState);

// return (seed, trueTheta)
std::tuple<EnvironmentConfig, int, int> unpackEnvConfig(const void* buf, size_t len);

// Euclidean distance between two agents (positions only)
HD INLINE float agentDist(const float* __restrict__ pos, int a, int b)
{
    float s = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float dx = pos[a * DIM + d] - pos[b * DIM + d];
        s += dx * dx;
    }

    return sqrtf(s);
}

// Speed (L2 norm of velocity)
HD INLINE float agentSpeed(const float* __restrict__ speed, int agent)
{
    float s = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float v = speed[agent * DIM + d];
        s += v * v;
    }
    return sqrtf(s);
}

// Linear interpolation of pre-sampled centerline
HD INLINE void sampleCenterline(const float* __restrict__ trackPoints, float s, float* __restrict__ out)
{
    s = s - floorf(s); // wrap to [0,1)
    float idx_f = s * N_TRACK_SAMPLES;
    int idx0 = (int)idx_f;
    int idx1 = (idx0 + 1) % N_TRACK_SAMPLES;
    float t = idx_f - idx0;
    for (int d = 0; d < DIM; d++)
        out[d] = (1.0f - t) * trackPoints[idx0 * DIM + d] + t * trackPoints[idx1 * DIM + d];
}

// Project position onto sampled track
// Returns best s in [0,1]; writes distance into bestDist
// closestOut may be nullptr
HD INLINE float projectOnTrack(const float* __restrict__ trackPoints, const float* __restrict__ pos, float* __restrict__ closestOut, float& bestDist)
{
    float bestS = 0.0f;
    float bestD2 = __FLT_MAX__;
    float sStep = 1.0f / N_TRACK_SAMPLES;
    for (int i = 0; i < N_TRACK_SAMPLES; i++)
    {
        float d2 = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = trackPoints[i * DIM + d] - pos[d];
            d2 += dx * dx;
        }
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestS = i * sStep;
            if (closestOut)
                for (int d = 0; d < DIM; d++)
                    closestOut[d] = trackPoints[i * DIM + d];
        }
    }
    bestDist = sqrtf(bestD2);
    return bestS;
}

// util function to return the distance from pos to the given trackPoints
HD INLINE float sqDistToSample(const float* __restrict__ trackPoints, int iTrack, const float* __restrict__ pos)
{
    float sqdist = 0.f;
    for (int d = 0; d < DIM; d++)
        sqdist += (trackPoints[iTrack * DIM + d] - pos[d]) * (trackPoints[iTrack * DIM + d] - pos[d]);
    return sqdist;
}

// util function to keep going in one direction until local maximum
HD INLINE int findMinAlongDirection(const float* __restrict__ trackPoints, int iTrack, const float* __restrict__ pos, int delta, float baseSqDist,
                                    float& bestDist)
{
    // greedy search + margin (should work if the track is not too weird)
    // TODO: maybe fixed window size is better (and more compiler friendly)
    const int margin = 10;

    int bestI = iTrack;
    bestDist = baseSqDist;

    int currentI = (iTrack + delta + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
    float currentDist = sqDistToSample(trackPoints, currentI, pos);

    while (currentDist < bestDist)
    {
        bestI = currentI;
        bestDist = currentDist;

        currentI = (currentI + delta + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
        currentDist = sqDistToSample(trackPoints, currentI, pos);
    }

    // additional margin
    for (int i = 0; i < margin; i++)
    {
        currentDist = sqDistToSample(trackPoints, currentI, pos);
        if (currentDist < bestDist)
        {
            bestI = currentI;
            bestDist = currentDist;
        }
        currentI = (currentI + delta + N_TRACK_SAMPLES) % N_TRACK_SAMPLES;
    }

    return bestI;
}

// return the closest S, and writes the closest track point in closestOut and its distance in bestDist
HD INLINE float fastProjectOnTrack(const float* __restrict__ trackPoints, const float* __restrict__ pos, float* __restrict__ closestOut,
                                   float& bestDist,
                                   float prevS = -1.0f // negative means unknown -> full scan
)
{
#ifdef CHECK_PROJECTION
    float testS, testDist;
    float testClosestOut[MAX_DIM];
    testS = projectOnTrack(trackPoints, pos, testClosestOut, testDist);
#endif

    if (prevS < 0)
        return projectOnTrack(trackPoints, pos, closestOut, bestDist);

    int baseI = (int)(prevS * N_TRACK_SAMPLES + 0.5f) % N_TRACK_SAMPLES;
    float baseDist = sqDistToSample(trackPoints, baseI, pos);

    // find best forwards and backwards distances
    float bestFDist, bestBDist;
    int bestBI = findMinAlongDirection(trackPoints, baseI, pos, -1, baseDist, bestBDist);
    int bestFI = findMinAlongDirection(trackPoints, baseI, pos, +1, baseDist, bestFDist);

    float bestS;

    if (bestFDist < bestBDist)
    {
        bestDist = sqrtf(bestFDist);
        if (closestOut)
            for (int d = 0; d < DIM; d++)
                closestOut[d] = trackPoints[bestFI * DIM + d];

        bestS = ((float)bestFI) / N_TRACK_SAMPLES;
    }
    else
    {
        bestDist = sqrtf(bestBDist);
        if (closestOut)
            for (int d = 0; d < DIM; d++)
                closestOut[d] = trackPoints[bestBI * DIM + d];

        bestS = ((float)bestBI) / N_TRACK_SAMPLES;
    }

#ifdef CHECK_PROJECTION
    // bool wrong = fabs(bestS - testS) > 1e-5 || fabsf(bestDist - testDist) > 1e-5;
    // for (int d = 0; d < DIM && closestOut; d++)
    //     wrong = wrong || abs(closestOut[d] - testClosestOut[d] > 1e-5);

    bool wrong = fabs(bestDist - testDist) > 1e-5f;

    if (wrong)
    {
        printf("WRONG PROJECTION for pos ");
        for (int d = 0; d < DIM; d++)
            printf("%f ", pos[d]);
        printf(" prevS %f\n: computed bestS %f\tbestDist %f\tclosestOut ", prevS, bestS, bestDist);
        for (int d = 0; d < DIM && closestOut; d++)
            printf("%f ", closestOut[d]);
        printf("\n: expected bestS %f\tbestDist %f\tclosestOut ", testS, testDist);
        for (int d = 0; d < DIM; d++)
            printf("%f ", testClosestOut[d]);
        printf("\n");
    }
#endif

    return bestS;
}

// Update belief if oppAction was observed, nomAction is the nominal action for each model (nTrueModels * dim)
HD INLINE void updateBelief(float* __restrict__ belief, const float* __restrict__ oppAction, const float* __restrict__ nomAction,
                            const PIDConfig* __restrict__ params, float maxPIDaccel)
{
    float sum = 0.0f;
    const float sigmaEnv = 0.2f; // account for clamping + various imperfections

    // opponent is following a normal distribution around nomAction, with given stddev
    for (int theta = 0; theta < N_TRUE_MODELS; theta++)
    {
        // compute sq value of nominal action
        float sqAccel = 0.f;
        for (int d = 0; d < DIM; d++)
            sqAccel += nomAction[theta * DIM + d] * nomAction[theta * DIM + d];

        float scale = 1.0f;
        if (sqAccel > maxPIDaccel * maxPIDaccel)
            scale = maxPIDaccel / sqrtf(sqAccel);

        float sqDist = 0.f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = oppAction[d] - nomAction[theta * DIM + d] * scale;
            sqDist += dx * dx;
        }

        float sigma = sqrtf(sigmaEnv * sigmaEnv + params[theta].actionNoise * params[theta].actionNoise);
        // d-dimensional normal law with diagonal sigma matrix (sigma^2, ...)

        // TODO: prob better to use other pow since dimension is integer & known (maybe even more efficient if dimension is known
        // at compile time)
        belief[theta] *= expf(-0.5f * sqDist / (sigma * sigma)) / (powf(2.0f * M_PIf32, (float)DIM / 2.0f) * powf(sigma, (float)DIM));
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
// radius. return -1.0f if other agent won
template <bool loseOnOppWin = true> HD INLINE float trackBoundaryDist(const SimState& state, const EnvironmentConfig& envConfig, int /* trueTheta */)
{
    if constexpr (loseOnOppWin)
        if (state.laps[1 - envConfig.iMppi] >= envConfig.nWinLaps)
            return -1.0f;

    const float* __restrict__ curPos = state.pos.data() + envConfig.iMppi * DIM;

    float minDist = INFINITY;
    for (int d = 0; d < DIM; d++)
        minDist = fminf(minDist, fminf(curPos[d] - envConfig.arenaMin[d], envConfig.arenaMax[d] - curPos[d]));

    for (int iObs = 0; iObs < N_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dist =
                fmaxf(0.0f, fmaxf(envConfig.obstacles[iObs * 2 * DIM + d] - curPos[d], curPos[d] - envConfig.obstacles[(iObs * 2 + 1) * DIM + d]));
            sqDist += dist * dist;
        }
        minDist = fminf(minDist, sqrtf(sqDist));
    }

    for (int iObs = 0; iObs < N_ROUND_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = curPos[d] - envConfig.roundObsCenters[iObs * DIM + d];
            sqDist += dx * dx;
        }
        minDist = fminf(minDist, sqrtf(sqDist) - envConfig.roundObsRadius[iObs]);
    }

    // distance to the other agent
    float sqDist = 0.0f;
    for (int d = 0; d < DIM; d++)
    {
        float dx = state.pos[d] - state.pos[d + DIM];
        sqDist += dx * dx;
    }

    minDist = fminf(minDist, sqrtf(sqDist) - envConfig.droneRadius);

    return minDist;
}

template <bool loseOnOppWin = true> HD INLINE bool isOutside(const SimState& state, const EnvironmentConfig& envConfig, float radius, int trueTheta)
// radius should be e.g. minDist/2 in the actual dynamics and (minDist * factor) / 2 for MPPI
{
    return trackBoundaryDist<loseOnOppWin>(state, envConfig, trueTheta) < radius;
}

HD INLINE bool isWinner(const SimState& state, const EnvironmentConfig& envConfig)
{
    // did we complete enough laps?

    if (state.laps[envConfig.iMppi] >= envConfig.nWinLaps)
        return true;

    // opponent outside?

    const float* __restrict__ curPos = state.pos.data() + (1 - envConfig.iMppi) * DIM;
    float radius = envConfig.droneRadius;

    for (int d = 0; d < DIM; d++)
        if (curPos[d] < envConfig.arenaMin[d] + radius || curPos[d] > envConfig.arenaMax[d] + radius)
            return true;

    for (int iObs = 0; iObs < N_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dist =
                fmaxf(0.0f, fmaxf(envConfig.obstacles[iObs * 2 * DIM + d] - curPos[d], curPos[d] - envConfig.obstacles[(iObs * 2 + 1) * DIM + d]));
            sqDist += dist * dist;
        }

        if (sqDist < radius * radius)
            return true;
    }

    for (int iObs = 0; iObs < N_ROUND_OBSTACLES; iObs++)
    {
        float sqDist = 0.0f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = curPos[d] - envConfig.roundObsCenters[iObs * DIM + d];
            sqDist += dx * dx;
        }

        if (sqDist < (radius + envConfig.roundObsRadius[iObs]) * (radius + envConfig.roundObsRadius[iObs]))
            return true;
    }

    return false;
}

HD INLINE bool isOppWinner(const SimState& state, const EnvironmentConfig& envConfig)
{
    for (int iAgent = 0; iAgent < N_AGENTS; iAgent++)
        if (iAgent != envConfig.iMppi && state.laps[iAgent] >= envConfig.nWinLaps)
            return true;

    return false;
}

// Advance = currentGate + nGates * laps + scale * (1 - normalizedDistToGate) (it is much better to pass through a gate than to
// just be close to it) (this is a rough measure, it doesn't include speed for example) (expect pos to be of size d, ie. pos[0]
// should be position of actual agent) 'pos' points to the agent's contiguous position array of length DIM
HD INLINE float getAnyAdvance(const SimState& state, const EnvironmentConfig& envConfig, int agent)
{
    float sqGateDist = 0.0f;     // sq dist between pos and next gate
    float sqConsGateDist = 0.0f; // sq dist between current gate and next gate
    int nextGate = (state.gates[agent] + 1) % N_GATES;

    float scale = 0.8f; // 1.0f means reward is continuous when going through a gate; 0.5f for example means that reward will be
                        // between 0.0 and 0.5 before the first gate, 1 and 1.5 between 1st and 2nd, etc

    const float* __restrict__ prevGateCenter = envConfig.gateCenters.data() + state.gates[agent] * DIM;
    const float* __restrict__ nextGateCenter = envConfig.gateCenters.data() + nextGate * DIM;
    const float* __restrict__ pos = state.pos.data() + agent * DIM;

    for (int d = 0; d < DIM; d++)
    {
        sqGateDist += (nextGateCenter[d] - pos[d]) * (nextGateCenter[d] - pos[d]);
        sqConsGateDist += (nextGateCenter[d] - prevGateCenter[d]) * (nextGateCenter[d] - prevGateCenter[d]);
    }

    return state.gates[agent] + N_GATES * state.laps[agent] + scale * (1.0f - sqrtf(sqGateDist / sqConsGateDist));
}

HD INLINE float getAdvance(const SimState& state, const EnvironmentConfig& envConfig)
{
    return getAnyAdvance(state, envConfig, envConfig.iMppi);
}

HD INLINE float getOppAdvance(const SimState& state, const EnvironmentConfig& envConfig)
{
    return getAnyAdvance(state, envConfig, 1 - envConfig.iMppi);
}

HD INLINE float getMaxAccel(const EnvironmentConfig& envConfig)
{
    return envConfig.maxAccel[envConfig.iMppi];
}

// if addMargin, use additional margin (we restrict to passing within gateRadius * margin of the gate center)
template <bool addMargin>
HD INLINE void updateGates(const EnvironmentConfig& envConfig, const float* __restrict__ pos, const float* __restrict__ prevPos,
                           float* __restrict__ currentS, int* __restrict__ currentGates, int* __restrict__ nLaps, float margin = 0.0f)
{
    for (int iAgent = 0; iAgent < N_AGENTS; iAgent++)
    {
        float dist;

        if (currentS[iAgent * N_RACELINES] >= 0.0f)
            for (int iRaceline = 0; iRaceline < N_RACELINES; iRaceline++)
            {
                float s = fastProjectOnTrack(envConfig.trackPoints + iRaceline * N_TRACK_SAMPLES * DIM, pos + iAgent * DIM, nullptr, dist,
                                             currentS[iAgent * N_RACELINES + iRaceline]);
                currentS[iAgent * N_RACELINES + iRaceline] = s;
            }

        // check if we passed through next gate: compute lambda = dot(vec, center - x_t) / dot(vec, x_{t+1} - x_t)
        int nextGate = (currentGates[iAgent] + 1) % N_GATES;
        float num = 0.f, denom = 0.f;
        for (int d = 0; d < DIM; d++)
        {
            num += envConfig.gateVectors[nextGate * DIM + d] * (envConfig.gateCenters[nextGate * DIM + d] - prevPos[iAgent * DIM + d]);
            denom += envConfig.gateVectors[nextGate * DIM + d] * (pos[iAgent * DIM + d] - prevPos[iAgent * DIM + d]);
        }

        // direction is inside the gate plan: cannot cross
        if (fabsf(denom) < 1e-10f)
            continue;

        float lambda = num / denom;
        // we cross if 0 <= lambda <= 1 and if the projection of the segment (x_t, x_t+1) on the gate plan (ie. (1 - lambda) * x_t
        // + lambda * x_t+1) is at distance <= radius from the center if we want to make sure we cross the gate in the right
        // direction, we have to check num >= 0 (<=> denom > 0). Here, we allows passing through in both directions

        if (lambda < 0.f || lambda > 1.f)
            continue;

        float sqDist = 0.f;
        for (int d = 0; d < DIM; d++)
        {
            float dx = (1.f - lambda) * prevPos[iAgent * DIM + d] + lambda * pos[iAgent * DIM + d] - envConfig.gateCenters[nextGate * DIM + d];
            sqDist += dx * dx;
        }

        float rad = envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate];
        if constexpr (addMargin)
            rad *= margin * margin;

        if (sqDist <= rad)
        {
            currentGates[iAgent]++;
            if (currentGates[iAgent] == N_GATES)
            {
                currentGates[iAgent] = 0;
                nLaps[iAgent]++;
            }
        }
    }
}

// Control function.Does not add noise, since this is different on CPU and GPU.
HD INLINE void computePIDAction(int agent, const SimState& state, const EnvironmentConfig& envConfig, const PIDConfig& pid, float* outAction)
{
    float target[DIM] = {};

    sampleCenterline(envConfig.trackPoints + N_TRACK_SAMPLES * pid.racelineIndex * DIM,
                     state.S[agent * N_RACELINES + pid.racelineIndex] + envConfig.targetDistance, target);

    const float* curPos = state.pos.data() + agent * DIM;
    const float* curVel = state.vel.data() + agent * DIM;

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
    if (pid.repulsionFactor != 0.0f)
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
            if (dist < pid.repulsionDistFact * envConfig.droneRadius * 2.0f)
            {
                float scale = pid.repulsionFactor / powf(dist * 0.5f / envConfig.droneRadius, pid.repulsionPower + 1.0f);
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

// compute opp. nominal actions, PID noise + env noise (only if applyNoise is True), env dynamics, belief update (only if
// shouldUpdateBelief) and branch update (only if considerBranching is true; if we become specialized, set corresponding branching
// time to t+1)
template <bool shouldUpdateBelief, bool considerBranching, bool addMargin, bool loseOnOppWin = true, bool applyEnvNoise = true, typename RNG>
HD INLINE TerminalType environmentStep(int t, int trueTheta, const EnvironmentConfig& envConfig,
                                       const float* __restrict__ egoAction, // (dim)
                                       bool applyNoise, // TODO: this could be a template (but probably doesn't matter if we're inlined anyway)
                                       SimState& state,
                                       BranchState& branchState, // belief is updated in-place if shouldUpdateBelief. branchingTime and branchUsed are
                                                                 // updated if considerBranching is true
                                       bool& branched, // set to true if we branched at this step. must be previously initialized to false. unused if
                                                       // considerBranching is false
                                       float minConfidence, // for branching. unused if considerBranching is false
                                       ScratchEnvBuffer& scratch, RNG& rng, float gateMargin = 0.0f)
{
    // 1. Copy current pos (for gate update)
    float prevPos[N_AGENTS * DIM];
    for (int i = 0; i < N_AGENTS * DIM; i++)
        prevPos[i] = state.pos[i];

    // 2. Build actions and nominal PID actions (for belief update)
    // TODO: with 2 agents, this should be optimized
    for (int a = 0; a < N_AGENTS; a++)
    {
        if (a == envConfig.iMppi)
            for (int d = 0; d < DIM; d++)
                scratch.actions[a * DIM + d] = egoAction[d];
        else
        {
            for (int thetaT = 0; thetaT < N_TRUE_MODELS; thetaT++)
                computePIDAction(a, state, envConfig, envConfig.oppPid[thetaT], scratch.nomPidActions.data() + thetaT * DIM);

            for (int d = 0; d < DIM; d++)
            {
                float u = scratch.nomPidActions[trueTheta * DIM + d];
                if (applyNoise && envConfig.oppPid[trueTheta].actionNoise != 0.0f)
                    u += envConfig.oppPid[trueTheta].actionNoise * sampleNormal(rng);

                scratch.actions[a * DIM + d] = u;
            }
        }
    }

    // 3. Clamp actions and add env action noise
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

            if constexpr (applyEnvNoise)
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
            state.vel[a * DIM + d] += envConfig.dt * scratch.actions[a * DIM + d];

    // 6. Cap speed
    for (int a = 0; a < N_AGENTS; a++)
    {
        float spd = agentSpeed(state.vel.data(), a);
        if (spd > envConfig.maxSpeed[a])
        {
            float sc = envConfig.maxSpeed[a] / spd;
            for (int d = 0; d < DIM; d++)
                state.vel[a * DIM + d] *= sc;
        }
    }

    // 7. State noise (pos + speed)
    if constexpr (applyEnvNoise)
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
    updateGates<addMargin>(envConfig, state.pos.data(), prevPos, state.S.data(), state.gates.data(), state.laps.data(), gateMargin);

    int oppAgent = 1 - envConfig.iMppi;

    // 9. Update belief & branching time
    if constexpr (shouldUpdateBelief)
        updateBelief(branchState.belief.data(), scratch.actions.data() + oppAgent * DIM, scratch.nomPidActions.data(), envConfig.oppPid.data(),
                     envConfig.maxAccel[oppAgent]);

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
} // namespace EnvDroneRace
