#pragma once

#include "common.h"

namespace EnvStratRace
{
inline constexpr int DIM = 2;
inline constexpr int N_OPP = 2;
inline constexpr int N_TRACK_SAMPLES = 512;

inline constexpr int N_AGENTS = N_OPP + 1;

inline constexpr int ACTION_DIM = DIM;

inline constexpr int N_MODEL_FACTORS = 1;

// CUDA does not like constexpr arrays, so we use constexpr inline functions

HD INLINE constexpr int MODEL_SIZE(int /* k */)
{
    // for several models with different sizes, tests are fine: return (k == 0) ? 2 : 3 or a switch, but here we can just fold it
    return 2;
}

HD INLINE constexpr int BRANCH_SIZE(int /* k */)
{
    // same comment as above
    return 3;
}

inline constexpr int N_TRUE_MODELS = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : MODEL_SIZE(0) * MODEL_SIZE(1);
inline constexpr int N_BRANCH_PLANS = N_MODEL_FACTORS == 1 ? BRANCH_SIZE(0) : BRANCH_SIZE(0) * BRANCH_SIZE(1);
inline constexpr int MAX_MODEL_SIZE = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : (MODEL_SIZE(0) > MODEL_SIZE(1) ? MODEL_SIZE(0) : MODEL_SIZE(1));

inline constexpr int WINDOW_SIZE = 10;    // on each side
inline constexpr float PID_TARGET = 3.0f; // as multiples of dt

struct OppConfig
{
    float s1[N_OPP];
    float s2[N_OPP];
    float s3[N_OPP];
    float speedScale[N_OPP];

    float actionNoise[N_OPP];
};

// Environment configuration
struct EnvironmentConfig
{
    float arenaMin[DIM]; // (dim)
    float arenaMax[DIM]; // (dim)

    float dt;

    bool sendStates;

    float droneRadius;
    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed[N_AGENTS]; // (nAgents)
    float maxAccel[N_AGENTS]; // (nAgents)

    // track
    float trackWidth;
    int nWinLaps;

    float initBelief[N_TRUE_MODELS];

    OppConfig oppConfigs[N_TRUE_MODELS];
    float kP;
    float kV;
    float maxOppLatDistFact; // max reference lateral displacement for opponents is trackWidth - maxOppLatDistFact * droneRadius

    float trackLength;
    float* p_grid;     // (nTrackSamples, dim)
    float* dp_grid;    // (nTrackSamples, dim) derivative of p_grid.
    float* t_grid;     // (nTrackSamples, dim) normalized tangential speed (norm(dp_grid)). normal direction is (-y, x) by convention (CCW)
    float* kappa_grid; // (nTrackSamples) curvature along the track
};

struct SimState
{
    float pos[N_AGENTS * DIM];
    float vel[N_AGENTS * DIM];
    float S[N_AGENTS];
    float latDist[N_AGENTS]; // signed lateral distance to the track

    int laps[N_AGENTS];
};

struct ScratchEnvBuffer
{
    float actions[N_AGENTS * DIM];                    // (nAgents, dim)
    float nomOppActions[N_TRUE_MODELS * N_OPP * DIM]; // (nTrueModels, nOpp, dim)
};
} // namespace EnvStratRace
