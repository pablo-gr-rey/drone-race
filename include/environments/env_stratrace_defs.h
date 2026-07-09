#pragma once

#include "common.h"

namespace EnvStratRace
{
inline constexpr int DIM = 2;
inline constexpr int N_OPP = 1;
inline constexpr int N_TRACK_SAMPLES = 512;

inline constexpr int N_AGENTS = N_OPP + 1;

inline constexpr int ACTION_DIM = DIM;

inline constexpr int N_MODEL_FACTORS = 1;

// CUDA does not like constexpr arrays, so we use constexpr inline functions

HD INLINE constexpr int MODEL_SIZE(int /* k */)
{
    // for several models with different sizes, tests are fine: return (k == 0) ? 2 : 3 or a switch, but here we can just fold it
    return 2;
    // return 1;
}

HD INLINE constexpr int BRANCH_SIZE(int /* k */)
{
    // same comment as above
    return 3;
    // return 2;
}

inline constexpr int N_TRUE_MODELS = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : MODEL_SIZE(0) * MODEL_SIZE(1);
inline constexpr int N_BRANCH_PLANS = N_MODEL_FACTORS == 1 ? BRANCH_SIZE(0) : BRANCH_SIZE(0) * BRANCH_SIZE(1);
inline constexpr int MAX_MODEL_SIZE = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : (MODEL_SIZE(0) > MODEL_SIZE(1) ? MODEL_SIZE(0) : MODEL_SIZE(1));

inline constexpr int WINDOW_SIZE = 10;    // on each side
inline constexpr float PID_TARGET = 3.0f; // as multiples of dt, ie. PID will reach for target at expected s in future time PID_TARGET * dt

struct OppConfig
{
    cuda::std::array<float, N_OPP> s1;
    cuda::std::array<float, N_OPP> s2;
    cuda::std::array<float, N_OPP> s3;
    cuda::std::array<float, N_OPP> speedScale;

    cuda::std::array<float, N_OPP> actionNoise;
};

// Environment configuration
struct EnvironmentConfig
{
    cuda::std::array<float, DIM> arenaMin; // (dim)
    cuda::std::array<float, DIM> arenaMax; // (dim)

    float dt;

    bool sendStates;

    float droneRadius;
    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    cuda::std::array<float, N_AGENTS> maxSpeed; // (nAgents)
    cuda::std::array<float, N_AGENTS> maxAccel; // (nAgents)

    // track
    float trackWidth;
    int nWinLaps;

    cuda::std::array<float, N_TRUE_MODELS> initBelief;

    cuda::std::array<OppConfig, N_TRUE_MODELS> oppConfigs;
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
    cuda::std::array<float, N_AGENTS * DIM> pos;
    cuda::std::array<float, N_AGENTS * DIM> vel;
    cuda::std::array<float, N_AGENTS> S;
    cuda::std::array<float, N_AGENTS> latDist; // signed lateral distance to the track

    cuda::std::array<int, N_AGENTS> laps;
};

struct ScratchEnvBuffer
{
    cuda::std::array<float, N_AGENTS * DIM> actions;                    // (nAgents, dim)
    cuda::std::array<float, N_TRUE_MODELS * N_OPP * DIM> nomOppActions; // (nTrueModels, nOpp, dim)
};
} // namespace EnvStratRace
