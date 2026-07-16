#pragma once

#include "common.h"

namespace EnvDroneRace
{
inline constexpr int DIM = 2;
inline constexpr int N_AGENTS = 2;
inline constexpr int N_GATES = 2;
inline constexpr int N_TRACK_SAMPLES = 512;
inline constexpr int N_OBSTACLES = 0;

inline constexpr int ACTION_DIM = DIM;

// 0/1 obstacle
inline constexpr int N_ROUND_OBSTACLES = 0;
// inline constexpr int N_ROUND_OBSTACLES = 1;
inline constexpr int N_RACELINES = 2;
inline constexpr int N_MODEL_FACTORS = 1;

// 2 obstacles
// inline constexpr int N_ROUND_OBSTACLES = 2;
// inline constexpr int N_RACELINES = 4;
// inline constexpr int N_MODEL_FACTORS = 2;

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

// PID parameters (used for opponent modelling)
struct PIDConfig
{
    float kp = 10.0f;
    float kd = 5.0f;

    float repulsionFactor = 20.0f;
    float repulsionPower = 2.0f;
    float repulsionDistFact = 10.0f;

    int racelineIndex = 0;

    float actionNoise = 0.0f;
};

// Environment configuration
// this should have the same layout as GateEnvironmentConfig in the Python side
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

    cuda::std::array<float, N_AGENTS> maxSpeed;
    cuda::std::array<float, N_AGENTS> maxAccel;

    // track
    int nWinLaps;
    float targetDistance;

    cuda::std::array<float, N_GATES * DIM> gateCenters; // (nGates * dim)
    cuda::std::array<float, N_GATES * DIM> gateVectors; // (nGates * dim)
    cuda::std::array<float, N_GATES> gateRadius;        // (nGates)

    cuda::std::array<float, N_OBSTACLES * DIM * 2> obstacles;         // (nObstacles * dim * 2): rectangle obstacles, i.e. [xmin, ymin, xmax, ymax]
    cuda::std::array<float, N_ROUND_OBSTACLES * DIM> roundObsCenters; // (nRoundObstacles * dim): center of obstacles
    cuda::std::array<float, N_ROUND_OBSTACLES> roundObsRadius;        // (nRoundObstacles): radius of obstacles

    float* trackPoints = nullptr; // (nTrackSamples * nRacelines, dim)  // this pointer is different on host & device!

    // TODO: move elsewhere (not useful in kernels)
    cuda::std::array<float, N_TRUE_MODELS> initBelief;

    cuda::std::array<PIDConfig, N_TRUE_MODELS> oppPid;
    int iMppi;
};

struct SimState
{
    cuda::std::array<float, N_AGENTS * DIM> pos;
    cuda::std::array<float, N_AGENTS * DIM> vel;
    cuda::std::array<float, N_AGENTS * N_RACELINES> S;

    cuda::std::array<int, N_AGENTS> laps;
    cuda::std::array<int, N_AGENTS> gates;
};

struct ScratchEnvBuffer
{
    cuda::std::array<float, N_AGENTS * DIM> actions;
    cuda::std::array<float, N_TRUE_MODELS * DIM> nomPidActions;
};
} // namespace EnvDroneRace
