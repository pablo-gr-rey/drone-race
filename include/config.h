#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <protocol.h>
#include <variant>
#include <type_traits>

#ifdef __CUDACC__
#define HD __host__ __device__
#define INLINE static __forceinline__ 
#else
#define HD
#define INLINE inline
#endif

constexpr bool USE_SPLINES = false;

// ── compile-time limits ──────────────────────────────────────────────
constexpr int N_AGENTS = 2;
constexpr int DIM = 2;
constexpr int N_GATES = 2;
constexpr int N_TRACK_SAMPLES = 512;
constexpr int N_OBSTACLES = 0;

constexpr int MAX_N_KNOTS = 60;

// 0/1 obstacle
constexpr int N_ROUND_OBSTACLES = 0;
// constexpr int N_ROUND_OBSTACLES = 1;
constexpr int N_RACELINES = 2;
constexpr int N_MODEL_FACTORS = 1;

// 2 obstacles
// constexpr int N_ROUND_OBSTACLES = 2;
// constexpr int N_RACELINES = 4;
// constexpr int N_MODEL_FACTORS = 2;

constexpr float MIN_COEFF_THRESHOLD = 0.01f;        // minimum probability threshold for samples to contribute
constexpr float DECAY = 0.95f;

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

constexpr int N_TRUE_MODELS = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : MODEL_SIZE(0) * MODEL_SIZE(1);
constexpr int N_BRANCH_PLANS = N_MODEL_FACTORS == 1 ? BRANCH_SIZE(0) : BRANCH_SIZE(0) * BRANCH_SIZE(1);
constexpr int MAX_MODEL_SIZE = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : (MODEL_SIZE(0) > MODEL_SIZE(1) ? MODEL_SIZE(0) : MODEL_SIZE(1));

// if defined, fastProjectOnTrack will be compared to projectOnTrack. use this to test that the margin is correct when changing track (it will be much slower, though)
// #define CHECK_PROJECTION

// ── CUDA error helper ────────────────────────────────────────────────
#ifdef DEBUG
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess)                                            \
            throw std::runtime_error(std::string("CUDA error at ")         \
                + __FILE__ + ":" + std::to_string(__LINE__) + " - "        \
                + cudaGetErrorString(err));                                \
    } while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

enum CONTROLLER_KIND : int32_t { CONT_MPPI, CONT_PRMPPI };

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

// MPPI configuration
struct MPPIConfig
{
    int nSamples = 100;

    int nTimesteps = 20;

    // spline configuration
    int nKnots;     // M
    int knots[MAX_N_KNOTS];       // tau_i for 0 <= i < M

    float invTemperature = 10.0f;

    float samplingNoise = 0.1f;
    float gateTraversalMargin = 0.95f;

    float collDistFactor = 1.0f;

    // running costs
    float oppDistWeight = 1.0f;
    float oppDistPower = 2.0f;
    float oppDistThresholdFactor = 3.0f;
    float boundaryCost = 10.0f;
    float boundaryThresholdFactor = 1.5f;
    float outsideCost = 1000.0f;
    float oppOutsideCost = 100.0f;
    float collisionCost = 10000.0f;
    float winCost = 1000.0f;

    // terminal costs
    float finalAdvWeight = 10.0f;
    float finalOppAdvWeight = 5.0f;

    float minConfidence = 0.9f;

    int nVerifSamples = 10000;
    float beta = 1e-6f;
    int verifHorizon = 40;

    float maxVerifEps = 0.01;

    void unpackHeader(Reader& reader);
};

// PRMPPI configuration
struct PRMPPIConfig
{
    int nSamples = 100;

    int nTimesteps = 20;

    float invTemperature = 10.0f;

    float samplingNoise = 0.1f;
    float gateTraversalMargin = 0.95f;

    float collDistFactor = 1.0f;

    // running costs
    float oppDistWeight = 1.0f;
    float oppDistPower = 2.0f;
    float oppDistThresholdFactor = 3.0f;
    float boundaryCost = 10.0f;
    float boundaryThresholdFactor = 1.5f;
    float oppOutsideCost = 100.0f;
    float winCost = 1000.0f;

    // terminal costs
    float finalAdvWeight = 10.0f;
    float finalOppAdvWeight = 5.0f;

    // safety cost
    float safetyWeight = 10000.0f;
    float minSafeDist = 0.0f;      // safety cost is lessened by this amount (i.e. we must be at distance at least minSafeDist from obstacles & opponents)

    // safety assurance (delta), and number of model samples P (=ceil((1-delta)/delta))
    float delta = 0.1f;
    float P = 10;

    void unpackHeader(Reader& reader);
};

using AnyControllerConfig = std::variant<MPPIConfig, PRMPPIConfig>;

AnyControllerConfig loadControllerConfig(const void* buf, size_t len);

// Environment configuration
// this should have the same layout as GateEnvironmentConfig in the Python side
struct EnvironmentConfig
{
    float dt;

    bool sendStates;

    float initPos[N_AGENTS * DIM];
    float initSpeed[N_AGENTS * DIM];
    float initS[N_AGENTS * N_RACELINES];    // (nAgents * nRacelines). if == -1.0f, will not be updated (since it is only useful for PIDs)
    int initLaps[N_AGENTS];
    int initGates[N_AGENTS];

    float minDist;
    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed[N_AGENTS];
    float maxAccel[N_AGENTS];

    // track
    int nWinLaps;
    float targetDistance;

    float gateCenters[N_GATES * DIM]; // (nGates * dim)
    float gateVectors[N_GATES * DIM]; // (nGates * dim)
    float gateRadius[N_GATES];  // (nGates)

    float arenaMin[DIM];    // (dim)
    float arenaMax[DIM];    // (dim)

    float obstacles[N_OBSTACLES * DIM * 2];   // (nObstacles * dim * 2): rectangle obstacles, i.e. [xmin, ymin, xmax, ymax]
    float roundObsCenters[N_ROUND_OBSTACLES * DIM];     // (nRoundObstacles * dim): center of obstacles
    float roundObsRadius[N_ROUND_OBSTACLES];        // (nRoundObstacles): radius of obstacles

    float* trackPoints = nullptr; // (nTrackSamples * nRacelines, dim)  // this pointer is different on host & device!

    float initBelief[N_TRUE_MODELS];

    PIDConfig oppPid[N_TRUE_MODELS];
    int iMppi;
    int trueTheta;

    EnvironmentConfig()
    {}

    int unpackHeader(const void* buf, size_t len);   // returns (seed, trackPoints)
};
