#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <protocol.h>
#include <variant>
#include <type_traits>

// ── compile-time limits ──────────────────────────────────────────────
constexpr int MAX_AGENTS = 2;
constexpr int MAX_DIM = 2;
constexpr int MAX_GATES = 5;
constexpr int MAX_MODELS = 2;
constexpr int MAX_OBSTACLES = 2;
constexpr int MAX_RACELINES = 2;

#ifdef __CUDACC__
#define HD __host__ __device__
#define INLINE static __forceinline__ 
#else
#define HD
#define INLINE inline
#endif

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

// ── Environment configuration ────────────────────────────────────────
// this should have the same layout as GateEnvironmentConfig in the Python side (including superclasses, ie. BaseEnvironmentConfig then GateEnvironmentConfig)
struct EnvironmentConfig
{
    int nAgents;
    int dim;
    float dt;

    bool sendStates;
    int nRacelines;
    int nGates;

    float initPos[MAX_AGENTS * MAX_DIM];
    float initSpeed[MAX_AGENTS * MAX_DIM];
    float initS[MAX_AGENTS * MAX_RACELINES];    // (nAgents * nRacelines). if == -1.0f, will not be updated (since it is only useful for PIDs)
    int initLaps[MAX_AGENTS];
    int initGates[MAX_AGENTS];

    float minDist;
    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed[MAX_AGENTS];
    float maxAccel[MAX_AGENTS];

    // track
    int nTrackSamples;
    int nWinLaps;
    float targetDistance;

    float gateCenters[MAX_GATES * MAX_DIM]; // (nGates * dim)
    float gateVectors[MAX_GATES * MAX_DIM]; // (nGates * dim)
    float gateRadius[MAX_GATES];  // (nGates)

    float arenaMin[MAX_DIM];    // (dim)
    float arenaMax[MAX_DIM];    // (dim)

    int nObstacles;
    float obstacles[MAX_OBSTACLES * MAX_DIM * 2];   // (nObstacles * dim * 2): rectangle obstacles, i.e. [xmin, ymin, xmax, ymax]

    // float* trackPoints = nullptr; // (nTrackSamples * nRacelines, dim)

    EnvironmentConfig()
    {}

    std::vector<float> unpackHeader(const void* buf, size_t len);   // returns trackPoints
};

struct VerifConfig
{
    int nVerifSamples;
    float beta;

    std::vector<int> K;     // assumed to be sorted

    void unpackHeader(const void* buf, size_t len);
};

// dummy controller parameters
struct DummyConfig {};

// ── Opponent model type (for GPU kernels) ────────────────────────────
enum ControllerKind
{
    CONT_DUMMY = 0,   // zero acceleration
    CONT_PID = 1,
    CONT_MPPI = 2
};

// ── PID parameters (also used for opponent modelling on GPU) ─────────
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

// ── MPPI configuration ───────────────────────────────────────────────
struct MPPIConfig
{
    int   nSamples = 100;
    int   nTimesteps = 20;
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
    float finalSpeedWeight = 5.0f;

    float minConfidence = 0.9f;

    int nModels;
    ControllerKind oppKind = CONT_DUMMY;

    PIDConfig oppPid[MAX_MODELS];
    float initBelief[MAX_MODELS];
};

using ControllerConfig = std::variant<DummyConfig, PIDConfig, MPPIConfig>;

// ── Controller specification (POD, used to construct controllers) ────
struct ControllerSpec
{
    std::string name = "dummy";
    ControllerConfig config = DummyConfig{};

    void unpackHeader(const void* buf, size_t len);
};
