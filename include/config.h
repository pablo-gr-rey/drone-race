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
constexpr int MAX_AGENTS = 4;
constexpr int MAX_DIM = 2;
constexpr int MAX_GATES = 5;
constexpr int MAX_PHYS_DIM = MAX_AGENTS * MAX_DIM * 2; // pos+vel
constexpr int MAX_ACTION_DIM = MAX_AGENTS * MAX_DIM;

#define DEBUG

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

    std::vector<float> initPos;
    std::vector<float> initSpeed;
    std::vector<float> initS;
    std::vector<int> initLaps;
    std::vector<int> initGates;

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

    std::vector<float> gateCenters; // (nGates * dim)
    std::vector<float> gateVectors; // (nGates * dim)
    std::vector<float> gateRadius;  // (nGates)

    std::vector<float> arenaMin;    // (dim)
    std::vector<float> arenaMax;    // (dim)

    std::vector<float> trackPoints; // (nTrackSamples * nRacelines, dim)

    // derived
    int physDim = 0;   // nAgents * dim * 2
    int actionDim = 0;   // nAgents * dim

    EnvironmentConfig()
    {}

    void unpackHeader(const void* buf, size_t len);

    void recompute()
    {
        physDim = nAgents * dim * 2;
        actionDim = nAgents * dim;
    }
};

// dummy controller parameters
struct DummyConfig {};

// ── PID parameters (also used for opponent modelling on GPU) ─────────
struct PIDConfig
{
    float kp = 10.0f;
    float kd = 5.0f;

    float repulsionFactor = 20.0f;
    float repulsionPower = 2.0f;
    float repulsionDistFact = 10.0f;

    int racelineIndex = 0;
};

using OpponentControllerConfig = std::variant<DummyConfig, PIDConfig>;

// ── MPPI configuration ───────────────────────────────────────────────
struct MPPIConfig
{
    int   nSamples = 100;
    int   nTimesteps = 20;
    float invTemperature = 10.0f;

    float samplingNoise = 0.1f;

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

    OpponentControllerConfig opponent = PIDConfig{};
};

// ── Opponent model type (for GPU kernels) ────────────────────────────
enum class OpponentModelType
{
    PID,
    Dummy,   // zero acceleration
};

using ControllerConfig = std::variant<DummyConfig, PIDConfig, MPPIConfig>;

// ── Controller specification (POD, used to construct controllers) ────
struct ControllerSpec
{
    // enum Kind { Dummy, PID_Kind, MPPI_Kind } kind = Dummy;
    std::string name = "dummy";
    ControllerConfig config = DummyConfig{};
    // PIDConfig  pidParams;
    // MPPIConfig mppiConfig;
    // OpponentModelType oppModel = OpponentModelType::PID;
    // PIDConfig  oppPidParams;

    void unpackHeader(const void* buf, size_t len);
};
