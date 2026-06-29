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

#define USE_ENV_DRONERACE

constexpr bool USE_SPLINES = true;

constexpr int MAX_N_KNOTS = 60;

constexpr float MIN_COEFF_THRESHOLD = 0.01f;        // minimum probability threshold for samples to contribute
constexpr float DECAY = 0.95f;

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

#include "environments/env_dronerace_defs.h"

#ifdef USE_ENV_DRONERACE
namespace Env = EnvDroneRace;
#else
#error "must define a valid environment!"
#endif

using EnvironmentConfig = Env::EnvironmentConfig;
using SimState = Env::SimState;
using ScratchEnvBuffer = Env::ScratchEnvBuffer;

constexpr int ACTION_DIM = Env::ACTION_DIM;

constexpr int N_MODEL_FACTORS = Env::N_MODEL_FACTORS;
constexpr int N_TRUE_MODELS = Env::N_TRUE_MODELS;
constexpr int N_BRANCH_PLANS = Env::N_BRANCH_PLANS;
constexpr int MAX_MODEL_SIZE = Env::MAX_MODEL_SIZE;

HD INLINE constexpr int MODEL_SIZE(int k)
{
    return Env::MODEL_SIZE(k);
}

HD INLINE constexpr int BRANCH_SIZE(int k)
{
    return Env::BRANCH_SIZE(k);
}

enum CONTROLLER_KIND : int32_t { CONT_MPPI, CONT_PRMPPI };

enum TerminalType
{
    TERM_NONE = 0,
    TERM_COLLISION,
    TERM_EGO_OUTSIDE,
    TERM_OPP_OUTSIDE,
    TERM_WIN,
    TERM_OPP_WIN
};

#include "environments/env_dronerace.h"

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
    float boundaryCost = 10.0f;
    float boundaryThresholdFactor = 1.5f;

    float outsideCost = 1000.0f;
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
    float boundaryCost = 10.0f;
    float boundaryThresholdFactor = 1.5f;

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
