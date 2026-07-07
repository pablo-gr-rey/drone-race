#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <variant>

#include <cuda/std/array>

#include "common.h"

// #define USE_ENV_DRONERACE
#define USE_ENV_HIDDENOBS
// #define USE_ENV_STRATRACE

inline constexpr bool USE_SPLINES = true;

inline constexpr int MAX_N_KNOTS = 60;

inline constexpr float MIN_COEFF_THRESHOLD = 0.01f; // minimum probability threshold for samples to contribute
inline constexpr float DECAY = 0.95f;

#ifdef USE_ENV_DRONERACE
#include "environments/env_dronerace_defs.h"
namespace Env = EnvDroneRace;
#elif defined(USE_ENV_HIDDENOBS)
#include "environments/env_hiddenobs_defs.h"
namespace Env = EnvHiddenObs;
#elif defined(USE_ENV_STRATRACE)
#include "environments/env_stratrace_defs.h"
namespace Env = EnvStratRace;
#else
#error "must define a valid environment!"
#endif

using EnvironmentConfig = Env::EnvironmentConfig;
using SimState = Env::SimState;
using ScratchEnvBuffer = Env::ScratchEnvBuffer;

inline constexpr int ACTION_DIM = Env::ACTION_DIM;

inline constexpr int N_MODEL_FACTORS = Env::N_MODEL_FACTORS;
inline constexpr int N_TRUE_MODELS = Env::N_TRUE_MODELS;
inline constexpr int N_BRANCH_PLANS = Env::N_BRANCH_PLANS;
inline constexpr int MAX_MODEL_SIZE = Env::MAX_MODEL_SIZE;

HD INLINE constexpr int MODEL_SIZE(int k)
{
    return Env::MODEL_SIZE(k);
}

HD INLINE constexpr int BRANCH_SIZE(int k)
{
    return Env::BRANCH_SIZE(k);
}

enum CONTROLLER_KIND : int32_t
{
    CONT_MPPI,
    CONT_PRMPPI
};
enum ENV_KIND : int32_t
{
    ENV_DRONERACE,
    ENV_HIDDENOBS,
    ENV_STRATRACE
};

enum TerminalType
{
    TERM_NONE = 0,
    TERM_LOSE,
    TERM_WIN,
};

struct BranchState
{
    cuda::std::array<float, N_TRUE_MODELS> belief;
    cuda::std::array<int, N_MODEL_FACTORS> predTheta;     // 0 if nominal for k, theta+1 if specialized (<=> marginal over theta_k=theta > threshold)
    cuda::std::array<int, N_MODEL_FACTORS> branchingTime; // -1 if not branched, 0 if initially committed, otherwise t+1 if branched at global time
                                                          // t (i.e. origin of new branch)
};

struct DeviceRNG
{
    curandState* state;
};

struct HostRNG
{
    std::normal_distribution<float>* nd;
    std::mt19937* rng;
};

#ifdef USE_ENV_DRONERACE
#include "environments/env_dronerace.h" // IWYU pragma: export
#elif defined(USE_ENV_HIDDENOBS)
#include "environments/env_hiddenobs.h" // IWYU pragma: export
#elif defined(USE_ENV_STRATRACE)
#include "environments/env_stratrace.h" // IWYU pragma: export
#else
#error "must define a valid environment!"
#endif

// MPPI configuration
struct MPPIConfig
{
    int nSamples = 100;

    int nTimesteps = 20;

    // spline configuration
    int nKnots;                               // M
    cuda::std::array<int, MAX_N_KNOTS> knots; // tau_i for 0 <= i < M

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
    float minSafeDist = 0.0f; // safety cost is lessened by this amount (i.e. we must be at
                              // distance at least minSafeDist from obstacles & opponents)

    // safety assurance (delta), and number of model samples P
    // (=ceil((1-delta)/delta))
    float delta = 0.1f;
    float P = 10;

    void unpackHeader(Reader& reader);
};

using AnyControllerConfig = std::variant<MPPIConfig, PRMPPIConfig>;

AnyControllerConfig loadControllerConfig(const void* buf, size_t len);
