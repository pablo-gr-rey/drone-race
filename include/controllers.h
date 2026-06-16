#pragma once

#include "config.h"
#include "state.h"

#include <vector>
#include <string>
#include <curand_kernel.h>
#include <random>
#include <optional>

// forward
class SimulationEngine;

// Abstract controller
class Controller
{
public:
    std::string name;
    const EnvironmentConfig* envConfig = nullptr;
    SimulationEngine* engine = nullptr;

    virtual ~Controller() = default;

    // Writes `dim` floats into outAction.
    virtual void getControl(
        int agent,
        const SimState& state,
        float* outAction,
        std::normal_distribution<float>& nd,
        std::mt19937& rng,
        std::optional<std::vector<float>> pastAction = std::nullopt,
        std::optional<SimState> pastState = std::nullopt) = 0;
};

// Dummy
class DummyController : public Controller
{
public:
    explicit DummyController(const EnvironmentConfig& c);

    void getControl(
        int agent,
        const SimState& state,
        float* outAction,
        std::normal_distribution<float>& nd,
        std::mt19937& rng,
        std::optional<std::vector<float>> pastAction = std::nullopt,
        std::optional<SimState> pastState = std::nullopt) override;
};

// PID
class PIDController : public Controller
{
public:
    PIDController(const EnvironmentConfig& c, const PIDConfig& p);

    void getControl(
        int agent,
        const SimState& state,
        float* outAction,
        std::normal_distribution<float>& nd,
        std::mt19937& rng,
        std::optional<std::vector<float>> pastAction = std::nullopt,
        std::optional<SimState> pastState = std::nullopt) override;

    PIDConfig params;
};

// MPPI
class MPPIController : public Controller
{
public:
    // if nominal (size (nModels+1) * T * dim) is not given, assumed 0
    MPPIController(
        const EnvironmentConfig& c,
        const MPPIConfig& mc,
        const VerifConfig& vConfig,
        float* d_trackPoints,
        int s,
        std::optional<std::vector<float>> nominal = std::nullopt);

    ~MPPIController();

    void getControl(
        int agent,
        const SimState& state,
        float* outAction,
        std::normal_distribution<float>& nd,
        std::mt19937& rng,
        std::optional<std::vector<float>> pastAction = std::nullopt,
        std::optional<SimState> pastState = std::nullopt) override;

    MPPIConfig mppiConfig;

    std::vector<float> h_nominal;   // host mirror (nModels+1, T, dim)
    std::vector<float> h_belief;    // host belief (nModels)

    std::vector<uint> failCountOld;    // size 2: nColl, nOutside (previous nominal, only MPPI outside is counted)
    std::vector<uint> failCountNew;    // size 2: nColl, nOutside (candidate new nominal, only MPPI outside is counted)
    std::vector<uint> failCount;       // size 2: nColl, nOutside (final nominal, only MPPI outside is counted)
    double epsilonPartial;             // epsilon obtained by verification at current iteration
    double epsilon;                    // epsilonPartial + verifConfig.horizon * verifConfig.maxEps
    double certifiedLoss;              // certified max loss if we use new plan instead of previous
    bool useNewPlan;

    float min_nu = 0.0005f;    // in terms of proportion of nSamples
    float max_nu = 0.001f;

private:
    EnvironmentConfig envConfig;
    VerifConfig verifConfig;

    int seed;

    // Device memory

    // Rollout side
    float* d_belief = nullptr;     // (nModels)

    float* d_noise = nullptr;      // (nModels+1, T, N, dim)

    // d_costs[0, s] = averaged / expected cost of sample s
    // d_costs[theta+1, s] = cost of sample s if opp is following theta
    float* d_costs = nullptr;      // (nModels+1, N)

    // d_branchUsed[theta, s] = -1 if MPPI did not switch, value of model it switched to otherwise
    int* d_branchUsed = nullptr;   // (nModels, N)

    // d_branchTime[theta, s] = branching time (or T if no branching)
    int* d_branchTime = nullptr;   // (nModels, N)

    float* d_nominal = nullptr;      // (nModels+1, T, dim)
    float* d_prevnominal = nullptr;  // (nModels+1, T, dim)

    // ((nModels+1), T, N): masked row costs used for branch/time-aware min reduction
    float* d_maskedCosts = nullptr;

    // ((nModels+1), T): branch/time-aware mins
    float* d_minCosts = nullptr;

    // (nModels+1): denom for generic at t=0 and each specialized branch at local time 0
    float* d_nu = nullptr;

    float* d_trackPts = nullptr;  // cached on device

    void* d_temp_storage = nullptr; // for min-reduce
    size_t temp_storage_bytes = 0;  // for min-reduce

    curandState* d_rng = nullptr;

    // Verification side
    uint* d_failCountNew = nullptr;  // 2 ints (first is numColl, second is numOutside)
    uint* d_failCountOld = nullptr;  // 2 ints
    curandState* d_verif_rng = nullptr;

    bool deviceReady = false;

    void allocDevice();
    void freeDevice();

    void computeEpsilon();       // compute epsilon based on failCount and verifConfig
    void computeCertifiedLoss(); // compute loss of new plan over old plan
};

// Shared PID control function. Does not add noise, since this is different on CPU and GPU.
HD INLINE void computePIDAction(
    int agent,
    const SimState& state,
    const EnvironmentConfig& envConfig,
    const PIDConfig& pid,
    const float* trackPoints,
    float* outAction)
{
    float target[DIM] = {};

    sampleCenterline(
        trackPoints + N_TRACK_SAMPLES * pid.racelineIndex * DIM,
        state.S[agent * N_RACELINES + pid.racelineIndex] + envConfig.targetDistance,
        target);

    const float* curPos = state.pos + agent * DIM;
    const float* curVel = state.vel + agent * DIM;

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
    if (pid.repulsionDistFact != 0.0f)
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
            if (dist < pid.repulsionDistFact * envConfig.minDist)
            {
                float scale = pid.repulsionFactor / powf(dist / envConfig.minDist, pid.repulsionPower + 1.0f);
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
