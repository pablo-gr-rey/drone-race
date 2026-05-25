#pragma once

#include "config.h"
#include "state.h"

#include <vector>
#include <string>
#include <curand_kernel.h>
#include <random>

// forward
class SimulationEngine;

// ── Abstract controller ──────────────────────────────────────────────
class Controller
{
public:
    std::string name;
    const EnvironmentConfig* envConfig = nullptr;
    SimulationEngine* engine = nullptr;

    virtual ~Controller() = default;

    // Writes `dim` floats into outAction.
    virtual void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng, std::optional<std::vector<float>> pastAction = std::nullopt, std::optional<std::vector<float>> pastPos = std::nullopt, std::optional<std::vector<float>> pastVel = std::nullopt, std::optional<std::vector<float>> pastS = std::nullopt) = 0;
};

// ── Dummy ────────────────────────────────────────────────────────────
class DummyController : public Controller
{
public:
    explicit DummyController(const EnvironmentConfig& c);
    void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng, std::optional<std::vector<float>> pastAction = std::nullopt, std::optional<std::vector<float>> pastPos = std::nullopt, std::optional<std::vector<float>> pastVel = std::nullopt, std::optional<std::vector<float>> pastS = std::nullopt) override;
};

// ── PID ──────────────────────────────────────────────────────────────
class PIDController : public Controller
{
public:
    PIDController(const EnvironmentConfig& c, const PIDConfig& p);
    void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng, std::optional<std::vector<float>> pastAction = std::nullopt, std::optional<std::vector<float>> pastPos = std::nullopt, std::optional<std::vector<float>> pastVel = std::nullopt, std::optional<std::vector<float>> pastS = std::nullopt) override;

    PIDConfig params;
};

// ── MPPI ─────────────────────────────────────────────────────────────
class MPPIController : public Controller
{
public:
    // if belief is not given, assumed uniform; if nominal (size (nModels+1) * T * dim) is not given, assumed 0
    MPPIController(const EnvironmentConfig& c, const MPPIConfig& mc, const VerifConfig& vConfig, float* d_trackPoints, std::optional<std::vector<float>> nominal = std::nullopt);
    ~MPPIController();
    void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng, std::optional<std::vector<float>> pastAction = std::nullopt, std::optional<std::vector<float>> pastPos = std::nullopt, std::optional<std::vector<float>> pastVel = std::nullopt, std::optional<std::vector<float>> pastS = std::nullopt) override;

    MPPIConfig mppiConfig;

    std::vector<float> h_nominal;   // host mirror (nModels+1, T, dim)
    std::vector<float> h_belief;    // host belief (nModels)

    std::vector<uint> failCount;    // size 2: nColl, nOutside (only MPPI outside is counted)
    double epsilon;

    float min_nu = 0.005;    // in terms of proportion of nSamples   // TODO: tune this better? (previously: 0.01, 0.05)
    float max_nu = 0.01;
private:
    EnvironmentConfig envConfig;
    VerifConfig verifConfig;

    // Device memory

    // Rollout side
    float* d_pos = nullptr;   // (nAgents, dim) - initial positions
    float* d_speed = nullptr; // (nAgents, dim) - initial speeds
    float* d_S = nullptr;      // (nAgents, nRacelines) - initial advance along the track
    int* d_laps = nullptr;   // (nAgents) - initial number of laps
    int* d_currentGates = nullptr;    // (nAgents) - initial gate progression
    float* d_belief = nullptr;  // (nModels) - initial belief

    float* d_noise = nullptr;  // (nModels+1, T, N, dim)

    float* d_costs = nullptr;  // (N)

    float* d_nominal = nullptr;  // (nModels+1, T, dim)
    float* d_minCost = nullptr;  // scalar
    float* d_nu = nullptr;  // scalar (sum of costs: useful for monitoring & live updating inv temp)

    float* d_trackPts = nullptr;  // cached on device

    void* d_temp_storage = nullptr; // for min-reduce
    size_t temp_storage_bytes = 0;  // for min-reduce

    curandState* d_rng = nullptr;

    // Verification side
    uint* d_failCount;  // 2 ints (first is numColl, second is numOutside)
    curandState* d_verif_rng = nullptr;

    bool deviceReady = false;

    void allocDevice();
    void freeDevice();

    void computeEpsilon();  // compute epsilon based on failCount and verifConfig
};

// Shared PID control function. Does not add noise, since this is different on CPU and GPU
HD INLINE void computePIDAction(
    int agent,
    const float* pos,
    const float* vel,
    const float* S,
    const EnvironmentConfig& envConfig,
    const PIDConfig& pid,
    const float* trackPoints,
    float* outAction)
{
    float target[MAX_DIM] = {};

    sampleCenterline(trackPoints + envConfig.nTrackSamples * pid.racelineIndex * envConfig.dim, envConfig.nTrackSamples, envConfig.dim, S[agent * envConfig.nRacelines + pid.racelineIndex] + envConfig.targetDistance, target);

    const float* curPos = pos + agent * envConfig.dim;
    const float* curVel = vel + agent * envConfig.dim;

    float sqError = 0.0f;
    for (int d = 0; d < envConfig.dim; d++)
    {
        float e = target[d] - curPos[d];
        sqError += e * e;
    }

    // Match CPU logic
    float invDist = 1.0f / sqrtf(sqError + 1e-5f);

    float vParallelMag = 0.0f;
    for (int d = 0; d < envConfig.dim; d++)
    {
        float dirD = (target[d] - curPos[d]) * invDist;
        vParallelMag += curVel[d] * dirD;
    }

    for (int d = 0; d < envConfig.dim; d++)
    {
        float error = (target[d] - curPos[d]) * invDist;
        float latVelError = curVel[d] - vParallelMag * error;
        outAction[d] = pid.kp * error + pid.kd * (-latVelError);
    }

    // PD (no integral)
    // for (int d = 0; d < envConfig.dim; d++)
    //     outAction[d] = pid.kp * (target[d] - pos[d]) + pid.kd * (-vel[d]);

    // repulsion
    if (pid.repulsionDistFact != 0.0f)
        for (int other = 0; other < envConfig.nAgents; other++)
        {
            if (other == agent)
                continue;

            float diff[MAX_DIM];
            float dist2 = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
            {
                diff[d] = pos[other * envConfig.dim + d] - curPos[d];
                dist2 += diff[d] * diff[d];
            }
            float dist = sqrtf(dist2) + 1e-8f;
            if (dist < pid.repulsionDistFact * envConfig.minDist)
            {
                float scale = pid.repulsionFactor / powf(dist, pid.repulsionPower + 1.0f);
                for (int d = 0; d < envConfig.dim; d++)
                    outAction[d] -= scale * diff[d];
            }
        }

    // normalize
    float sqAccel = 0.0f;
    for (int d = 0; d < envConfig.dim; d++)
        sqAccel += outAction[d] * outAction[d];

    if (sqAccel > envConfig.maxAccel[agent] * envConfig.maxAccel[agent])
    {
        float fact = envConfig.maxAccel[agent] / sqrtf(sqAccel);
        for (int d = 0; d < envConfig.dim; d++)
            outAction[d] *= fact;
    }
}
