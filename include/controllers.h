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
    virtual void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng) = 0;

    virtual void reset() {}
};

// ── Dummy ────────────────────────────────────────────────────────────
class DummyController : public Controller
{
public:
    explicit DummyController(const EnvironmentConfig& c);
    void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng) override;
};

// ── PID ──────────────────────────────────────────────────────────────
class PIDController : public Controller
{
public:
    PIDController(const EnvironmentConfig& c, const PIDConfig& p);
    void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng) override;

    PIDConfig params;
};

// ── MPPI ─────────────────────────────────────────────────────────────
class MPPIController : public Controller
{
public:
    MPPIController(const EnvironmentConfig& c, const MPPIConfig& mc, float* d_trackPoints);
    ~MPPIController();
    void getControl(int agent, const float* pos, const float* speed, const float* S, const int* laps, const int* currentGates, float* outAction, std::normal_distribution<float>& nd, std::mt19937& rng) override;
    void reset() override;

    MPPIConfig mppiCfg;
private:
    EnvironmentConfig envConfig;

    // device memory
    float* d_pos = nullptr;   // (nAgents * dim) - initial positions
    float* d_speed = nullptr; // (nAgents * dim) - initial speeds
    float* d_S = nullptr;      // (nAgents) - initial advance along the track
    int* d_laps = nullptr;   // (nAgents) - initial number of laps
    int* d_currentGates = nullptr;    // (nAgents) - initial gate progression
    float* d_sampPos = nullptr;   // (N, nAgents * dim) - final positions
    float* d_sampSpeed = nullptr; // (N, nAgents * dim) - final speeds
    float* d_sampS = nullptr;  // (N, nAgents) - final advance along the track 
    int* d_sampLaps = nullptr;  // (N, nAgents) - final number of laps
    int* d_sampGates = nullptr; // (N, nAgents) - final gate progression
    float* d_newS = nullptr;
    float* d_newLaps = nullptr;
    float* d_noise = nullptr;  // (T, N, dim)
    float* d_costs = nullptr;  // (N)
    float* d_nominal = nullptr;  // (T, dim)
    float* d_actions = nullptr;  // (N, actionDim)
    float* d_minCost = nullptr;  // scalar
    float* d_trackPts = nullptr;  // cached on device
    void* d_temp_storage = nullptr; // for min-reduce
    size_t temp_storage_bytes = 0;  // for min-reduce
    curandState* d_rng = nullptr;

    std::vector<float> h_nominal;   // host mirror (T * dim)
    float* host_trackPoints;    // original value of EnvironmentConfig.trackPoints
    bool deviceReady = false;

    void allocDevice();
    void freeDevice();
};

// Shared PID control function. Does not add noise, since this is different on CPU and GPU
HD inline void computePIDAction(
    int agent,
    const float* pos,
    const float* vel,
    const float* S,
    const int* currentGates,
    const EnvironmentConfig& envConfig,
    const PIDConfig& pid,
    const float* trackPoints,
    float* outAction)
{
    float target[MAX_DIM];

    if (pid.racelineIndex >= 0)
        sampleCenterline(trackPoints + envConfig.nTrackSamples * pid.racelineIndex * envConfig.dim, envConfig.nTrackSamples, envConfig.dim, S[agent] + envConfig.targetDistance, target);
    else
    {
        int nextGate = (currentGates[agent] + 1) % envConfig.nGates;
        for (int d = 0; d < envConfig.dim; d++)
            target[d] = envConfig.gateCenters[nextGate * envConfig.dim + d];
    }

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
    for (int other = 0; other < envConfig.nAgents; other++)
    {
        if (other == agent) continue;
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
            float scale = pid.repulsionFactor / powf(dist, pid.repulsionPower);
            for (int d = 0; d < envConfig.dim; d++)
                outAction[d] -= scale * diff[d];
        }
    }
}
