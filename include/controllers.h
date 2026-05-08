#pragma once

#include "config.h"
#include "device_config.cuh"
#include <vector>
#include <string>
#include <curand_kernel.h>

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
    virtual void getControl(int agent,
        const float* phys,
        const float* S,
        const float* laps,
        float* outAction) = 0;

    virtual void reset() {}
};

// ── Dummy ────────────────────────────────────────────────────────────
class DummyController : public Controller
{
public:
    explicit DummyController(const EnvironmentConfig& c);
    void getControl(int agent, const float* phys,
        const float* S, const float* laps,
        float* outAction) override;
};

// ── PID ──────────────────────────────────────────────────────────────
class PIDController : public Controller
{
public:
    PIDController(const EnvironmentConfig& c, const PIDConfig& p);
    void getControl(int agent, const float* phys,
        const float* S, const float* laps,
        float* outAction) override;

    PIDConfig params;
};

// ── MPPI ─────────────────────────────────────────────────────────────
class MPPIController : public Controller
{
public:
    MPPIController(const EnvironmentConfig& c, const MPPIConfig& mc);
    ~MPPIController();
    void getControl(int agent, const float* phys,
        const float* S, const float* laps,
        float* outAction) override;
    void reset() override;

    MPPIConfig mppiCfg;
private:
    OpponentModelType oppModel;
    PIDConfig oppPidParams;

    DeviceEnvironmentConfig deviceEnvConfig;
    DeviceMPPIConfig deviceMPPIConfig;

    // device memory
    float* d_phys = nullptr;   // (physDim) - initial single current state
    float* d_S = nullptr;      // (nAgents) - initial advance along the track
    float* d_laps = nullptr;   // (nAgents) - initial number of laps
    float* d_sampPhys = nullptr;  // (N, physDim) - final state (optional?)
    float* d_sampS = nullptr;  // (N, nAgents) - final advance along the track 
    float* d_sampLaps = nullptr;  // (N, nAgents) - final number of laps
    float* d_newPhys = nullptr;
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
    bool deviceReady = false;

    void allocDevice();
    void freeDevice();
    void uploadTrack();
};
