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

class MPPIController
{
public:
    // if nominal (size (nModels+1) * T * dim) is not given, assumed 0
    MPPIController() {};

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
        float* outAction);

    std::string name;
    SimulationEngine* engine = nullptr;

    MPPIConfig mppiConfig;

    std::vector<float> h_nominal;   // host mirror (nModels+1, T, dim)
    std::array<float, N_MODELS> h_belief;    // host belief (nModels). now, this is updated by the engine

    std::array<uint, 2> failCountOld;    // size 2: nColl, nOutside (previous nominal, only MPPI outside is counted)
    std::array<uint, 2> failCountNew;    // size 2: nColl, nOutside (candidate new nominal, only MPPI outside is counted)
    std::array<uint, 2> failCount;       // size 2: nColl, nOutside (final nominal, only MPPI outside is counted)
    double epsilonPartial;             // epsilon obtained by verification at current iteration
    double epsilon;                    // epsilonPartial + verifConfig.horizon * verifConfig.maxEps
    double certifiedLoss;              // certified max loss if we use new plan instead of previous
    bool useNewPlan;

    float min_nu = 0.0005f;    // in terms of proportion of nSamples
    float max_nu = 0.001f;

private:
    float* h_trackPoints;
    EnvironmentConfig envConfig;    // this contains the device track points
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

    // float* d_trackPts = nullptr;  // cached on device // now part of EnvironmentConfig

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
