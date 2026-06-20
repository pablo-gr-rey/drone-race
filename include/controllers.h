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

    std::vector<float> h_nominal;   // host mirror (nBranchPlans, T, dim)
    std::array<float, N_TRUE_MODELS> h_belief;    // host belief (nTrueModels). this is updated by the engine

    std::array<uint, 2> failCountOld;    // size 2: nColl, nOutside (previous nominal, only MPPI outside is counted)
    std::array<uint, 2> failCountNew;    // size 2: nColl, nOutside (candidate new nominal, only MPPI outside is counted)
    std::array<uint, 2> failCount;       // size 2: nColl, nOutside (final nominal, only MPPI outside is counted)
    double epsilonPartial;             // epsilon obtained by verification at current iteration
    double epsilon;                    // epsilonPartial + verifConfig.horizon * verifConfig.maxEps
    double certifiedLoss;              // certified max loss if we use new plan instead of previous
    bool useNewPlan;

    float min_nu = 0.0005f;    // in terms of proportion of nSamples
    float max_nu = 0.001f;

    std::vector<float> buildSplineMatrix();     // return a T*M matrix B such that for u spline control points, B*u computes the (natural) spline for each value of 0 <= t < T (in particular, if tau_i is integer, (Bu)_(tau_i) = u_i)

private:
    std::vector<float> h_B;     // spline matrix

    float* h_trackPoints;
    EnvironmentConfig envConfig;    // this contains the device track points
    VerifConfig verifConfig;

    int seed;

    // Device memory

    float* d_B = nullptr;       // spline matrix (T, M)

    // Rollout side
    float* d_belief = nullptr;     // (nTrueModels)

    float* d_noise = nullptr;      // (nBranchPlans, M, N, dim)

    // d_costs[(nom0/theta0, nom1/theta1, ...), s] = cost of sample s, averaged over unknown parameters (i.e. d_costs[(nom0, theta1), s] = cost of sample s under theta1, averaged over theta0)
    float* d_costs = nullptr;      // (nBranchPlans, N)

    // d_costsTrue[trueModel, s] = cost of sample s over true model (it will then be averaged to compute d_costs)
    float* d_costsTrue = nullptr;   // (nTrueModels, N)

    // d_branchUsed[trueTheta, s, k] = 0 if sample s stayed in nominal for parameter k in case of trueTheta, otherwise 1 + value of model it switched to
    int* d_branchUsed = nullptr;   // (nTrueModels, N, nModelFactors)

    // d_branchTime[trueTheta, s, k] = branching time for parameter k in case of trueTheta (or T if no branching)
    int* d_branchTime = nullptr;   // (nTrueModels, N, nModelFactors)

    float* d_splineNominal = nullptr;       // (nBranchPlans, M, dim): spline nominal
    float* d_tempSplineNominal = nullptr;       // (nBranchPlans, M, dim): temp spline nominal (for shifting)
    float* d_nominal = nullptr;      // (nBranchPlans, T, dim): dense time nominal
    float* d_prevnominal = nullptr;  // (nBranchPlans, T, dim): dense time prev nominal

    // masked row costs used for branch/time-aware min reduction
    float* d_maskedCosts = nullptr; // (nBranchPlans, T, N)

    // branch/time aware minimums
    float* d_minCosts = nullptr;    // (nBranchPlans, T)
    float* d_minSplineCosts = nullptr;   // size N_BRANCH_PLANS * M

    // denominator for each branch tuple
    float* d_nu = nullptr;  // (nBranchPlans)

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
