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

class Controller
{
public:
    virtual ~Controller() = default;

    virtual void getControl(int agent, const SimState& state, float* outAction) = 0;

    SimulationEngine* engine = nullptr;
    std::array<float, N_TRUE_MODELS> h_belief;    // host belief (nTrueModels). this is updated by the engine
};

class MPPIController : public Controller
{
public:
    // if nominal (size (nModels+1) * T * dim) is not given, assumed 0
    MPPIController() {};

    MPPIController(
        const EnvironmentConfig& c,
        const MPPIConfig& mc,
        float* d_trackPoints,
        int s,
        std::optional<std::vector<float>> nominal = std::nullopt);

    ~MPPIController();

    void getControl(int agent, const SimState& state, float* outAction) override;

    MPPIConfig mppiConfig;

    std::vector<float> h_nominal;   // host mirror (nBranchPlans, T, dim)

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

    // d_branchUsed[trueTheta, s, k] = 0 if sample s stayed in nominal for parameter k in case of trueTheta (and corresponding branchTime = -1), otherwise 1 + value of model it switched to. if we are already committed at time 0, then equal to 1 + theta (and corresponding branchTime = 0)
    int* d_branchUsed = nullptr;   // (nTrueModels, N, nModelFactors)

    // d_branchTime[trueTheta, s, k] = branching time for parameter k in case of trueTheta (or -1 if no branching, 0 if we are already branched at time 0). note that this is ABSOLUTE time, and that it corresponds to the first time in the new branch (i.e. if we become certain on the first step, then branchingTime=1)
    int* d_branchTime = nullptr;   // (nTrueModels, N, nModelFactors)

    float* d_splineNominal = nullptr;       // (nBranchPlans, M, dim): spline nominal
    float* d_tempSplineNominal = nullptr;       // (nBranchPlans, M, dim): temp spline nominal (for shifting)
    float* d_nominal = nullptr;      // (nBranchPlans, T, dim): dense time nominal
    float* d_prevnominal = nullptr;  // (nBranchPlans, T, dim): dense time prev nominal

    // branch/time aware minimums
    float* d_minCosts = nullptr;    // (nBranchPlans, T or M)

    // denominator for each branch tuple
    float* d_nu = nullptr;  // (nBranchPlans)

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

// controller of the paper "Parameter-Robust MPPI for Safe Online Learning of Unknown Parameters"
class PRMPPIController : public Controller
{
public:
    // if nominal (size (nModels+1) * T * dim) is not given, assumed 0
    PRMPPIController() {};

    PRMPPIController(
        const EnvironmentConfig& c,
        const PRMPPIConfig& mc,
        float* d_trackPoints,
        int s,
        std::optional<std::vector<float>> nominal = std::nullopt);

    ~PRMPPIController();

    void getControl(int agent, const SimState& state, float* outAction) override;

    PRMPPIConfig mppiConfig;

    std::vector<float> h_nom_nominal;   // host mirror (T, dim)
    std::vector<float> h_rob_nominal;   // host mirror (T, dim)

    bool resetNom;      // true if the new nominal is optimized from rob_nominal (happens if the nominal has a very high safety cost) (i.e. we use cand2 for new nom_nominal)
    bool useNomPlan;    // true if we use nom plan (otherwise, nom was deemed unsafe, and we used rob_nominal)

    float min_nu = 0.0005f;    // in terms of proportion of nSamples
    float max_nu = 0.001f;

private:
    float* h_trackPoints;
    EnvironmentConfig envConfig;    // this contains the device track points

    float invTempNomFull;
    float invTempRobFull;
    float invTempRobSafe;

    int seed;

    // Device memory

    // Rollout side
    float* d_belief = nullptr;     // (nTrueModels)

    float* d_noise = nullptr;      // (T, N, dim)

    float* d_cost_nom = nullptr;      // (P, N, 2) (stores cost, safetyCost)
    float* d_cost_rob = nullptr;      // (P, N, 2) (stores cost, safetyCost)

    float* d_nom_nominal = nullptr;      // (T, dim)
    float* d_rob_nominal = nullptr;  // (T, dim)

    float* d_cand1_nominal = nullptr;   // (T, dim) candidate nom_nominal (computed from d_nom_nominal and d_cost_nom full)
    float* d_cand2_nominal = nullptr;   // (T, dim) candidate nom_nominal (computed from d_rob_nominal and d_cost_rob full)
    float* d_new_rob_nominal = nullptr; // (T, dim) candidate rob_nominal (computed from d_rob_nominal and d_cost_rob safe)

    float* d_minCosts = nullptr;    // (3): nom_full, rob_full, rob_safe
    float* d_candCosts = nullptr;   // (2*P): full cost of cand1, cand2

    int* d_thetas = nullptr;    // (P)

    // denominator for each branch tuple
    float* d_nu = nullptr;  // (3): nom_full, rob_full, rob_safe

    curandState* d_rng = nullptr;   // (P, N)
    curandState* d_theta_rng = nullptr;     // (P)

    bool deviceReady = false;

    void allocDevice();
    void freeDevice();
};
