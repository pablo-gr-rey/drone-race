#include "config.h"
#include "controllers.h"
#include "kernels.cuh"

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <format>
#include <stdexcept>
#include <vector>

// Construction
PRMPPIController::PRMPPIController(const EnvironmentConfig& c, const PRMPPIConfig& mc, int s, std::optional<std::vector<float>> nominal)
{
    std::cout << "PRMPPI INIT" << std::endl;
    envConfig = c;
    mppiConfig = mc;
    seed = s;

    // h_belief = std::to_array(envConfig.initBelief);
    std::copy(envConfig.initBelief.begin(), envConfig.initBelief.end(), h_belief.begin());

    if (nominal)
    {
        if ((int)nominal->size() != mc.nTimesteps * ACTION_DIM)
            throw std::runtime_error(
                std::format("Invalid PRMPPI construction: expected nominal size {}, got {}", mc.nTimesteps * ACTION_DIM, nominal->size()));

        h_nom_nominal = nominal.value();
        h_rob_nominal = nominal.value();
    }
    else
    {
        h_nom_nominal.assign(mc.nTimesteps * ACTION_DIM, 0.0f);
        h_rob_nominal.assign(mc.nTimesteps * ACTION_DIM, 0.0f);
    }

    invTempNomFull = invTempRobFull = invTempRobSafe = mppiConfig.invTemperature;
}

PRMPPIController::~PRMPPIController()
{
    freeDevice();
}

void PRMPPIController::allocDevice()
{
    if (deviceReady)
        return;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int P = mppiConfig.P;

    // Single authoritative state (uploaded each call)
    CUDA_CHECK(cudaMalloc(&d_belief, N_TRUE_MODELS * sizeof(float)));

    // Noise
    CUDA_CHECK(cudaMalloc(&d_noise, T * N * ACTION_DIM * sizeof(float)));

    // Costs
    CUDA_CHECK(cudaMalloc(&d_cost_nom, P * N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost_rob, P * N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candCosts, 2 * P * sizeof(float)));

    // Nominal action
    CUDA_CHECK(cudaMalloc(&d_nom_nominal, T * ACTION_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rob_nominal, T * ACTION_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand1_nominal, T * ACTION_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand2_nominal, T * ACTION_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_rob_nominal, T * ACTION_DIM * sizeof(float)));

    // Min-reduction / nu
    CUDA_CHECK(cudaMalloc(&d_minCosts, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thetas, P * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nu, 3 * sizeof(float)));

    // Per-sample RNG states
    CUDA_CHECK(cudaMalloc(&d_rng, P * N * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_rng, 0, P * N * sizeof(curandState)));

    CUDA_CHECK(cudaMalloc(&d_theta_rng, P * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_theta_rng, 0, P * sizeof(curandState)));

    // Initialise RNG
    int blk = 256;
    int grd = (P * N + blk - 1) / blk;
    initRNGKernel<<<grd, blk>>>(d_rng, seed, P * N);
    int tGrd = (P + blk - 1) / blk;
    initRNGKernel<<<tGrd, blk>>>(d_theta_rng, seed, P);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_nom_nominal, h_nom_nominal.data(), h_nom_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rob_nominal, h_rob_nominal.data(), h_rob_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));

    deviceReady = true;
}

void PRMPPIController::freeDevice()
{
    auto safe_free = [](auto*& p)
    {
        if (p)
        {
            CUDA_CHECK(cudaFree(p));
            p = nullptr;
        }
    };

    safe_free(d_belief);

    safe_free(d_noise);

    safe_free(d_cost_nom);
    safe_free(d_cost_rob);
    safe_free(d_candCosts);

    safe_free(d_nom_nominal);
    safe_free(d_rob_nominal);
    safe_free(d_cand1_nominal);
    safe_free(d_cand2_nominal);
    safe_free(d_new_rob_nominal);

    safe_free(d_minCosts);

    safe_free(d_thetas);
    safe_free(d_nu);

    safe_free(d_rng);
    safe_free(d_theta_rng);

    deviceReady = false;
}

void PRMPPIController::getControl(const SimState& state, float* outAction)
{
    if (!engine)
        throw std::runtime_error("PRMPPI: engine not set");

    if (!deviceReady)
        allocDevice();

    std::cout << "PRMPPI BEGIN" << std::endl;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int P = mppiConfig.P;

    // now, belief update is done in engine

    std::cout << "PRMPPI belief: ";
    for (float v : h_belief)
        std::cout << v << " ";
    std::cout << "\n";

    // 0. shift nominals, and upload them
    for (int t = 0; t < T - 1; ++t)
        for (int d = 0; d < ACTION_DIM; ++d)
        {
            h_nom_nominal[t * ACTION_DIM + d] = h_nom_nominal[(t + 1) * ACTION_DIM + d];
            h_rob_nominal[t * ACTION_DIM + d] = h_rob_nominal[(t + 1) * ACTION_DIM + d];
        }

    for (int d = 0; d < ACTION_DIM; ++d)
    {
        h_nom_nominal[(T - 1) * ACTION_DIM + d] = 0.0f;
        h_rob_nominal[(T - 1) * ACTION_DIM + d] = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_nom_nominal, h_nom_nominal.data(), T * ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_rob_nominal, h_rob_nominal.data(), T * ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // 1. upload belief (now, current state is passed as argument to the kernels)
    CUDA_CHECK(cudaMemcpy(d_belief, h_belief.data(), N_TRUE_MODELS * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_nu, 0, 3 * sizeof(float)));

    int blk = 256;

    // 2. generate all noise at once: (T, N, dim)
    int noiseGrd = (N + blk - 1) / blk;
    PRMPPIgenerateNoiseKernel<<<noiseGrd, blk>>>(d_noise, d_rng, mppiConfig.samplingNoise, T, N);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 3. generate values of thetas
    int thGenGrd = (P + blk - 1) / blk;
    PRMPPIsampleThetaValues<<<thGenGrd, blk>>>(d_belief, d_thetas, d_theta_rng, P);

    // 4. rollout (2*N*P launches, first half for nom, second half for rob)
    int rollGrd = (2 * N * P + blk - 1) / blk;
    PRMPPIfullRolloutKernel<<<rollGrd, blk>>>(envConfig, mppiConfig, state, d_nom_nominal, d_rob_nominal, d_noise, d_cost_nom, d_cost_rob, d_thetas,
                                              d_rng);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 5. For each sample s, compute expCost := avg(cost[p, s, 0]) and safeCost := max(cost[p, s, 1]); stores cost[0, s, 0] := expCost + weight * (1
    // if safeCost < 0), cost[0, s, 1] = safeCost. 2*N threads (1st part for nom, 2nd part for rob)
    int avgGrid = (2 * N + blk - 1) / blk;
    PRMPPIcostAvgKernel<<<avgGrid, blk>>>(d_cost_nom, d_cost_rob, N, P, mppiConfig.safetyWeight);

    // 6. compute min costs (3 blocks for nom_full, rob_full, rob_safe)
    PRMPPIcomputeMinCostsKernel<<<3, blk, blk * sizeof(float)>>>(d_cost_nom, d_cost_rob, d_minCosts, N);

    // 7. compute weights, and update nominals (nom_nominal + cost_nom_full&minCosts[0] -> cand1; rob_nominal + cost_rob_full&mincosts[1] -> cand2;
    // rob_nominal + cost_rob_safe&mincosts[2] -> rob_nominal), 3*T*ACTION_DIM blocks
    PRMPPIWeightedAverageKernel<<<3 * T * ACTION_DIM, blk, 2 * blk * sizeof(float)>>>(d_nom_nominal, d_rob_nominal, d_noise, d_cost_nom, d_cost_rob,
                                                                                      d_minCosts, d_cand1_nominal, d_cand2_nominal, d_new_rob_nominal,
                                                                                      N, T, invTempNomFull, invTempRobFull, invTempRobSafe, d_nu);

    std::swap(d_rob_nominal, d_new_rob_nominal);

    // 8. compute full cost for the 2 candidates nominals and all models, 2 * P threads (using the rng of the first 2*P rollouts)
    int candGrd = (2 * P + blk - 1) / blk;
    PRMPPIcomputeCandidateCostKernel<<<candGrd, blk>>>(envConfig, mppiConfig, state, d_cand1_nominal, d_cand2_nominal, d_thetas, d_candCosts, d_rng);

    // 9. Compare the averaged cost of the two candidates, and copy the best one into d_nom_nominal (or rather swap the pointers)
    std::vector<float> candCosts(2 * P);

    CUDA_CHECK(cudaMemcpy(candCosts.data(), d_candCosts, 2 * P * sizeof(float), cudaMemcpyDeviceToHost));

    float cand1Cost = 0.0f;
    float cand2Cost = 0.0f;

    for (int p = 0; p < P; ++p)
    {
        cand1Cost += candCosts[p];
        cand2Cost += candCosts[P + p];
    }

    cand1Cost /= (float)P;
    cand2Cost /= (float)P;

    std::cout << "Averaged full cost for candidate 1 (from d_nom_nominal): " << cand1Cost << "\n";
    std::cout << "Averaged full cost for candidate 2 (from d_rob_nominal): " << cand2Cost << "\n";

    resetNom = cand2Cost < cand1Cost;

    if (cand1Cost <= cand2Cost)
        std::swap(d_nom_nominal, d_cand1_nominal);
    else
        std::swap(d_nom_nominal, d_cand2_nominal);

    // 10. compute safe cost for the nominal model, P threads (using the rng of the first P rollouts)
    int safeGrd = (P + blk - 1) / blk;
    PRMPPIcomputeSafeCostKernel<<<safeGrd, blk>>>(envConfig, mppiConfig, state, d_nom_nominal, d_thetas, d_candCosts, d_rng);

    // 11. Check safety: if any averaged over p safety costs is negative, then nominal model is unsafe, and copy d_rob_nominal into d_nom_nominal (and
    // set useNomPlan to false, otherwise true)
    std::vector<float> safeCosts(P);

    CUDA_CHECK(cudaMemcpy(safeCosts.data(), d_candCosts, P * sizeof(float), cudaMemcpyDeviceToHost));

    useNomPlan = true;

    for (float c : safeCosts)
        if (c > 0.0f)
        {
            useNomPlan = false;
            break;
        }

    std::cout << "Averaged safety cost of r_nom_nominal: ";
    for (float c : safeCosts)
        std::cout << c << " ";
    std::cout << "\n";

    if (useNomPlan)
        std::cout << "USING NOMINAL PLAN\n";
    else
        std::cout << "USING ROBUST PLAN\n";

    // we DO NOT copy the robust plan into the nominal even if nominal is unsafe, simply send the first action of robust
    // if (!useNomPlan)
    //     CUDA_CHECK(cudaMemcpy(
    //         d_nom_nominal,
    //         d_rob_nominal,
    //         T * ACTION_DIM * sizeof(float),
    //         cudaMemcpyDeviceToDevice));

    // 12. Copy d_nom_nominal and d_rob_nominal into the corresponding host vectors; copy the first step of the chosen nominal into outAction

    CUDA_CHECK(cudaMemcpy(h_nom_nominal.data(), d_nom_nominal, T * ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_rob_nominal.data(), d_rob_nominal, T * ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    if (useNomPlan)
        std::copy(h_nom_nominal.begin(), h_nom_nominal.begin() + ACTION_DIM, outAction);
    else
        std::copy(h_rob_nominal.begin(), h_rob_nominal.begin() + ACTION_DIM, outAction);

    // 13. Update inverse temperatures

    std::array<float, 3> nu;
    std::array<float, 3> minCosts;

    CUDA_CHECK(cudaMemcpy(nu.data(), d_nu, nu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(minCosts.data(), d_minCosts, minCosts.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\tSubmitted action: \t";

    for (int d = 0; d < ACTION_DIM; d++)
        std::cout << outAction[d] << " ";

    std::cout << "\n\tNu values for nom + full_cost, rob + full_cost, rob + safe_cost:\t";
    for (float n : nu)
        std::cout << n << " ";
    std::cout << "\n";
    std::cout << "\tMin costs for nom + full_cost, rob + full_cost, rob + safe_cost:\t";
    for (float n : minCosts)
        std::cout << n << " ";
    std::cout << "\n";

    auto adaptTemperature = [&](float& invTemp, float usedNu, std::string name)
    {
        if (usedNu > max_nu * (float)N && invTemp > 0.01f)
        {
            std::cout << "\tdecreasing " << name << " inverse temperature from " << invTemp << " to ";
            invTemp *= 0.9f;
            std::cout << invTemp << "\n";
        }
        else if (usedNu < min_nu * (float)N && invTemp < 100.0f)
        {
            std::cout << "\tincreasing " << name << " inverse temperature from " << invTemp << " to ";
            invTemp *= 1.2f;
            std::cout << invTemp << "\n";
        }
    };

    adaptTemperature(invTempNomFull, nu[0], "nom/full");
    adaptTemperature(invTempRobFull, nu[1], "rob/full");
    adaptTemperature(invTempRobSafe, nu[2], "rob/safe");

    std::cout << "PRMPPI DONE" << std::endl;
}
