#include "controllers.h"
#include "engine.h"
#include "config.h"
#include "state.h"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <stdexcept>
#include <variant>
#include <type_traits>
#include <cub/cub.cuh>
#include <format>

// Construction
MPPIController::MPPIController(
    const EnvironmentConfig& c,
    const MPPIConfig& mc,
    const VerifConfig& vC,
    float* d_trackPoints,
    int s,
    std::optional<std::vector<float>> nominal)
{
    std::cout << "MPPI INIT" << std::endl;
    name = "MPPI";
    envConfig = c;
    mppiConfig = mc;
    verifConfig = vC;
    seed = s;

    h_trackPoints = c.trackPoints;
    envConfig.trackPoints = d_trackPoints;

    h_belief = std::to_array(mc.initBelief);

    if (nominal)
    {
        if ((int) nominal->size() != (N_MODELS + 1) * mc.nTimesteps * DIM)
            throw std::runtime_error(std::format("Invalid MPPI construction: expected nominal size %d, got %d", (N_MODELS + 1) * mc.nTimesteps * DIM, nominal->size()));

        h_nominal = nominal.value();
    }
    else
        h_nominal.assign((N_MODELS + 1) * mc.nTimesteps * DIM, 0.0f);

    failCountNew = failCountOld = failCount = { 0u, 0u };
}

MPPIController::~MPPIController()
{
    freeDevice();
}

void MPPIController::allocDevice()
{
    if (deviceReady)
        return;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;

    // Single authoritative state (uploaded each call)
    CUDA_CHECK(cudaMalloc(&d_belief, N_MODELS * sizeof(float)));

    // Noise: (nModels+1, T, N, dim)
    CUDA_CHECK(cudaMalloc(&d_noise, (N_MODELS + 1) * T * N * DIM * sizeof(float)));

    // Costs
    CUDA_CHECK(cudaMalloc(&d_costs, (N_MODELS + 1) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_branchUsed, N_MODELS * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_branchTime, N_MODELS * N * sizeof(int)));

    // Nominal action: (nModels+1, T, dim)
    CUDA_CHECK(cudaMalloc(&d_nominal, (N_MODELS + 1) * T * DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_nominal, 0, (N_MODELS + 1) * T * DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_prevnominal, (N_MODELS + 1) * T * DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_prevnominal, 0, (N_MODELS + 1) * T * DIM * sizeof(float)));

    // Min-reduction / masked costs / nu
    CUDA_CHECK(cudaMalloc(&d_minCosts, (N_MODELS + 1) * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maskedCosts, (N_MODELS + 1) * T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nu, (N_MODELS + 1) * sizeof(float)));

    // Per-sample RNG states
    CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_rng, 0, N * sizeof(curandState)));

    // Verification side
    CUDA_CHECK(cudaMalloc(&d_failCountOld, 2 * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_failCountNew, 2 * sizeof(uint)));

    CUDA_CHECK(cudaMalloc(&d_verif_rng, verifConfig.nVerifSamples * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_verif_rng, 0, verifConfig.nVerifSamples * sizeof(curandState)));

    // Initialise RNG
    int blk = 256;
    int grd = (N + blk - 1) / blk;
    initRNGKernel << <grd, blk >> > (d_rng, seed, N);

    int grdVerif = (verifConfig.nVerifSamples + blk - 1) / blk;
    initRNGKernel << <grdVerif, blk >> > (d_verif_rng, seed + 1, verifConfig.nVerifSamples);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

    // Upload initial nominal
    CUDA_CHECK(cudaMemcpy(
        d_nominal,
        h_nominal.data(),
        h_nominal.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    // Temp storage size for reducing N elements
    cub::DeviceReduce::Min(
        nullptr,
        temp_storage_bytes,
        (const float*) nullptr,
        (float*) nullptr,
        N);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    deviceReady = true;
}

void MPPIController::freeDevice()
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
    safe_free(d_costs);
    safe_free(d_branchUsed);
    safe_free(d_branchTime);

    safe_free(d_nominal);
    safe_free(d_prevnominal);

    safe_free(d_minCosts);
    safe_free(d_maskedCosts);
    safe_free(d_nu);

    safe_free(d_rng);
    safe_free(d_temp_storage);

    safe_free(d_failCountOld);
    safe_free(d_failCountNew);
    safe_free(d_verif_rng);

    temp_storage_bytes = 0;
    deviceReady = false;
}

void MPPIController::getControl(
    int agent,
    const SimState& state,
    float* outAction)
{
    if (!engine)
        throw std::runtime_error("MPPI: engine not set");

    if (!deviceReady)
        allocDevice();

    std::cout << "MPPI BEGIN" << std::endl;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;

    int blk = 256;
    int grd = (N + blk - 1) / blk;

    // now, belief update is done in engine

    std::cout << "MPPI belief: ";
    for (float v : h_belief)
        std::cout << v << " ";
    std::cout << "\n";

    // 1. upload belief (now, current state is passed as argument to the kernels)
    CUDA_CHECK(cudaMemcpy(d_belief, h_belief.data(), N_MODELS * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_failCountOld, 0, 2 * sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_failCountNew, 0, 2 * sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_nu, 0, (N_MODELS + 1) * sizeof(float)));

    // 2. generate all noise at once: (nModels+1, T, N, dim)
    generateNoiseKernel << <grd, blk >> > (
        d_noise,
        d_rng,
        mppiConfig.samplingNoise,
        T, N);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 3. rollout
    fullRolloutKernel << <grd, blk >> > (
        agent,
        envConfig,
        mppiConfig,
        state,
        d_belief,
        d_nominal,
        d_noise,
        d_costs,
        d_branchUsed,
        d_branchTime,
        d_rng);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 4. build masked costs
    int maskTotal = (N_MODELS + 1) * T * N;
    int maskGrd = (maskTotal + blk - 1) / blk;

    buildMaskedCostsKernel << <maskGrd, blk >> > (
        d_costs,
        d_branchUsed,
        d_branchTime,
        d_belief,
        d_maskedCosts,
        N, T);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 4.5 reduce mins over each (branch,time) row
    minReduceCUB(
        d_maskedCosts,
        d_minCosts,
        N,
        (N_MODELS + 1) * T,
        d_temp_storage,
        temp_storage_bytes);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 5. Weighted average update
    int wGrid = (N_MODELS + 1) * T * DIM;
    weightedAverageKernelUnified << <wGrid, blk, 2 * blk * sizeof(float) >> > (
        d_costs,
        d_minCosts,
        d_noise,
        d_branchUsed,
        d_branchTime,
        d_belief,
        d_nominal,
        mppiConfig.invTemperature,
        N, T,
        d_nu);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 5.5 Clamp nominal action sequences
    int nVecs = (N_MODELS + 1) * T;
    int clampGrd = (nVecs + blk - 1) / blk;
    clampNominalKernel << <clampGrd, blk >> > (
        d_nominal,
        envConfig.maxAccel[agent],
        T);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 5.75 Verification
    int verifGrd = (verifConfig.nVerifSamples + blk - 1) / blk;

    verifyNominalFailureKernel << <verifGrd, blk >> > (
        agent,
        verifConfig.nVerifSamples,
        envConfig,
        mppiConfig,
        verifConfig.horizon,
        state,
        d_belief,
        d_nominal,
        d_verif_rng,
        d_failCountNew);

    CUDA_CHECK(cudaMemcpy(
        failCountNew.data(),
        d_failCountNew,
        2 * sizeof(uint),
        cudaMemcpyDeviceToHost));

    verifyNominalFailureKernel << <verifGrd, blk >> > (
        agent,
        verifConfig.nVerifSamples,
        envConfig,
        mppiConfig,
        verifConfig.horizon,
        state,
        d_belief,
        d_prevnominal,
        d_verif_rng,
        d_failCountOld);

    CUDA_CHECK(cudaMemcpy(
        failCountOld.data(),
        d_failCountOld,
        2 * sizeof(uint),
        cudaMemcpyDeviceToHost));

    computeCertifiedLoss();
    useNewPlan = certifiedLoss < verifConfig.maxEps;

    if (useNewPlan)
        std::cout << "USING NEW PLAN\n";
    else
        std::cout << "USING PREVIOUS PLAN\n";

    failCount = useNewPlan ? failCountNew : failCountOld;
    computeEpsilon();

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 6. Download chosen nominal
    if (useNewPlan)
    {
        CUDA_CHECK(cudaMemcpy(
            h_nominal.data(),
            d_nominal,
            h_nominal.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(
            h_nominal.data(),
            d_prevnominal,
            h_nominal.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    std::vector<float> hostMin((N_MODELS + 1) * T), nu(N_MODELS + 1);
    CUDA_CHECK(cudaMemcpy(
        hostMin.data(),
        d_minCosts,
        (N_MODELS + 1) * T * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        nu.data(),
        d_nu,
        (N_MODELS + 1) * sizeof(float),
        cudaMemcpyDeviceToHost));

    int predTheta = findConfident(h_belief.data(), mppiConfig.minConfidence);

    for (int d = 0; d < DIM; d++)
        outAction[d] = h_nominal[(predTheta + 1) * T * DIM + d];

    if (predTheta == -1)
        std::cout << "Submitting nominal action: ";
    else
        std::cout << "Submitting action for predicted model " << predTheta << ": ";

    for (int d = 0; d < DIM; d++)
        std::cout << outAction[d] << " ";
    std::cout << std::endl;

    float usedNu = nu[predTheta + 1];

    std::cout << "Minimum cost (for submitted action): "
        << hostMin[(predTheta + 1) * T]
        << std::endl;

    std::cout << "Sum of computed sample costs w_k (nu) (for submitted action): "
        << usedNu << "\n";

    if (usedNu > max_nu * (float) N)
    {
        std::cout << "\tdecreasing inverse temperature from "
            << mppiConfig.invTemperature << " to ";
        mppiConfig.invTemperature *= 0.9f;
        std::cout << mppiConfig.invTemperature << "\n";
    }
    else if (usedNu < min_nu * (float) N)
    {
        std::cout << "\tincreasing inverse temperature from "
            << mppiConfig.invTemperature << " to ";
        mppiConfig.invTemperature *= 1.2f;
        std::cout << mppiConfig.invTemperature << "\n";
    }

    std::cout << "Verification samples: failed "
        << failCountNew[0] + failCountNew[1]
        << " out of " << verifConfig.nVerifSamples
        << " (collision: " << failCountNew[0]
        << "; outside: " << failCountNew[1]
        << ")" << std::endl;

    // 7. Shift nominal action sequence left by one timestep (warm start)
    for (int t = 0; t < T - 1; t++)
        for (int d = 0; d < DIM; d++)
            h_nominal[t * DIM + d] = h_nominal[(t + 1) * DIM + d];

    for (int d = 0; d < DIM; d++)
        h_nominal[(T - 1) * DIM + d] = 0.0f;

    if (predTheta >= 0)
    {
        for (int t = 0; t < T - 1; t++)
            for (int d = 0; d < DIM; d++)
                h_nominal[((predTheta + 1) * T + t) * DIM + d] =
                h_nominal[((predTheta + 1) * T + (t + 1)) * DIM + d];

        for (int d = 0; d < DIM; d++)
            h_nominal[((predTheta + 1) * T + T - 1) * DIM + d] = 0.0f;
    }

    // Upload shifted nominal back to device and keep prevnominal in sync
    CUDA_CHECK(cudaMemcpy(
        d_nominal,
        h_nominal.data(),
        h_nominal.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_prevnominal,
        d_nominal,
        h_nominal.size() * sizeof(float),
        cudaMemcpyDeviceToDevice));

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    std::cout << "MPPI DONE" << std::endl;
}

void MPPIController::computeCertifiedLoss()
{
    uint totFailOld = failCountOld[0] + failCountOld[1];
    uint totFailNew = failCountNew[0] + failCountNew[1];

    int n = verifConfig.nVerifSamples;

    double eps1 = clopperPearsonUpperBound(totFailNew, n, verifConfig.beta / 2.0);
    double eps2 = clopperPearsonLowerBound(totFailOld, n, verifConfig.beta / 2.0);

    certifiedLoss = eps1 - eps2;

    std::cout << std::fixed << std::setprecision(5)
        << "Certified loss: " << certifiedLoss
        << " eps1 " << eps1
        << " eps2 " << eps2
        << " (totFailOld totFailNew n "
        << totFailOld << " " << totFailNew << " " << n << ")\n";
}

void MPPIController::computeEpsilon()
{
    uint totFail = failCount[0] + failCount[1];
    int n = verifConfig.nVerifSamples;

    epsilonPartial = clopperPearsonUpperBound(totFail, n, verifConfig.beta);
    epsilon = epsilonPartial + verifConfig.horizon * verifConfig.maxEps;

    std::cout << std::fixed << std::setprecision(5)
        << "For fail count " << totFail
        << " among " << n
        << " found epsilon = " << epsilon
        << " partial " << epsilonPartial
        << std::endl;
}
