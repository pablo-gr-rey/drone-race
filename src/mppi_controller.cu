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
MPPIController::MPPIController(const EnvironmentConfig& c, const MPPIConfig& mc, float* d_trackPoints, std::optional<std::vector<float>> nominal)
{
    std::cout << "MPPI INIT" << std::endl;
    name = "MPPI";
    envConfig = c;
    mppiConfig = mc;
    d_trackPts = d_trackPoints;

    // if (belief)
    // {
    //     if ((int) belief->size() != mc.nModels)
    //         throw std::runtime_error(std::format("Invalid MPPI construction: expected belief size %d, got %d", mc.nModels, belief->size()));
    //     h_belief = belief.value();
    // }
    // else
    //     h_belief.assign(mc.nModels, 1.0f / mc.nModels);
    h_belief.assign(mc.initBelief, mc.initBelief + mc.nModels);

    if (nominal)
    {
        if ((int) nominal->size() != (mc.nModels + 1) * mc.nTimesteps * envConfig.dim)
            throw std::runtime_error(std::format("Invalid MPPI construction: expected nominal size %d, got %d", (mc.nModels + 1) * mc.nTimesteps * envConfig.dim, nominal->size()));
        h_nominal = nominal.value();
    }
    else
        h_nominal.assign((mc.nModels + 1) * mc.nTimesteps * envConfig.dim, 0.);
}

MPPIController::~MPPIController()
{
    freeDevice();
}

// ═════════════════════════════════════════════════════════════════════
// Device memory management
// ═════════════════════════════════════════════════════════════════════
void MPPIController::allocDevice()
{
    if (deviceReady)
        return;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int dim = envConfig.dim;
    int nAgents = envConfig.nAgents;
    int nModels = mppiConfig.nModels;

    // Single authoritative state (uploaded each call)
    CUDA_CHECK(cudaMalloc(&d_pos, nAgents * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_speed, nAgents * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, nAgents * envConfig.nRacelines * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_laps, nAgents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_currentGates, nAgents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_belief, nModels * sizeof(float)));

    // Noise: (nModels+1, T, N, dim)
    CUDA_CHECK(cudaMalloc(&d_noise, (nModels + 1) * T * N * dim * sizeof(float)));

    // Costs
    CUDA_CHECK(cudaMalloc(&d_costs, N * sizeof(float)));

    // Nominal action: (nModels+1, T, dim)
    CUDA_CHECK(cudaMalloc(&d_nominal, (nModels + 1) * T * dim * sizeof(float)));

    // Scalar for min reduction
    CUDA_CHECK(cudaMalloc(&d_minCost, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nu, sizeof(float)));

    // Per-sample RNG states
    CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_rng, 0, N * sizeof(curandState)));  // otherwise, memory is flagged as unitialized even though initRng does get called

    // Initialise RNG
    int blk = 256;
    int grd = (N + blk - 1) / blk;
    initRNGKernel << <grd, blk >> > (d_rng, 12345ULL, N);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

    // Upload initial (zero) nominal action
    CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(),
        h_nominal.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    // allocate temp storage for min reduce

    // get temporary storage size for min reduce
    cub::DeviceReduce::Min(
        nullptr,
        temp_storage_bytes,
        (const float*) nullptr,
        (float*) nullptr,
        N
    );

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    deviceReady = true;
}


void MPPIController::freeDevice()
{
    auto safe_free = [](auto*& p) {
        if (p)
        {
            CUDA_CHECK(cudaFree(p));
            p = nullptr;
        }
        };

    safe_free(d_pos);
    safe_free(d_speed);
    safe_free(d_S);
    safe_free(d_laps);
    safe_free(d_currentGates);
    safe_free(d_belief);

    safe_free(d_noise);
    safe_free(d_costs);
    safe_free(d_nominal);
    safe_free(d_minCost);
    safe_free(d_nu);
    safe_free(d_rng);
    safe_free(d_temp_storage);

    temp_storage_bytes = 0;

    deviceReady = false;
}

// ═════════════════════════════════════════════════════════════════════
// getControl — the main MPPI routine
// ═════════════════════════════════════════════════════════════════════
void MPPIController::getControl(int agent,
    const float* pos,
    const float* speed,
    const float* S,
    const int* laps,
    const int* currentGates,
    float* outAction,
    std::normal_distribution<float>& /* nd */, std::mt19937& /* rng */,
    std::optional<std::vector<float>> pastAction,
    std::optional<std::vector<float>> pastPos,
    std::optional<std::vector<float>> pastVel,
    std::optional<std::vector<float>> pastS)
{
    if (!engine)
        throw std::runtime_error("MPPI: engine not set");

    // Lazy allocation (needs engine pointer for track data)
    if (!deviceReady)
        allocDevice();

    std::cout << "MPPI BEGIN" << std::endl;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int dim = envConfig.dim;
    int nTP = envConfig.nTrackSamples;
    int nModels = mppiConfig.nModels;

    int blk = 256;
    int grd = (N + blk - 1) / blk;

    // 0. Update belief
    if (pastAction)
    {
        // compute nominal actions
        std::vector<float> nomPidAction(mppiConfig.nModels * envConfig.dim);

        for (int thetaT = 0; thetaT < mppiConfig.nModels; thetaT++)
        {
            // std::cout << "computing PID action for model " << thetaT << std::endl;
            computePIDAction(1 - agent, pastPos->data(), pastVel->data(), pastS->data(), envConfig, mppiConfig.oppPid[thetaT], engine->trackPoints.data(), nomPidAction.data() + thetaT * envConfig.dim);
        }

        updateBelief(h_belief.data(), pastAction->data() + (1 - agent) * envConfig.dim, nomPidAction.data(), mppiConfig.oppPid, mppiConfig.nModels, envConfig.dim, envConfig.maxAccel[1 - agent]);
    }

    std::cout << "MPPI belief: ";
    for (float v : h_belief)
        std::cout << v << " ";
    std::cout << "\n";

    // 1. upload current state
    CUDA_CHECK(cudaMemcpy(d_pos, pos, (size_t) envConfig.nAgents * envConfig.dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_speed, speed, (size_t) envConfig.nAgents * envConfig.dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, S, envConfig.nAgents * envConfig.nRacelines * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_laps, laps, envConfig.nAgents * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_currentGates, currentGates, envConfig.nAgents * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_belief, h_belief.data(), nModels * sizeof(float), cudaMemcpyHostToDevice));

    // 2. generate all noise at once: (T, N, dim)
    generateNoiseKernel << <grd, blk >> > (
        d_noise, d_rng,
        mppiConfig.samplingNoise,
        T, N, dim, nModels);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 3. Full rollout - single kernel, each thread = one sample (nominal + all models)
    //    Each thread loads the shared initial state into registers,
    //    loops over T timesteps & nModels models (dynamics + belief update + cost per-model),
    //    then writes out total cost and final state
    fullRolloutKernel << <grd, blk >> > (
        agent,
        envConfig,
        mppiConfig,
        d_pos, d_speed, d_S, d_laps, d_currentGates, d_belief,        // initial state (broadcast by reads)
        d_nominal,                      // (T, dim)
        d_noise,                        // (T, N, dim)
        d_costs,                        // (N,) output: total cost per sample
        d_rng,
        d_trackPts,
        nTP,
        N);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 4. find minimum cost (parallel reduction)
    float initMin = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_minCost, &initMin, sizeof(float), cudaMemcpyHostToDevice));

    // int rGrid = (N + blk * 2 - 1) / (blk * 2);
    // minReduceKernel << <rGrid, blk, blk * sizeof(float) >> > (
    //     d_costs, d_minCost, N);
    minReduceCUB(d_costs, d_minCost, N, d_temp_storage, temp_storage_bytes);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    float hostMin;
    CUDA_CHECK(cudaMemcpy(&hostMin, d_minCost, sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Weighted average of noise -> update nominal action
    //    One block per (timestep × dim) entry.
    int wGrid = (nModels + 1) * T * dim;
    weightedAverageKernel << <wGrid, blk, 2 * blk * sizeof(float) >> > (
        d_costs, d_noise, d_nominal,
        hostMin, mppiConfig.invTemperature,
        nModels, N, T, dim, d_nu);

    float nu;
    CUDA_CHECK(cudaMemcpy(&nu, d_nu, sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 6. Download updated nominal action sequence
    // TODO: we don't need to copy everything, we could just copy the interesting action and do the shift on GPU
    CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, h_nominal.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int predTheta = findConfident(h_belief.data(), mppiConfig.nModels, mppiConfig.minConfidence);
    // if predTheta == -1, submit nominal general action; otherwise, submit nominal action corresponding to this hypothesis

    for (int d = 0; d < dim; d++)
        outAction[d] = h_nominal[(predTheta + 1) * T * envConfig.dim + d];

    if (predTheta == -1)
        std::cout << "Submitting nominal action: ";
    else
        std::cout << "Submitting action for predicted model " << predTheta << ": ";
    for (int d = 0; d < dim; d++)
        std::cout << outAction[d] << " ";
    std::cout << std::endl;

    std::cout << "Minimum cost: " << hostMin << std::endl;
    std::cout << "Sum of computed sample costs w_k (nu): " << nu << "\n";
    if (nu > max_nu * (float) N)
    {
        std::cout << "\tdecreasing inverse temperature from " << mppiConfig.invTemperature << " to ";
        mppiConfig.invTemperature *= 0.9f;
        std::cout << mppiConfig.invTemperature << "\n";
    }
    else if (nu < min_nu * (float) N)
    {
        std::cout << "\tincreasing inverse temperature from " << mppiConfig.invTemperature << " to ";
        mppiConfig.invTemperature *= 1.2f;
        std::cout << mppiConfig.invTemperature << "\n";
    }

    // 7. Shift nominal action sequence left by one timestep (warm-start for next call)
    for (int t = 0; t < T - 1; t++)
        for (int d = 0; d < dim; d++)
            h_nominal[t * dim + d] = h_nominal[(t + 1) * dim + d];
    // Last timestep becomes zero
    for (int d = 0; d < dim; d++)
        h_nominal[(T - 1) * dim + d] = 0.0f;

    // if we submitted a nominal action, shift this one as well
    if (predTheta >= 0)
    {
        for (int t = 0; t < T - 1; t++)
            for (int d = 0; d < dim; d++)
                h_nominal[((predTheta + 1) * T + t) * dim + d] = h_nominal[((predTheta + 1) * T + (t + 1)) * dim + d];
        for (int d = 0; d < dim; d++)
            h_nominal[((predTheta + 1) * T + T - 1) * dim + d] = 0.0f;
    }

    // Upload shifted nominal back to device for next call
    CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), h_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    std::cout << "MPPI DONE" << std::endl;
}
