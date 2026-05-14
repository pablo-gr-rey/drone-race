#include "controllers.h"
#include "engine.h"
#include "config.h"
#include "device_config.cuh"
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

// Construction
MPPIController::MPPIController(const EnvironmentConfig& c, const MPPIConfig& mc)
{
    std::cout << "MPPI INIT" << std::endl;
    name = "MPPI";
    envConfig = &c;
    mppiCfg = mc;

    deviceEnvConfig = DeviceEnvironmentConfig(c);
    deviceMPPIConfig = DeviceMPPIConfig(mc);

    h_nominal.resize(mc.nTimesteps * c.dim, 0.0f);

    std::visit([&](auto&& opp)
        {
            using O = std::decay_t<decltype(opp)>;

            if constexpr (std::is_same_v<O, DummyConfig>)
                oppModel = OpponentModelType::Dummy;
            else if constexpr (std::is_same_v<O, PIDConfig>)
            {
                oppModel = OpponentModelType::PID;
                oppPidParams = opp;
            }
        }, mc.opponent);
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
    if (deviceReady) return;

    const auto& c = *envConfig;
    int N = mppiCfg.nSamples;
    int T = mppiCfg.nTimesteps;

    // Single authoritative state (uploaded each call)
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t) c.nAgents * c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_speed, (size_t) c.nAgents * c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, c.nAgents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_laps, c.nAgents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_currentGates, c.nAgents * sizeof(int)));

    // Final state output buffers (one per sample, optional / for debug)
    CUDA_CHECK(cudaMalloc(&d_sampPos, (size_t) N * c.nAgents * c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sampSpeed, (size_t) N * c.nAgents * c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sampS, (size_t) N * c.nAgents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sampLaps, (size_t) N * c.nAgents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sampGates, (size_t) N * c.nAgents * sizeof(int)));

    // Noise: (T, N, dim)
    CUDA_CHECK(cudaMalloc(&d_noise, (size_t) T * N * c.dim * sizeof(float)));

    // Costs: (N,)
    CUDA_CHECK(cudaMalloc(&d_costs, N * sizeof(float)));

    // Nominal action: (T, dim)
    CUDA_CHECK(cudaMalloc(&d_nominal, T * c.dim * sizeof(float)));

    // Scalar for min reduction
    CUDA_CHECK(cudaMalloc(&d_minCost, sizeof(float)));

    // Per-sample RNG states
    CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(curandState)));

    // Initialise RNG
    int blk = 256;
    int grd = (N + blk - 1) / blk;
    initRNGKernel << <grd, blk >> > (d_rng, 12345ULL, N);
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

void MPPIController::uploadTrack()
{
    if (d_trackPts)
        return;                 // already uploaded
    if (!engine)
        throw std::runtime_error("MPPI: engine not set, cannot upload track");

    size_t bytes = engine->envConfig.trackPoints.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_trackPts, bytes));
    CUDA_CHECK(cudaMemcpy(d_trackPts, engine->envConfig.trackPoints.data(),
        bytes, cudaMemcpyHostToDevice));
}

void MPPIController::freeDevice()
{
    auto safe_free = [](auto*& p) {
        if (p)
        {
            cudaFree(p);
            p = nullptr;
        }
        };

    safe_free(d_pos);
    safe_free(d_speed);
    safe_free(d_S);
    safe_free(d_laps);
    safe_free(d_currentGates);
    safe_free(d_sampPos);
    safe_free(d_sampSpeed);
    safe_free(d_sampS);
    safe_free(d_sampLaps);
    safe_free(d_sampGates);
    safe_free(d_noise);
    safe_free(d_costs);
    safe_free(d_nominal);
    safe_free(d_minCost);
    safe_free(d_trackPts);
    safe_free(d_rng);
    safe_free(d_temp_storage);

    temp_storage_bytes = 0;

    deviceReady = false;
}

void MPPIController::reset()
{
    std::fill(h_nominal.begin(), h_nominal.end(), 0.0f);
    if (d_nominal)
        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(),
            h_nominal.size() * sizeof(float),
            cudaMemcpyHostToDevice));
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
    float* outAction)
{
    if (!engine)
        throw std::runtime_error("MPPI: engine not set");

    // Lazy allocation (needs engine pointer for track data)
    allocDevice();
    uploadTrack();

    const auto& c = *envConfig;
    int N = mppiCfg.nSamples;
    int T = mppiCfg.nTimesteps;
    int dim = c.dim;
    int nTP = c.nTrackSamples;

    int blk = 256;
    int grd = (N + blk - 1) / blk;

    // 1. upload current state
    CUDA_CHECK(cudaMemcpy(d_pos, pos, (size_t) c.nAgents * c.dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_speed, speed, (size_t) c.nAgents * c.dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, S, c.nAgents * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_laps, laps, c.nAgents * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_currentGates, currentGates, c.nAgents * sizeof(int), cudaMemcpyHostToDevice));

    // 2. generate all noise at once: (T, N, dim)
    generateNoiseKernel << <grd, blk >> > (
        d_noise, d_rng,
        mppiCfg.samplingNoise,
        T, N, dim);

    // 3. Full rollout - single kernel, each thread = one sample
    //    Each thread loads the shared initial state into registers,
    //    loops over T timesteps (dynamics + opponent prediction + cost),
    //    then writes out total cost and final state
    fullRolloutKernel << <grd, blk >> > (
        agent,
        deviceEnvConfig,
        deviceMPPIConfig,
        oppModel, oppPidParams,
        d_pos, d_speed, d_S, d_laps, d_currentGates,         // initial state (broadcast by reads)
        d_nominal,                      // (T, dim)
        d_noise,                        // (T, N, dim)
        d_costs,                        // (N,) output: total cost per sample
        d_sampPos, d_sampSpeed, d_sampS, d_sampLaps, d_sampGates, // (N, ...) output: final states
        d_rng,
        d_trackPts, nTP,
        N);

    // 4. find minimum cost (parallel reduction)
    float initMin = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_minCost, &initMin, sizeof(float), cudaMemcpyHostToDevice));

    // int rGrid = (N + blk * 2 - 1) / (blk * 2);
    // minReduceKernel << <rGrid, blk, blk * sizeof(float) >> > (
    //     d_costs, d_minCost, N);
    minReduceCUB(d_costs, d_minCost, N, d_temp_storage, temp_storage_bytes);

    float hostMin;
    CUDA_CHECK(cudaMemcpy(&hostMin, d_minCost, sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Weighted average of noise -> update nominal action
    //    One block per (timestep × dim) entry.
    int wGrid = T * dim;
    weightedAverageKernel << <wGrid, blk, 2 * blk * sizeof(float) >> > (
        d_costs, d_noise, d_nominal,
        hostMin, mppiCfg.invTemperature,
        N, T, dim);

    // 6. Download updated nominal action sequence
    std::vector<float> buf(T * dim);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_nominal, T * dim * sizeof(float), cudaMemcpyDeviceToHost));

    // First timestep's action is the output
    for (int d = 0; d < dim; d++)
        outAction[d] = buf[d];

    // printf("Minimum cost: %f\n", hostMin);
    // for (int d = 0; d < dim; d++)
    //     printf("%.5f", outAction[d]);
    // printf("\n");

    // 7. Shift nominal action sequence left by one timestep (warm-start for next call)
    h_nominal.assign(buf.begin(), buf.end());

    for (int t = 0; t < T - 1; t++)
        for (int d = 0; d < dim; d++)
            h_nominal[t * dim + d] = h_nominal[(t + 1) * dim + d];

    // Last timestep becomes zero
    for (int d = 0; d < dim; d++)
        h_nominal[(T - 1) * dim + d] = 0.0f;

    // Upload shifted nominal back to device for next call
    CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(),
        h_nominal.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaGetLastError());
}
