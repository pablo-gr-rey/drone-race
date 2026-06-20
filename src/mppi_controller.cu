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

    h_belief = std::to_array(envConfig.initBelief);

    if (nominal)
    {
        if ((int) nominal->size() != N_BRANCH_PLANS * mc.nTimesteps * DIM)
            throw std::runtime_error(std::format("Invalid MPPI construction: expected nominal size %d, got %d", N_BRANCH_PLANS * mc.nTimesteps * DIM, nominal->size()));

        h_nominal = nominal.value();
    }
    else
        h_nominal.assign(N_BRANCH_PLANS * mc.nTimesteps * DIM, 0.0f);

    failCountNew = failCountOld = failCount = { 0u, 0u };

    h_B = buildSplineMatrix();
}

MPPIController::~MPPIController()
{
    freeDevice();
}

std::vector<float> MPPIController::buildSplineMatrix()
{
    int M = mppiConfig.nKnots, T = mppiConfig.nTimesteps;

    std::vector<float> B(T * M, 0.0f);

    std::vector<int> tau(mppiConfig.knots, mppiConfig.knots + mppiConfig.nKnots);

    // precompute h(j) = tau_(j+1) - tau_j
    std::vector<float> h(M - 1);
    for (int j = 0; j < M - 1; j++)
        h[j] = float(tau[j + 1] - tau[j]);

    // for each basis vector e_i, compute the spline, and evaluate it to get the i-th column of B
    for (int basis = 0; basis < M; basis++)
    {
        std::vector<float> y(M, 0.0f);
        y[basis] = 1.0f;

        // Natural cubic spline second derivatives m[0...M-1] with m[0]=m[M-1]=0
        std::vector<float> m(M, 0.0f);

        if (M > 2)
        {
            const int N = M - 2; // number of interior unknowns

            std::vector<float> a(N), b(N), c(N), d(N);      // coefficients of the polynomials

            for (int i = 0; i < N; ++i)
            {
                const int j = i + 1; // interior knot index
                a[i] = h[j - 1];
                b[i] = 2.0f * (h[j - 1] + h[j]);
                c[i] = h[j];
                d[i] = 6.0f * ((y[j + 1] - y[j]) / h[j] - (y[j] - y[j - 1]) / h[j - 1]);
            }

            // Thomas algorithm
            for (int i = 1; i < N; ++i)
            {
                float w = a[i] / b[i - 1];
                b[i] -= w * c[i - 1];
                d[i] -= w * d[i - 1];
            }

            std::vector<float> x(N);
            x[N - 1] = d[N - 1] / b[N - 1];
            for (int i = N - 2; i >= 0; --i)
                x[i] = (d[i] - c[i] * x[i + 1]) / b[i];

            for (int i = 0; i < N; ++i)
                m[i + 1] = x[i];
        }

        // Evaluate spline at each dense time t = 0..T-1
        // Outside [tau_0, tau_{M-1}] we do simple endpoint hold
        int seg = 0;
        for (int t = 0; t < T; ++t)
        {
            float val = 0.0f;

            if (t <= tau.front())
                val = y.front();
            else if (t >= tau.back())
                val = y.back();
            else
            {
                while (!(tau[seg] <= t && t <= tau[seg + 1]))
                    ++seg;

                const float x0 = float(tau[seg]);
                const float x1 = float(tau[seg + 1]);
                const float hh = x1 - x0;
                const float A = (x1 - t) / hh;
                const float BB = (t - x0) / hh;

                val =
                    m[seg] * (A * A * A - A) * (hh * hh) / 6.0f +
                    m[seg + 1] * (BB * BB * BB - BB) * (hh * hh) / 6.0f +
                    y[seg] * A +
                    y[seg + 1] * BB;
            }

            B[t * M + basis] = val;
        }
    }

    return B;
}

void MPPIController::allocDevice()
{
    if (deviceReady)
        return;

    int N = mppiConfig.nSamples;
    int T = mppiConfig.nTimesteps;
    int M = mppiConfig.nKnots;

    // Spline matrix
    CUDA_CHECK(cudaMalloc(&d_B, M * T * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Single authoritative state (uploaded each call)
    CUDA_CHECK(cudaMalloc(&d_belief, N_TRUE_MODELS * sizeof(float)));

    // Noise
    CUDA_CHECK(cudaMalloc(&d_noise, N_BRANCH_PLANS * M * N * DIM * sizeof(float)));

    // Costs
    CUDA_CHECK(cudaMalloc(&d_minSplineCosts, N_BRANCH_PLANS * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, N_BRANCH_PLANS * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costsTrue, N_TRUE_MODELS * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_branchUsed, N_TRUE_MODELS * N * N_MODEL_FACTORS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_branchTime, N_TRUE_MODELS * N * N_MODEL_FACTORS * sizeof(int)));

    // Nominal action
    CUDA_CHECK(cudaMalloc(&d_splineNominal, N_BRANCH_PLANS * M * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tempSplineNominal, N_BRANCH_PLANS * M * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nominal, N_BRANCH_PLANS * T * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prevnominal, N_BRANCH_PLANS * T * DIM * sizeof(float)));

    // Min-reduction / masked costs / nu
    CUDA_CHECK(cudaMalloc(&d_minCosts, N_BRANCH_PLANS * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maskedCosts, N_BRANCH_PLANS * T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nu, N_BRANCH_PLANS * sizeof(float)));

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
    CUDA_CHECK(cudaMemset(d_splineNominal, 0, N_BRANCH_PLANS * M * DIM * sizeof(float)));       // TODO: initialize this properly
    CUDA_CHECK(cudaMemcpy(
        d_nominal,
        h_nominal.data(),
        h_nominal.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_prevnominal,
        h_nominal.data(),
        h_nominal.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

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

    safe_free(d_B);

    safe_free(d_belief);

    safe_free(d_noise);
    safe_free(d_costs);
    safe_free(d_costsTrue);
    safe_free(d_branchUsed);
    safe_free(d_branchTime);

    safe_free(d_splineNominal);
    safe_free(d_tempSplineNominal);
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
    int M = mppiConfig.nKnots;

    // now, belief update is done in engine

    std::cout << "MPPI belief: ";
    for (float v : h_belief)
        std::cout << v << " ";
    std::cout << "\n";

    // 1. upload belief (now, current state is passed as argument to the kernels)
    CUDA_CHECK(cudaMemcpy(d_belief, h_belief.data(), N_TRUE_MODELS * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_failCountOld, 0, 2 * sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_failCountNew, 0, 2 * sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_nu, 0, N_BRANCH_PLANS * sizeof(float)));

    int blk = 256;
    int grd = (N + blk - 1) / blk;

    // 2. generate all noise at once: (nModels+1, M or T, N, dim)
    generateNoiseKernel << <grd, blk >> > (
        d_noise,
        d_rng,
        mppiConfig.samplingNoise,
        USE_SPLINES ? M : T,
        N);

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
        USE_SPLINES ? d_splineNominal : d_nominal,
        d_noise,
        d_B,
        d_costsTrue,
        d_branchUsed,
        d_branchTime,
        d_rng);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 3.5 Compute costs from costsTrue
    int aggTotal = N_BRANCH_PLANS * N;
    int aggGrd = (aggTotal + blk - 1) / blk;
    aggregateBranchCostsKernel << <aggGrd, blk >> > (
        d_costsTrue,
        d_belief,
        d_costs,
        N);

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 4. Compute the minimum cost for each sample (with masking ie. only considering valid samples)

    // 5. Weighted average update
    if (USE_SPLINES)
    {
        int nMinBlocks = N_BRANCH_PLANS * T;
        computeMaskedMinSplineCostsKernel << <nMinBlocks, blk, blk * sizeof(float) >> > (
            d_costs,
            d_branchUsed,
            d_branchTime,
            d_belief,
            d_B,
            d_minSplineCosts,
            mppiConfig);

        int wGrid = N_BRANCH_PLANS * M * DIM;
        weightedAverageSplineKernelUnified << <wGrid, blk, 2 * blk * sizeof(float) >> > (
            d_costs,
            d_minSplineCosts,
            d_noise,
            d_branchUsed,
            d_branchTime,
            d_belief,
            d_splineNominal,
            mppiConfig,
            d_B,
            d_nu
            );
    }
    else
    {
        int nMinBlocks = N_BRANCH_PLANS * T;
        computeMaskedMinCostsKernel << <nMinBlocks, blk, blk * sizeof(float) >> > (
            d_costs,
            d_branchUsed,
            d_branchTime,
            d_belief,
            d_minCosts,
            N,
            T);

#ifdef DEBUG
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        int wGrid = N_BRANCH_PLANS * T * DIM;
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
    }

#ifdef DEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // 5.25 If we use splines, then re-compute actual nominal
    // 5.5 Clamp nominal
    // 5.626 If we use splines, then shift spline nominal for next step warm-start
    if (USE_SPLINES)
    {
        int interpGrd = (N_BRANCH_PLANS * T * DIM + blk - 1) / blk;
        interpolateSplineKernel << < interpGrd, blk >> > (d_splineNominal, d_nominal, d_B, T, M);

#ifdef DEBUG
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        int clampGrd = (N_BRANCH_PLANS * T + blk - 1) / blk;
        clampNominalKernel << <clampGrd, blk >> > (d_splineNominal, envConfig.maxAccel[agent], M);
    }
    else
    {
        int clampGrd = (N_BRANCH_PLANS * T + blk - 1) / blk;
        clampNominalKernel << <clampGrd, blk >> > (
            d_nominal,
            envConfig.maxAccel[agent],
            T);
    }

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

    std::vector<float> hostMin(N_BRANCH_PLANS * T), nu(N_BRANCH_PLANS);
    CUDA_CHECK(cudaMemcpy(
        hostMin.data(),
        d_minCosts,
        N_BRANCH_PLANS * T * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        nu.data(),
        d_nu,
        N_BRANCH_PLANS * sizeof(float),
        cudaMemcpyDeviceToHost));

    // int predTheta = findConfident(h_belief.data(), mppiConfig.minConfidence);
    int predTheta[N_MODEL_FACTORS];
    findConfident(h_belief.data(), mppiConfig.minConfidence, predTheta);

    int branchIdx = flattenBranchIndex(predTheta);

    for (int d = 0; d < DIM; d++)
        outAction[d] = h_nominal[branchIdx * T * DIM + d];

    std::cout << "Chosen branch (-1=nominal, theta=committed to theta): ";
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        std::cout << (predTheta[k] - 1) << ' ';
    std::cout << "\n\tSubmitted action: \t";

    for (int d = 0; d < DIM; d++)
        std::cout << outAction[d] << " ";
    std::cout << std::endl;

    float usedNu = nu[branchIdx];

    std::cout << "Minimum cost (for submitted action): "
        << hostMin[branchIdx * T]
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
    // this ensures that the verification logic makes sense: if we stop optimizing (useNewPlan=false), then shifting corresponds
    // to simply time passing, such that the predicted plan matches the actual behavior

    int base = branchIdx * T * DIM;
    for (int t = 0; t < T - 1; t++)
        for (int d = 0; d < DIM; d++)
            h_nominal[base + t * DIM + d] = h_nominal[base + (t + 1) * DIM + d];

    for (int d = 0; d < DIM; d++)
        h_nominal[base + (T - 1) * DIM + d] = 0.0f;

    // 8. If we use splines, also shift the corresponding branch (while keeping the other)

    if (USE_SPLINES)
    {
        int shiftGrd = (M * DIM + blk - 1) / blk;

        CUDA_CHECK(cudaMemcpy(d_tempSplineNominal, d_splineNominal, N_BRANCH_PLANS * M * DIM * sizeof(float), cudaMemcpyDeviceToDevice));

        shiftSplineKernel << <shiftGrd, blk >> > (d_splineNominal, d_tempSplineNominal, d_B, mppiConfig, branchIdx);

        std::swap(d_tempSplineNominal, d_splineNominal);        // we don't need to copy, we can just alternatively use the buffers!

        // TODO: we probably should keep d_oldSplineNominal as well (well, it is just d_tempSplineNominal!)
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
        << ": totFailOld "
        << totFailOld << " (" << failCountOld[0] << "+" << failCountOld[1] << ") totFailNew " << totFailNew << " (" << failCountNew[0] << "+" << failCountNew[1] << "), n " << n << ")\n";
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
