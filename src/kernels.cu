#include "config.h"
#include "state.h"
#include "track.cuh"
#include "costs.cuh"
#include "opponent_models.cuh"
#include "kernels.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>

// RNG init
__global__ void initRNGKernel(curandState* states,
    unsigned long long seed, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

// Noise generation
__global__ void generateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int T, int N, int dim)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    curandState local = rng[s];
    for (int t = 0; t < T; t++)
        for (int d = 0; d < dim; d++)
            noise[(t * N + s) * dim + d] = curand_normal(&local) * stddev;
    rng[s] = local;
}

// Fused rollout step
__global__ void fullRolloutKernel(
    int controlAgent,
    const DeviceEnvironmentConfig envConfig,
    const DeviceMPPIConfig mc,
    OpponentModelType oppModel,
    const PIDConfig oppPid,
    const float* __restrict__ initPhys,
    const float* __restrict__ initS,
    const int* __restrict__ initLaps,
    const int* __restrict__ initGates,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    float* __restrict__ finalPhys,
    float* __restrict__ finalS,
    int* __restrict__ finalLaps,
    int* __restrict__ finalGates,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts,
    int nTP, int N)
{
    // TODO: this does not handle gates!!
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;

    // ── Load initial state into registers ────────────────────────────
    float phys[MAX_PHYS_DIM];
    float S[MAX_AGENTS];
    int laps[MAX_AGENTS];
    int currentGates[MAX_AGENTS];

    float prevPos[MAX_AGENTS * MAX_DIM];

    for (int i = 0; i < envConfig.physDim; i++)
        phys[i] = initPhys[i];
    for (int a = 0; a < envConfig.nAgents; a++)
    {
        S[a] = initS[a];
        laps[a] = initLaps[a];
        currentGates[a] = initGates[a];
    }

    curandState rng = rngStates[s];
    float cost = 0.0f;
    bool stop = false;

    // ── Main rollout loop ────────────────────────────────────────────
    for (int t = 0; t < mc.nTimesteps && !stop; t++)
    {
        // 0. Update prevPos
        for (int i = 0; i < envConfig.nAgents * envConfig.dim; i++)
            prevPos[i] = phys[2 * i];

        // 1. Build actions
        float actions[MAX_ACTION_DIM];
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            if (a == controlAgent)
            {
                for (int d = 0; d < envConfig.dim; d++)
                    actions[a * envConfig.dim + d] = nominal[t * envConfig.dim + d] + noise[(t * N + s) * envConfig.dim + d];
            }
            else
                predictOpponent(oppModel, a, phys, S, laps, currentGates, envConfig, oppPid, trackPts, nTP, actions + a * envConfig.dim);
        }

        // TODO: cap actions properly (L_2 norm instead of L_inf)
        // 2. Clamp + action noise
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            float sqNorm = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
                sqNorm += actions[a * envConfig.dim + d] * actions[a * envConfig.dim + d];

            float factor = 1.0f;
            if (sqNorm > envConfig.maxAccel[a] * envConfig.maxAccel[a])
                factor = envConfig.maxAccel[a] / sqrtf(sqNorm);

            for (int d = 0; d < envConfig.dim; d++)
            {
                float& v = actions[a * envConfig.dim + d];
                v *= factor;
                v += curand_normal(&rng) * envConfig.actionNoiseLevel;
            }
        }

        // 3. Integrate position
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
                setPos(phys, a, d, envConfig.dim, getPos(phys, a, d, envConfig.dim) + envConfig.dt * getVel(phys, a, d, envConfig.dim));

        // 4. Integrate velocity
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
                setVel(phys, a, d, envConfig.dim, getVel(phys, a, d, envConfig.dim) + envConfig.dt * actions[a * envConfig.dim + d]);

        // 5. Cap speed
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            float spd = agentSpeed(phys, a, envConfig.dim);
            if (spd > envConfig.maxSpeed[a])
            {
                float sc = envConfig.maxSpeed[a] / spd;
                for (int d = 0; d < envConfig.dim; d++)
                    setVel(phys, a, d, envConfig.dim, getVel(phys, a, d, envConfig.dim) * sc);
            }
        }

        // 6. State noise
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
            {
                setPos(phys, a, d, envConfig.dim, getPos(phys, a, d, envConfig.dim) + curand_normal(&rng) * envConfig.posNoiseLevel);
                setVel(phys, a, d, envConfig.dim, getVel(phys, a, d, envConfig.dim) + curand_normal(&rng) * envConfig.speedNoiseLevel);
            }

        // 7. Track S / laps / gates update
        for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        {
            // TODO: we could avoid copying into pos (using stride 2)
            float pos[MAX_DIM];
            for (int d = 0; d < envConfig.dim; d++)
                pos[d] = getPos(phys, iAgent, d, envConfig.dim);

            float dist;
            // float newSa = projectOnTrack(trackPts, nTP, cfg.dim, pos, nullptr, dist);
            float newSa = fastProjectOnTrack(trackPts, nTP, envConfig.dim, pos, nullptr, dist, S[iAgent]);
            S[iAgent] = newSa;

            dist = trackBoundaryDist(envConfig.arenaMin, envConfig.arenaMax, pos, envConfig.dim);

            // one agent is outside: break (but still update other agents & costs)
            if (dist < 0.)
                stop = true;

            int nextGate = (currentGates[iAgent] + 1) % envConfig.nGates;
            float num = 0., denom = 0.;
            for (int d = 0; d < envConfig.dim; d++)
            {
                num += envConfig.gateVectors[nextGate * envConfig.dim + d] * (envConfig.gateCenters[nextGate * envConfig.dim + d] - prevPos[iAgent * envConfig.dim + d]);
                denom += envConfig.gateVectors[nextGate * envConfig.dim + d] * (pos[d] - prevPos[iAgent * envConfig.dim + d]);
            }

            // direction is inside the gate plan: cannot cross
            if (fabsf(denom) < 1e-10f)
                continue;

            float lambda = num / denom;
            // we cross if 0 <= lambda <= 1 and if the projection of the segment (x_t, x_t+1) on the gate plan (ie. (1 - lambda) * x_t + lambda * x_t+1) is at distance <= radius from the center
            // if we want to make sure we cross the gate in the right direction, we have to check num >= 0 (<=> denom > 0)

            if (lambda < 0.f || lambda > 1.f)
                continue;

            float sqDist = 0.;
            for (int d = 0; d < envConfig.dim; d++)
            {
                float dx = (1. - lambda) * prevPos[iAgent * envConfig.dim + d] + lambda * pos[d] - envConfig.gateCenters[nextGate * envConfig.dim + d];
                sqDist += dx * dx;
            }

            // std::cout.precision(5);
            // std::cout << std::fixed << "\tsqDist = " << sqDist << " sq radius " << envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate] << "\n";

            if (sqDist <= envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate])
            {
                currentGates[iAgent]++;
                if (currentGates[iAgent] == envConfig.nGates)
                {
                    currentGates[iAgent] = 0;
                    laps[iAgent]++;
                }
            }
        }

        // 8. Running cost
        cost += stateCost(controlAgent, phys, S, laps, currentGates, t, envConfig, mc, trackPts, nTP);
    }

    // ── Terminal cost ────────────────────────────────────────────────
    cost += finalCost(controlAgent, phys, S, laps, currentGates, envConfig, mc, trackPts, nTP);

    // ── Write outputs ────────────────────────────────────────────────
    totalCosts[s] = cost;

    // We don't actually need the final state for the MPPI update,
    // but if you want to inspect it for debugging:
    // (can be removed to save bandwidth)
    float* op = finalPhys + s * envConfig.physDim;
    float* oS = finalS + s * envConfig.nAgents;
    int* oL = finalLaps + s * envConfig.nAgents;
    int* oG = finalGates + s * envConfig.nAgents;

    for (int i = 0; i < envConfig.physDim; i++)
        op[i] = phys[i];
    for (int a = 0; a < envConfig.nAgents; a++)
    {
        oS[a] = S[a];
        oL[a] = laps[a];
        oG[a] = currentGates[a];
    }

    rngStates[s] = rng;
}

// cuda reduce min
void minReduceCUB(const float* __restrict__ d_costs,
    float* __restrict__ d_minCost,
    int N,
    void* __restrict__ d_temp_storage,
    size_t temp_storage_bytes)
{
    CUDA_CHECK(cub::DeviceReduce::Min(
        d_temp_storage,
        temp_storage_bytes,
        d_costs,
        d_minCost,
        N
    ));
}

// weighted average
__global__ void weightedAverageKernel(
    const float* __restrict__ costs,
    const float* __restrict__ noise,
    float* __restrict__ nominal,
    float minCost, float invTemp,
    int N, int T, int dim)
{
    int td = blockIdx.x;
    if (td >= T * dim) return;
    int t = td / dim, d = td % dim;

    extern __shared__ float sh[];
    float* s_wn = sh;
    float* s_w = sh + blockDim.x;

    float wSum = 0.0f, wnSum = 0.0f;
    for (int s = threadIdx.x; s < N; s += blockDim.x)
    {
        float w = expf(-(costs[s] - minCost) / invTemp);
        wSum += w;
        wnSum += w * noise[(t * N + s) * dim + d];
    }
    s_wn[threadIdx.x] = wnSum;
    s_w[threadIdx.x] = wSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if ((int) threadIdx.x < stride)
        {
            s_wn[threadIdx.x] += s_wn[threadIdx.x + stride];
            s_w[threadIdx.x] += s_w[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && s_w[0] > 1e-30f)
        nominal[t * dim + d] += s_wn[0] / s_w[0];

    // if (threadIdx.x == 0 && td == 0)
    //     printf("Sum of w_k (nu): %f\n", s_w[0]);
}
