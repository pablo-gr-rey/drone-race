#include "config.h"
#include "state.h"
#include "track.cuh"
#include "costs.cuh"
#include "opponent_models.cuh"
#include "reduction.cuh"

#include <curand_kernel.h>
#include <cfloat>

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

// Broadcast initial state to all samples
__global__ void broadcastKernel(const float* srcPhys,
    const float* srcS,
    const float* srcLaps,
    float* dstPhys, float* dstS, float* dstLaps,
    int physDim, int nAgents, int N)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    for (int i = 0; i < physDim; i++)  dstPhys[s * physDim + i] = srcPhys[i];
    for (int i = 0; i < nAgents; i++)  dstS[s * nAgents + i] = srcS[i];
    for (int i = 0; i < nAgents; i++)  dstLaps[s * nAgents + i] = srcLaps[i];
}

// Fused rollout step
__global__ void fullRolloutKernel(
    int controlAgent,
    const DeviceEnvironmentConfig cfg,
    const DeviceMPPIConfig mc,
    OpponentModelType oppModel,
    const PIDConfig oppPid,
    const float* __restrict__ initPhys,
    const float* __restrict__ initS,
    const float* __restrict__ initLaps,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    float* __restrict__ finalPhys,
    float* __restrict__ finalS,
    float* __restrict__ finalLaps,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts,
    int nTP, int N)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;

    // ── Load initial state into registers ────────────────────────────
    float phys[MAX_PHYS_DIM];
    float S[MAX_AGENTS];
    float laps[MAX_AGENTS];

    for (int i = 0; i < cfg.physDim; i++)  phys[i] = initPhys[i];
    for (int a = 0; a < cfg.nAgents; a++)  S[a] = initS[a];
    for (int a = 0; a < cfg.nAgents; a++)  laps[a] = initLaps[a];

    curandState rng = rngStates[s];
    float cost = 0.0f;
    bool stop = false;

    // ── Main rollout loop ────────────────────────────────────────────
    for (int t = 0; t < mc.nTimesteps && !stop; t++)
    {
        // 1. Build actions
        float actions[MAX_ACTION_DIM];
        for (int a = 0; a < cfg.nAgents; a++)
        {
            if (a == controlAgent)
            {
                for (int d = 0; d < cfg.dim; d++)
                    actions[a * cfg.dim + d] =
                    nominal[t * cfg.dim + d]
                    + noise[(t * N + s) * cfg.dim + d];
            }
            else
            {
                predictOpponent(oppModel, a, phys, S, laps,
                    cfg, oppPid, trackPts, nTP,
                    actions + a * cfg.dim);
            }
        }

        // 2. Clamp + action noise
        for (int a = 0; a < cfg.nAgents; a++)
            for (int d = 0; d < cfg.dim; d++)
            {
                float& v = actions[a * cfg.dim + d];
                v = fminf(fmaxf(v, -cfg.maxAccel[a]), cfg.maxAccel[a]);
                v += curand_normal(&rng) * cfg.actionNoiseLevel;
            }

        // 3. Integrate position
        for (int a = 0; a < cfg.nAgents; a++)
            for (int d = 0; d < cfg.dim; d++)
                setPos(phys, a, d, cfg.dim,
                    getPos(phys, a, d, cfg.dim)
                    + cfg.dt * getVel(phys, a, d, cfg.dim));

        // 4. Integrate velocity
        for (int a = 0; a < cfg.nAgents; a++)
            for (int d = 0; d < cfg.dim; d++)
                setVel(phys, a, d, cfg.dim,
                    getVel(phys, a, d, cfg.dim)
                    + cfg.dt * actions[a * cfg.dim + d]);

        // 5. Cap speed
        for (int a = 0; a < cfg.nAgents; a++)
        {
            float spd = agentSpeed(phys, a, cfg.dim);
            if (spd > cfg.maxSpeed[a])
            {
                float sc = cfg.maxSpeed[a] / spd;
                for (int d = 0; d < cfg.dim; d++)
                    setVel(phys, a, d, cfg.dim,
                        getVel(phys, a, d, cfg.dim) * sc);
            }
        }

        // 6. State noise
        for (int a = 0; a < cfg.nAgents; a++)
            for (int d = 0; d < cfg.dim; d++)
            {
                setPos(phys, a, d, cfg.dim,
                    getPos(phys, a, d, cfg.dim)
                    + curand_normal(&rng) * cfg.posNoiseLevel);
                setVel(phys, a, d, cfg.dim,
                    getVel(phys, a, d, cfg.dim)
                    + curand_normal(&rng) * cfg.speedNoiseLevel);
            }

        // 7. Track S / laps update
        for (int a = 0; a < cfg.nAgents; a++)
        {
            // TODO: cnul !! faut aller plus vite
            float pos[MAX_DIM];
            for (int d = 0; d < cfg.dim; d++)
                pos[d] = getPos(phys, a, d, cfg.dim);
            float dist;
            // float newSa = projectOnTrack(trackPts, nTP, cfg.dim, pos, nullptr, dist);
            float newSa = fastProjectOnTrack(trackPts, nTP, cfg.dim, pos, nullptr, dist, S[a]);

            // float dist2;
            // float newSa2 = fastProjectOnTrack(trackPts, nTP, cfg.dim, pos, nullptr, dist, S[a]);

            // if (abs(dist - dist2) > 1e-6)
            // {
            //     printf("WRONG fastdist for pos %f %f prevS %f : got (S, d) = (%f, %f) instead of (%f, %f)\n", pos[0], pos[1], S[a], newSa2, dist2, newSa, dist);
            // }

            if (newSa > S[a] + 0.5f) laps[a] -= 1.0f;
            if (newSa < S[a] - 0.5f) laps[a] += 1.0f;
            S[a] = newSa;

            // one agent is outside: break
            if (dist > cfg.trackWidth * 0.5f)
                stop = true;
        }

        // 8. Running cost
        cost += stateCost(controlAgent, phys, S, laps,
            t, cfg, mc, trackPts, nTP);
    }

    // ── Terminal cost ────────────────────────────────────────────────
    cost += finalCost(controlAgent, phys, S, laps,
        cfg, mc, trackPts, nTP);

    // ── Write outputs ────────────────────────────────────────────────
    totalCosts[s] = cost;

    // We don't actually need the final state for the MPPI update,
    // but if you want to inspect it for debugging:
    // (can be removed to save bandwidth)
    float* op = finalPhys + s * cfg.physDim;
    float* oS = finalS + s * cfg.nAgents;
    float* oL = finalLaps + s * cfg.nAgents;
    for (int i = 0; i < cfg.physDim; i++) op[i] = phys[i];
    for (int a = 0; a < cfg.nAgents; a++) { oS[a] = S[a]; oL[a] = laps[a]; }

    rngStates[s] = rng;
}

// ═════════════════════════════════════════════════════════════════════
// Float atomicMin via CAS
// ═════════════════════════════════════════════════════════════════════
__device__ inline void atomicMinFloat(float* addr, float val)
{
    int* ia = (int*) addr;
    int old = *ia, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(ia, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// ═════════════════════════════════════════════════════════════════════
// Min-reduce
// ═════════════════════════════════════════════════════════════════════
__global__ void minReduceKernel(const float* __restrict__ costs,
    float* __restrict__ outMin, int n)
{
    extern __shared__ float sd[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float val = FLT_MAX;
    if (i < n)                val = costs[i];
    if (i + blockDim.x < n)  val = fminf(val, costs[i + blockDim.x]);
    sd[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sd[tid] = fminf(sd[tid], sd[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMinFloat(outMin, sd[0]);
}

// ═════════════════════════════════════════════════════════════════════
// Weighted average
// ═════════════════════════════════════════════════════════════════════
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
