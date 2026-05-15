#include "config.h"
#include "state.h"
#include "costs.cuh"
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
    const EnvironmentConfig envConfig,
    // const DeviceEnvironmentConfig envConfig,
    const MPPIConfig mc,
    const float* __restrict__ initPos,
    const float* __restrict__ initVel,
    const float* __restrict__ initS,
    const int* __restrict__ initLaps,
    const int* __restrict__ initGates,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    float* __restrict__ finalPos,
    float* __restrict__ finalVel,
    float* __restrict__ finalS,
    int* __restrict__ finalLaps,
    int* __restrict__ finalGates,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts,
    int nTP, int N)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;

    // ── Load initial state into registers ────────────────────────────
    float pos[MAX_AGENTS * MAX_DIM];
    float vel[MAX_AGENTS * MAX_DIM];
    float S[MAX_AGENTS];
    int laps[MAX_AGENTS];
    int currentGates[MAX_AGENTS];
    float prevPos[MAX_AGENTS * MAX_DIM];

    for (int a = 0; a < envConfig.nAgents; ++a)
        for (int d = 0; d < envConfig.dim; ++d)
        {
            pos[a * envConfig.dim + d] = initPos[a * envConfig.dim + d];
            vel[a * envConfig.dim + d] = initVel[a * envConfig.dim + d];
        }
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
            prevPos[i] = pos[i];

        // 1. Build actions
        float actions[MAX_AGENTS * MAX_DIM];
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            if (a == controlAgent)
            {
                for (int d = 0; d < envConfig.dim; d++)
                    actions[a * envConfig.dim + d] = nominal[t * envConfig.dim + d] + noise[(t * N + s) * envConfig.dim + d];
            }
            else
                switch (mc.oppKind)
                {
                case ControllerKind::CONT_DUMMY: {
                    for (int d = 0; d < envConfig.dim; d++)
                        actions[a * envConfig.dim + d] = 0.0f;
                    break;
                }

                case ControllerKind::CONT_PID: {
                    // TODO: should use belief
                    computePIDAction(a, pos, vel, S, currentGates, envConfig, mc.oppPid[mc.oppPidStrat], trackPts, actions + a * envConfig.dim);
                    for (int d = 0; d < envConfig.dim; d++)
                        actions[a * envConfig.dim + d] += mc.oppPid[mc.oppPidStrat].actionNoise * curand_normal(&rng);
                    break;
                }
                }
        }

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
                pos[a * envConfig.dim + d] += envConfig.dt * vel[a * envConfig.dim + d];

        // 4. Integrate velocity
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
                vel[a * envConfig.dim + d] += envConfig.dt * actions[a * envConfig.dim + d];

        // 5. Cap speed
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            float spd = agentSpeed(vel, a, envConfig.dim);
            if (spd > envConfig.maxSpeed[a])
            {
                float sc = envConfig.maxSpeed[a] / spd;
                for (int d = 0; d < envConfig.dim; d++)
                    vel[a * envConfig.dim + d] = vel[a * envConfig.dim + d] * sc;
            }
        }

        // 6. State noise
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
            {
                pos[a * envConfig.dim + d] += curand_normal(&rng) * envConfig.posNoiseLevel;
                vel[a * envConfig.dim + d] += curand_normal(&rng) * envConfig.speedNoiseLevel;
            }

        // 7. Track S / laps / gates update
        for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        {
            const float* curPos = pos + iAgent * envConfig.dim;

            // if agent is outside, stop rollout (subsequent samples do not matter)
            for (int d = 0; d < envConfig.dim; d++)
                if (curPos[d] < envConfig.arenaMin[d] || curPos[d] > envConfig.arenaMax[d])
                    stop = true;

            float dist;
            float newSa = fastProjectOnTrack(trackPts, nTP, envConfig.dim, curPos, nullptr, dist, S[iAgent]);
            S[iAgent] = newSa;

            int nextGate = (currentGates[iAgent] + 1) % envConfig.nGates;
            float num = 0., denom = 0.;
            for (int d = 0; d < envConfig.dim; d++)
            {
                num += envConfig.gateVectors[nextGate * envConfig.dim + d] * (envConfig.gateCenters[nextGate * envConfig.dim + d] - prevPos[iAgent * envConfig.dim + d]);
                denom += envConfig.gateVectors[nextGate * envConfig.dim + d] * (pos[iAgent * envConfig.dim + d] - prevPos[iAgent * envConfig.dim + d]);
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
                float dx = (1. - lambda) * prevPos[iAgent * envConfig.dim + d] + lambda * pos[iAgent * envConfig.dim + d] - envConfig.gateCenters[nextGate * envConfig.dim + d];
                sqDist += dx * dx;
            }

            // std::cout.precision(5);
            // std::cout << std::fixed << "\tsqDist = " << sqDist << " sq radius " << envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate] << "\n";

            float margin = iAgent == controlAgent ? mc.gateTraversalMargin * mc.gateTraversalMargin : 1.0f;

            if (sqDist <= envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate] * margin)
            {
                currentGates[iAgent]++;
                if (currentGates[iAgent] == envConfig.nGates)
                {
                    currentGates[iAgent] = 0;
                    laps[iAgent]++;
                }
            }

            // if agent won, stop rollout
            if (laps[iAgent] >= envConfig.nWinLaps)
                stop = true;
        }

        // is there a collision?
        for (int iAgent1 = 0; iAgent1 < envConfig.nAgents; iAgent1++)
            for (int iAgent2 = iAgent1 + 1; iAgent2 < envConfig.nAgents; iAgent2++)
            {
                float dist2 = 0.f;
                for (int d = 0; d < envConfig.dim; ++d)
                {
                    float dx = pos[iAgent1 * envConfig.dim + d] - pos[iAgent2 * envConfig.dim + d];
                    dist2 += dx * dx;
                }
                if (dist2 < envConfig.minDist * envConfig.minDist)
                {
                    stop = true;
                    break;
                }
            }

        // 8. Running cost
        cost += stateCost(controlAgent, pos, vel, S, laps, currentGates, t, envConfig, mc, trackPts, nTP);
    }

    // ── Terminal cost ────────────────────────────────────────────────
    cost += finalCost(controlAgent, pos, vel, S, laps, currentGates, envConfig, mc, trackPts, nTP);

    // ── Write outputs ────────────────────────────────────────────────
    totalCosts[s] = cost;

    // float sqRad = 0.0f;
    // for (int d = 0; d < envConfig.dim; d++)
    //     sqRad += pos[(1 - controlAgent) * envConfig.dim + d] * pos[(1 - controlAgent) * envConfig.dim + d];
    // printf("Final computed position of PID: rad %f (x %f y %f) (stopped early=%d)\n", sqrtf(sqRad), pos[(1 - controlAgent) * envConfig.dim], pos[(1 - controlAgent) * envConfig.dim + 1], stop);

    // We don't actually need the final state for the MPPI update,
    // but if you want to inspect it for debugging:
    // (can be removed to save bandwidth)
    float* op = finalPos + s * envConfig.nAgents * envConfig.dim;
    float* ov = finalVel + s * envConfig.nAgents * envConfig.dim;
    float* oS = finalS + s * envConfig.nAgents;
    int* oL = finalLaps + s * envConfig.nAgents;
    int* oG = finalGates + s * envConfig.nAgents;
    for (int i = 0; i < envConfig.nAgents * envConfig.dim; i++)
        op[i] = pos[i];
    for (int i = 0; i < envConfig.nAgents * envConfig.dim; i++)
        ov[i] = vel[i];
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
