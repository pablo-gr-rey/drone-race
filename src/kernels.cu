#include "config.h"
#include "state.h"
#include "costs.cuh"
#include "kernels.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>

// RNG init
__global__ void initRNGKernel(curandState* states, unsigned long long seed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        // printf("initializing RNG %d\n", i);
        curand_init(seed, i, 0, &states[i]);
    }
}

// Noise generation
__global__ void generateNoiseKernel(float* noise, curandState* rng,
    float stddev,
    int nTimesteps, int N, int dim, int nModels)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    // printf("Accessing RNG %d\n", s);
    curandState local = rng[s];
    for (int theta = 0; theta <= nModels; theta++)      // generate for nominal + 1 for each model
        for (int t = 0; t < nTimesteps; t++)
            for (int d = 0; d < dim; d++)
                noise[((theta * nTimesteps + t) * N + s) * dim + d] = curand_normal(&local) * stddev;

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
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,
    const float* __restrict__ noise,
    float* __restrict__ totalCosts,
    curandState* __restrict__ rngStates,
    const float* __restrict__ trackPts)
{
    int N = mc.nSamples;

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N)
        return;

    float pos[MAX_AGENTS * MAX_DIM];
    float vel[MAX_AGENTS * MAX_DIM];
    float currentS[MAX_AGENTS * MAX_RACELINES];
    int laps[MAX_AGENTS];
    int currentGates[MAX_AGENTS];
    float belief[MAX_MODELS];
    float prevPos[MAX_AGENTS * MAX_DIM];

    float nomPidAction[MAX_MODELS * MAX_DIM];

    totalCosts[s] = 0.0f;
    curandState rng = rngStates[s];

    // find out if we are already committed
    // if so, we only consider that plan (switch to normal MPPI); otherwise, cost at the end could incur slight perturbations for the other branches

    for (int thetaT = 0; thetaT < mc.nModels; thetaT++)
        belief[thetaT] = initBelief[thetaT];

    int initPredTheta = findConfident(belief, mc.nModels, mc.minConfidence);

    // TODO: if there are many models, we could skip them if they have small probability
    for (int theta = 0; theta < mc.nModels; theta++)    // theta is the model the opponent is actually following
    {
        if (initPredTheta != -1 && theta != initPredTheta)
            continue;

        // copy initial state
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            for (int d = 0; d < envConfig.dim; d++)
            {
                pos[a * envConfig.dim + d] = initPos[a * envConfig.dim + d];
                vel[a * envConfig.dim + d] = initVel[a * envConfig.dim + d];
            }

            for (int r = 0; r < envConfig.nRacelines; r++)
                currentS[a * envConfig.nRacelines + r] = initS[a * envConfig.nRacelines + r];

            laps[a] = initLaps[a];
            currentGates[a] = initGates[a];
        }

        for (int thetaT = 0; thetaT < mc.nModels; thetaT++)
            belief[thetaT] = initBelief[thetaT];

        float cost = 0.0f;
        bool stop = false;

        int predTheta = findConfident(belief, mc.nModels, mc.minConfidence);    // -1 if we have not branched, otherwise the predicted model
        // hopefully, this will be -1 or theta, but we can't be sure of it so we have to take into account the possibility that we mispredict (since the actual MPPI controller will not know whether it has mispredicted)
        int branchingTime = 0;      // unused if we have not branched (and set at branching time), we can use 0 in both cases

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
                    // nominal actions has shape (nModels+1, nTimesteps, dim) with first part = before branching time, then for theta_0, theta_1, etc
                    // noise has shape (nModels+1, nTimesteps, nSamples, dim)
                    // if we have branched, the time origin of the specialized nominal actions is the branching time (otherwise, branchingTime=0)
                    for (int d = 0; d < envConfig.dim; d++)
                    {
                        float nom = nominal[((predTheta + 1) * mc.nTimesteps + t - branchingTime) * envConfig.dim + d];
                        float noisef = noise[(((predTheta + 1) * mc.nTimesteps + t - branchingTime) * mc.nSamples + s) * envConfig.dim + d];
                        actions[a * envConfig.dim + d] = nom + noisef;
                        // actions[a * envConfig.dim + d] = nominal[(predTheta * mc.nModels + t - branchingTime) * envConfig.dim + d] + noise[((predTheta * mc.nModels + t - branchingTime) * mc.nSamples + s) * envConfig.dim + d];
                    }
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
                        // we compute the predicted nominal action (without noise), to update the belief (which we do after sending our own action, this will only be used at the end of the control loop as MPPI does not have an instantaneous information advantage)
                        // if the nominal action are not too close, then we can get a good idea of which strategy the opponent is using since it will be the one corresponding to the nominal action closest to the actual action
                        // note: this assumes only 1 other agent!
                        for (int thetaT = 0; thetaT < mc.nModels; thetaT++)
                            computePIDAction(a, pos, vel, currentS, envConfig, mc.oppPid[thetaT], trackPts, nomPidAction + thetaT * envConfig.dim);

                        // we use theta for the actual action (which is assumed to be the real model), and add noise (reuse the same PID instead of recomputing it)
                        // computePIDAction(a, pos, vel, currentS, envConfig, mc.oppPid[theta], trackPts, actions + a * envConfig.dim);
                        for (int d = 0; d < envConfig.dim; d++)
                            actions[a * envConfig.dim + d] = nomPidAction[theta * envConfig.dim + d] + mc.oppPid[theta].actionNoise * curand_normal(&rng);
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

            updateGates(envConfig, pos, prevPos, currentS, currentGates, laps, trackPts);

            // if agent won or is outside, or if there is a collision, stop rollout
            for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
                stop = stop || (laps[iAgent] >= envConfig.nWinLaps) || (isOutside(envConfig, pos + iAgent * envConfig.dim, envConfig.minDist / 2.0f));

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
            cost += stateCost(controlAgent, pos, vel, laps, currentGates, t, envConfig, mc);

            // 9. Update belief & potential branching time
            updateBelief(belief, actions + (1 - controlAgent) * envConfig.dim, nomPidAction, mc.oppPid, mc.nModels, envConfig.dim, envConfig.maxAccel[1 - controlAgent]);

            if (predTheta == -1 && (predTheta = findConfident(belief, mc.nModels, mc.minConfidence)) != -1)
                branchingTime = t + 1;
        }

        // ── Terminal cost ────────────────────────────────────────────────
        cost += finalCost(controlAgent, pos, vel, laps, currentGates, envConfig, mc);

        // actual cost is dependent on the probability that the opponent is actually following theta, ie. initBelief[theta], unless we are already committed
        if (initPredTheta == -1)
            totalCosts[s] += initBelief[theta] * cost;
        else
            totalCosts[s] = cost;
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
    const float* __restrict__ minCost, float invTemp,
    int nModels, int N, int T, int dim,
    float* __restrict__ nu)
{
    int mtd = blockIdx.x;
    if (mtd >= (nModels + 1) * T * dim)
        return;

    int d = mtd % dim;
    mtd /= dim;
    int t = mtd % T;
    int theta = mtd / T;

    extern __shared__ float sh[];
    float* s_wn = sh;
    float* s_w = sh + blockDim.x;

    float wSum = 0.0f;
    float wnSum = 0.0f;

    float minC = *minCost;

    for (int s = threadIdx.x; s < N; s += blockDim.x)
    {
        float w = expf(-(costs[s] - minC) / invTemp);
        wSum += w;
        wnSum += w * noise[((theta * T + t) * N + s) * dim + d];
    }

    s_wn[threadIdx.x] = wnSum;
    s_w[threadIdx.x] = wSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_wn[threadIdx.x] += s_wn[threadIdx.x + stride];
            s_w[threadIdx.x] += s_w[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && s_w[0] > 1e-30f)
        nominal[(theta * T + t) * dim + d] += s_wn[0] / s_w[0];

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *nu = s_w[0];
    }
}

// clamp nominals
__global__ void clampNominalKernel(
    float* nominal,
    float maxAccel,
    int nModels,
    int T,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nVecs = (nModels + 1) * T;
    if (idx >= nVecs)
        return;

    float sqNorm = 0.0f;
    int base = idx * dim;
    for (int d = 0; d < dim; ++d)
    {
        float v = nominal[base + d];
        sqNorm += v * v;
    }

    float maxSq = maxAccel * maxAccel;
    if (sqNorm > maxSq)
    {
        float scale = maxAccel / sqrtf(sqNorm);
        for (int d = 0; d < dim; ++d)
            nominal[base + d] *= scale;
    }
}

__global__ void verifyNominalFailureKernel(
    int controlAgent,
    int nVerif,
    EnvironmentConfig envConfig,
    MPPIConfig mc,
    const float* __restrict__ initPos,
    const float* __restrict__ initVel,
    const float* __restrict__ initS,
    const int* __restrict__ initLaps,
    const int* __restrict__ initGates,
    const float* __restrict__ initBelief,
    const float* __restrict__ nominal,     // (nModels+1, T, dim)
    const float* __restrict__ trackPts,
    curandState* __restrict__ rngStates,
    unsigned int* __restrict__ failCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nVerif)
        return;

    curandState rng = rngStates[s];

    // Local rollout state
    float pos[MAX_AGENTS * MAX_DIM];
    float vel[MAX_AGENTS * MAX_DIM];
    float currentS[MAX_AGENTS * MAX_RACELINES];
    float prevPos[MAX_AGENTS * MAX_DIM];
    int laps[MAX_AGENTS];
    int currentGates[MAX_AGENTS];

    float belief[MAX_MODELS];
    float nomPidAction[MAX_MODELS * MAX_DIM];
    float actions[MAX_AGENTS * MAX_DIM];

    // Copy initial state
    for (int a = 0; a < envConfig.nAgents; a++)
    {
        for (int d = 0; d < envConfig.dim; d++)
        {
            pos[a * envConfig.dim + d] = initPos[a * envConfig.dim + d];
            vel[a * envConfig.dim + d] = initVel[a * envConfig.dim + d];
        }

        for (int r = 0; r < envConfig.nRacelines; r++)
            currentS[a * envConfig.nRacelines + r] = initS[a * envConfig.nRacelines + r];

        laps[a] = initLaps[a];
        currentGates[a] = initGates[a];
    }

    for (int k = 0; k < mc.nModels; k++)
        belief[k] = initBelief[k];

    // Sample actual opponent model according to initial belief
    int theta = sampleModelFromBelief(initBelief, mc.nModels, &rng);

    int predTheta = findConfident(belief, mc.nModels, mc.minConfidence);
    int branchingTime = 0;

    int failed = 0;     // 1 if collision, 2 if outside

    // Dynamics rollout
    for (int t = 0; t < mc.nTimesteps && !failed; t++)
    {
        // Save previous positions for gate update
        for (int i = 0; i < envConfig.nAgents * envConfig.dim; i++)
            prevPos[i] = pos[i];

        // Build actions
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            if (a == controlAgent)
            {
                int localT = t - branchingTime;
                int startInd = ((predTheta + 1) * mc.nTimesteps + localT) * envConfig.dim;

                for (int d = 0; d < envConfig.dim; d++)
                    actions[a * envConfig.dim + d] = nominal[startInd + d];
            }
            else
            {
                // predicted nominal PID actions for all models
                for (int thetaT = 0; thetaT < mc.nModels; thetaT++)
                    computePIDAction(a, pos, vel, currentS, envConfig, mc.oppPid[thetaT], trackPts, nomPidAction + thetaT * envConfig.dim);

                // actual opponent action = nominal PID + PID model noise
                for (int d = 0; d < envConfig.dim; d++)
                    actions[a * envConfig.dim + d] = nomPidAction[theta * envConfig.dim + d] + mc.oppPid[theta].actionNoise * curand_normal(&rng);
            }
        }

        // Clamp accelerations and add environment action noise
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            float sqNorm = 0.0f;
            for (int d = 0; d < envConfig.dim; d++)
            {
                float v = actions[a * envConfig.dim + d];
                sqNorm += v * v;
            }

            float factor = 1.0f;
            float maxSq = envConfig.maxAccel[a] * envConfig.maxAccel[a];
            if (sqNorm > maxSq)
                factor = envConfig.maxAccel[a] / sqrtf(sqNorm);

            for (int d = 0; d < envConfig.dim; d++)
            {
                float& v = actions[a * envConfig.dim + d];
                v *= factor;
                v += envConfig.actionNoiseLevel * curand_normal(&rng);
            }
        }

        // Integrate position
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
                pos[a * envConfig.dim + d] += envConfig.dt * vel[a * envConfig.dim + d];

        // Integrate velocity
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
                vel[a * envConfig.dim + d] += envConfig.dt * actions[a * envConfig.dim + d];

        // Speed cap
        for (int a = 0; a < envConfig.nAgents; a++)
        {
            float spd = agentSpeed(vel, a, envConfig.dim);
            if (spd > envConfig.maxSpeed[a])
            {
                float sc = envConfig.maxSpeed[a] / spd;
                for (int d = 0; d < envConfig.dim; d++)
                    vel[a * envConfig.dim + d] *= sc;
            }
        }

        // State noise
        for (int a = 0; a < envConfig.nAgents; a++)
            for (int d = 0; d < envConfig.dim; d++)
            {
                pos[a * envConfig.dim + d] += envConfig.posNoiseLevel * curand_normal(&rng);
                vel[a * envConfig.dim + d] += envConfig.speedNoiseLevel * curand_normal(&rng);
            }

        // Update gates / laps / S
        updateGates(envConfig, pos, prevPos, currentS, currentGates, laps, trackPts);

        // Update belief from observed opponent action
        int oppAgent = 1 - controlAgent;
        updateBelief(
            belief,
            actions + oppAgent * envConfig.dim,
            nomPidAction,
            mc.oppPid,
            mc.nModels,
            envConfig.dim,
            envConfig.maxAccel[oppAgent]);

        if (predTheta == -1)
        {
            int conf = findConfident(belief, mc.nModels, mc.minConfidence);
            if (conf != -1)
            {
                predTheta = conf;
                branchingTime = t + 1;
            }
        }

        // Failure checks

        // 1. MPPI goes outside
        if (isOutside(envConfig, pos + controlAgent * envConfig.dim, envConfig.minDist / 2.0f))
        {
            failed = 2;
            break;
        }

        // 2. Collision
        for (int a1 = 0; a1 < envConfig.nAgents && !failed; a1++)
            for (int a2 = a1 + 1; a2 < envConfig.nAgents; a2++)
            {
                float dist2 = 0.0f;
                for (int d = 0; d < envConfig.dim; d++)
                {
                    float dx = pos[a1 * envConfig.dim + d] - pos[a2 * envConfig.dim + d];
                    dist2 += dx * dx;
                }

                if (dist2 < envConfig.minDist * envConfig.minDist)
                {
                    failed = 1;
                    break;
                }
            }

        // If MPPI wins, end rollout (but it does not count as a failure)

        if (isOutside(envConfig, pos + (1 - controlAgent) * envConfig.dim, envConfig.minDist / 2.0f))
            break;

        bool shouldBreak = false;
        for (int a = 0; a < envConfig.nAgents; a++)
            if (laps[a] >= envConfig.nWinLaps)
                shouldBreak = true;

        if (shouldBreak)
            break;
    }

    if (failed)
        atomicAdd(failCount + failed - 1, 1u);

    rngStates[s] = rng;
}
