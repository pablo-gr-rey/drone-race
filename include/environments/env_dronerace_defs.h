#pragma once

namespace EnvDroneRace
{
    constexpr int DIM = 2;
    constexpr int N_AGENTS = 2;
    constexpr int N_GATES = 2;
    constexpr int N_TRACK_SAMPLES = 512;
    constexpr int N_OBSTACLES = 0;

    constexpr int ACTION_DIM = DIM;

    // 0/1 obstacle
    constexpr int N_ROUND_OBSTACLES = 0;
    // constexpr int N_ROUND_OBSTACLES = 1;
    constexpr int N_RACELINES = 2;
    constexpr int N_MODEL_FACTORS = 1;

    // 2 obstacles
    // constexpr int N_ROUND_OBSTACLES = 2;
    // constexpr int N_RACELINES = 4;
    // constexpr int N_MODEL_FACTORS = 2;


    // CUDA does not like constexpr arrays, so we use constexpr inline functions

    HD INLINE constexpr int MODEL_SIZE(int /* k */)
    {
        // for several models with different sizes, tests are fine: return (k == 0) ? 2 : 3 or a switch, but here we can just fold it
        return 2;
    }

    HD INLINE constexpr int BRANCH_SIZE(int /* k */)
    {
        // same comment as above
        return 3;
    }

    constexpr int N_TRUE_MODELS = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : MODEL_SIZE(0) * MODEL_SIZE(1);
    constexpr int N_BRANCH_PLANS = N_MODEL_FACTORS == 1 ? BRANCH_SIZE(0) : BRANCH_SIZE(0) * BRANCH_SIZE(1);
    constexpr int MAX_MODEL_SIZE = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : (MODEL_SIZE(0) > MODEL_SIZE(1) ? MODEL_SIZE(0) : MODEL_SIZE(1));

    // PID parameters (used for opponent modelling)
    struct PIDConfig
    {
        float kp = 10.0f;
        float kd = 5.0f;

        float repulsionFactor = 20.0f;
        float repulsionPower = 2.0f;
        float repulsionDistFact = 10.0f;

        int racelineIndex = 0;

        float actionNoise = 0.0f;
    };

    // Environment configuration
    // this should have the same layout as GateEnvironmentConfig in the Python side
    struct EnvironmentConfig
    {
        float dt;

        bool sendStates;

        float initPos[N_AGENTS * DIM];
        float initSpeed[N_AGENTS * DIM];
        float initS[N_AGENTS * N_RACELINES];    // (nAgents * nRacelines). if == -1.0f, will not be updated (since it is only useful for PIDs)
        int initLaps[N_AGENTS];
        int initGates[N_AGENTS];

        float minDist;
        float posNoiseLevel;
        float speedNoiseLevel;
        float actionNoiseLevel;

        float maxSpeed[N_AGENTS];
        float maxAccel[N_AGENTS];

        // track
        int nWinLaps;
        float targetDistance;

        float gateCenters[N_GATES * DIM]; // (nGates * dim)
        float gateVectors[N_GATES * DIM]; // (nGates * dim)
        float gateRadius[N_GATES];  // (nGates)

        float arenaMin[DIM];    // (dim)
        float arenaMax[DIM];    // (dim)

        float obstacles[N_OBSTACLES * DIM * 2];   // (nObstacles * dim * 2): rectangle obstacles, i.e. [xmin, ymin, xmax, ymax]
        float roundObsCenters[N_ROUND_OBSTACLES * DIM];     // (nRoundObstacles * dim): center of obstacles
        float roundObsRadius[N_ROUND_OBSTACLES];        // (nRoundObstacles): radius of obstacles

        float* trackPoints = nullptr; // (nTrackSamples * nRacelines, dim)  // this pointer is different on host & device!

        // TODO: move elsewhere (not useful in kernels)
        float initBelief[N_TRUE_MODELS];

        PIDConfig oppPid[N_TRUE_MODELS];
        int iMppi;
        int trueTheta;
    };

    struct SimState
    {
        float pos[N_AGENTS * DIM];
        float vel[N_AGENTS * DIM];
        float S[N_AGENTS * N_RACELINES];

        int laps[N_AGENTS];
        int gates[N_AGENTS];
    };

    struct ScratchEnvBuffer
    {
        float actions[N_AGENTS * DIM];
        float nomPidActions[N_TRUE_MODELS * DIM];
    };
}
