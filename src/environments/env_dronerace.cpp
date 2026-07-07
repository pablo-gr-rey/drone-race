#include "config.h" // IWYU pragma: export

#ifdef USE_ENV_DRONERACE

#include <iostream>

#include "protocol.h"

namespace EnvDroneRace
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig)
{
    EnvironmentConfig d_config = hostConfig;

    size_t bytes = N_RACELINES * N_TRACK_SAMPLES * DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_config.trackPoints, bytes));
    CUDA_CHECK(cudaMemcpy(d_config.trackPoints, hostConfig.trackPoints, bytes, cudaMemcpyHostToDevice));

    return d_config;
}

void freeDeviceConfig(EnvironmentConfig& d_config)
{
    if (d_config.trackPoints != nullptr)
    {
        CUDA_CHECK(cudaFree(d_config.trackPoints));
        d_config.trackPoints = nullptr;
    }
}

void pushSimState(Writer& writer, const SimState& state)
{
    writer.pushFloatArray(state.pos);
    writer.pushFloatArray(state.vel);
    writer.pushFloatArray(state.S);

    writer.pushIntArray<int>(state.laps);
    writer.pushIntArray<int>(state.gates);
}

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds)
{
    std::vector<float> fullPos(preds.size() * N_AGENTS * DIM);

    for (size_t t = 0; t < preds.size(); t++)
        std::copy(preds[t].pos.data(), preds[t].pos.data() + N_AGENTS * DIM, fullPos.begin() + t * N_AGENTS * DIM);

    writer.pushFloatArray(fullPos);
}

void printState(const SimState& state)
{
    for (int a = 0; a < N_AGENTS; a++)
    {
        std::cout << "\tAgent " << (a + 1) << ": currentS:";
        for (int r = 0; r < N_RACELINES; r++)
            std::cout << " " << state.S[a * N_RACELINES + r];

        std::cout << "\tcurrentGates: " << state.gates[a] << "\tnLaps: " << state.laps[a] << "\tposition";

        for (int d = 0; d < ACTION_DIM; d++)
            std::cout << " " << state.pos[a * ACTION_DIM + d];

        std::cout << "\tspeed:";
        for (int d = 0; d < ACTION_DIM; d++)
            std::cout << " " << state.vel[a * ACTION_DIM + d];

        std::cout << std::endl;
    }
}

SimState unpackSimState(const void* buf, size_t len, const EnvironmentConfig& envConfig)
{
    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(
            std::format("Expected header message type for environment config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    SimState state;
    reader.readArray(state.pos);
    reader.readArray(state.vel);
    reader.readArray(state.S);
    reader.readArray(state.laps);
    reader.readArray(state.gates);

    std::cout << "Read initial simstate: ";
    Env::printState(state);

    reader.assertFinished();

    for (int k = 0; k < N_RACELINES; k++)
        state.S[envConfig.iMppi * N_RACELINES + k] = -1.0f;

    return state;
}

PIDConfig unpackPIDConfig(Reader& reader)
{
    PIDConfig pidconfig;

    pidconfig.kp = reader.readFloat();
    pidconfig.kd = reader.readFloat();

    pidconfig.repulsionFactor = reader.readFloat();
    pidconfig.repulsionPower = reader.readFloat();
    pidconfig.repulsionDistFact = reader.readFloat();

    pidconfig.racelineIndex = reader.readInt32();

    pidconfig.actionNoise = reader.readFloat();

    std::cout << "loaded PID repulsionFactor " << pidconfig.repulsionFactor << " racelineIndex " << pidconfig.racelineIndex << " noise level "
              << pidconfig.actionNoise << '\n';

    return pidconfig;
}

// return (seed, trueTheta)
std::tuple<EnvironmentConfig, int, int> unpackEnvConfig(const void* buf, size_t len)
{
    EnvironmentConfig envConfig;

    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(
            std::format("Expected header message type for environment config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    int envKind = reader.readInt32();
    if (envKind != ENV_DRONERACE)
        throw std::runtime_error(
            std::format("Environment kind mismatch: received kind {}, but compiled environment is ENV_DRONERACE ({}). Recompile the code with the "
                        "correct environment set in config.h",
                        envKind, static_cast<int>(ENV_DRONERACE)));

    int nModelFactors = reader.readInt32();
    if (nModelFactors != N_MODEL_FACTORS)
        throw std::runtime_error(
            std::format("Received MPPI config for {} environment parameters but N_MODEL_FACTORS is set to {}. Edit this constant and recompile",
                        nModelFactors, N_MODEL_FACTORS));

    std::vector<float> modelSizes = reader.readArray<float>();
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if ((int)modelSizes[k] != MODEL_SIZE(k))
            throw std::runtime_error(
                std::format("Received invalid model size for parameter {}: got {}, but MODEL_SIZE({}) is set to {}. Edit this constant and recompile",
                            k, (int)modelSizes[k], k, MODEL_SIZE(k)));

    reader.readArray(envConfig.arenaMin);
    reader.readArray(envConfig.arenaMax);

    int nAgents = (int)reader.readInt32();
    int dim = (int)reader.readInt32();
    envConfig.dt = reader.readFloat();

    envConfig.sendStates = (bool)reader.readInt32();
    int nRacelines = (int)reader.readInt32();
    int nGates = (int)reader.readInt32();

    if (nAgents != N_AGENTS)
        throw std::runtime_error(
            std::format("Received config for {} agents but N_AGENTS is set to {}. Edit this constant and recompile", nAgents, N_AGENTS));
    if (dim != DIM)
        throw std::runtime_error(std::format("Received config for dimension {} but DIM is set to {}. Edit this constant and recompile", dim, DIM));
    if (nRacelines != N_RACELINES)
        throw std::runtime_error(
            std::format("Received config for {} racelines but N_RACELINES is set to {}. Edit this constant and recompile", nRacelines, N_RACELINES));
    if (nGates != N_GATES)
        throw std::runtime_error(
            std::format("Received config for {} gates but N_GATES is set to {}. Edit this constant and recompile", nGates, N_GATES));

    envConfig.droneRadius = reader.readFloat();
    envConfig.posNoiseLevel = reader.readFloat();
    envConfig.speedNoiseLevel = reader.readFloat();
    envConfig.actionNoiseLevel = reader.readFloat();

    reader.readArray(envConfig.maxSpeed);
    reader.readArray(envConfig.maxAccel);

    int nTrackSamples = (int)reader.readInt32();
    if (nTrackSamples != N_TRACK_SAMPLES)
        throw std::runtime_error(
            std::format("Received config for {} track samples but N_TRACK_SAMPLES is set to {}. Edit this constant and recompile", nTrackSamples,
                        N_TRACK_SAMPLES));

    envConfig.nWinLaps = (int)reader.readInt32();
    envConfig.targetDistance = reader.readFloat();

    reader.readArray(envConfig.gateCenters);
    reader.readArray(envConfig.gateVectors);
    reader.readArray(envConfig.gateRadius);

    int nObstacles = reader.readInt32();
    if (nObstacles != N_OBSTACLES)
        throw std::runtime_error(std::format("Received config for {} rect obstacles but N_OBSTACLES is set to {}. Edit this constant and recompile",
                                             nObstacles, N_OBSTACLES));

    int nObsCoords = reader.readArray(envConfig.obstacles);
    if (nObsCoords != N_OBSTACLES * DIM * 2)
        throw std::runtime_error(std::format("Received config for {} rect obstacles coordinates but expected N_OBSTACLES * DIM * 2 = {}", nObsCoords,
                                             N_OBSTACLES * DIM * 2));

    int nRoundObs = reader.readInt32();
    if (nRoundObs != N_ROUND_OBSTACLES)
        throw std::runtime_error(
            std::format("Received config for {} round obstacles but N_ROUND_OBSTACLES is set to {}. Edit this constant and recompile", nRoundObs,
                        N_ROUND_OBSTACLES));

    int nRoundObsCoords = reader.readArray(envConfig.roundObsCenters);
    if (nRoundObsCoords != N_ROUND_OBSTACLES * DIM)
        throw std::runtime_error(std::format("Received config for {} round obstacles centers but expected N_ROUND_OBSTACLES * DIM = {}",
                                             nRoundObsCoords, N_ROUND_OBSTACLES * DIM));
    int nRadius = reader.readArray(envConfig.roundObsRadius);
    if (nRadius != N_ROUND_OBSTACLES)
        throw std::runtime_error(
            std::format("Received config for {} round obstacles radius but expected N_ROUND_OBSTACLES = {}", nRadius, N_ROUND_OBSTACLES));

    int seed = reader.readInt32();

    reader.readArray(envConfig.initBelief);

    for (int i = 0; i < N_TRUE_MODELS; i++)
        envConfig.oppPid[i] = unpackPIDConfig(reader);

    envConfig.iMppi = reader.readInt32();

    int trueTheta = reader.readInt32();

    std::vector<float> vecTrackPoints = reader.readArray<float>();
    if (vecTrackPoints.size() != N_RACELINES * N_TRACK_SAMPLES * DIM)
        throw std::runtime_error(std::format("Received {} track samples coordinates but expected N_RACELINES * N_TRACK_SAMPLES * DIM = {}",
                                             vecTrackPoints.size(), N_RACELINES * N_TRACK_SAMPLES * DIM));

    envConfig.trackPoints = (float*)malloc(N_RACELINES * N_TRACK_SAMPLES * DIM * sizeof(float));
    std::copy(vecTrackPoints.begin(), vecTrackPoints.end(), envConfig.trackPoints);

    reader.assertFinished();

    std::cout << "loaded envConfig, nAgents " << nAgents << " nWinLaps " << envConfig.nWinLaps << " true theta " << trueTheta << " max speed "
              << envConfig.maxSpeed[0] << ' ' << envConfig.maxSpeed[1] << " nTrackSamples" << nTrackSamples << " gate vectors:";
    for (int i = 0; i < nGates * dim; i++)
        std::cout << envConfig.gateVectors[i] << (i % dim ? " " : ";  ");
    std::cout << "\n";

    return {envConfig, seed, trueTheta};
}
} // namespace EnvDroneRace

#endif
