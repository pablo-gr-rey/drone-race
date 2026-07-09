#include "config.h" // IWYU pragma: export

#ifdef USE_ENV_STRATRACE

#include <iostream>

#include "protocol.h"

namespace EnvStratRace
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig)
{
    EnvironmentConfig d_config = hostConfig;

    auto alloc_copy = [](float*& p, size_t bytes)
    {
        float* newp;
        CUDA_CHECK(cudaMalloc(&newp, bytes));
        CUDA_CHECK(cudaMemcpy(newp, p, bytes, cudaMemcpyHostToDevice));
        p = newp;
    };

    alloc_copy(d_config.p_grid, N_TRACK_SAMPLES * DIM * sizeof(float));
    alloc_copy(d_config.dp_grid, N_TRACK_SAMPLES * DIM * sizeof(float));
    alloc_copy(d_config.t_grid, N_TRACK_SAMPLES * DIM * sizeof(float));
    alloc_copy(d_config.kappa_grid, N_TRACK_SAMPLES * sizeof(float));

    return d_config;
}

void freeDeviceConfig(EnvironmentConfig& d_config)
{
    auto safe_free = [](auto*& p)
    {
        if (p)
        {
            CUDA_CHECK(cudaFree(p));
            p = nullptr;
        }
    };

    safe_free(d_config.p_grid);
    safe_free(d_config.dp_grid);
    safe_free(d_config.t_grid);
    safe_free(d_config.kappa_grid);
}

void pushSimState(Writer& writer, const SimState& state)
{
    writer.pushFloatArray(state.pos);
    writer.pushFloatArray(state.vel);
    writer.pushFloatArray(state.S);
    writer.pushFloatArray(state.latDist);

    writer.pushIntArray<int>(state.laps);
}

void pushAddInfo(Writer& writer, const SimState& /* state */, const SimState& prevState, const EnvironmentConfig& envConfig, int trueTheta)
{
    // push lateral distance goal for each opponent
    std::array<float, N_OPP> latDist;

    for (int iOpp = 1; iOpp <= N_OPP; iOpp++)
        latDist[iOpp - 1] = computeOppTargetLatDist<true>(iOpp, prevState, envConfig, trueTheta);

    writer.pushFloatArray(latDist);
}

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds)
{
    std::vector<float> fullPos(preds.size() * N_AGENTS * DIM);

    for (size_t t = 0; t < preds.size(); t++)
        std::copy(preds[t].pos.begin(), preds[t].pos.begin() + N_AGENTS * DIM, fullPos.begin() + t * N_AGENTS * DIM);

    writer.pushFloatArray(fullPos);
}

void printState(const SimState& state)
{
    for (int a = 0; a < N_AGENTS; a++)
    {
        std::cout << "\tAgent " << a << ": currentS:" << state.S[a] << "\tlatDist " << state.latDist[a] << "\tnLaps: " << state.laps[a]
                  << "\tposition";

        for (int d = 0; d < DIM; d++)
            std::cout << " " << state.pos[a * DIM + d];

        std::cout << "\tspeed:";
        for (int d = 0; d < DIM; d++)
            std::cout << " " << state.vel[a * DIM + d];

        std::cout << std::endl;
    }
}

SimState unpackSimState(const void* buf, size_t len, const EnvironmentConfig& /* envConfig */)
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
    reader.readArray(state.latDist);

    reader.readArray(state.laps);

    reader.assertFinished();

    return state;
}

OppConfig unpackOppConfig(Reader& reader)
{
    OppConfig oppConfig;

    reader.readArray(oppConfig.s1);
    reader.readArray(oppConfig.s2);
    reader.readArray(oppConfig.s3);
    reader.readArray(oppConfig.speedScale);

    reader.readArray(oppConfig.actionNoise);

    return oppConfig;
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
    if (envKind != ENV_STRATRACE)
        throw std::runtime_error(
            std::format("Environment kind mismatch: received kind {}, but compiled environment is ENV_STRATRACE ({}). Recompile the code with the "
                        "correct environment set in config.h",
                        envKind, static_cast<int>(ENV_STRATRACE)));

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

    int nOpp = (int)reader.readInt32();
    int dim = (int)reader.readInt32();
    envConfig.dt = reader.readFloat();

    envConfig.sendStates = (bool)reader.readInt32();

    if (nOpp != N_OPP)
        throw std::runtime_error(
            std::format("Received config for {} opponents but N_OPP is set to {}. Edit this constant and recompile", nOpp, N_OPP));
    if (dim != DIM)
        throw std::runtime_error(std::format("Received config for dimension {} but DIM is set to {}. Edit this constant and recompile", dim, DIM));

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

    envConfig.trackWidth = reader.readFloat();
    envConfig.nWinLaps = (int)reader.readInt32();

    int seed = reader.readInt32();

    reader.readArray(envConfig.initBelief);

    for (int i = 0; i < N_TRUE_MODELS; i++)
        envConfig.oppConfigs[i] = unpackOppConfig(reader);

    int trueTheta = reader.readInt32();

    envConfig.kP = reader.readFloat();
    envConfig.kV = reader.readFloat();
    envConfig.maxOppLatDistFact = reader.readFloat();

    envConfig.trackLength = reader.readFloat();

    reader.allocReadArray(envConfig.p_grid, N_TRACK_SAMPLES * DIM);
    reader.allocReadArray(envConfig.dp_grid, N_TRACK_SAMPLES * DIM);
    reader.allocReadArray(envConfig.t_grid, N_TRACK_SAMPLES * DIM);
    reader.allocReadArray(envConfig.kappa_grid, N_TRACK_SAMPLES);

    reader.assertFinished();

    std::cout << "loaded stratRace envConfig, nOpp " << nOpp << " nWinLaps " << envConfig.nWinLaps << " true theta " << trueTheta << " max speeds ";
    for (float v : envConfig.maxSpeed)
        std::cout << v << " ";
    std::cout << " track length " << envConfig.trackLength << "\n";

    return {envConfig, seed, trueTheta};
}
} // namespace EnvStratRace

#endif
