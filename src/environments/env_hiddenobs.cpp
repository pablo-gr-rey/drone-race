#include "config.h" // IWYU pragma: export

#ifdef USE_ENV_HIDDENOBS

#include <iostream>

#include "protocol.h"

namespace EnvHiddenObs
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig)
{
    // we do not need additional GPU memory
    return hostConfig;
}

void freeDeviceConfig(EnvironmentConfig& /* d_config */)
{
}

void pushSimState(Writer& writer, const SimState& state)
{
    writer.pushFloatArray(state.pos);
    writer.pushFloatArray(state.vel);

    writer.pushInt32(state.laps);
    writer.pushInt32(state.gates);
}

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds)
{
    std::vector<float> fullPos(preds.size() * DIM);

    for (size_t t = 0; t < preds.size(); t++)
        std::copy(preds[t].pos.data(), preds[t].pos.data() + DIM, fullPos.begin() + t * DIM);

    writer.pushFloatArray(fullPos);
}

void printState(const SimState& state)
{
    std::cout << "\tcurrentGates: " << state.gates << "\tnLaps: " << state.laps << "\tposition";

    for (int d = 0; d < ACTION_DIM; d++)
        std::cout << " " << state.pos[d];

    std::cout << "\tspeed:";
    for (int d = 0; d < ACTION_DIM; d++)
        std::cout << " " << state.vel[d];

    std::cout << std::endl;
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

    state.laps = reader.readInt32();
    state.gates = reader.readInt32();

    reader.assertFinished();

    return state;
}

std::tuple<EnvironmentConfig, int, int> unpackEnvConfig(const void* buf, size_t len) // return seed, trueTheta
{
    EnvironmentConfig envConfig;

    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(
            std::format("Expected header message type for environment config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    int envKind = reader.readInt32();
    if (envKind != ENV_HIDDENOBS)
        throw std::runtime_error(
            std::format("Environment kind mismatch: received kind {}, but compiled environment is ENV_HIDDENOBS ({}). Recompile the code with the "
                        "correct environment set in config.h",
                        kind, static_cast<int>(ENV_DRONERACE)));

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

    int dim = (int)reader.readInt32();
    envConfig.dt = reader.readFloat();

    envConfig.sendStates = (bool)reader.readInt32();

    int nGates = (int)reader.readInt32();

    if (dim != DIM)
        throw std::runtime_error(std::format("Received config for dimension {} but DIM is set to {}. Edit this constant and recompile", dim, DIM));
    if (nGates != N_GATES)
        throw std::runtime_error(
            std::format("Received config for {} gates but N_GATES is set to {}. Edit this constant and recompile", nGates, N_GATES));

    envConfig.droneRadius = reader.readFloat();
    envConfig.posNoiseLevel = reader.readFloat();
    envConfig.speedNoiseLevel = reader.readFloat();
    envConfig.actionNoiseLevel = reader.readFloat();

    envConfig.maxSpeed = reader.readFloat();
    envConfig.maxAccel = reader.readFloat();

    envConfig.nWinLaps = (int)reader.readInt32();

    reader.readArray(envConfig.gateCenters);
    reader.readArray(envConfig.gateVectors);
    reader.readArray(envConfig.gateRadius);

    int nRectObstacles = reader.readInt32();
    if (nRectObstacles != N_RECT_OBSTACLES)
        throw std::runtime_error(
            std::format("Received config for {} rect obstacles but N_RECT_OBSTACLES is set to {}. Edit this constant and recompile", nRectObstacles,
                        N_RECT_OBSTACLES));

    int nObsCoords = reader.readArray(envConfig.rectObstacles);
    if (nObsCoords != N_RECT_OBSTACLES * DIM * 2)
        throw std::runtime_error(std::format("Received config for {} rect obstacles coordinates but expected N_RECT_OBSTACLES * DIM * 2 = {}",
                                             nObsCoords, N_RECT_OBSTACLES * DIM * 2));

    int nUncertainObs = reader.readInt32();
    if (nUncertainObs != N_UNCERTAIN_OBSTACLES)
        throw std::runtime_error(
            std::format("Received config for {} round obstacles but N_UNCERTAIN_OBSTACLES is set to {}. Edit this constant and recompile",
                        nUncertainObs, N_UNCERTAIN_OBSTACLES));

    int nRoundObsCoords = reader.readArray(envConfig.hiddenObsCenters);
    if (nRoundObsCoords != N_UNCERTAIN_OBSTACLES * DIM)
        throw std::runtime_error(std::format("Received config for {} round obstacles centers but expected N_UNCERTAIN_OBSTACLES * DIM = {}",
                                             nRoundObsCoords, N_UNCERTAIN_OBSTACLES * DIM));
    int nRadius = reader.readArray(envConfig.hiddenObsRadius);
    if (nRadius != N_UNCERTAIN_OBSTACLES)
        throw std::runtime_error(
            std::format("Received config for {} round obstacles radius but expected N_UNCERTAIN_OBSTACLES = {}", nRadius, N_UNCERTAIN_OBSTACLES));

    int nDblLimits = reader.readArray(envConfig.dblHpLimits);
    int nDblLambda = reader.readArray(envConfig.dblHpLambdas);
    if (nDblLimits != 4)
        throw std::runtime_error(std::format("Received config for {} diag-half-plane limits, expected 4 (see env_hiddenobs_defs.h)", nDblLimits));
    if (nDblLambda != 4)
        throw std::runtime_error(std::format("Received config for {} diag-half-plane lambdas, expected 4 (see env_hiddenobs_defs.h)", nDblLambda));

    int nAnnCenter = reader.readArray(envConfig.annulusCenter);
    int nAnnRadius = reader.readArray(envConfig.annulusRadius);
    if (nAnnCenter != DIM)
        throw std::runtime_error(std::format("Received config for {} annulus center coords, expected {}", nAnnCenter, DIM));
    if (nAnnRadius != 2)
        throw std::runtime_error(std::format("Received config for {} annulus center coords, expected 2", nAnnRadius));

    int seed = reader.readInt32();

    reader.readArray(envConfig.initBelief);

    int trueTheta = reader.readInt32();

    reader.assertFinished();

    std::cout << "loaded hidden-obs envConfig, droneRadius " << envConfig.droneRadius << " nWinLaps " << envConfig.nWinLaps << " true theta "
              << trueTheta << " max speed " << envConfig.maxSpeed << " annulus radius " << envConfig.annulusRadius[0] << ' '
              << envConfig.annulusRadius[1] << " gate vectors ";

    for (int i = 0; i < nGates * dim; i++)
        std::cout << envConfig.gateVectors[i] << (i % dim ? " " : ";  ");
    std::cout << "\n";

    return {envConfig, seed, trueTheta};
}
} // namespace EnvHiddenObs

#endif
