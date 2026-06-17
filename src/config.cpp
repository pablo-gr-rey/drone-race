#include "config.h"
#include <iostream>
#include <format>

static PIDConfig unpackPIDConfig(Reader& reader)
{
    PIDConfig pidconfig;

    pidconfig.kp = reader.readFloat();
    pidconfig.kd = reader.readFloat();

    pidconfig.repulsionFactor = reader.readFloat();
    pidconfig.repulsionPower = reader.readFloat();
    pidconfig.repulsionDistFact = reader.readFloat();

    pidconfig.racelineIndex = reader.readInt32();

    pidconfig.actionNoise = reader.readFloat();

    std::cout << "loaded PID repulsionFactor " << pidconfig.repulsionFactor << " racelineIndex " << pidconfig.racelineIndex << " noise level " << pidconfig.actionNoise << '\n';

    return pidconfig;
}

int EnvironmentConfig::unpackHeader(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for environment config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    int nAgents = (int) reader.readInt32();
    int dim = (int) reader.readInt32();
    dt = reader.readFloat();

    sendStates = (bool) reader.readInt32();
    int nRacelines = (int) reader.readInt32();
    int nGates = (int) reader.readInt32();

    if (nAgents != N_AGENTS)
        throw std::runtime_error(std::format("Received config for {} agents but N_AGENTS is set to {}. Edit this constant and recompile", nAgents, N_AGENTS));
    if (dim != DIM)
        throw std::runtime_error(std::format("Received config for dimension {} but DIM is set to {}. Edit this constant and recompile", dim, DIM));
    if (nRacelines != N_RACELINES)
        throw std::runtime_error(std::format("Received config for {} racelines but N_RACELINES is set to {}. Edit this constant and recompile", nRacelines, N_RACELINES));
    if (nGates != N_GATES)
        throw std::runtime_error(std::format("Received config for {} gates but N_GATES is set to {}. Edit this constant and recompile", nGates, N_GATES));

    reader.readFloatArray(initPos);
    reader.readFloatArray(initSpeed);

    reader.readFloatArray(initS);
    reader.readIntArray(initLaps);
    reader.readIntArray(initGates);

    minDist = reader.readFloat();
    posNoiseLevel = reader.readFloat();
    speedNoiseLevel = reader.readFloat();
    actionNoiseLevel = reader.readFloat();

    reader.readFloatArray(maxSpeed);
    reader.readFloatArray(maxAccel);

    int nTrackSamples = (int) reader.readInt32();
    if (nTrackSamples != N_TRACK_SAMPLES)
        throw std::runtime_error(std::format("Received config for {} track samples but N_TRACK_SAMPLES is set to {}. Edit this constant and recompile", nTrackSamples, N_TRACK_SAMPLES));

    nWinLaps = (int) reader.readInt32();
    targetDistance = reader.readFloat();

    reader.readFloatArray(gateCenters);
    reader.readFloatArray(gateVectors);
    reader.readFloatArray(gateRadius);

    reader.readFloatArray(arenaMin);
    reader.readFloatArray(arenaMax);

    int nObstacles = reader.readInt32();
    if (nObstacles != N_OBSTACLES)
        throw std::runtime_error(std::format("Received config for {} obstacles but N_OBSTACLES is set to {}. Edit this constant and recompile", nObstacles, N_OBSTACLES));
    reader.readFloatArray(obstacles);

    int seed = reader.readInt32();

    for (int i = 0; i < N_TRUE_MODELS; i++)
        oppPid[i] = unpackPIDConfig(reader);

    iMppi = reader.readInt32();
    trueTheta = reader.readInt32();

    std::vector<float> vecTrackPoints = reader.readFloatArray();
    if (vecTrackPoints.size() != N_RACELINES * N_TRACK_SAMPLES * DIM)
        throw std::runtime_error(std::format("Received {} track samples coordinates but expected N_RACELINES * N_TRACK_SAMPLES * DIM = {}", vecTrackPoints.size(), N_RACELINES * N_TRACK_SAMPLES * DIM));

    trackPoints = (float*) malloc(N_RACELINES * N_TRACK_SAMPLES * DIM * sizeof(float));
    std::copy(vecTrackPoints.begin(), vecTrackPoints.end(), trackPoints);

    reader.assertFinished();

    std::cout << "read nAgents " << nAgents << " nWinLaps " << nWinLaps << " max speed " << maxSpeed[0] << ' ' << maxSpeed[1] << " nTrackSamples" << nTrackSamples << "\ngate vectors:";
    for (int i = 0; i < nGates * dim; i++)
        std::cout << gateVectors[i] << (i % dim ? " " : ";  ");
    std::cout << "\n";

    return seed;
}

void VerifConfig::unpackHeader(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for environment config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    nVerifSamples = reader.readInt32();
    beta = reader.readFloat();
    horizon = reader.readInt32();

    maxEps = reader.readFloat();

    reader.assertFinished();

    std::cout << "loaded verification N " << nVerifSamples << " beta " << beta << " horizon " << horizon << " maxEps " << maxEps << "\n";
}

void MPPIConfig::unpackHeader(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for MPPI config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    nSamples = (int) reader.readInt32();
    nTimesteps = (int) reader.readInt32();
    invTemperature = reader.readFloat();

    samplingNoise = reader.readFloat();
    gateTraversalMargin = reader.readFloat();

    collDistFactor = reader.readFloat();

    oppDistWeight = reader.readFloat();
    oppDistPower = reader.readFloat();
    oppDistThresholdFactor = reader.readFloat();
    boundaryCost = reader.readFloat();
    boundaryThresholdFactor = reader.readFloat();
    outsideCost = reader.readFloat();
    oppOutsideCost = reader.readFloat();
    collisionCost = reader.readFloat();
    winCost = reader.readFloat();

    finalAdvWeight = reader.readFloat();
    finalOppAdvWeight = reader.readFloat();
    finalSpeedWeight = reader.readFloat();

    minConfidence = reader.readFloat();
    int nModels = reader.readInt32();

    if (nModels != N_TRUE_MODELS)
        throw std::runtime_error(std::format("Received MPPI config for %d PID strategies but N_TRUE_MODELS is set to %d. Edit this constant and recompile", nModels, N_TRUE_MODELS));

    reader.readFloatArray(initBelief);

    reader.assertFinished();

    std::cout << "loaded MPPI samples " << nSamples << " timesteps " << nTimesteps << " collDistFactor " << collDistFactor << " with " << N_TRUE_MODELS << " opponent strats\n";
}
