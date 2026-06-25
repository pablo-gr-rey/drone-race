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
        throw std::runtime_error(std::format("Received config for {} rect obstacles but N_OBSTACLES is set to {}. Edit this constant and recompile", nObstacles, N_OBSTACLES));

    int nObsCoords = reader.readFloatArray(obstacles);
    if (nObsCoords != N_OBSTACLES * DIM * 2)
        throw std::runtime_error(std::format("Received config for {} rect obstacles coordinates but expected N_OBSTACLES * DIM * 2 = {}", nObsCoords, N_OBSTACLES * DIM * 2));

    int nRoundObs = reader.readInt32();
    if (nRoundObs != N_ROUND_OBSTACLES)
        throw std::runtime_error(std::format("Received config for {} round obstacles but N_ROUND_OBSTACLES is set to {}. Edit this constant and recompile", nRoundObs, N_ROUND_OBSTACLES));

    int nRoundObsCoords = reader.readFloatArray(roundObsCenters);
    if (nRoundObsCoords != N_ROUND_OBSTACLES * DIM)
        throw std::runtime_error(std::format("Received config for {} round obstacles centers but expected N_ROUND_OBSTACLES * DIM = {}", nRoundObsCoords, N_ROUND_OBSTACLES * DIM));
    int nRadius = reader.readFloatArray(roundObsRadius);
    if (nRadius != N_ROUND_OBSTACLES)
        throw std::runtime_error(std::format("Received config for {} round obstacles radius but expected N_ROUND_OBSTACLES = {}", nRadius, N_ROUND_OBSTACLES));

    int seed = reader.readInt32();

    int nModelFactors = reader.readInt32();
    if (nModelFactors != N_MODEL_FACTORS)
        throw std::runtime_error(std::format("Received MPPI config for {} environment parameters but N_MODEL_FACTORS is set to {}. Edit this constant and recompile", nModelFactors, N_MODEL_FACTORS));

    std::vector<float> modelSizes = reader.readFloatArray();
    for (int k = 0; k < N_MODEL_FACTORS; k++)
        if ((int) modelSizes[k] != MODEL_SIZE(k))
            throw std::runtime_error(std::format("Received invalid model size for parameter {}: got {}, but MODEL_SIZE({}) is set to {}. Edit this constant and recompile", k, (int) modelSizes[k], k, MODEL_SIZE(k)));

    reader.readFloatArray(initBelief);

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

    std::cout << "loaded envConfig, nAgents " << nAgents << " nWinLaps " << nWinLaps << " true theta " << trueTheta << " max speed " << maxSpeed[0] << ' ' << maxSpeed[1] << " nTrackSamples" << nTrackSamples << " gate vectors:";
    for (int i = 0; i < nGates * dim; i++)
        std::cout << gateVectors[i] << (i % dim ? " " : ";  ");
    std::cout << "\n";

    return seed;
}

void MPPIConfig::unpackHeader(Reader& reader)
{
    nSamples = (int) reader.readInt32();
    nTimesteps = (int) reader.readInt32();
    nKnots = (int) reader.readInt32();
    if (nKnots > MAX_N_KNOTS)
        throw std::runtime_error(std::format("Received invalid number of knots: got {}, but MAX_N_KNOTS is set to {}. Edit this constant and recompile", nKnots, MAX_N_KNOTS));

    int nRecv = reader.readIntArray(knots);
    if (nRecv != nKnots)
        throw std::runtime_error(std::format("Received invalid number of knots: got {} elements, but received nKnots={}", nRecv, nKnots));

    bool useSplines = reader.readInt32();
    if (useSplines != USE_SPLINES)
        throw std::runtime_error(std::format("Received useSplines={}, but constexpr USE_SPLINES is set to {}. Edit this constant and recompile", useSplines, USE_SPLINES));

    if (!USE_SPLINES)
        nKnots = nTimesteps;        // for MPPI device array allocationsmy

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

    minConfidence = reader.readFloat();

    nVerifSamples = reader.readInt32();
    beta = reader.readFloat();
    verifHorizon = reader.readInt32();
    maxVerifEps = reader.readFloat();

    std::cout << "loaded MPPI samples " << nSamples << " timesteps " << nTimesteps << " collDistFactor " << collDistFactor << " invTemp " << invTemperature << " with " << N_TRUE_MODELS << " opponent strats\n";
}

void PRMPPIConfig::unpackHeader(Reader& reader)
{
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
    oppOutsideCost = reader.readFloat();
    winCost = reader.readFloat();

    finalAdvWeight = reader.readFloat();
    finalOppAdvWeight = reader.readFloat();

    safetyWeight = reader.readFloat();
    minSafeDist = reader.readFloat();

    delta = reader.readFloat();
    P = reader.readInt32();

    std::cout << "loaded PRMPPI samples " << nSamples << " timesteps " << nTimesteps << " safetyWeight " << safetyWeight << " with delta = " << delta << " P = " << P << "\n";
}

AnyControllerConfig loadControllerConfig(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int msg_kind = reader.readInt32();
    if (msg_kind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for controller config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), msg_kind));

    int cont_kind = reader.readInt32();

    if (cont_kind == CONT_MPPI)
    {
        MPPIConfig mppiConfig;
        mppiConfig.unpackHeader(reader);
        reader.assertFinished();

        return mppiConfig;
    }
    else if (cont_kind == CONT_PRMPPI)
    {
        PRMPPIConfig mppiConfig;
        mppiConfig.unpackHeader(reader);
        reader.assertFinished();

        return mppiConfig;
    }
    else
        throw std::runtime_error(std::format("Received invalid controller kind {}", cont_kind));
}
