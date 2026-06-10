#include "config.h"
#include <iostream>
#include <format>

std::pair<int, std::vector<float>> EnvironmentConfig::unpackHeader(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for environment config (type {}) but got type {} instead", static_cast<int>(MSG_HEADER), kind));

    nAgents = (int) reader.readInt32();
    dim = (int) reader.readInt32();
    dt = reader.readFloat();

    sendStates = (bool) reader.readInt32();
    nRacelines = (int) reader.readInt32();
    nGates = (int) reader.readInt32();

    if (nAgents > MAX_AGENTS)
        throw std::runtime_error(std::format("Received config for {} agents but MAX_AGENTS is set to {}. Edit this constant and recompile", nAgents, MAX_AGENTS));
    if (dim > MAX_DIM)
        throw std::runtime_error(std::format("Received config for dimension {} but MAX_DIM is set to {}. Edit this constant and recompile", dim, MAX_DIM));
    if (nRacelines > MAX_RACELINES)
        throw std::runtime_error(std::format("Received config for {} racelines but MAX_RACELINES is set to {}. Edit this constant and recompile", nRacelines, MAX_RACELINES));
    if (nGates > MAX_GATES)
        throw std::runtime_error(std::format("Received config for {} gates but MAX_GATES is set to {}. Edit this constant and recompile", nGates, MAX_GATES));

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

    nTrackSamples = (int) reader.readInt32();
    nWinLaps = (int) reader.readInt32();
    targetDistance = reader.readFloat();

    reader.readFloatArray(gateCenters);
    reader.readFloatArray(gateVectors);
    reader.readFloatArray(gateRadius);

    reader.readFloatArray(arenaMin);
    reader.readFloatArray(arenaMax);

    nObstacles = reader.readInt32();
    if (nObstacles > MAX_OBSTACLES)
        throw std::runtime_error(std::format("Received config for {} obstacles but MAX_OBSTACLES is set to {}. Edit this constant and recompile", nObstacles, MAX_OBSTACLES));
    reader.readFloatArray(obstacles);

    int seed = reader.readInt32();

    std::vector<float> trackPoints = reader.readFloatArray();

    reader.assertFinished();

    std::cout << "read nAgents " << nAgents << " nWinLaps " << nWinLaps << " max speed " << maxSpeed[0] << ' ' << maxSpeed[1] << " nTrackSamples" << nTrackSamples << "\ngate vectors:";
    for (int i = 0; i < nGates * dim; i++)
        std::cout << gateVectors[i] << (i % dim ? " " : ";  ");
    std::cout << "\n";

    return { seed, trackPoints };
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

// does not unpack opponents
static MPPIConfig unpackMPPIConfig(Reader& reader)
{
    MPPIConfig mppiconfig;

    mppiconfig.nSamples = (int) reader.readInt32();
    mppiconfig.nTimesteps = (int) reader.readInt32();
    mppiconfig.invTemperature = reader.readFloat();

    mppiconfig.samplingNoise = reader.readFloat();
    mppiconfig.gateTraversalMargin = reader.readFloat();

    mppiconfig.collDistFactor = reader.readFloat();

    mppiconfig.oppDistWeight = reader.readFloat();
    mppiconfig.oppDistPower = reader.readFloat();
    mppiconfig.oppDistThresholdFactor = reader.readFloat();
    mppiconfig.boundaryCost = reader.readFloat();
    mppiconfig.boundaryThresholdFactor = reader.readFloat();
    mppiconfig.outsideCost = reader.readFloat();
    mppiconfig.oppOutsideCost = reader.readFloat();
    mppiconfig.collisionCost = reader.readFloat();
    mppiconfig.winCost = reader.readFloat();

    mppiconfig.finalAdvWeight = reader.readFloat();
    mppiconfig.finalOppAdvWeight = reader.readFloat();
    mppiconfig.finalSpeedWeight = reader.readFloat();

    mppiconfig.minConfidence = reader.readFloat();
    mppiconfig.nModels = reader.readInt32();
    mppiconfig.oppKind = (ControllerKind) reader.readInt32();

    if (mppiconfig.nModels > MAX_MODELS)
        throw std::runtime_error(std::format("Received MPPI config for %d PID strategies but MAX_MODELS is set to %d. Edit this constant and recompile", mppiconfig.nModels, MAX_MODELS));

    for (int i = 0; i < mppiconfig.nModels; i++)
    {
        ControllerKind kind = (ControllerKind) reader.readInt32();
        if (kind != CONT_PID)
            throw std::runtime_error(std::format("For opponent %d of MPPI config, received kind %d instead of CONT_PID (%d)", i, (int) kind, (int) CONT_PID));

        mppiconfig.oppPid[i] = unpackPIDConfig(reader);
    }

    reader.readFloatArray(mppiconfig.initBelief);

    std::cout << "loaded MPPI samples " << mppiconfig.nSamples << " timesteps " << mppiconfig.nTimesteps << " collDistFactor " << mppiconfig.collDistFactor << " with " << mppiconfig.nModels << " opponent strats:\n";
    for (int i = 0; i < mppiconfig.nModels; i++)
        std::cout << "\tConfig " << i << ": initBelief " << mppiconfig.initBelief[i] << " repulsionFactor " << mppiconfig.oppPid[i].repulsionFactor << " action noise " << mppiconfig.oppPid[i].actionNoise << "\n";

    return mppiconfig;
}

void ControllerSpec::unpackHeader(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int msgKind = reader.readInt32();
    if (msgKind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for environment config (type %d) but got type %d instead", static_cast<int>(MSG_HEADER), msgKind));

    int contKind = reader.readInt32();

    if (contKind == ControllerKind::CONT_DUMMY)
        config = DummyConfig{};
    else if (contKind == ControllerKind::CONT_PID)
        config = unpackPIDConfig(reader);
    else if (contKind == ControllerKind::CONT_MPPI)
    {
        // read scalar parameters
        MPPIConfig mppiconfig = unpackMPPIConfig(reader);

        // read opponent
        // mppiconfig.oppKind = (ControllerKind) reader.readInt32();
        // if (mppiconfig.oppKind == CONT_DUMMY)
        // {
        // }
        // //     mppiconfig.opponent = DummyConfig{};
        // else if (mppiconfig.oppKind == CONT_PID)
        //     mppiconfig.oppPid = unpackPIDConfig(reader);
        // else
        //     throw std::runtime_error("MPPI opponent kind unsupported");

        config = mppiconfig;
    }

    reader.assertFinished();
}
