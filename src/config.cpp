#include "config.h"
#include <iostream>
#include <format>

std::vector<float> EnvironmentConfig::unpackHeader(const void* buf, size_t len)
{
    Reader reader(buf, len);

    int kind = reader.readInt32();
    if (kind != MSG_HEADER)
        throw std::runtime_error(std::format("Expected header message type for environment config (type %d) but got type %d instead", static_cast<int>(MSG_HEADER), kind));

    nAgents = (int) reader.readInt32();
    dim = (int) reader.readInt32();
    dt = reader.readFloat();

    sendStates = (bool) reader.readInt32();
    nRacelines = (int) reader.readInt32();
    nGates = (int) reader.readInt32();

    if (nAgents > MAX_AGENTS)
        throw std::runtime_error(std::format("Received config for %d agents but MAX_AGENTS is set to %d. Edit this constant and recompile", nAgents, MAX_AGENTS));
    if (dim > MAX_DIM)
        throw std::runtime_error(std::format("Received config for dimension %d but MAX_DIM is set to %d. Edit this constant and recompile", dim, MAX_DIM));
    if (nGates > MAX_GATES)
        throw std::runtime_error(std::format("Received config for %d gates but MAX_GATES is set to %d. Edit this constant and recompile", nGates, MAX_GATES));

    // initPos = reader.readFloatArray();
    // initSpeed = reader.readFloatArray();

    // initS = reader.readFloatArray();
    // initLaps = reader.readIntArray();
    // initGates = reader.readIntArray();

    reader.readFloatArray(initPos);
    reader.readFloatArray(initSpeed);

    reader.readFloatArray(initS);
    reader.readIntArray(initLaps);
    reader.readIntArray(initGates);

    minDist = reader.readFloat();
    posNoiseLevel = reader.readFloat();
    speedNoiseLevel = reader.readFloat();
    actionNoiseLevel = reader.readFloat();

    // std::vector<float> ms = reader.readFloatArray(); // expecting length nAgents
    // std::vector<float> ma = reader.readFloatArray(); // expecting length nAgents

    // if ((int) ms.size() != nAgents || (int) ma.size() != nAgents)
    //     throw std::runtime_error("Error unpacking EnvironmentConfig: maxSpeed/maxAccel length must equal nAgents");
    // if (nAgents > MAX_AGENTS)
    //     throw std::runtime_error("Error unpacking EnvironmentConfig: nAgents > MAX_AGENTS");

    // std::memcpy(maxSpeed, ms.data(), ms.size() * sizeof(float));
    // std::memcpy(maxAccel, ma.data(), ma.size() * sizeof(float));

    reader.readFloatArray(maxSpeed);
    reader.readFloatArray(maxAccel);

    nTrackSamples = (int) reader.readInt32();
    nWinLaps = (int) reader.readInt32();
    targetDistance = reader.readFloat();

    // gateCenters = reader.readFloatArray();
    // gateVectors = reader.readFloatArray();
    // gateRadius = reader.readFloatArray();

    // arenaMin = reader.readFloatArray();
    // arenaMax = reader.readFloatArray();

    reader.readFloatArray(gateCenters);
    reader.readFloatArray(gateVectors);
    reader.readFloatArray(gateRadius);

    reader.readFloatArray(arenaMin);
    reader.readFloatArray(arenaMax);

    // float* trackPoints = (float*) malloc(nRacelines * nTrackSamples * dim * sizeof(float));
    // reader.readFloatArray(trackPoints);
    std::vector<float> trackPoints = reader.readFloatArray();

    reader.assertFinished();

    std::cout << "read nAgents " << nAgents << " nWinLaps " << nWinLaps << " max speed " << maxSpeed[0] << ' ' << maxSpeed[1] << " nTrackSamples" << nTrackSamples << "\ngate vectors:";
    for (float val : gateVectors)
        std::cout << val << " ";
    std::cout << "\n";

    return trackPoints;
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

    std::cout << "loaded MPPI samples " << mppiconfig.nSamples << " samplingNoise " << mppiconfig.samplingNoise << " finalAdvWeight " << mppiconfig.finalAdvWeight << " finalOppAdvWeight " << mppiconfig.finalOppAdvWeight << '\n';

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
        mppiconfig.oppKind = (ControllerKind) reader.readInt32();
        if (mppiconfig.oppKind == CONT_DUMMY)
        {
        }
        //     mppiconfig.opponent = DummyConfig{};
        else if (mppiconfig.oppKind == CONT_PID)
            mppiconfig.oppPid = unpackPIDConfig(reader);
        else
            throw std::runtime_error("MPPI opponent kind unsupported");

        config = mppiconfig;
    }

    reader.assertFinished();
}
