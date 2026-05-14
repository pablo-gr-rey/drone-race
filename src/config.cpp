#include "config.h"
#include <iostream>
#include <format>

// static void print_vector(std::string name, std::vector<float> vec)
// {
//     std::cout << name << " size " << vec.size() << "\t";
//     for (float v : vec)
//         std::cout << v << " ";
//     std::cout << "\n";
// }

void EnvironmentConfig::unpackHeader(const void* buf, size_t len)
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

    initPos = reader.readFloatArray();
    initSpeed = reader.readFloatArray();

    initS = reader.readFloatArray();
    initLaps = reader.readIntArray();
    initGates = reader.readIntArray();

    minDist = reader.readFloat();
    posNoiseLevel = reader.readFloat();
    speedNoiseLevel = reader.readFloat();
    actionNoiseLevel = reader.readFloat();

    std::vector<float> ms = reader.readFloatArray(); // expecting length nAgents
    std::vector<float> ma = reader.readFloatArray(); // expecting length nAgents

    if ((int) ms.size() != nAgents || (int) ma.size() != nAgents)
        throw std::runtime_error("Error unpacking EnvironmentConfig: maxSpeed/maxAccel length must equal nAgents");
    if (nAgents > MAX_AGENTS)
        throw std::runtime_error("Error unpacking EnvironmentConfig: nAgents > MAX_AGENTS");

    std::memcpy(maxSpeed, ms.data(), ms.size() * sizeof(float));
    std::memcpy(maxAccel, ma.data(), ma.size() * sizeof(float));

    nTrackSamples = (int) reader.readInt32();
    nWinLaps = (int) reader.readInt32();
    targetDistance = reader.readFloat();

    gateCenters = reader.readFloatArray();
    gateVectors = reader.readFloatArray();
    gateRadius = reader.readFloatArray();

    arenaMin = reader.readFloatArray();
    arenaMax = reader.readFloatArray();

    trackPoints = reader.readFloatArray();

    reader.assertFinished();

    std::cout << "read nAgents " << nAgents << " actionNoiseLevel " << actionNoiseLevel << " max speed " << maxSpeed[0] << ' ' << maxSpeed[1] << " length of track points " << trackPoints.size() << "\ngate vectors:";
    for (float val : gateVectors)
        std::cout << val << " ";
    std::cout << "\n";

    recompute();
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

    std::cout << "loaded PID repulsionFactor " << pidconfig.repulsionFactor << " racelineIndex " << pidconfig.racelineIndex << '\n';

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

    if (contKind == CONT_DUMMY)
        config = DummyConfig{};
    else if (contKind == CONT_PID)
        config = unpackPIDConfig(reader);
    else if (contKind == CONT_MPPI)
    {
        // read scalar parameters
        MPPIConfig mppiconfig = unpackMPPIConfig(reader);

        // read opponent
        int oppKind = (int) reader.readInt32();
        if (oppKind == CONT_DUMMY)
            mppiconfig.opponent = DummyConfig{};
        else if (oppKind == CONT_PID)
            mppiconfig.opponent = unpackPIDConfig(reader);
        else
            throw std::runtime_error("MPPI opponent kind unsupported");

        config = mppiconfig;
    }

    reader.assertFinished();
}
