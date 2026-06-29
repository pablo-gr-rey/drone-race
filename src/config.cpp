#include "config.h"
#include <iostream>
#include <format>

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

    boundaryCost = reader.readFloat();
    boundaryThresholdFactor = reader.readFloat();

    outsideCost = reader.readFloat();
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

    boundaryCost = reader.readFloat();
    boundaryThresholdFactor = reader.readFloat();

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
