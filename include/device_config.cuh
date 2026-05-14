#pragma once
#include "config.h"   // for MAX_AGENTS, OpponentModelType, etc.

// CUDA-safe: no std::vector, no std::variant, no std::string
struct DeviceEnvironmentConfig
{
    int nAgents;
    int dim;
    float dt;

    int nGates;

    float minDist;
    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed[MAX_AGENTS];
    float maxAccel[MAX_AGENTS];

    int nTrackSamples;
    int nWinLaps;
    float targetDistance;

    // track points are directly allocated on the device

    int physDim;
    int actionDim;

    float gateCenters[MAX_GATES * MAX_DIM];
    float gateVectors[MAX_GATES * MAX_DIM];
    float gateRadius[MAX_GATES];

    float arenaMin[MAX_DIM];
    float arenaMax[MAX_DIM];

    DeviceEnvironmentConfig() = default;
    explicit DeviceEnvironmentConfig(const EnvironmentConfig& h)
    {
        nAgents = h.nAgents;
        dim = h.dim;
        dt = h.dt;
        nGates = h.nGates;

        minDist = h.minDist;
        posNoiseLevel = h.posNoiseLevel;
        speedNoiseLevel = h.speedNoiseLevel;
        actionNoiseLevel = h.actionNoiseLevel;

        for (int i = 0; i < MAX_AGENTS;i++)
        {
            maxSpeed[i] = h.maxSpeed[i];
            maxAccel[i] = h.maxAccel[i];
        }

        nTrackSamples = h.nTrackSamples;
        nWinLaps = h.nWinLaps;
        targetDistance = h.targetDistance;

        physDim = h.physDim;
        actionDim = h.actionDim;

        for (int i = 0; i < h.nGates; i++)
        {
            gateRadius[i] = h.gateRadius[i];
            for (int d = 0; d < h.dim; d++)
            {
                gateCenters[i * h.dim + d] = h.gateCenters[i * h.dim + d];
                gateVectors[i * h.dim + d] = h.gateVectors[i * h.dim + d];
            }
        }

        for (int d = 0; d < h.dim; d++)
        {
            arenaMin[d] = h.arenaMin[d];
            arenaMax[d] = h.arenaMax[d];
        }
    }
};

struct DeviceMPPIConfig
{
    int   nSamples;
    int   nTimesteps;
    float invTemperature;

    float samplingNoise;
    float gateTraversalMargin;

    float collDistFactor;

    // running costs
    float oppDistWeight;
    float oppDistPower;
    float oppDistThresholdFactor;
    float boundaryCost;
    float boundaryThresholdFactor;
    float outsideCost;
    float oppOutsideCost;
    float collisionCost;
    float winCost;

    // terminal costs
    float finalAdvWeight;
    float finalOppAdvWeight;
    float finalSpeedWeight;

    DeviceMPPIConfig() = default;
    explicit DeviceMPPIConfig(const MPPIConfig& h)
    {
        nSamples = h.nSamples;
        nTimesteps = h.nTimesteps;
        invTemperature = h.invTemperature;

        samplingNoise = h.samplingNoise;
        gateTraversalMargin = h.gateTraversalMargin;
        collDistFactor = h.collDistFactor;

        oppDistWeight = h.oppDistWeight;
        oppDistPower = h.oppDistPower;
        oppDistThresholdFactor = h.oppDistThresholdFactor;
        boundaryCost = h.boundaryCost;
        boundaryThresholdFactor = h.boundaryThresholdFactor;
        outsideCost = h.outsideCost;
        oppOutsideCost = h.oppOutsideCost;
        collisionCost = h.collisionCost;
        winCost = h.winCost;

        finalAdvWeight = h.finalAdvWeight;
        finalOppAdvWeight = h.finalOppAdvWeight;
        finalSpeedWeight = h.finalSpeedWeight;
    }
};
