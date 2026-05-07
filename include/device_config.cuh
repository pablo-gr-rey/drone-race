#pragma once
#include "config.h"   // for MAX_AGENTS, OpponentModelType, etc.

// CUDA-safe: no std::vector, no std::variant, no std::string
struct DeviceEnvironmentConfig
{
    int nAgents;
    int dim;
    float dt;

    float minDist;
    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed[MAX_AGENTS];
    float maxAccel[MAX_AGENTS];

    float trackWidth;
    int nTrackSamples;
    int nWinLaps;
    float targetDistance;

    int physDim;
    int actionDim;

    DeviceEnvironmentConfig() = default;
    explicit DeviceEnvironmentConfig(const EnvironmentConfig& h)
    {
        nAgents = h.nAgents;
        dim = h.dim;
        dt = h.dt;

        minDist = h.minDist;
        posNoiseLevel = h.posNoiseLevel;
        speedNoiseLevel = h.speedNoiseLevel;
        actionNoiseLevel = h.actionNoiseLevel;

        for (int i = 0;i < MAX_AGENTS;i++)
        {
            maxSpeed[i] = h.maxSpeed[i];
            maxAccel[i] = h.maxAccel[i];
        }

        trackWidth = h.trackWidth;
        nTrackSamples = h.nTrackSamples;
        nWinLaps = h.nWinLaps;
        targetDistance = h.targetDistance;

        physDim = h.physDim;
        actionDim = h.actionDim;
    }
};

struct DeviceMPPIConfig
{
    int   nSamples;
    int   nTimesteps;
    float invTemperature;

    float samplingNoise;

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
