#!/usr/bin/env python3
"""ZMQ subscriber that receives simulation data from the C++ drone_race
executable and renders it live using matplotlib.

Usage:
    python viewer.py [zmq_address]
"""

import json
import types
from typing import Any

import numpy as np
from protocol import ZMQRecv
from utils import (
    ControllerConfig,
    GateEnvironmentConfig,
    MPPIConfig,
    PIDConfig,
    PRMPPIConfig,
    SimState,
)


class NpJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, types.FunctionType):
            return None

        return super().default(o)


def tinyGateEnv(
    roundObs: bool = True, afraid: bool = False, useSplines: bool = True, usePR: bool = False
) -> tuple[GateEnvironmentConfig, SimState, ControllerConfig, list[list[str]]]:
    nAgents = 2
    dim = 2
    nTrackSamples = 512

    # startS = np.linspace(0.15, 0.0, nAgents)
    startS = np.linspace(0.1, 0.02, nAgents)
    nGates = 2

    length = 25 if roundObs else 30
    height = 0.5
    obsSize = 0.2
    raceDelay = 0.1
    margin = 0.05
    heightFactor = 3

    gateCenters, gateVectors, gateRadius = (
        np.array([[length, 0], [length * 0.01, 0]]),
        np.array([[1, 0], [1, 0]]),
        np.array([1, 1]),
    )

    repulsion = 30 if afraid else 0
    pid0 = PIDConfig(kp=5, kd=20, repulsionFactor=repulsion, racelineIndex=0, actionNoise=1)
    pid1 = PIDConfig(kp=5, kd=20, repulsionFactor=repulsion, racelineIndex=1, actionNoise=1)

    config = GateEnvironmentConfig(
        nAgents=nAgents,
        dim=dim,
        nRaceLines=2,
        nWinLaps=1,
        nGates=nGates,
        maxSpeed=np.linspace(2.05, 2.5, nAgents),
        # maxSpeed=np.linspace(2, 3, nAgents),
        maxAccel=np.linspace(3, 3, nAgents),
        nTrackSamples=nTrackSamples,
        targetDistance=0.05,
        gateCenters=gateCenters,
        gateVectors=gateVectors,
        gateRadius=gateRadius,
        minDist=1.5,
        arenaMin=np.array([-0.1 * length, -2 * height * heightFactor]),
        arenaMax=np.array([1.1 * length, 2 * height * heightFactor]),
        seed=42,
        opponentPidConfigs=(pid0, pid1),
        nModelFactors=1,
        modelSizes=np.array([2]),
        initBelief=np.array([0.5, 0.5]),
        actionNoiseLevel=0.1,
    )

    if roundObs:
        config.nRoundObstacles = 1
        config.roundObsCenters = np.array([length / 2, 0.0])
        # config.roundObsRadius = np.array([height * 1.5])
        config.roundObsRadius = np.array([height * 0.5])
    else:
        config.nObstacles = 1
        config.obstacles = np.array([length * (0.5 - obsSize / 2), -height, length * (0.5 + obsSize / 2), height])

    assert config.trackPoints is not None  # it is built automatically in GateEnvironmentConfig

    def getHeight(x):
        x_norm = x / length
        if x_norm > 1.0:
            return 0.0

        if x_norm > 0.5:
            x_norm = 1 - x_norm  # track is symmetrical relative to x = 0.5

        if x_norm < 0.5 - obsSize / 2 - raceDelay:
            return 0.0
        elif x_norm < 0.5 - obsSize / 2.0 - margin:
            x_norm_small = (x_norm - (0.5 - obsSize / 2 - raceDelay)) / (raceDelay - margin)  # between 0 and 1
            return np.sin(np.pi / 2.0 * x_norm_small) ** 2  # smooth between 0 and 1
        else:
            return 1.0

    base_x = np.linspace(0.0, length * 1.1, nTrackSamples)
    base_y = np.array([getHeight(x) for x in base_x]) * height * heightFactor

    config.trackPoints = np.concat(
        [
            np.column_stack((base_x, base_y)),
            np.column_stack((base_x, -base_y)),
        ]
    )

    initState = SimState(
        pos=np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten(),
        vel=np.zeros(config.dim * config.nAgents),
        S=np.repeat(startS, 2),
        laps=np.zeros(nAgents),
        gates=np.array([1, 1]),
    )

    # if afraid:
    #     config.init_pos = np.array([10.0, 4.0, 0.1, 0.0])a

    if useSplines and not usePR:
        mppiconfig = MPPIConfig(
            nSamples=2**16,
            # nSamples=3,
            nTimesteps=60,
            inv_temperature=10,
            # samplingNoise=3,
            samplingNoise=2,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.1,
            # collDistFactor=1.3,
            finalAdvWeight=200,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=0.0,
            # boundaryCost=0.0,
            boundaryThresholdFactor=1.7,
            outsideCost=1e6,
            winCost=1e6,
            minConfidence=0.95,
            # initBelief=np.array([0.7, 0.3]),
            nKnots=10,
        )

    elif not usePR:
        mppiconfig = MPPIConfig(
            nSamples=2**15,
            # nSamples=1,
            nTimesteps=60,
            inv_temperature=10,
            samplingNoise=3,
            # samplingNoise=0.5,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.1,
            # collDistFactor=1.3,
            finalAdvWeight=200,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=0,
            # boundaryCost=0.0,
            boundaryThresholdFactor=1,
            # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
            outsideCost=1e6,
            winCost=1e6,
            minConfidence=0.95,
            # initBelief=np.array([0.7, 0.3]),
            nKnots=15,
        )

    else:
        mppiconfig = PRMPPIConfig(
            nSamples=2**15,
            # nSamples=1,
            nTimesteps=60,
            inv_temperature=10,
            samplingNoise=3,
            # samplingNoise=0.5,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.0,
            # collDistFactor=1.3,
            finalAdvWeight=200,
            # finalAdvWeight=0,=
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=10,
            # boundaryCost=0.0,
            boundaryThresholdFactor=2,
            # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
            winCost=1e6,
            safetyWeight=1e5,
            minSafeDist=0.0,
            delta=0.1,
        )

    return config, initState, mppiconfig, [["Top", "Bottom"]]


def tinyGateEnv2Models(
    roundObs: bool = True, useSplines: bool = True, usePR: bool = False
) -> tuple[GateEnvironmentConfig, SimState, ControllerConfig, list[list[str]]]:
    nAgents = 2
    dim = 2
    nTrackSamples = 512

    # startS = np.linspace(0.15, 0.0, nAgents)
    startS = np.linspace(0.1, 0.02, nAgents)
    nGates = 2

    totLength = 20 if roundObs else 20

    height = 0.5
    heightFactor = 3

    obsLength = 0.1

    obsx1 = 0.55
    obsx2 = 0.8

    curveLength = 0.06
    curveOffset = 0.08

    gateCenters, gateVectors, gateRadius = (
        np.array([[totLength, 0], [totLength * 0.01, 0]]),
        np.array([[1, 0], [1, 0]]),
        np.array([1, 1]),
    )

    pids = [
        PIDConfig(
            kp=5,
            kd=20,
            repulsionFactor=0,
            racelineIndex=index,
            actionNoise=2,
        )
        for index in range(4)
    ]

    obstacles = np.column_stack(
        [
            totLength * np.array([obsx1 - obsLength / 2, obsx1 + obsLength / 2, obsx2 - obsLength / 2, obsx2 + obsLength / 2]),
            height * np.array([-1, 1, -1, 1]),
        ]
    )

    config = GateEnvironmentConfig(
        nAgents=nAgents,
        dim=dim,
        nRaceLines=4,
        nWinLaps=1,
        nGates=nGates,
        maxSpeed=np.linspace(2, 2.5, nAgents),
        # maxSpeed=np.linspace(2, 3, nAgents),
        maxAccel=np.linspace(3, 3, nAgents),
        nTrackSamples=nTrackSamples,
        targetDistance=0.1,
        gateCenters=gateCenters,
        gateVectors=gateVectors,
        gateRadius=gateRadius,
        minDist=0.6,
        arenaMin=np.array([-0.1 * totLength, -2 * height * heightFactor]),
        arenaMax=np.array([1.1 * totLength, 2 * height * heightFactor]),
        seed=42,
        nModelFactors=2,
        modelSizes=np.array([2, 2]),
        opponentPidConfigs=tuple(pids),
    )

    if roundObs:
        config.nRoundObstacles = 2
        config.roundObsCenters = np.array([obsx1, 0, obsx2, 0]) * totLength
        config.roundObsRadius = np.array([height, height]) * 1.5
    else:
        config.nObstacles = 2
        config.obstacles = obstacles

    def getHeightStay(x: float) -> float:
        "Goes on top and stay there for the 2 obstacles"
        x_norm = x / totLength

        if x_norm < obsx1 - obsLength / 2 - curveOffset:
            return 0.0
        elif x_norm < obsx1 - obsLength / 2 - curveOffset + curveLength:
            x_norm_s = (x_norm - obsx1 + obsLength / 2 + curveOffset) / curveLength  # between 0 and 1
            return np.sin(np.pi / 2.0 * x_norm_s) ** 2  # smooth between 0 and 1
        elif x_norm < obsx2 + obsLength / 2 + curveOffset - curveLength:
            return 1.0
        elif x_norm < obsx2 + obsLength / 2 + curveOffset:
            x_norm_s = (x_norm - obsx2 - obsLength / 2 - curveOffset + curveLength) / curveLength
            return np.sin(np.pi / 2.0 * (1 - x_norm_s))
        else:
            return 0.0

    def getHeightSwitch(x: float) -> float:
        "Goes on top then on bottom"
        x_norm = x / totLength

        if x_norm < obsx1 - obsLength / 2 - curveOffset:
            return 0.0

        elif x_norm < obsx1 - obsLength / 2 - curveOffset + curveLength:
            x_norm_s = (x_norm - obsx1 + obsLength / 2 + curveOffset) / curveLength  # between 0 and 1
            return np.sin(np.pi / 2.0 * x_norm_s) ** 2  # smooth between 0 and 1
        elif x_norm < obsx1 + obsLength / 2 + curveOffset - curveLength:
            return 1.0
        elif x_norm < obsx1 + obsLength / 2 + curveOffset:
            x_norm_s = (x_norm - obsx1 - obsLength / 2 - curveOffset + curveLength) / curveLength
            return np.sin(np.pi / 2.0 * (1 - x_norm_s))

        if x_norm < obsx2 - obsLength / 2 - curveOffset:
            return 0.0

        elif x_norm < obsx2 - obsLength / 2 - curveOffset + curveLength:
            x_norm_s = (x_norm - obsx2 + obsLength / 2 + curveOffset) / curveLength  # between 0 and 1
            return -(np.sin(np.pi / 2.0 * x_norm_s) ** 2)  # smooth between 0 and 1
        elif x_norm < obsx2 + obsLength / 2 + curveOffset - curveLength:
            return -1.0
        elif x_norm < obsx2 + obsLength / 2 + curveOffset:
            x_norm_s = (x_norm - obsx2 - obsLength / 2 - curveOffset + curveLength) / curveLength
            return -np.sin(np.pi / 2.0 * (1 - x_norm_s))

        else:
            return 0.0

    base_x = np.linspace(0.0, totLength * 1.1, nTrackSamples)
    base_y_stay = np.array([getHeightStay(x) for x in base_x]) * height * heightFactor
    base_y_switch = np.array([getHeightSwitch(x) for x in base_x]) * height * heightFactor

    config.trackPoints = np.concat(
        [
            np.column_stack((base_x, base_y_stay)),
            np.column_stack((base_x, base_y_switch)),
            np.column_stack((base_x, -base_y_stay)),
            np.column_stack((base_x, -base_y_switch)),
        ]
    )

    initState = SimState(
        pos=np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten(),
        vel=np.zeros(2),
        S=np.repeat(startS, 2),
        laps=np.zeros(nAgents),
        gates=np.array([1, 1]),
    )

    # if afraid:
    #     config.init_pos = np.array([10.0, 4.0, 0.1, 0.0])
    if useSplines and not usePR:
        mppiconfig = MPPIConfig(
            nSamples=2**15,
            # nSamples=1,
            nTimesteps=60,
            inv_temperature=10,
            # samplingNoise=3,
            samplingNoise=1,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.1,
            # collDistFactor=1.3,
            finalAdvWeight=200,
            # finalAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=0,
            # boundaryCost=0.0,
            boundaryThresholdFactor=1,
            # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
            outsideCost=1e6,
            winCost=1e6,
            minConfidence=0.95,
            # initBelief=np.array([0.7, 0.3]),
            nKnots=12,
        )

    elif not usePR:
        mppiconfig = MPPIConfig(
            nSamples=2**15,
            # nSamples=1,
            nTimesteps=60,
            inv_temperature=10,
            samplingNoise=3,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            # collDistFactor=1.1,
            collDistFactor=1.3,
            finalAdvWeight=200,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=5000,
            # boundaryCost=0.01,
            boundaryCost=100.0,
            boundaryThresholdFactor=3,
            outsideCost=1e7,
            winCost=1e5,
            minConfidence=0.95,
            # initBelief=np.array([0.5, 0.5]),
            # initBelief=np.array([0.7, 0.3]),
        )

    else:
        mppiconfig = PRMPPIConfig(
            nSamples=2**15,
            # nSamples=1,
            nTimesteps=60,
            inv_temperature=10,
            samplingNoise=3,
            # samplingNoise=0.5,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.0,
            # collDistFactor=1.3,
            finalAdvWeight=200,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=0,
            # boundaryCost=0.0,
            boundaryThresholdFactor=2,
            # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
            winCost=1e6,
            safetyWeight=1e5,
            minSafeDist=0.0,
            delta=0.1,
        )

    return config, initState, mppiconfig, [["1st=Top", "1st=Bottom"], ["2nd=Top", "2nd=Bottom"]]


def activeEnv() -> tuple[GateEnvironmentConfig, SimState, MPPIConfig, list[list[str]]]:
    nAgents = 2
    dim = 2
    nTrackSamples = 1000

    startS = np.linspace(0.2, 0.0, nAgents)
    # startS = np.linspace(0.1, 0.02, nAgents)
    nGates = 2

    length = 40
    corridor = 3
    fullHeight = 5

    gateCenters, gateVectors, gateRadius = (
        np.array([[length, 0], [0, 0]]),
        np.array([[1, 0], [1, 0]]),
        np.array([1, 1]),
    )

    pid0 = PIDConfig(
        kp=5, kd=20, repulsionFactor=120, racelineIndex=0, actionNoise=1, repulsionDistFactor=2.5, repulsionPower=1.5
    )
    pid1 = PIDConfig(kp=5, kd=20, repulsionFactor=0, racelineIndex=0, actionNoise=1, repulsionDistFactor=2.5)

    config = GateEnvironmentConfig(
        nAgents=nAgents,
        dim=dim,
        nRaceLines=1,
        nWinLaps=1,
        nGates=nGates,
        maxSpeed=np.linspace(2, 4, nAgents),
        maxAccel=np.linspace(3, 5, nAgents),
        nTrackSamples=nTrackSamples,
        targetDistance=0.05,
        gateCenters=gateCenters,
        gateVectors=gateVectors,
        gateRadius=gateRadius,
        minDist=1.9,
        arenaMin=np.array([-0.1 * length, -fullHeight]),
        arenaMax=np.array([1.1 * length, fullHeight]),
        nObstacles=2,
        obstacles=np.array(
            [
                [-0.2 * length, corridor, 1.2 * length, 1.1 * fullHeight],
                [-0.2 * length, -1.1 * fullHeight, 1.2 * length, -corridor],
            ]
        ),
        opponentPidConfigs=(pid0, pid1),
        nModelFactors=1,
        modelSizes=np.array([2]),
        initBelief=np.array([0.5, 0.5]),
    )

    config.trackPoints = np.column_stack([np.linspace(0, length * 1.2, nTrackSamples), np.zeros(nTrackSamples)])

    initState = SimState(
        pos=np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten(),
        vel=np.zeros(2),
        S=np.repeat(startS, 2),
        laps=np.zeros(nAgents),
        gates=np.array([1, 1]),
    )

    mppiconfig = MPPIConfig(
        nSamples=2**18,
        nTimesteps=60,
        inv_temperature=10,
        samplingNoise=3,
        gateTraversalMargin=0.95,
        collDistFactor=1.1,
        # collDistFactor=1.3,
        finalAdvWeight=50000,
        # finalAdvWeight=0,
        finalOppAdvWeight=0,
        # finalOppAdvWeight=500,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        outsideCost=1e6,
        winCost=100000,
        minConfidence=0.95,
    )

    return config, initState, mppiconfig, [["Afraid", "Bold"]]


def highInertiaEnv(
    afraid: bool = False, usePR: bool = False, useSplines: bool = False
) -> tuple[GateEnvironmentConfig, SimState, ControllerConfig, list[list[str]]]:
    nAgents = 2
    dim = 2
    nTrackSamples = 512

    # startS = np.linspace(0.15, 0.0, nAgents)
    startS = np.linspace(0.15, 0.02, nAgents)
    nGates = 2

    length = 30
    obsHeight = 1.5
    obsPos = 0.6
    halfSize = 0.1
    curveLength = 0.05
    trackHeight = 3

    gateCenters, gateVectors, gateRadius = (
        np.array([[length, 0], [length * 0.01, 0]]),
        np.array([[1, 0], [1, 0]]),
        np.array([1, 1]),
    )

    repulsion = 30 if afraid else 0
    pid0 = PIDConfig(kp=5, kd=20, repulsionFactor=repulsion, racelineIndex=0, actionNoise=0.01)
    pid1 = PIDConfig(kp=5, kd=20, repulsionFactor=repulsion, racelineIndex=1, actionNoise=0.01)

    config = GateEnvironmentConfig(
        nAgents=nAgents,
        dim=dim,
        dt=0.1,
        nRaceLines=2,
        nWinLaps=1,
        nGates=nGates,
        maxSpeed=np.linspace(1.2, 3, nAgents),
        # maxSpeed=np.linspace(2, 3, nAgents),
        maxAccel=np.linspace(2, 0.3, nAgents),
        nTrackSamples=nTrackSamples,
        targetDistance=0.05,
        gateCenters=gateCenters,
        gateVectors=gateVectors,
        gateRadius=gateRadius,
        minDist=2.2,
        arenaMin=np.array([-0.1 * length, -trackHeight]),
        arenaMax=np.array([1.1 * length, trackHeight]),
        seed=42,
        opponentPidConfigs=(pid0, pid1),
        nModelFactors=1,
        modelSizes=np.array([2]),
        initBelief=np.array([0.5, 0.5]),
    )

    def getHeight(x):
        x_norm = x / length
        if x_norm > 1.0:
            return 0.0

        if x_norm < obsPos - halfSize:
            return 0.0
        elif x_norm < obsPos - halfSize + curveLength:
            x_norm_small = (x_norm - (obsPos - halfSize)) / curveLength
            return np.sin(np.pi / 2.0 * x_norm_small) ** 2  # smooth between 0 and 1
        elif x_norm < obsPos + halfSize - curveLength:
            return 1.0
        elif x_norm < obsPos + halfSize:
            x_norm_small = (x_norm - (obsPos + halfSize - curveLength)) / curveLength
            return np.sin(np.pi / 2.0 * (1 - x_norm_small)) ** 2
        else:
            return 0.0

    base_x = np.linspace(0.0, length * 1.1, nTrackSamples)
    base_y = np.array([getHeight(x) for x in base_x]) * obsHeight

    config.trackPoints = np.concat(
        [
            np.column_stack((base_x, base_y)),
            np.column_stack((base_x, -base_y)),
        ]
    )

    initState = SimState(
        pos=np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten(),
        vel=np.zeros(config.dim * config.nAgents),
        S=np.repeat(startS, 2),
        laps=np.zeros(nAgents),
        gates=np.array([1, 1]),
    )

    # if afraid:
    #     config.init_pos = np.array([10.0, 4.0, 0.1, 0.0])a

    if not usePR and not useSplines:
        mppiconfig = MPPIConfig(
            useSplines=False,
            nSamples=2**16,
            # nSamples=1,
            nTimesteps=70,
            inv_temperature=1,
            samplingNoise=0.1,
            # samplingNoise=0.5,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.2,
            # collDistFactor=1.3,
            finalAdvWeight=500,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=0,
            # boundaryCost=0.0,
            boundaryThresholdFactor=1,
            outsideCost=1e6,
            winCost=1e6,
            minConfidence=0.95,
        )

    elif not usePR:
        mppiconfig = MPPIConfig(
            useSplines=True,
            nSamples=2**16,
            # nSamples=1,
            nTimesteps=70,
            inv_temperature=1,
            samplingNoise=0.05,
            # samplingNoise=0.5,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.1,
            # collDistFactor=1.2,
            finalAdvWeight=500,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            # boundaryCost=5,
            # boundaryCost=10.0,
            boundaryCost=10.0,
            boundaryThresholdFactor=1.4,
            outsideCost=1e7,
            winCost=1e6,
            minConfidence=0.95,
            nKnots=20,
        )

    else:
        mppiconfig = PRMPPIConfig(
            nSamples=2**16,
            # nSamples=1,
            nTimesteps=60,
            inv_temperature=1,
            samplingNoise=0.3,
            # samplingNoise=0.5,
            gateTraversalMargin=0.9,  # restrict 5% on each side
            collDistFactor=1.1,
            # collDistFactor=1.3,
            finalAdvWeight=500,
            # finalAdvWeight=0,
            finalOppAdvWeight=0,
            # finalOppAdvWeight=500,
            boundaryCost=10.0,
            # boundaryCost=0.0,
            boundaryThresholdFactor=1.4,
            # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
            safetyWeight=1e4,
            delta=0.1,
        )

    return config, initState, mppiconfig, [["Top", "Bottom"]]


def mainGate():
    # envConfig, mppiconfig, pid0, pid1, oppNames = standardGateEnv()  # pid0 = afraid; pid1 = bold
    # envConfig, mppiconfig, pids, oppNames = tinyGateEnv(afraid=False, useSplines=True)  # pid0 = top; pid1 = bottom
    # envConfig, mppiconfig, pids, oppNames = tinyGateEnv(afraid=False, useSplines=False)  # pid0 = top; pid1 = bottom
    # envConfig, mppiconfig, pids, oppNames = tinyGateEnv(afraid=False, usePR=True)  # pid0 = top; pid1 = bottom
    # envConfig, mppiconfig, pids, oppNames = tinyGateEnv2Models(roundObs=True, usePR=False)
    # envConfig, mppiconfig, pid0, pid1, oppNames = activeEnv()  # pid0 = afraid; pid1 = bold

    envConfig, initState, mppiconfig, oppNames = highInertiaEnv(usePR=False, useSplines=True)

    envConfig.trueTheta = 1
    envConfig.iMppi = 1

    if isinstance(mppiconfig, MPPIConfig):
        mppiconfig.nVerifSamples = 2**17
        mppiconfig.beta = 1e-5
        mppiconfig.verifHorizon = 60
        mppiconfig.maxVerifEps = 0.0001
        # mppiconfig.maxVerifEps = 10

    envConfig.sendStates = True

    z = ZMQRecv()

    evt, res = z.runSim(envConfig, mppiconfig, initState, render=envConfig.sendStates, oppNames=oppNames)

    print(f"result: {evt.name} {res}")

    z.close()


if __name__ == "__main__":
    # mainZMQ()
    mainGate()
    # buildOptimalRaceline()
    # computeResults(1, False)
    # plotResults(1, False)
