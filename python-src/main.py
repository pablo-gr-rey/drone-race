#!/usr/bin/env python3
"""ZMQ subscriber that receives simulation data from the C++ drone_race
executable and renders it live using matplotlib.

Usage:
    python viewer.py [zmq_address]
"""

import dataclasses
import json
import sys
import types
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import zmq
from controllers import DummyController, MPPIConfig, PIDConfig
from environment import EnvironmentRenderer, TrackEnvironment, TrackEnvironmentConfig
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from protocol import EVENT_TYPE, MSG_TYPE, ByteUnpacker, ZMQRecv, encodeConfig, unpackState
from tqdm import tqdm
from utils import ControllerConfig, DummyConfig, lissajous


class NpJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, types.FunctionType):
            return None

        return super().default(o)


def buildOptimalRaceline():
    nAgents = 1
    dim = 2
    trackWidth = 1

    centerline: Callable[[float], np.ndarray] = lambda s: lissajous(s, 10, 2, 8)  # noqa: E731

    startS = [0.0]

    init_state = [
        centerline(startS[iAgent])[d] * (1 - isSpeed) for iAgent in range(nAgents) for d in range(dim) for isSpeed in range(2)
    ]

    config = TrackEnvironmentConfig(
        # centerline=roundTrack,
        centerline=centerline,
        nAgents=nAgents,
        nTrackSamples=1000,
        init_state=init_state,
        add_state=(np.array(startS), np.zeros(nAgents)),
        # posNoiseLevel=0.01,
        # speedNoiseLevel=0.05,
        # actionNoiseLevel=0.1,
        trackWidth=trackWidth,
        # trackWidth=5,
        maxAccel=np.array([5]),
        # maxAccel=np.array([0.2]),
        # maxAccel=np.array([5 - 3 * i / (nAgents - 1) for i in range(nAgents)]),
        # maxSpeed=np.array([2 - 0 * i / (nAgents - 1) for i in range(nAgents)]),
        maxSpeed=np.array([2]),
        # maxSpeed=np.array([2, 2]),
        targetDistance=0.02,
        nWinLaps=1,
        minDist=1.2,
        # minDist=1.75,
        dt=0.1,
    )

    mppisolo = MPPIConfig(
        nSamples=100000,
        nTimesteps=50,
        samplingNoise=np.max(config.maxAccel) * 0.5,
        inv_temperature=0.5,
        finalAdvWeight=100,
        finalSpeedWeight=0,
        oppDistWeight=0,
        boundaryCost=0,
        collisionCost=0,
        oppOutsideCost=0,
        outsideCost=100000,
        winCost=10000,
        # winCost=0,
    )

    mppisolo.opponentConfig = DummyConfig()

    z = ZMQRecv()
    z.runSim(config, [mppisolo])

    posLog = [phys[::2] for phys in z.stateLog]
    print(posLog[:10])

    info = {
        "title": f"Optimal raceline with trackWidth={trackWidth}",
        "trackWidth": trackWidth,
        "length": len(posLog),
        "points": posLog,
        "envConfig": dataclasses.asdict(config),
        "mppiConfig": dataclasses.asdict(mppisolo),
    }

    with open(f"data/opt-line-trackWidth-{trackWidth}.json", "w") as f:
        json.dump(info, f, cls=NpJsonEncoder, indent=4)


def loadRaceline(trackWidth: float | int) -> np.ndarray:
    with open(f"data/opt-line-trackWidth-{trackWidth}.json") as f:
        data = json.load(f)

    return data["points"]


def mainZMQ():
    nAgents = 2
    # nAgents = 1
    dim = 2

    centerline: Callable[[float], np.ndarray] = lambda s: lissajous(s, 10, 2, 8)  # noqa: E731
    raceline = loadRaceline(2)
    # centerline: Callable[[float], np.ndarray] = lambda s: flower(s, 10.0, 2.0)

    if nAgents > 1:
        startS = [0.1 * i / (nAgents - 1) for i in range(nAgents)]
    else:
        startS = [0.0]

    init_state = [
        centerline(startS[iAgent])[d] * (1 - isSpeed) for iAgent in range(nAgents) for d in range(dim) for isSpeed in range(2)
    ]

    config = TrackEnvironmentConfig(
        # centerline=roundTrack,
        centerline=centerline,
        nAgents=nAgents,
        nTrackSamples=1000,
        init_state=init_state,
        add_state=(np.array(startS), np.zeros(nAgents)),
        # posNoiseLevel=0.01,
        # speedNoiseLevel=0.05,
        # actionNoiseLevel=0.1,
        trackWidth=3,
        # trackWidth=5,
        maxAccel=np.array([5, 3]),
        # maxAccel=np.array([0.2]),
        # maxAccel=np.array([5 - 3 * i / (nAgents - 1) for i in range(nAgents)]),
        # maxSpeed=np.array([2 - 0 * i / (nAgents - 1) for i in range(nAgents)]),
        maxSpeed=np.array([2, 1.5]),
        # maxSpeed=np.array([2, 2]),
        targetDistance=0.02,
        nWinLaps=1,
        minDist=1.2,
        # minDist=1.75,
        dt=0.1,
    )

    mppiconfig = MPPIConfig(
        nSamples=10000,
        nTimesteps=40,
        inv_temperature=2,
        samplingNoise=3,
        # collDistFactor=1.5,
        collDistFactor=1,
        finalAdvWeight=200,
        # finalAdvWeight=0,
        # finalSpeedWeight=50,
        finalSpeedWeight=0,
        # oppDistWeight=0.01,
        oppDistWeight=0.0,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=50,
        # finalOppAdvWeight=550,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        # oppOutsideCost=1000,
        oppOutsideCost=1000,
        outsideCost=5000,
        collisionCost=1000,
        winCost=1000,
    )

    mppiconfig2 = MPPIConfig(
        nSamples=10000,
        nTimesteps=30,
        samplingNoise=3,
        collDistFactor=1.5,
        inv_temperature=2,
        # finalAdvWeight=200,
        finalAdvWeight=0,
        finalSpeedWeight=50,
        # oppDistWeight=0.01,
        oppDistWeight=0.0,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=400,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        oppOutsideCost=1000,
        # oppOutsideCost=0,
        outsideCost=1000,
        # collisionCost=10,
        collisionCost=0,
        winCost=1000,
    )

    assert config.trackPoints is not None
    config.nRaceLines = 2
    config.trackPoints = np.concatenate([config.trackPoints, raceline])

    mppisolo = MPPIConfig(
        nSamples=100000,
        nTimesteps=50,
        samplingNoise=np.max(config.maxAccel) * 0.5,
        inv_temperature=1,
        finalAdvWeight=100,
        finalSpeedWeight=0,
        oppDistWeight=0,
        boundaryCost=0,
        collisionCost=0,
        oppOutsideCost=0,
        outsideCost=100000,
        # winCost=10000,
        winCost=0,
    )

    pidconfig = PIDConfig(kp=10, kd=5, repulsionFactor=20)
    blindpidconfig = PIDConfig(kp=10, kd=5, repulsionFactor=0)

    dummyconfig = DummyConfig()

    # mppiconfig.opponentConfig = blindpidconfig
    mppiconfig.opponentConfig = pidconfig
    # mppiconfig.opponentConfig = dummyconfig
    # mppiconfig2.opponentConfig = pidconfig
    # mppiconfig.opponentConfig = blindpidconfig

    cont_configs: list[ControllerConfig] = [pidconfig, mppiconfig]
    # cont_configs: list[ControllerConfig] = [blindpidconfig, mppiconfig]
    # cont_configs: list[ControllerConfig] = [mppiconfig, mppiconfig2]
    # cont_configs: list[ControllerConfig] = [mppiconfig, blindpidconfig]
    # cont_configs: list[ControllerConfig] = [mppiconfig, blindpidconfig]
    # cont_configs: list[ControllerConfig] = [mppiconfig] + [pidconfig] * (nAgents - 1)  # type: ignore

    config.sendStates = True

    z = ZMQRecv()

    evt, res = z.runSim(config, cont_configs, render=config.sendStates)

    print(f"result: {evt.name} {res}")

    z.close()


def computeSensResult(collDistFactor: float = 1, fastMPPI: bool = True):
    nAgents = 2
    dim = 2

    centerline: Callable[[float], np.ndarray] = lambda s: lissajous(s, 10, 2, 8)  # noqa: E731
    # centerline: Callable[[float], np.ndarray] = lambda s: flower(s, 10.0, 2.0)

    startS = [0.0, 0.1] if fastMPPI else [0.1, 0.0]
    # values = [0.0, 5.0, 10.0]
    values = np.linspace(0, 20, 5)

    init_state = [
        centerline(startS[iAgent])[d] * (1 - isSpeed) for iAgent in range(nAgents) for d in range(dim) for isSpeed in range(2)
    ]

    config = TrackEnvironmentConfig(
        # centerline=roundTrack,
        centerline=centerline,
        nAgents=nAgents,
        nTrackSamples=1000,
        init_state=init_state,
        add_state=(np.array(startS), np.zeros(nAgents)),
        # posNoiseLevel=0.01,
        # speedNoiseLevel=0.05,
        # actionNoiseLevel=0.1,
        trackWidth=3,
        # trackWidth=5,
        maxAccel=np.array([5, 3]) if fastMPPI else np.array([3, 5]),
        # maxAccel=np.array([0.2]),
        # maxAccel=np.array([5 - 3 * i / (nAgents - 1) for i in range(nAgents)]),
        # maxSpeed=np.array([2 - 0 * i / (nAgents - 1) for i in range(nAgents)]),
        maxSpeed=np.array([2, 1.5]) if fastMPPI else np.array([1.5, 2]),
        # maxSpeed=np.array([2, 2]),
        targetDistance=0.02,
        nWinLaps=1,
        minDist=1.5,
        # minDist=1.75,
        dt=0.1,
    )

    mppiconfig = MPPIConfig(
        nSamples=10000,
        nTimesteps=40,
        inv_temperature=2,
        samplingNoise=3,
        # collDistFactor=1.5,
        collDistFactor=collDistFactor,
        finalAdvWeight=200,
        # finalAdvWeight=0,
        # finalSpeedWeight=50,
        finalSpeedWeight=0,
        # oppDistWeight=0.01,
        oppDistWeight=0.0,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=50,
        # finalOppAdvWeight=550,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        # oppOutsideCost=1000,
        oppOutsideCost=1000,
        outsideCost=5000,
        collisionCost=1000,
        winCost=1000,
    )

    config.sendStates = False

    z = ZMQRecv()

    pidconfigReal = PIDConfig(kp=10, kd=5, repulsionFactor=20)
    pidconfigSupp = PIDConfig(kp=10, kd=5, repulsionFactor=0)

    mppiconfig.opponentConfig = pidconfigSupp

    cont_configs: list[ControllerConfig] = [mppiconfig, pidconfigReal]

    # results[realSens][suppSens]
    results: list[list[tuple[EVENT_TYPE, int]]] = [
        [(EVENT_TYPE.EVT_TRUNCATED, -1) for i in range(len(values))] for j in range(len(values))
    ]

    with tqdm(desc="Running simulation grid...", total=len(values) ** 2) as pbar:
        for i, realSens in enumerate(values):
            for j, suppSens in enumerate(values):
                pidconfigReal.repulsionFactor = realSens
                pidconfigSupp.repulsionFactor = suppSens

                results[i][j] = z.runSim(config, cont_configs, render=config.sendStates)

                tqdm.write(f"result for real {realSens} supp {suppSens}: {results[i][j][0].name, results[i][j][1]}")
                pbar.update(1)

    # before dumping to JSON
    config.centerline = None
    mppiconfig.opponentPredictors = []

    desc = "fast" if fastMPPI else "slow"

    info = {
        "title": f"Sensitivity analysis with collDistFactor={mppiconfig.collDistFactor} ({desc} MPPI)",
        "values": values,
        "info": f"{desc} MPPI, fast PID. values[i][j] is with real = values[i], supposed = values[j]",
        "results": results,
        "envConfig": dataclasses.asdict(config),
        "mppiConfig": dataclasses.asdict(mppiconfig),
        "pidConfig": dataclasses.asdict(pidconfigReal),
    }

    with open(f"data/sens-collDistFactor-{collDistFactor}-{desc}.json", "w") as f:
        json.dump(info, f, cls=NpJsonEncoder, indent=4)

    z.close()


def plotSensResults(collDistFactor: float = 1, fastMPPI: bool = True):
    desc = "fast" if fastMPPI else "slow"

    def getValue(evt_type: EVENT_TYPE, arg: int) -> int:
        if evt_type == EVENT_TYPE.EVT_COLLISION:
            return 2
        elif evt_type == EVENT_TYPE.EVT_OUTSIDE:
            return 3 + arg
        elif evt_type == EVENT_TYPE.EVT_WINNER:
            return arg
        return 5

    with open(f"data/sens-collDistFactor-{collDistFactor}-{desc}.json") as f:
        info = json.load(f)

    # Example data
    results = info["results"]
    results_val = [[getValue(*x) for x in ln] for ln in results]

    # Coordinates (same length as results dimension)
    values = info["values"]

    # colors[k] is used for value k in results
    labels = ["MPPI wins", "PID wins", "Collision", "MPPI outside", "PID outside", "Truncated"]
    colors = ["lawngreen", "blue", "red", "darkblue", "darkgreen", "orange"]

    # Build discrete colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), len(labels))  # bins centered at 0,1,...,5

    fig, ax = plt.subplots(figsize=(7, 5))

    # extent maps array indices to your coordinate values
    # each cell centered on values[i], values[j]
    dx = values[1] - values[0] if len(values) > 1 else 1
    x0, x1 = values[0] - dx / 2, values[-1] + dx / 2
    y0, y1 = values[0] - dx / 2, values[-1] + dx / 2

    ax.imshow(results_val, cmap=cmap, norm=norm, origin="lower", interpolation="nearest", extent=(x0, x1, y0, y1), aspect="equal")

    # Ticks at your coordinate values
    ax.set_xticks(values)
    ax.set_yticks(values)
    ax.set_xlabel("Supposed avoidance")
    ax.set_ylabel("Real avoidance")
    ax.set_title(info["title"])

    # Optional: draw cell borders
    ax.set_xticks(np.linspace(x0, x1, len(results[0]) + 1), minor=True)
    ax.set_yticks(np.linspace(y0, y1, len(results) + 1), minor=True)
    ax.grid(which="minor", color="k", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Legend
    handles = [Patch(facecolor=colors[k], edgecolor="k", label=labels[k]) for k in range(6)]
    ax.legend(handles=handles, title="Value", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"figures/sens-collDistFactor-{collDistFactor}.png")
    plt.show()


if __name__ == "__main__":
    mainZMQ()
    # buildOptimalRaceline()
    # computeResults(1, False)
    # plotResults(1, False)
