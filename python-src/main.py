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
    CONTROLLER_TYPE,
    ControllerConfig,
    GateEnvironmentConfig,
    MPPIConfig,
    PIDConfig,
    VerifConfig,
    circularGateTrack,
)


class NpJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, types.FunctionType):
            return None

        return super().default(o)


# TODO: adapt this to gate environment

# def buildOptimalRaceline():
#     nAgents = 1
#     dim = 2
#     trackWidth = 1

#     centerline: Callable[[float], np.ndarray] = lambda s: lissajous(s, 10, 2, 8)  # noqa: E731

#     startS = [0.0]

#     init_state = [
#         centerline(startS[iAgent])[d] * (1 - isSpeed) for iAgent in range(nAgents) for d in range(dim) for isSpeed in range(2)
#     ]

#     config = TrackEnvironmentConfig(
#         # centerline=roundTrack,
#         centerline=centerline,
#         nAgents=nAgents,
#         nTrackSamples=1000,
#         init_state=init_state,
#         add_state=(np.array(startS), np.zeros(nAgents)),
#         # posNoiseLevel=0.01,
#         # speedNoiseLevel=0.05,
#         # actionNoiseLevel=0.1,
#         trackWidth=trackWidth,
#         # trackWidth=5,
#         maxAccel=np.array([5]),
#         # maxAccel=np.array([0.2]),
#         # maxAccel=np.array([5 - 3 * i / (nAgents - 1) for i in range(nAgents)]),
#         # maxSpeed=np.array([2 - 0 * i / (nAgents - 1) for i in range(nAgents)]),
#         maxSpeed=np.array([2]),
#         # maxSpeed=np.array([2, 2]),
#         targetDistance=0.02,
#         nWinLaps=1,
#         minDist=1.2,
#         # minDist=1.75,
#         dt=0.1,
#     )

#     mppisolo = MPPIConfig(
#         nSamples=100000,
#         nTimesteps=50,
#         samplingNoise=np.max(config.maxAccel) * 0.5,
#         inv_temperature=0.5,
#         finalAdvWeight=100,
#         finalSpeedWeight=0,
#         oppDistWeight=0,
#         boundaryCost=0,
#         collisionCost=0,
#         oppOutsideCost=0,
#         outsideCost=100000,
#         winCost=10000,
#         # winCost=0,
#     )

#     mppisolo.opponentConfig = DummyConfig()

#     z = ZMQRecv()
#     z.runSim(config, [mppisolo])

#     posLog = z.posLog
#     print(posLog[:10])

#     info = {
#         "title": f"Optimal raceline with trackWidth={trackWidth}",
#         "trackWidth": trackWidth,
#         "length": len(posLog),
#         "points": posLog,
#         "envConfig": dataclasses.asdict(config),
#         "mppiConfig": dataclasses.asdict(mppisolo),
#     }

#     with open(f"../data/opt-line-trackWidth-{trackWidth}.json", "w") as f:
#         json.dump(info, f, cls=NpJsonEncoder, indent=4)


# def loadRaceline(centerline: np.ndarray, trackWidth: float | int) -> np.ndarray:
#     "load the raceline and return its sampled version arr of shape nSamples*dim, such that arr[k] is the point of the raceline (linearly interpolated) closest to centerline[k]. this allows keeping a single S value across the track for raceline-following PID"
#     with open(f"../data/opt-line-trackWidth-{trackWidth}.json") as f:
#         data = json.load(f)

#     points = np.array(data["points"])

#     # we want to interpolate points to have nSamples data and keep them regularly spaced
#     # since points are generated from a trajectory, they won't have the same distance between two consecutive points

#     nSamples, dim = centerline.shape

#     seg_start = points
#     seg_end = np.roll(points, -1, axis=0)

#     seg_vec = seg_end - seg_start
#     seg_len2 = np.clip(np.sum(seg_vec**2, axis=1), a_min=1e-12, a_max=None)

#     new_points = np.zeros((nSamples, dim))

#     for k in range(nSamples):
#         p = centerline[k]
#         w = p - seg_start
#         t = np.sum(w * seg_vec, axis=1) / seg_len2
#         t = np.clip(t, 0.0, 1.0)
#         candidates = seg_start + t[:, None] * seg_vec
#         d2 = np.sum((candidates - p)**2, axis=1)
#         new_points[k] = candidates[np.argmin(d2)]

#     return new_points

#     # # we compute the cumulated distance between points
#     # diffs = np.diff(points, axis=0)
#     # distances = np.sqrt(np.sum(diffs**2, axis=1))
#     # accumulated_dist = np.concatenate(([0], np.cumsum(distances)))

#     # # new grid samples
#     # new_dist = np.linspace(0, accumulated_dist[-1], nSamples)

#     # # interpolate
#     # new_points = np.zeros((nSamples, dim))
#     # for d in range(dim):
#     #     new_points[:, d] = np.interp(new_dist, accumulated_dist, points[:, d])

#     # return new_points

# def computeSensResult(collDistFactor: float = 1, fastMPPI: bool = True):
#     nAgents = 2
#     dim = 2

#     centerline: Callable[[float], np.ndarray] = lambda s: lissajous(s, 10, 2, 8)  # noqa: E731
#     # centerline: Callable[[float], np.ndarray] = lambda s: flower(s, 10.0, 2.0)

#     startS = [0.0, 0.1] if fastMPPI else [0.1, 0.0]
#     # values = [0.0, 5.0, 10.0]
#     values = np.linspace(0, 20, 5)

#     init_state = [
#         centerline(startS[iAgent])[d] * (1 - isSpeed) for iAgent in range(nAgents) for d in range(dim) for isSpeed in range(2)
#     ]

#     config = TrackEnvironmentConfig(
#         # centerline=roundTrack,
#         centerline=centerline,
#         nAgents=nAgents,
#         nTrackSamples=1000,
#         init_state=init_state,
#         add_state=(np.array(startS), np.zeros(nAgents)),
#         # posNoiseLevel=0.01,
#         # speedNoiseLevel=0.05,
#         # actionNoiseLevel=0.1,
#         trackWidth=3,
#         # trackWidth=5,
#         maxAccel=np.array([5, 3]) if fastMPPI else np.array([3, 5]),
#         # maxAccel=np.array([0.2]),
#         # maxAccel=np.array([5 - 3 * i / (nAgents - 1) for i in range(nAgents)]),
#         # maxSpeed=np.array([2 - 0 * i / (nAgents - 1) for i in range(nAgents)]),
#         maxSpeed=np.array([2, 1.5]) if fastMPPI else np.array([1.5, 2]),
#         # maxSpeed=np.array([2, 2]),
#         targetDistance=0.02,
#         nWinLaps=1,
#         minDist=1.5,
#         # minDist=1.75,
#         dt=0.1,
#     )

#     mppiconfig = MPPIConfig(
#         nSamples=10000,
#         nTimesteps=40,
#         inv_temperature=2,
#         samplingNoise=3,
#         # collDistFactor=1.5,
#         collDistFactor=collDistFactor,
#         finalAdvWeight=200,
#         # finalAdvWeight=0,
#         # finalSpeedWeight=50,
#         finalSpeedWeight=0,
#         # oppDistWeight=0.01,
#         oppDistWeight=0.0,
#         oppDistThresholdFactor=2,
#         finalOppAdvWeight=50,
#         # finalOppAdvWeight=550,
#         # boundaryCost=0.01,
#         boundaryCost=0.0,
#         boundaryThresholdFactor=2,
#         # oppOutsideCost=1000,
#         oppOutsideCost=1000,
#         outsideCost=5000,
#         collisionCost=1000,
#         winCost=1000,
#     )

#     config.sendStates = False

#     z = ZMQRecv()

#     pidconfigReal = PIDConfig(kp=10, kd=5, repulsionFactor=20)
#     pidconfigSupp = PIDConfig(kp=10, kd=5, repulsionFactor=0)

#     mppiconfig.opponentConfig = pidconfigSupp

#     cont_configs: list[ControllerConfig] = [mppiconfig, pidconfigReal]

#     # results[realSens][suppSens]
#     results: list[list[tuple[EVENT_TYPE, int]]] = [
#         [(EVENT_TYPE.EVT_TRUNCATED, -1) for i in range(len(values))] for j in range(len(values))
#     ]

#     with tqdm(desc="Running simulation grid...", total=len(values) ** 2) as pbar:
#         for i, realSens in enumerate(values):
#             for j, suppSens in enumerate(values):
#                 pidconfigReal.repulsionFactor = realSens
#                 pidconfigSupp.repulsionFactor = suppSens

#                 results[i][j] = z.runSim(config, cont_configs, render=config.sendStates)

#                 tqdm.write(f"result for real {realSens} supp {suppSens}: {results[i][j][0].name, results[i][j][1]}")
#                 pbar.update(1)

#     # before dumping to JSON
#     config.centerline = None
#     mppiconfig.opponentPredictors = []

#     desc = "fast" if fastMPPI else "slow"

#     info = {
#         "title": f"Sensitivity analysis with collDistFactor={mppiconfig.collDistFactor} ({desc} MPPI)",
#         "values": values,
#         "info": f"{desc} MPPI, fast PID. values[i][j] is with real = values[i], supposed = values[j]",
#         "results": results,
#         "envConfig": dataclasses.asdict(config),
#         "mppiConfig": dataclasses.asdict(mppiconfig),
#         "pidConfig": dataclasses.asdict(pidconfigReal),
#     }

#     with open(f"../data/sens-collDistFactor-{collDistFactor}-{desc}.json", "w") as f:
#         json.dump(info, f, cls=NpJsonEncoder, indent=4)

#     z.close()


# def plotSensResults(collDistFactor: float = 1, fastMPPI: bool = True):
#     desc = "fast" if fastMPPI else "slow"

#     def getValue(evt_type: EVENT_TYPE, arg: int) -> int:
#         if evt_type == EVENT_TYPE.EVT_COLLISION:
#             return 2
#         elif evt_type == EVENT_TYPE.EVT_OUTSIDE:
#             return 3 + arg
#         elif evt_type == EVENT_TYPE.EVT_WINNER:
#             return arg
#         return 5

#     with open(f"../data/sens-collDistFactor-{collDistFactor}-{desc}.json") as f:
#         info = json.load(f)

#     # Example data
#     results = info["results"]
#     results_val = [[getValue(*x) for x in ln] for ln in results]

#     # Coordinates (same length as results dimension)
#     values = info["values"]

#     # colors[k] is used for value k in results
#     labels = ["MPPI wins", "PID wins", "Collision", "MPPI outside", "PID outside", "Truncated"]
#     colors = ["lawngreen", "blue", "red", "darkblue", "darkgreen", "orange"]

#     # Build discrete colormap
#     cmap = ListedColormap(colors)
#     norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), len(labels))  # bins centered at 0,1,...,5

#     fig, ax = plt.subplots(figsize=(7, 5))

#     # extent maps array indices to your coordinate values
#     # each cell centered on values[i], values[j]
#     dx = values[1] - values[0] if len(values) > 1 else 1
#     x0, x1 = values[0] - dx / 2, values[-1] + dx / 2
#     y0, y1 = values[0] - dx / 2, values[-1] + dx / 2

#     ax.imshow(results_val, cmap=cmap, norm=norm, origin="lower", interpolation="nearest", extent=(x0, x1, y0, y1), aspect="equal")

#     # Ticks at your coordinate values
#     ax.set_xticks(values)
#     ax.set_yticks(values)
#     ax.set_xlabel("Supposed avoidance")
#     ax.set_ylabel("Real avoidance")
#     ax.set_title(info["title"])

#     # Optional: draw cell borders
#     ax.set_xticks(np.linspace(x0, x1, len(results[0]) + 1), minor=True)
#     ax.set_yticks(np.linspace(y0, y1, len(results) + 1), minor=True)
#     ax.grid(which="minor", color="k", linewidth=0.5)
#     ax.tick_params(which="minor", bottom=False, left=False)

#     # Legend
#     handles = [Patch(facecolor=colors[k], edgecolor="k", label=labels[k]) for k in range(6)]
#     ax.legend(handles=handles, title="Value", bbox_to_anchor=(1.05, 1), loc="upper left")

#     plt.tight_layout()
#     plt.savefig(f"figures/sens-collDistFactor-{collDistFactor}.png")
#     plt.show()


def standardGateEnv() -> tuple[GateEnvironmentConfig, MPPIConfig, PIDConfig, PIDConfig, list[str]]:
    nAgents = 2
    # nAgents = 1
    dim = 2
    nTrackSamples = 1000
    trackRadius = 8

    startS = np.linspace(0.2, 0, nAgents)

    nGates = 5
    # gateCenters = np.array([[0, 0], [3, 3], [0, 6], [-3, 3]])
    # gateVectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    # gateRadius = np.ones(nGates)

    gateCenters, gateVectors, gateRadius = circularGateTrack(nGates, trackRadius, 1, None if dim == 2 else 0.0)

    config = GateEnvironmentConfig(
        nAgents=nAgents,
        dim=dim,
        nRaceLines=1,
        nWinLaps=2,
        nGates=nGates,
        maxSpeed=np.linspace(2, 3, nAgents),
        maxAccel=np.linspace(3, 5, nAgents),
        nTrackSamples=nTrackSamples,
        targetDistance=0.05,
        gateCenters=gateCenters,
        gateVectors=gateVectors,
        gateRadius=gateRadius,
        minDist=1.5,
    )

    assert config.trackPoints is not None  # it is built automatically in GateEnvironmentConfig

    # add init state and add_state
    # config.init_state = np.array(
    #     [np.stack([config.trackPoints[int(s * config.nTrackSamples)], np.zeros(dim)], axis=1).flatten() for s in startS]
    # ).flatten()
    config.init_pos = np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten()
    config.initS = np.repeat(startS, 2)
    config.initnLaps = np.zeros(nAgents)
    config.initGates = np.array([int(s * nGates) for s in startS], dtype=np.float32)

    angles = np.linspace(0, 2 * np.pi, config.nTrackSamples, endpoint=False)

    config.nRaceLines = 2
    config.trackPoints = np.concat(
        [
            trackRadius * np.stack([np.cos(angles), np.sin(angles)], axis=1),
            (trackRadius - 0.5 * gateRadius[0])
            * np.stack(
                [np.cos(angles), np.sin(angles)],
                axis=1,
            ),
        ]
    )

    pidafraid = PIDConfig(
        kp=5, kd=20, repulsionFactor=30, repulsionDistFactor=3, racelineIndex=1, actionNoise=2, repulsionPower=2.0
    )
    pidbold = PIDConfig(kp=5, kd=20, repulsionFactor=0, repulsionDistFactor=3, racelineIndex=1, actionNoise=2, repulsionPower=2.0)

    mppiconfig = MPPIConfig(
        nSamples=10000,
        nTimesteps=60,
        inv_temperature=10,
        samplingNoise=3,
        gateTraversalMargin=0.95,
        collDistFactor=1.1,
        # collDistFactor=1.3,
        finalAdvWeight=200,
        # finalAdvWeight=0,
        # finalSpeedWeight=50,
        finalSpeedWeight=0,
        # oppDistWeight=1,
        oppDistWeight=0.0,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=0,
        # finalOppAdvWeight=500,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        oppOutsideCost=0,
        # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
        outsideCost=1000000,
        collisionCost=1000000,
        winCost=100000,
        minConfidence=0.95,
        oppKind=CONTROLLER_TYPE.CONT_PID,
        nModels=2,
        opponentPidConfigs=(pidafraid, pidbold),
        initBelief=np.array([0.5, 0.5]),
        # initBelief=np.array([1, 0]),
    )

    return config, mppiconfig, pidafraid, pidbold, ["Afraid", "Bold"]


def tinyGateEnv(afraid: bool = False) -> tuple[GateEnvironmentConfig, MPPIConfig, PIDConfig, PIDConfig, list[str]]:
    nAgents = 2
    dim = 2
    nTrackSamples = 512

    # startS = np.linspace(0.15, 0.0, nAgents)
    startS = np.linspace(0.1, 0.02, nAgents)
    nGates = 2

    length = 30
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

    config = GateEnvironmentConfig(
        nAgents=nAgents,
        dim=dim,
        nRaceLines=2,
        nWinLaps=1,
        nGates=nGates,
        maxSpeed=np.linspace(2, 2.5, nAgents),
        # maxSpeed=np.linspace(2, 3, nAgents),
        maxAccel=np.linspace(3, 3, nAgents),
        nTrackSamples=nTrackSamples,
        targetDistance=0.05,
        gateCenters=gateCenters,
        gateVectors=gateVectors,
        gateRadius=gateRadius,
        minDist=1,
        arenaMin=np.array([-0.1 * length, -2 * height * heightFactor]),
        arenaMax=np.array([1.1 * length, 2 * height * heightFactor]),
        nObstacles=1,
        obstacles=np.array([length * (0.5 - obsSize / 2), -height, length * (0.5 + obsSize / 2), height]),
        seed=42,
    )

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

    config.init_pos = np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten()
    config.initS = np.repeat(startS, 2)
    print(config.initS)
    config.initnLaps = np.zeros(nAgents)
    config.initGates = np.array([1, 1])

    # if afraid:
    #     config.init_pos = np.array([10.0, 4.0, 0.1, 0.0])

    repulsion = 30 if afraid else 0
    pid0 = PIDConfig(kp=5, kd=20, repulsionFactor=repulsion, racelineIndex=0, actionNoise=2)
    pid1 = PIDConfig(kp=5, kd=20, repulsionFactor=repulsion, racelineIndex=1, actionNoise=2)

    mppiconfig = MPPIConfig(
        nSamples=2**15,
        nTimesteps=60,
        inv_temperature=10,
        samplingNoise=3,
        gateTraversalMargin=0.9,  # restrict 5% on each side
        collDistFactor=1.1,
        # collDistFactor=1.3,
        finalAdvWeight=200,
        # finalAdvWeight=0,
        # finalSpeedWeight=50,
        finalSpeedWeight=0,
        oppDistWeight=0,
        # oppDistWeight=0.0,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=0,
        # finalOppAdvWeight=500,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        oppOutsideCost=0,
        # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
        outsideCost=1e6,
        collisionCost=1e6,
        winCost=1e10,
        minConfidence=0.95,
        oppKind=CONTROLLER_TYPE.CONT_PID,
        nModels=2,
        opponentPidConfigs=(pid0, pid1),
        initBelief=np.array([0.5, 0.5]),
        # initBelief=np.array([0.7, 0.3]),
    )

    return config, mppiconfig, pid0, pid1, ["Top", "Bottom"]


def activeEnv() -> tuple[GateEnvironmentConfig, MPPIConfig, PIDConfig, PIDConfig, list[str]]:
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
    )

    config.trackPoints = np.column_stack([np.linspace(0, length * 1.2, nTrackSamples), np.zeros(nTrackSamples)])

    config.init_pos = np.array([config.trackPoints[int(s * config.nTrackSamples)] for s in startS]).flatten()
    config.initS = np.repeat(startS, 2)
    config.initnLaps = np.zeros(nAgents)
    config.initGates = np.array([1, 1])

    pid0 = PIDConfig(
        kp=5, kd=20, repulsionFactor=120, racelineIndex=0, actionNoise=1, repulsionDistFactor=2.5, repulsionPower=1.5
    )
    pid1 = PIDConfig(kp=5, kd=20, repulsionFactor=0, racelineIndex=0, actionNoise=1, repulsionDistFactor=2.5)

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
        # finalSpeedWeight=50,
        finalSpeedWeight=0,
        # oppDistWeight=1,
        oppDistWeight=0.0,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=0,
        # finalOppAdvWeight=500,
        # boundaryCost=0.01,
        boundaryCost=0.0,
        boundaryThresholdFactor=2,
        oppOutsideCost=0,
        # oppOutsideCost=1000,  # with this, it's too competitive and will push the opponent out of the arena
        outsideCost=1e6,
        collisionCost=1e6,
        winCost=100000,
        minConfidence=0.95,
        oppKind=CONTROLLER_TYPE.CONT_PID,
        nModels=2,
        opponentPidConfigs=(pid0, pid1),
        initBelief=np.array([0.5, 0.5]),
        # initBelief=np.array([0, 1]),
    )

    return config, mppiconfig, pid0, pid1, ["Afraid", "Bold"]


def mainGate():
    # envConfig, mppiconfig, pid0, pid1, oppNames = standardGateEnv()  # pid0 = afraid; pid1 = bold
    envConfig, mppiconfig, pid0, pid1, oppNames = tinyGateEnv(afraid=False)  # pid0 = top; pid1 = bottom
    # envConfig, mppiconfig, pid0, pid1, oppNames = activeEnv()  # pid0 = afraid; pid1 = bold

    # dummyconfig = DummyConfig()

    cont_configs: list[ControllerConfig] = []

    cont_configs = [pid0, mppiconfig]
    # cont_configs = [pid1, mppiconfig]

    # cont_configs = [mppiconfig, pid0]
    # cont_configs = [mppiconfig, pid1]

    # cont_configs = [mppiconfig, mppiconfig]

    # cont_configs = [blindpidconfig, mppiconfig]
    # cont_configs = [mppiconfig, mppiconfig2]
    # cont_configs = [mppiconfig, blindpidconfig]
    # cont_configs = [mppiconfig, blindpidconfig]
    # cont_configs = [mppiconfig] + [pidconfig] * (nAgents - 1)  # type: ignore

    verifConfig = VerifConfig(N=2**17, beta=1e-6, horizon=40, maxEps=0.001)

    envConfig.sendStates = True

    z = ZMQRecv()

    evt, res = z.runSim(envConfig, verifConfig, cont_configs, render=envConfig.sendStates, oppNames=oppNames)

    print(f"result: {evt.name} {res}")

    z.close()


if __name__ == "__main__":
    # mainZMQ()
    mainGate()
    # buildOptimalRaceline()
    # computeResults(1, False)
    # plotResults(1, False)
