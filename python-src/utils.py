from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    pass


def roundTrack(s: float) -> np.ndarray:
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)]) * 10


def lissajous(s: float, radius: float, s_radius: float, period: int) -> np.ndarray:
    R = radius + s_radius * np.sin(period * np.pi * s)
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)]) * R


def flower(s: float, radius: float, s_radius: float) -> np.ndarray:
    theta = 2 * np.pi * s
    x, y = np.cos(theta) * radius, np.sin(theta) * radius
    return np.array([x + s_radius * np.sin(theta * theta), y + s_radius * np.cos(theta * theta)])


def circularGateTrack(
    nGates: int, trackRadius: float, gateRadius: float, height: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns gateCenters, gateVectors, gateRadius for nGates around a circle of given radius.
    If height is None, returns 2D gates; otherwise, returns 3D gates with given height.
    """

    angles = np.linspace(0, 2 * np.pi, nGates, endpoint=False)

    dir_x = -np.sin(angles)
    dir_y = np.cos(angles)

    cent_x = trackRadius * np.cos(angles)
    cent_y = trackRadius * np.sin(angles)

    if height is not None:  # 3D
        centers = np.stack([cent_x, cent_y, np.full(nGates, height)], axis=1)
        vectors = np.stack([dir_x, dir_y, np.zeros(nGates)], axis=1)
    else:  # 2D
        centers = np.stack([cent_x, cent_y], axis=1)
        vectors = np.stack([dir_x, dir_y], axis=1)

    radius = np.full(nGates, gateRadius)

    return centers, vectors, radius


class MSG_TYPE(IntEnum):
    MSG_HEADER = 0
    MSG_STATE = 1
    MSG_EVENT = 2
    MSG_DONE = 3


class EVENT_TYPE(IntEnum):
    EVT_COLLISION = 0
    EVT_OUTSIDE = 1
    EVT_WINNER = 2
    EVT_TRUNCATED = 3


class CONTROLLER_TYPE(IntEnum):
    CONT_DUMMY = 0
    CONT_PID = 1
    CONT_MPPI = 2

    @classmethod
    def fromConfig(cls, config: "ControllerConfig") -> "CONTROLLER_TYPE":
        if isinstance(config, DummyConfig):
            return CONTROLLER_TYPE.CONT_DUMMY
        elif isinstance(config, PIDConfig):
            return CONTROLLER_TYPE.CONT_PID
        elif isinstance(config, MPPIConfig):
            return CONTROLLER_TYPE.CONT_MPPI

        raise ValueError("Unknown controller config")


# @dataclass
# class BaseEnvironmentConfig(Generic[AddStateType]):
#     nAgents: int = 2
#     dim: int = 2
#     dt: float = 0.1

#     sendStates: bool = True
#     nRaceLines: int = 1  # number of race lines (at least 1, centerline; can specify more for PID following a given line)
#     # they all should be concatenated & specified in trackPoints (which contains nLines arrays of size nSamples * dim), and then the line config in PID specifies the offset (offset=0: following centerline from 0 to nSamples-1; offset=1: following arbitrary raceline from nSamples to 2*nSamples-1, etc)
#     nGates: int = 0

#     init_pos: list | np.ndarray = field(default_factory=lambda: [])
#     init_vel: list | np.ndarray = field(default_factory=lambda: [])
#     add_state: Optional[AddStateType] = None

#     minDist: float = 0.2
#     posNoiseLevel: float = 0.0
#     speedNoiseLevel: float = 0.0
#     actionNoiseLevel: float = 0.0

#     maxSpeed: np.ndarray = field(default_factory=lambda: np.array([]))  # in L_2 norm
#     maxAccel: np.ndarray = field(default_factory=lambda: np.array([]))  # in L_inf norm

#     def __post_init__(self) -> None:
#         defaultMaxSpeed: float = 0.5
#         defaultAccel: float = 1

#         self.posDim = self.nAgents * self.dim
#         self.velDim = self.nAgents * self.dim
#         self.stateDim = self.posDim + self.velDim
#         self.actionDim = self.nAgents * self.dim

#         if self.maxSpeed.shape == (0,):
#             self.maxSpeed = np.full(self.nAgents, defaultMaxSpeed)
#         if self.maxAccel.shape == (0,):
#             self.maxAccel = np.full(self.nAgents, defaultAccel)

#         self.init_pos = np.array(self.init_pos).flatten().astype(np.float32)
#         self.init_vel = np.array(self.init_vel).flatten().astype(np.float32)

#         # If still empty, default to zeros
#         if self.init_pos.size == 0:
#             self.init_pos = np.zeros(self.posDim, dtype=np.float32)
#         if self.init_vel.size == 0:
#             self.init_vel = np.zeros(self.velDim, dtype=np.float32)

#         if self.init_pos.size != self.posDim:
#             raise ValueError(f"Invalid init_pos size: got {self.init_pos.size}, expected {self.posDim}")
#         if self.init_vel.size != self.velDim:
#             raise ValueError(f"Invalid init_vel size: got {self.init_vel.size}, expected {self.velDim}")


# ConfigType = TypeVar("ConfigType", bound=BaseEnvironmentConfig)


# @dataclass
# class SimpleEnvironmentConfig(BaseEnvironmentConfig):
#     gateRadius: float = 0.3

#     arenaMinY: float = -5
#     arenaSide: float = 2

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         self.arenaMin = np.array([-self.arenaSide if i != 1 else self.arenaMinY for i in range(self.dim)])
#         self.arenaMax = np.array([self.arenaSide if i != 1 else 0 for i in range(self.dim)])


# @dataclass
# class TrackEnvironmentConfig(BaseEnvironmentConfig):
#     centerline: Optional[Callable[[float], np.ndarray]] = None  # function [0,1] -> middle of the track
#     trackWidth: float = 2.0
#     nTrackSamples: int = 500  # track is discretized with this number of samples
#     nWinLaps: int = 1

#     targetDistance: float = 0.1  # simple controllers will try to go to the track point at s + targetDistance

#     trackPoints: Optional[np.ndarray] = None

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         if self.centerline is not None and self.trackPoints is None:
#             # sample centerline
#             sGrid = np.linspace(0, 1, self.nTrackSamples, endpoint=False)
#             self.trackPoints = np.array([self.centerline(s) for s in sGrid])
#             # mm, m = 0.0, np.inf
#             # for i in range(self.nTrackSamples):
#             #     d = np.linalg.norm(self.trackPoints[(i + 1) % self.nTrackSamples] - self.trackPoints[i])
#             #     mm = max(mm, d)
#             #     m = min(m, d)
#             # print(f"minimum distance between 2 consecutive points: {m} max: {mm}")


@dataclass
class GateEnvironmentConfig:
    nAgents: int = 2
    dim: int = 2
    dt: float = 0.1

    sendStates: bool = True
    nRaceLines: int = 1  # number of race lines (at least 1, centerline; can specify more for PID following a given line)
    # they all should be concatenated & specified in trackPoints (which contains nLines arrays of size nSamples * dim), and then the line config in PID specifies the offset (offset=0: following centerline from 0 to nSamples-1; offset=1: following arbitrary raceline from nSamples to 2*nSamples-1, etc)
    nGates: int = 0

    init_pos: list | np.ndarray = field(default_factory=lambda: [])
    init_vel: list | np.ndarray = field(default_factory=lambda: [])
    # add_state: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # S, currentLaps, currentGates
    initS: np.ndarray = field(default_factory=lambda: np.array([]))
    initnLaps: np.ndarray = field(default_factory=lambda: np.array([]))
    initGates: np.ndarray = field(default_factory=lambda: np.array([]))

    minDist: float = 0.2
    posNoiseLevel: float = 0.0
    speedNoiseLevel: float = 0.0
    actionNoiseLevel: float = 0.0

    maxSpeed: np.ndarray = field(default_factory=lambda: np.array([]))  # in L_2 norm
    maxAccel: np.ndarray = field(default_factory=lambda: np.array([]))  # in L_inf norm

    nTrackSamples: int = 500  # track is discretized with this number of samples
    nWinLaps: int = 1

    targetDistance: float = 0.1  # simple controllers will try to go to the track point at s + targetDistance

    gateCenters: np.ndarray = field(default_factory=lambda: np.array([]))  # (nGates, dim)
    gateVectors: np.ndarray = field(default_factory=lambda: np.array([]))  # (nGates, dim)
    gateRadius: np.ndarray = field(default_factory=lambda: np.array([]))  # (nGates)

    arenaMin: np.ndarray = field(default_factory=lambda: np.array([]))
    arenaMax: np.ndarray = field(default_factory=lambda: np.array([]))

    trackPoints: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.init_pos = np.array(self.init_pos).flatten().astype(np.float32)
        self.init_vel = np.array(self.init_vel).flatten().astype(np.float32)

        totDim = self.dim * self.nAgents  # dimension of the pos & vel arrays

        # If still empty, default to zeros
        if self.init_pos.size == 0:
            self.init_pos = np.zeros(totDim, dtype=np.float32)
        if self.init_vel.size == 0:
            self.init_vel = np.zeros(totDim, dtype=np.float32)

        if self.init_pos.size != totDim:
            raise ValueError(f"Invalid init_pos size: got {self.init_pos.size}, expected {totDim}")
        if self.init_vel.size != totDim:
            raise ValueError(f"Invalid init_vel size: got {self.init_vel.size}, expected {totDim}")

        if self.trackPoints is None and self.nGates > 0:
            # sample simple centerline as linear points going through the gates
            self.trackPoints = np.concatenate(
                [
                    np.linspace(
                        self.gateCenters[i],
                        self.gateCenters[(i + 1) % self.nGates],
                        self.nTrackSamples // self.nGates + (i >= self.nGates - self.nTrackSamples % self.nGates),
                    )
                    for i in range(self.nGates)
                ]
            )

        if self.arenaMin.shape == (0,):
            self.arenaMin = np.min(self.gateCenters, axis=0) - np.max(self.gateRadius) * 5
            self.arenaMax = np.max(self.gateCenters, axis=0) + np.max(self.gateRadius) * 5


@dataclass
class ControllerConfig:
    kind: CONTROLLER_TYPE = CONTROLLER_TYPE.CONT_DUMMY

    def __post_init__(self):
        self.kind = CONTROLLER_TYPE.fromConfig(self)

    def getDefaultName(self) -> str:
        if isinstance(self, DummyConfig):
            return "Dummy"
        elif isinstance(self, PIDConfig):
            return "PID"
        elif isinstance(self, MPPIConfig):
            return "MPPI"

        raise ValueError("Unknown controller config")


@dataclass
class DummyConfig(ControllerConfig):
    pass


@dataclass
class PIDConfig(ControllerConfig):
    kp: float = 1.0
    kd: float = 0.5

    repulsionFactor: float = 0.5
    repulsionPower: float = 2.0
    repulsionDistFactor: float = 5.0

    racelineIndex: int = 0

    actionNoise: float = 0.0


@dataclass
class MPPIConfig(ControllerConfig):
    nSamples: int = 100
    nTimesteps: int = 20
    inv_temperature: float = 10

    samplingNoise: float = 1.0
    gateTraversalMargin: float = 0.95

    collDistFactor: float = 1.0

    # running cost is the sum of the distance to the opponents oppDistCost / dist^oppDistCost, or 0 if dist > oppDistThreshold * config.minDist
    oppDistWeight: float = 1
    oppDistPower: float = 2
    oppDistThresholdFactor: float = 3
    boundaryCost: float = 10
    boundaryThresholdFactor: float = 1.5
    outsideCost: float = 1000
    oppOutsideCost: float = 100
    collisionCost: float = 10000
    winCost: float = 1000

    # final cost: - distance to the gate * finalDistWeight + min(opp. dist to the gate) * finalOppDistWeight - finalSpeedWeight * dot(finalSpeed, targetDirection)
    finalAdvWeight: float = 10
    finalOppAdvWeight: float = 5
    finalSpeedWeight: float = 5

    # only used by local Python simulation
    # opponentPredictors: Optional[list[Callable[[int, np.ndarray, AddStateType], np.ndarray]]] = (
    #     None  # should be the list of modeled getControl() method of opponents
    # )

    # obviously, should not be MPPIConfig
    opponentConfig: DummyConfig | PIDConfig = field(default_factory=lambda: DummyConfig())


# class Controller(ABC):
#     def __init__(self, name: str, envConfig: GateEnvironmentConfig):
#         self.name = name
#         self.envConfig = envConfig
#         self.environment: "Optional[GateEnvironment]" = None

#     def setEnvironment(self, env: "GateEnvironment"):
#         self.environment = env

#     @classmethod
#     def fromConfig(cls, envConfig: GateEnvironmentConfig, config: ControllerConfig, name: Optional[str] = None) -> "Controller":
#         from controllers import MPPIController, PIDController

#         if name is None:
#             name = config.getDefaultName()

#         if isinstance(config, PIDConfig):
#             return PIDController(envConfig, config, name)
#         elif isinstance(config, MPPIConfig):
#             return MPPIController(envConfig, config, name)

#         raise ValueError("Unknown config")
