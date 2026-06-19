from dataclasses import dataclass, field
from enum import IntEnum
import random
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

    nObstacles: int = 0
    obstacles: np.ndarray = field(default_factory=lambda: np.array([]))

    nRoundObstacles: int = 0
    roundObsCenters: np.ndarray = field(default_factory=lambda: np.array([]))
    roundObsRadius: np.ndarray = field(default_factory=lambda: np.array([]))

    seed: int = 42  # if -1, then it will be set to a random value

    nModelFactors: int = 1
    nTrueModels: int = field(init=False)
    modelSizes: np.ndarray = field(default_factory=lambda: np.array([]))
    initBelief: np.ndarray = field(default_factory=lambda: np.array([]))

    opponentPidConfigs: tuple["PIDConfig", ...] = ()
    iMppi: int = 0
    trueTheta: int = 0

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

        if self.seed == -1:
            self.seed = random.randrange(2**31)

        self.modelSizes = self.modelSizes.astype(np.int32)
        self.nTrueModels = int(np.prod(self.modelSizes))

        if self.initBelief.size == 0:
            self.initBelief = np.full(self.nTrueModels, 1.0 / self.nTrueModels)

    def flattenTheta(self, thetaTuple: list[int]) -> int:
        "Flatten thetaTuple (0 <= theta[k] < modelSize[k]) into 0 <= trueTheta < nTrueModels"
        trueTheta, stride = 0, 1

        for k in range(self.nModelFactors - 1, -1, -1):
            trueTheta += thetaTuple[k] * stride
            stride *= self.modelSizes[k]

        return trueTheta

    def unflattenTheta(self, theta: int) -> list[int]:
        "Unflatten 0 <= trueTheta < nTrueModels into thetaTuple (0 <= theta[k] < modelSize[k])"
        ans: list[int] = [0] * self.nModelFactors
        for k in range(self.nModelFactors - 1, -1, -1):
            ans[k] = theta % self.modelSizes[k]
            theta //= self.modelSizes[k]

        return ans

    def computeMarginal(self, belief: np.ndarray, k: int) -> np.ndarray:
        "Compute belief marginalized over parameter k"
        marginal = np.zeros(self.modelSizes[k])

        for theta in range(self.nTrueModels):
            marginal[self.unflattenTheta(theta)[k]] += belief[theta]

        return marginal


@dataclass
class VerifConfig:
    N: int = int(1e6)
    beta: float = 1e-6
    horizon: int = 40

    maxEps: float = 0.01


# TODO: this is now useless
@dataclass
class ControllerConfig:
    def getDefaultName(self) -> str:
        if isinstance(self, PIDConfig):
            return "PID"
        elif isinstance(self, MPPIConfig):
            return "MPPI"

        raise ValueError("Unknown controller config")


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

    minConfidence: float = 0.9


@dataclass
class MPPIStatePredInfo:
    initPredTheta: np.ndarray
    branchTime: np.ndarray
    predTheta: np.ndarray
    fullPos: np.ndarray
    stopReason: EVENT_TYPE
    stopTime: int
    stopAgent: int


@dataclass
class MPPIStateInfo:
    belief: np.ndarray
    failCount: np.ndarray

    epsilon: float
    epsilonPartial: float
    useNewPlan: bool
    certifiedLoss: float

    nModels: int
    preds: list[MPPIStatePredInfo] = field(metadata={"len": "nModels"})


@dataclass
class FullStateInfo:
    step: int
    pos: np.ndarray
    speed: np.ndarray
    currentS: np.ndarray
    nLaps: np.ndarray
    currentGates: np.ndarray

    mppiInfo: MPPIStateInfo

    #     int,


#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     Optional[tuple[int, np.ndarray, np.ndarray, float, list[tuple[int, int, np.ndarray]]]],
# ]:
#     "Return (step, pos, vel, currentS, nLaps, currentGates, Optional[iMppi, belief, failCount, eps, list[(branchingTime, predTheta, fullPos, )]]) from bytes"
