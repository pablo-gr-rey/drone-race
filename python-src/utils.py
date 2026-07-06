from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
import math
import random
from typing import Any, ClassVar, Optional
from matplotlib import patches
from matplotlib.path import Path
import matplotlib.pyplot as plt

import numpy as np


class MSG_TYPE(IntEnum):
    MSG_HEADER = 0
    MSG_STATE = 1
    MSG_EVENT = 2
    MSG_DONE = 3


class EVENT_TYPE(IntEnum):
    EVT_OUTSIDE = 0
    EVT_WINNER = 1
    EVT_TRUNCATED = 2


class CONTROLLER_KIND(IntEnum):
    CONT_INVALID = -1
    CONT_MPPI = 0
    CONT_PRMPPI = 1


class ENV_KIND(IntEnum):
    ENV_INVALID = -1
    ENV_DRONERACE = 0
    ENV_HIDDENOBS = 1


@dataclass
class BaseControllerConfig(ABC):
    contKind: CONTROLLER_KIND = field(init=False, default=CONTROLLER_KIND.CONT_INVALID)

    @abstractmethod
    def getDefaultName(self) -> str: ...


@dataclass
class BaseEnvironmentConfig(ABC):
    envKind: ENV_KIND = field(init=False, default=ENV_KIND.ENV_INVALID)

    actionDim: int = field(init=False, default=-1, metadata={"send": False})

    nModelFactors: int = 1
    modelSizes: np.ndarray = field(default_factory=lambda: np.array([]))
    nTrueModels: int = field(init=False, metadata={"send": False})

    arenaMin: np.ndarray = field(default_factory=lambda: np.array([]))
    arenaMax: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.modelSizes = self.modelSizes.astype(np.int32)
        self.nTrueModels = int(np.prod(self.modelSizes))

    @abstractmethod
    def getMaxAccel(self) -> float: ...

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
class BaseSimState: ...


class BaseEnvironmentRenderer[EnvConfigT: BaseEnvironmentConfig](ABC):
    def __init__(
        self,
        envConfig: EnvConfigT,
        contConfig: BaseControllerConfig,
        ax: plt.Axes,  # type: ignore
        ax_status: plt.Axes,  # type: ignore
        oppNames: list[list[str]],
        **kwargs: Any,
    ):
        self.envConfig = envConfig
        self.contConfig = contConfig
        self.ax = ax
        self.ax_status = ax_status
        self.oppNames = oppNames

        self.postInit(**kwargs)

    @abstractmethod
    def postInit(self, **kwargs) -> None: ...

    @abstractmethod
    def drawBackground(self) -> None: ...

    @abstractmethod
    def drawFrame(self, stateLog: list["FullStateInfo"], iFrame: int) -> None: ...

    @abstractmethod
    def getZoomPos(self, stateLog: list["FullStateInfo"], iFrame: int) -> tuple[float, float]: ...

    def drawRectangle(self, omin: tuple[float, float], omax: tuple[float, float]) -> None:
        self.ax.add_patch(
            patches.Rectangle(
                omin,
                width=omax[0] - omin[0],
                height=omax[1] - omin[1],
                facecolor="gray",
                hatch="/",
                fill=True,
            )
        )

    def drawCircle(
        self, center: tuple[float, float], radius: float, fill: bool = True, edge: bool = True, dottedEdge: bool = False
    ) -> None:
        self.ax.add_patch(
            patches.Circle(
                center,
                radius,
                linewidth=3 if edge else 0,
                edgecolor="black",
                facecolor="gray",
                hatch="/",
                fill=fill,
                linestyle="dotted" if dottedEdge else "solid",
            )
        )

    def drawTriangle(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> None:
        self.ax.add_patch(
            patches.Polygon(
                [p1, p2, p3],
                facecolor="gray",
                hatch="/",
                fill=True,
            )
        )

    def drawHalfPlane(self, minx: float, maxx: float, lambdaTop: float, lambdaBot: float, isRight: bool) -> None:
        "isRight should be False for first dbl-half-plane and True for 2nd"

        xFact = 1 if isRight else -1  # plane is xFact*x +- y = lambdaTop/Bot

        # top part
        yTop = lambdaTop + (-minx if isRight else maxx)
        self.drawTriangle((minx, lambdaTop - xFact * minx), (maxx, lambdaTop - xFact * maxx), (maxx if isRight else minx, yTop))
        self.drawRectangle((minx, yTop), (maxx, self.envConfig.arenaMax[1] + 1))

        # bottom part
        yBot = -lambdaBot + (minx if isRight else -maxx)
        self.drawTriangle((minx, -lambdaBot + xFact * minx), (maxx, -lambdaBot + xFact * maxx), (maxx if isRight else minx, yBot))
        self.drawRectangle((minx, self.envConfig.arenaMin[1] - 1), (maxx, yBot))

    def drawAnnulus(self, xcenter: float, ycenter: float, minr: float, maxr: float) -> None:
        # inner circle
        self.ax.add_patch(
            patches.Wedge(
                (xcenter, ycenter),
                minr,
                90,
                180,
                facecolor="gray",
                hatch="/",
                fill=True,
                linewidth=0,
            )
        )

        # outer circle: we have to discretize the path
        nThetas = 20

        thetas = np.linspace(np.pi / 2, np.pi, nThetas)
        arc_x = maxr * np.cos(thetas) + xcenter
        arc_y = maxr * np.sin(thetas) + ycenter

        # path: topleft -> topright -> arc -> bottomleft -> back to topleft
        minx = self.envConfig.arenaMin[0] - 1
        maxy = self.envConfig.arenaMax[1] + 1

        path = Path(
            [
                (minx, maxy),
                (xcenter, maxy),
                (xcenter, maxr + ycenter),
                *zip(arc_x, arc_y),
                (-maxr + xcenter, ycenter),
                (minx, ycenter),
                (minx, maxy),
            ],
            closed=True,
        )

        self.ax.add_patch(
            patches.PathPatch(
                path,
                facecolor="gray",
                hatch="/",
                fill=True,
            )
        )


@dataclass
class DroneRaceEnvironmentConfig(BaseEnvironmentConfig):
    @dataclass
    class PIDConfig:  # not considered as a ControllerConfig, as these are meant to represent independant controllers (and PID isn't really interesting as a main character)
        kp: float = 1.0
        kd: float = 0.5

        repulsionFactor: float = 0.5
        repulsionPower: float = 2.0
        repulsionDistFactor: float = 5.0

        racelineIndex: int = 0

        actionNoise: float = 0.0

        def getDefaultName(self) -> str:
            return "PID"

    nAgents: int = 2
    dim: int = 2
    dt: float = 0.1

    sendStates: bool = True
    nRaceLines: int = 1  # number of race lines (at least 1, centerline; can specify more for PID following a given line)
    # they all should be concatenated & specified in trackPoints (which contains nLines arrays of size nSamples * dim), and then the line config in PID specifies the offset (offset=0: following centerline from 0 to nSamples-1; offset=1: following arbitrary raceline from nSamples to 2*nSamples-1, etc)
    nGates: int = 0

    droneRadius: float = 0.2
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

    nObstacles: int = 0
    obstacles: np.ndarray = field(default_factory=lambda: np.array([]))

    nRoundObstacles: int = 0
    roundObsCenters: np.ndarray = field(default_factory=lambda: np.array([]))
    roundObsRadius: np.ndarray = field(default_factory=lambda: np.array([]))

    seed: int = 42  # if -1, then it will be set to a random value

    initBelief: np.ndarray = field(default_factory=lambda: np.array([]))

    opponentPidConfigs: tuple[PIDConfig, ...] = ()
    iMppi: int = 0
    trueTheta: int = 0

    trackPoints: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.envKind = ENV_KIND.ENV_DRONERACE
        self.actionDim = self.dim

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

        if self.initBelief.size == 0:
            self.initBelief = np.full(self.nTrueModels, 1.0 / self.nTrueModels)

    def getMaxAccel(self) -> float:
        return self.maxAccel[self.iMppi]


@dataclass
class DroneRaceSimState(BaseSimState):
    pos: np.ndarray
    vel: np.ndarray
    S: np.ndarray
    laps: np.ndarray
    gates: np.ndarray


@dataclass
class HiddenObsEnvironmentConfig(BaseEnvironmentConfig):
    dim: int = 2
    dt: float = 0.1

    sendStates: bool = True

    nGates: int = 0

    droneRadius: float = 0.2

    posNoiseLevel: float = 0.0
    speedNoiseLevel: float = 0.0
    actionNoiseLevel: float = 0.0

    maxSpeed: float = 1.0
    maxAccel: float = 2.0

    nWinLaps: int = 1

    gateCenters: np.ndarray = field(default_factory=lambda: np.array([]))  # (nGates, dim)
    gateVectors: np.ndarray = field(default_factory=lambda: np.array([]))  # (nGates, dim)
    gateRadius: np.ndarray = field(default_factory=lambda: np.array([]))  # (nGates)

    nRectObstacles: int = 0
    rectObstacles: np.ndarray = field(default_factory=lambda: np.array([]))

    nHiddenObstacles: int = 0
    hiddenObsCenters: np.ndarray = field(default_factory=lambda: np.array([]))
    hiddenObsRadius: np.ndarray = field(default_factory=lambda: np.array([]))

    dblHpLimits: np.ndarray = field(default_factory=lambda: np.array([]))
    dblHpLambdas: np.ndarray = field(default_factory=lambda: np.array([]))

    annulusCenter: np.ndarray = field(default_factory=lambda: np.array([]))
    annulusRadius: np.ndarray = field(default_factory=lambda: np.array([]))

    seed: int = 42  # if -1, then it will be set to a random value

    initBelief: np.ndarray = field(default_factory=lambda: np.array([]))

    trueTheta: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()

        self.envKind = ENV_KIND.ENV_HIDDENOBS
        self.actionDim = self.dim

        if self.arenaMin.shape == (0,):
            self.arenaMin = np.min(self.gateCenters, axis=0) - np.max(self.gateRadius) * 5
            self.arenaMax = np.max(self.gateCenters, axis=0) + np.max(self.gateRadius) * 5

        if self.seed == -1:
            self.seed = random.randrange(2**31)

        if self.initBelief.size == 0:
            self.initBelief = np.full(self.nTrueModels, 1.0 / self.nTrueModels)

    def getMaxAccel(self) -> float:
        return self.maxAccel


@dataclass
class HiddenObsSimState(BaseSimState):
    pos: np.ndarray
    vel: np.ndarray

    laps: int
    gates: int


@dataclass
class MPPIConfig(BaseControllerConfig):
    nSamples: int = 10000
    nTimesteps: int = 60
    nKnots: int = 6
    knots: np.ndarray = field(default_factory=lambda: np.array([]))

    useSplines: bool = True

    inv_temperature: float = 10

    samplingNoise: float = 1.0
    gateTraversalMargin: float = 0.95

    collDistFactor: float = 1.0

    # running cost is the sum of the distance to the opponents oppDistCost / dist^oppDistCost, or 0 if dist > oppDistThreshold * config.minDist
    boundaryCost: float = 10
    boundaryThresholdFactor: float = 1.5

    outsideCost: float = 1000
    winCost: float = 1000

    # final cost: - distance to the gate * finalDistWeight + min(opp. dist to the gate) * finalOppDistWeight - finalSpeedWeight * dot(finalSpeed, targetDirection)
    finalAdvWeight: float = 10
    finalOppAdvWeight: float = 5

    minConfidence: float = 0.9

    # verification details

    nVerifSamples: int = int(1e6)
    beta: float = 1e-6
    verifHorizon: int = 40

    maxVerifEps: float = 0.01

    def __post_init__(self):
        self.contKind = CONTROLLER_KIND.CONT_MPPI
        if len(self.knots) == 0:
            self.knots = np.arange(0, self.nKnots) * int((self.nTimesteps - 2) / (self.nKnots - 1))  # we need tau_(M-1) <= T-2
            print(self.knots, self.nKnots)

    def getDefaultName(self) -> str:
        return "Branching MPPI"


@dataclass
class PRMPPIConfig(BaseControllerConfig):
    nSamples: int = 100
    nTimesteps: int = 20

    inv_temperature: float = 10

    samplingNoise: float = 1.0
    gateTraversalMargin: float = 0.95

    collDistFactor: float = 1.0

    # running cost is the sum of the distance to the opponents oppDistCost / dist^oppDistCost, or 0 if dist > oppDistThreshold * config.minDist
    boundaryCost: float = 10
    boundaryThresholdFactor: float = 1.5

    winCost: float = 1000

    # final cost: - distance to the gate * finalDistWeight + min(opp. dist to the gate) * finalOppDistWeight - finalSpeedWeight * dot(finalSpeed, targetDirection)
    finalAdvWeight: float = 10
    finalOppAdvWeight: float = 5

    # safety costs
    safetyWeight: float = 1e6
    minSafeDist: float = 0.0

    delta: float = 0.1
    P: int = 0

    def __post_init__(self):
        self.contKind = CONTROLLER_KIND.CONT_PRMPPI

        if self.P == 0:
            self.P = math.ceil((1 - self.delta) / self.delta)

    def getDefaultName(self) -> str:
        return "Parameter-robust MPPI"


@dataclass
class MPPIStatePredInfo:
    initPredTheta: np.ndarray
    branchTime: np.ndarray
    predTheta: np.ndarray
    fullPos: np.ndarray
    egoActions: np.ndarray
    stopReason: EVENT_TYPE
    stopTime: int


@dataclass
class MPPIStateInfo:
    belief: np.ndarray
    failCount: int

    epsilon: float
    epsilonPartial: float
    useNewPlan: bool
    certifiedLoss: float

    nModels: int
    preds: list[MPPIStatePredInfo] = field(metadata={"len": "nModels"})


@dataclass
class PRMPPIStatePredInfo:
    fullPos: np.ndarray
    egoActions: np.ndarray
    stopReason: EVENT_TYPE
    stopTime: int


@dataclass
class PRMPPIStateInfo:
    prevPos: np.ndarray
    belief: np.ndarray

    useNomPlan: bool
    resetNom: bool

    nModelsPlusRob: int
    preds: list[PRMPPIStatePredInfo] = field(metadata={"len": "nModelsPlusRob"})


@dataclass
class FullStateInfo[SimStateT: BaseSimState, contInfoT]:
    # kind of a hack, but this is a class member which should be set by the protocol part based on the actual controller type
    # this way, we can automatically unpack the state information corresponding to the given controller
    SimStateType: ClassVar[type[DroneRaceSimState] | type[HiddenObsSimState]]
    ContStateType: ClassVar[type[MPPIStateInfo] | type[PRMPPIStateInfo]]

    step: int

    state: SimStateT = field(metadata={"class": "SimStateType"})

    egoAction: np.ndarray

    contInfo: contInfoT = field(metadata={"class": "ContStateType"})

    @classmethod
    def updateTypes(cls, envConfig: BaseEnvironmentConfig, contConfig: BaseControllerConfig) -> None:
        if envConfig.envKind == ENV_KIND.ENV_DRONERACE:
            cls.SimStateType = DroneRaceSimState
        elif envConfig.envKind == ENV_KIND.ENV_HIDDENOBS:
            cls.SimStateType = HiddenObsSimState
        else:
            raise ValueError(f"Unknown environment kind {envConfig.envKind}")

        if contConfig.contKind == CONTROLLER_KIND.CONT_MPPI:
            cls.ContStateType = MPPIStateInfo
        elif contConfig.contKind == CONTROLLER_KIND.CONT_PRMPPI:
            cls.ContStateType = PRMPPIStateInfo
        else:
            raise ValueError(f"Unknown controller kind {contConfig.contKind}")
