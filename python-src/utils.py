from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
import math
import random
from typing import Callable, ClassVar, Literal, Optional, Self
from scipy.interpolate import CubicSpline


import numpy as np


def _errorTrack(_: float) -> np.ndarray:
    raise ValueError("Undefined track function")


class MSG_TYPE(IntEnum):
    MSG_HEADER = 0
    MSG_STATE = 1
    MSG_EVENT = 2
    MSG_DONE = 3


class EVENT_TYPE(IntEnum):
    EVT_OUTSIDE = 0
    EVT_WINNER = 1
    EVT_OPP_WINNER = 2
    EVT_TRUNCATED = 3


class CONTROLLER_KIND(IntEnum):
    CONT_INVALID = -1
    CONT_MPPI = 0
    CONT_PRMPPI = 1


class ENV_KIND(IntEnum):
    ENV_INVALID = -1
    ENV_DRONERACE = 0
    ENV_HIDDENOBS = 1
    ENV_STRATRACE = 2


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
class StratRaceEnvironmentConfig(BaseEnvironmentConfig):
    @dataclass
    class OppConfig:
        s1_i: np.ndarray  # = field(default_factory=lambda: np.array([]))
        s2_i: np.ndarray  # = field(default_factory=lambda: np.array([]))
        s3_i: np.ndarray  # = field(default_factory=lambda: np.array([]))
        speedScale: np.ndarray  # = field(default_factory=lambda: np.array([]))

        actionNoise: np.ndarray  # = field(default_factory=lambda: np.array([]))

        @classmethod
        def preset(
            cls,
            droneRadius: float,
            nOpps: int,
            s1: float | np.ndarray | Literal["insensitive"] | Literal["default"] | Literal["conservative"] = "default",
            s2: float | np.ndarray | Literal["infinite"] | Literal["default"] | Literal["insensitive"] = "default",
            s3: float | np.ndarray | Literal["no-block"] | Literal["default"] = "default",
            speedScale: float | np.ndarray = 1.0,
            actionNoise: float | np.ndarray = 0.1,
        ) -> Self:
            values: dict[str, np.ndarray] = {}

            def set_val(attr_name: str, val: float | np.ndarray | str, defaultValues: dict[str, float]) -> None:
                if isinstance(val, str) and val in defaultValues:
                    val = defaultValues[val]
                elif isinstance(val, str):
                    raise ValueError(
                        f"Invalid string literal provided for {attr_name} (got {val}, possible values are {'; '.join(defaultValues.keys())})"
                    )

                if isinstance(val, float):
                    values[attr_name] = np.full(nOpps, val)
                elif isinstance(val, np.ndarray):
                    values[attr_name] = val
                else:
                    raise ValueError(f"Invalid value for {attr_name} (got {val})")

            set_val(
                "s1_i",
                s1,
                {
                    "insensitive": 0.0,
                    "default": 2.5 * droneRadius,
                    "conservative": 4 * droneRadius,
                },
            )

            set_val(
                "s2_i",
                s2,
                {
                    "infinite": 0.0,
                    "default": 1 / (4 * droneRadius) ** 2,
                    "insensitive": 1 / (1e-5 * droneRadius) ** 2,
                },
            )

            set_val("s3_i", s3, {"no-block": 0.0, "default": 20 / droneRadius})

            set_val("speedScale", speedScale, {})
            set_val("actionNoise", actionNoise, {})

            return cls(**values)

    nOppAgents: int = 2
    dim: int = 2
    dt: float = 0.1

    sendStates: bool = True

    droneRadius: float = 0.2
    posNoiseLevel: float = 0.0
    speedNoiseLevel: float = 0.0
    actionNoiseLevel: float = 0.0

    maxSpeed: np.ndarray = field(default_factory=lambda: np.array([]))  # in L_2 norm, MPPI is first
    maxAccel: np.ndarray = field(default_factory=lambda: np.array([]))  # in L_inf norm, MPPI is first

    nTrackSamples: int = 512  # track is discretized with this number of samples
    trackWidth: float = 2
    nWinLaps: int = 1

    seed: int = 42  # if -1, then it will be set to a random value

    initBelief: np.ndarray = field(default_factory=lambda: np.array([]))

    opponentConfigs: tuple[OppConfig, ...] = ()
    trueTheta: int = 0

    kP: float = 10.0
    kV: float = -1.0
    maxOppLatDistFact: float = 1.1

    trackFunction: Callable[[float], np.ndarray] = field(metadata={"send": False}, default=_errorTrack)

    trackLength: float = field(init=False)
    p_grid: np.ndarray = field(init=False)
    dp_grid: np.ndarray = field(init=False)
    t_grid: np.ndarray = field(init=False)
    kappa_grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.envKind = ENV_KIND.ENV_STRATRACE
        self.actionDim = self.dim

        if self.kV == -1.0:
            self.kV = 2 * self.kP**0.5

        # compute track data
        q = np.array([self.trackFunction(k / self.nTrackSamples) for k in range(self.nTrackSamples)])
        diffs = np.diff(q, axis=0, append=q[0:1])
        d_k = np.linalg.norm(diffs, axis=1)

        # s_k = cumulative sum with s_0 = 0
        s_k = np.zeros(self.nTrackSamples + 1)
        s_k[1:] = np.cumsum(d_k)

        self.trackLength = s_k[-1]

        q_periodic = np.vstack([q, q[0:1]])

        # Compute periodic cubic splines x(s), y(s)
        cs_x = CubicSpline(s_k, q_periodic[:, 0], bc_type="periodic")
        cs_y = CubicSpline(s_k, q_periodic[:, 1], bc_type="periodic")

        # Uniform grid in arc-length
        s_grid = np.linspace(0, self.trackLength, self.nTrackSamples, endpoint=False)

        # p_grid = [x(s), y(s)]
        self.p_grid = np.column_stack([cs_x(s_grid), cs_y(s_grid)])

        # dp_grid = [x'(s), y'(s)] (first derivatives)
        dx = cs_x(s_grid, 1)
        dy = cs_y(s_grid, 1)
        self.dp_grid = np.column_stack([dx, dy])

        # t_grid = normalize(dp_grid)
        dp_norm = np.linalg.norm(self.dp_grid, axis=1, keepdims=True)
        self.t_grid = self.dp_grid / np.maximum(dp_norm, 1e-6)

        # Second derivatives for curvature
        ddx = cs_x(s_grid, 2)
        ddy = cs_y(s_grid, 2)

        # kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^1.5
        self.kappa_grid = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)

        if self.arenaMin.shape == (0,):
            self.arenaMin = np.min(self.p_grid, axis=0) - self.trackWidth * 2
            self.arenaMax = np.max(self.p_grid, axis=0) + self.trackWidth * 2

        if self.seed == -1:
            self.seed = random.randrange(2**31)

        if self.initBelief.size == 0:
            self.initBelief = np.full(self.nTrueModels, 1.0 / self.nTrueModels)

    def getMaxAccel(self) -> float:
        return self.maxAccel[0]


@dataclass
class StratRaceSimState(BaseSimState):
    pos: np.ndarray
    vel: np.ndarray
    S: np.ndarray
    latDist: np.ndarray
    laps: np.ndarray

    latDistTarget: np.ndarray = field(
        default_factory=lambda: np.array([]), metadata={"send": False}
    )  # proxy for the received additional info, is not sent


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
    oppWinCost: float = 1000

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
    oppWinCost: float = 1000

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
    SimStateType: ClassVar[type[DroneRaceSimState] | type[HiddenObsSimState] | type[StratRaceSimState]]
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
        elif envConfig.envKind == ENV_KIND.ENV_STRATRACE:
            cls.SimStateType = StratRaceSimState
        else:
            raise ValueError(f"Unknown environment kind {envConfig.envKind}")

        if contConfig.contKind == CONTROLLER_KIND.CONT_MPPI:
            cls.ContStateType = MPPIStateInfo
        elif contConfig.contKind == CONTROLLER_KIND.CONT_PRMPPI:
            cls.ContStateType = PRMPPIStateInfo
        else:
            raise ValueError(f"Unknown controller kind {contConfig.contKind}")
