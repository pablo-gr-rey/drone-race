import warnings
from abc import ABC, abstractmethod
from types import NoneType
from typing import Generic, Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Axes  # type: ignore
from renderer import EnvironmentRenderer
from utils import AddStateType, ConfigType, Controller, GateEnvironmentConfig, SimpleEnvironmentConfig, TrackEnvironmentConfig


class BaseEnvironment(ABC, Generic[ConfigType, AddStateType]):
    "Simple environment, only specifies the dynamics"

    def __init__(
        self,
        config: ConfigType,
        controllers: list[Controller],
        *,
        live_render: bool = False,
    ):
        self.config = config
        self.controllers = controllers

        for cont in controllers:
            cont.setEnvironment(self)

        # state format (2 agents, 3 dims): [x1 vx1 y1 vy1 z1 vz1 x2 vx2 y2 vy2 z2 vz2]
        self.state = np.array(config.init_state).flatten().astype(np.float32)
        if self.state.shape != (self.config.stateDim,):
            raise ValueError(f"Invalid shape for initial state: got {self.state.shape}, expected {self.config.stateDim}")

        self.reset()

        self.addState = self.initAddState(config.add_state)

        self.stateLog.append(self.state.copy())
        self.addStateLog.append(self.copyAddState())

        # live rendering support (non-blocking window)
        self.live_render = live_render
        self._renderer = None
        if self.live_render:
            # EnvironmentRenderer is defined later in this file; instantiate lazily
            self._renderer = EnvironmentRenderer(self)  # type: ignore
            # show non-blocking window for live updates
            self._renderer.show()

    def reset(self):
        self.stateLog: list[np.ndarray] = []
        self.addStateLog: list[AddStateType] = []
        self.actionLog: list[np.ndarray] = []

        self.nSteps = 0

        self.hasCollided = False
        self.winner: Optional[int] = None
        self.outside: Optional[int] = None

    @abstractmethod
    def initAddState(self, add_state: Optional[AddStateType]) -> AddStateType: ...

    def copyAddState(self) -> AddStateType:
        return self.addState

    def getStateBlock(self, agent: int, state: Optional[np.ndarray] = None) -> np.ndarray:
        "Return the state block [x, vx, y, vy, z, vz] corresponding to the given agent"
        if state is None:
            state = self.state
        return state[agent * self.config.dim * 2 : (agent + 1) * self.config.dim * 2]

    def checkCollision(self, state: Optional[np.ndarray] = None, addState: Optional[AddStateType] = None) -> bool:
        "Return True if there is a collision between any pair of agents. This base method does not consider the additional state, but may be overriden"

        for agent1 in range(self.config.nAgents):
            for agent2 in range(agent1 + 1, self.config.nAgents):
                block1, block2 = self.getStateBlock(agent1, state), self.getStateBlock(agent2, state)
                if np.linalg.norm(block1[::2] - block2[::2]) < self.config.minDist:
                    return True

        return False

    @abstractmethod
    def checkOutside(self, state: Optional[np.ndarray] = None, addState: Optional[AddStateType] = None) -> Optional[int]: ...

    @abstractmethod
    def checkWinner(self, state: Optional[np.ndarray] = None, addState: Optional[AddStateType] = None) -> Optional[int]: ...

    @abstractmethod
    def getTarget(self, agent: int, state: Optional[np.ndarray] = None, addState: Optional[AddStateType] = None) -> np.ndarray:
        "Return a target for the given agent (depending on the environment)"
        ...

    @abstractmethod
    def getAdvance(self, agent: int, state: Optional[np.ndarray] = None, addState: Optional[AddStateType] = None) -> float:
        "Return an absolute indicator of advancement in the arena (arbitrarily normalized, might be negative), ex: advancement through the track"
        ...

    @abstractmethod
    def closestBoundaryDist(
        self, agent: int, state: Optional[np.ndarray] = None, addState: Optional[AddStateType] = None
    ) -> float:
        "Return the distance of the agent to the closest boundary (for cost purposes)"
        ...

    def addDynStep(self, state: np.ndarray, addState: AddStateType) -> tuple[np.ndarray, AddStateType]:
        "Additional part of the dynamic related to the additional state. By default, does nothing."
        return state, addState

    def dynStep(
        self, action: list | np.ndarray, state: list | np.ndarray, addState: AddStateType
    ) -> tuple[np.ndarray, AddStateType]:
        "Run a step of the simulation dynamics; return newState, newAddState. Does not log anything. Calls addDynStep to run the part of the dynamic"

        # action format (2 agents, 3 dims): [ax1 ay1 az1 ax2 ay2 az2]
        action = np.array(action).flatten().astype(np.float32)
        if action.shape != (self.config.actionDim,):
            raise ValueError(f"Invalid shape for action: got {action.shape}, expected {self.config.actionDim}")

        # sanitize actions (each component is capped by config.maxAccel[agent])
        for agent in range(self.config.nAgents):
            block = action[agent * self.config.dim : (agent + 1) * self.config.dim]
            block[:] = np.clip(block, -self.config.maxAccel[agent], self.config.maxAccel[agent])

        # add noise
        action += np.random.normal(0, self.config.actionNoiseLevel, action.shape)

        newState = np.array(state)

        for agent in range(self.config.nAgents):
            stateBlock = self.getStateBlock(agent, newState)
            actionBlock = action[agent * self.config.dim : (agent + 1) * self.config.dim]

            # update positions
            stateBlock[::2] += self.config.dt * stateBlock[1::2]

            # update speeds
            stateBlock[1::2] += self.config.dt * actionBlock

            # sanitize speeds (the speed should be less than config.maxSpeed[agent], as the L2 norm)
            if np.linalg.norm(stateBlock[1::2]) > self.config.maxSpeed[agent]:
                stateBlock[1::2] *= self.config.maxSpeed[agent] / np.linalg.norm(stateBlock[1::2])

            # add noise
            stateBlock[::2] += np.random.normal(0, self.config.posNoiseLevel, self.config.dim)
            stateBlock[1::2] += np.random.normal(0, self.config.speedNoiseLevel, self.config.dim)

        return self.addDynStep(newState, addState)

    def step(self) -> tuple[bool, Optional[int], Optional[int]]:
        "Run a step of the simulation using the given controllers; return [isCollision, isOutside, winner]"
        action = [controller.getControl(i, self.state, self.addState) for (i, controller) in enumerate(self.controllers)]
        newState, newAddState = self.dynStep(action, self.state, self.addState)
        self.state = newState
        self.addState = newAddState

        self.actionLog.append(np.array(action).flatten())
        self.stateLog.append(self.state.copy())
        self.addStateLog.append(self.copyAddState())
        self.nSteps += 1

        # Notify live renderer (non-intrusive; does not affect dynamics)
        if self.live_render and self._renderer is not None:
            try:
                self._renderer.onNewState()
            except Exception as e:
                print(f"Renderer error: {e}")

        winner = self.checkWinner()  # collision & outside check are disabled if there is a winner
        collision, outside = self.checkCollision(), self.checkOutside()

        if self._renderer is not None:
            self._renderer.winner = winner
            self._renderer.collision = collision
            self._renderer.outside = outside
        return collision, outside, winner

    @abstractmethod
    def getBounds(self) -> tuple[np.ndarray, np.ndarray]:
        "Return min and max bounds for the arena (for display purposes)"
        ...

    @abstractmethod
    def renderBackground(self, ax: Axes) -> None: ...

    def render(self, axis=(0, 1)) -> None:
        "Render the position of the drones (supports arbitrary number of agents). This function is blocking; for live rendering, provide live_render=True at initialization"
        if self.live_render:
            warnings.warn("Should not call Environment.render() when live rendering is active")
            return

        renderer = EnvironmentRenderer(self, axis, autoplay=False)  # type: ignore  # renderer is now only compatible with gate environment... TODO: just delete everything else
        renderer.finish()
        return


class SimpleEnvironment(BaseEnvironment[SimpleEnvironmentConfig, NoneType]):
    "Simple drone arena where the goal is to reach a target, can handle arbitrary number of agents"

    def initAddState(self, add_state: None) -> None:
        # this is kind of stupid, but you need to implement it to make it possible for classes which do need additional states
        return None

    def checkOutside(self, state: Optional[np.ndarray] = None, addState=None) -> Optional[int]:
        "If an agent is outside of the arena (loses), return its id"

        for agent in range(self.config.nAgents):
            block = self.getStateBlock(agent, state)
            if np.any(block[::2] < self.config.arenaMin) or np.any(block[::2] > self.config.arenaMax):
                return agent

    def checkWinner(self, state: Optional[np.ndarray] = None, addState=None) -> Optional[int]:
        """Return the id of the winner (reaching the target set) or None otherwise

        Target set is defined as a drone having all coordinates except Y less than config.gateRadius (in abs value), and coordinate Y greater than 0"""

        for agent in range(self.config.nAgents):
            stateBlock = self.getStateBlock(agent, state)
            if stateBlock[2] > 0 and np.max(np.abs(np.concatenate([[stateBlock[0]], stateBlock[4::2]]))) < self.config.gateRadius:
                return agent

    def getTarget(self, agent: int, state: Optional[np.ndarray] = None, addState=None) -> np.ndarray:
        return np.zeros(self.config.dim)

    def getAdvance(self, agent: int, state: Optional[np.ndarray] = None, addState=None) -> float:
        return -np.linalg.norm(self.getStateBlock(agent, state)[::2])  # type: ignore

    def getBounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.config.arenaMin, self.config.arenaMax

    def closestBoundaryDist(self, agent: int, state: Optional[np.ndarray] = None, addState=None) -> float:
        block = self.getStateBlock(agent, state)
        distToMin = block[::2] - self.config.arenaMin
        distToMax = self.config.arenaMax - block[::2]
        return min(np.min(distToMin), np.min(distToMax))  # type: ignore

    def renderBackground(self, ax: Axes) -> None:
        # draw the gate
        ax.plot([-self.config.gateRadius, self.config.gateRadius], [0, 0], color="green", linewidth=5)


class TrackEnvironment(BaseEnvironment[TrackEnvironmentConfig, tuple[np.ndarray, np.ndarray]]):
    # additional state is (currentS, nLaps)
    def __init__(
        self,
        config: TrackEnvironmentConfig,
        controllers: list[Controller],
        *,
        live_render: bool = False,
    ):
        if config.trackPoints is not None:
            self.trackPoints = config.trackPoints
        else:
            if config.centerline is None:
                raise ValueError("Should either specify config.trackPoints or config.centerline")
            sGrid = np.linspace(0, 1, config.nTrackSamples, endpoint=False)
            self.trackPoints = np.array([config.centerline(s) for s in sGrid])

        super().__init__(config, controllers, live_render=live_render)

    def initAddState(self, add_state: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
        if add_state is None:
            return (
                np.array([self.getProgress(self.getStateBlock(agent)[::2]) for agent in range(self.config.nAgents)]),
                np.zeros(self.config.nAgents),
            )
        return add_state

    def copyAddState(self) -> tuple[np.ndarray, np.ndarray]:
        return self.addState[0].copy(), self.addState[1].copy()

    def projectOnTrack(self, pos: np.ndarray) -> tuple[float, float, np.ndarray]:
        "Project a point on the track; return (s, dist, trackPoint)"
        diffs = self.trackPoints - pos
        dists = np.linalg.norm(diffs, axis=1)

        idx = np.argmin(dists)
        return float(idx) / self.config.nTrackSamples, dists[idx], self.trackPoints[idx]

    def getProgress(self, pos: np.ndarray) -> float:
        s, _, _ = self.projectOnTrack(pos)
        return s

    def isOnTrack(self, pos: np.ndarray) -> bool:
        _, dist, _ = self.projectOnTrack(pos)
        return dist <= self.config.trackWidth / 2

    def checkOutside(
        self, state: Optional[np.ndarray] = None, addState: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> Optional[int]:
        for agent in range(self.config.nAgents):
            if not self.isOnTrack(self.getStateBlock(agent, state)[::2]):
                return agent

    def checkWinner(
        self, state: Optional[np.ndarray] = None, addState: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> Optional[int]:
        laps = addState[1] if addState is not None else self.addState[1]
        for agent in range(self.config.nAgents):
            if laps[agent] == self.config.nWinLaps:
                return agent

    def getTarget(
        self, agent: int, state: Optional[np.ndarray] = None, addState: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        "For target, return the track point at the distance config.targetDistance (as in f(s + targetDistance))"
        currentS = addState[0] if addState is not None else self.addState[0]
        return self.trackPoints[int((currentS[agent] + self.config.targetDistance) % 1 * self.config.nTrackSamples)]
        # return self.config.centerline((currentS[agent] + self.config.targetDistance) % 1)

    def getAdvance(
        self, agent: int, state: Optional[np.ndarray] = None, addState: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> float:
        currentS, laps = addState if addState is not None else self.addState
        return currentS[agent] + laps[agent]

    def getBounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.min(self.trackPoints, axis=0) - self.config.trackWidth * 2,
            np.max(self.trackPoints, axis=0) + self.config.trackWidth * 2,
        )

    def closestBoundaryDist(
        self, agent: int, state: Optional[np.ndarray] = None, addState: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> float:
        _, dist, _ = self.projectOnTrack(self.getStateBlock(agent, state)[::2])
        return self.config.trackWidth / 2 - dist

    def addDynStep(
        self, state: np.ndarray, addState: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        "Update the S position and the number of laps for the additional state"
        newS = addState[0].copy()
        newLaps = addState[1].copy()

        for agent in range(self.config.nAgents):
            s = self.getProgress(self.getStateBlock(agent, state)[::2])
            if s > newS[agent] + 0.5:  # went through start line backwards
                newLaps[agent] -= 1
            if s < newS[agent] - 0.5:  # went through start line forwards
                newLaps[agent] += 1

            newS[agent] = s

        return state, (newS, newLaps)

    def renderBackground(self, ax: Axes) -> None:
        trackPoints = self.trackPoints[: self.config.nTrackSamples, :]

        dpts = np.gradient(trackPoints, axis=0)

        normals = np.stack([-dpts[:, 1], dpts[:, 0]], axis=1)

        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals /= norms

        half_w = self.config.trackWidth / 2
        left = trackPoints + half_w * normals
        right = trackPoints - half_w * normals

        ax.plot(trackPoints[:, 0], trackPoints[:, 1], "k--")
        for i in range(1, self.config.nRaceLines):
            pts = self.trackPoints[i * self.config.nTrackSamples : (i + 1) * self.config.nTrackSamples, :]
            ax.plot(pts[:, 0], pts[:, 1], "g--", alpha=0.5)

        ax.plot(left[:, 0], left[:, 1], "black", linewidth=2)
        ax.plot(right[:, 0], right[:, 1], "black", linewidth=2)

        ax.plot([left[0, 0], right[0, 0]], [left[0, 1], right[0, 1]], "purple", linewidth=1)


class GateEnvironment(BaseEnvironment[GateEnvironmentConfig, tuple[np.ndarray, np.ndarray, np.ndarray]]):
    # additional state is (currentS, nLaps, nGates)
    # this environment cannot be used on the Python side (methods are not implemented), it is meant to be used through cuda and rendered only
    def __init__(
        self,
        config: GateEnvironmentConfig,
        controllers: list[Controller],
        *,
        live_render: bool = False,
    ):
        super().__init__(config, controllers, live_render=live_render)

    def initAddState(
        self,
        add_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if add_state is None:
            # we should properly project here to have a meaningful s, hopefully it will correct itself (this is only a fallback anyway)
            return np.zeros(self.config.nAgents), np.zeros(self.config.nAgents), np.zeros(self.config.nAgents)
        return add_state

    def checkOutside(
        self, state: np.ndarray | None = None, addState: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> int | None: ...

    def checkWinner(
        self, state: np.ndarray | None = None, addState: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> int | None: ...

    def checkCollision(
        self, state: np.ndarray | None = None, addState: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> bool: ...

    def getTarget(
        self, agent: int, state: np.ndarray | None = None, addState: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> np.ndarray: ...

    def getAdvance(
        self, agent: int, state: np.ndarray | None = None, addState: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> float: ...

    def closestBoundaryDist(
        self, agent: int, state: np.ndarray | None = None, addState: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> float: ...

    def getBounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.config.arenaMin, self.config.arenaMax

    def renderBackground(self, ax: Axes, display_raceline: Optional[list[bool]] = None) -> None:
        # draw gates
        gatePts: list[list] = []
        for i in range(self.config.nGates):
            if self.config.dim == 2:
                vec = self.config.gateRadius[i] * np.array([-self.config.gateVectors[i, 1], self.config.gateVectors[i, 0]])
                gatePts.append([self.config.gateCenters[i] - vec, self.config.gateCenters[i] + vec])
            lc = LineCollection(gatePts, colors=[1.0, 0.0, 0.0, 1.0], linewidth=3)
            ax.add_collection(lc)

        # draw race lines
        if self.config.trackPoints is not None:
            for i in range(self.config.nRaceLines):
                if display_raceline is None or display_raceline[i]:
                    pts = self.config.trackPoints[i * self.config.nTrackSamples : (i + 1) * self.config.nTrackSamples, :]
                    ax.plot(pts[:, 0], pts[:, 1], "g--", alpha=0.5)
