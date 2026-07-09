from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.text import Text
from utils import (
    BaseControllerConfig,
    BaseEnvironmentConfig,
    DroneRaceEnvironmentConfig,
    DroneRaceSimState,
    FullStateInfo,
    HiddenObsEnvironmentConfig,
    HiddenObsSimState,
    StratRaceEnvironmentConfig,
    StratRaceSimState,
)
from matplotlib import patches
from matplotlib.path import Path
import matplotlib.pyplot as plt


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


class BaseRaceEnvRenderer[EnvConfigT: DroneRaceEnvironmentConfig | StratRaceEnvironmentConfig](
    BaseEnvironmentRenderer[EnvConfigT]
):
    def basePostInit(
        self,
        iMppi: int,
        nAgents: int,
        contNames: list[str],
        renderTrails: bool = True,
        **kwargs: Any,
    ) -> None:
        self.iMppi = iMppi
        self.nAgents = nAgents
        self.renderTrails = renderTrails

        self.contNames = contNames

    def drawBackground(self) -> None:
        # color maps and patch/marker colors for drones
        self.cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "BuPu"]
        self.colors = ["blue", "red", "green", "purple", "orange", "gray", "brown", "pink"]

        # line collections for trajectories
        self.lcs: list[LineCollection] = []

        for i_agent in range(self.nAgents):
            lc = LineCollection([], cmap=self.cmaps[i_agent % self.nAgents], linewidth=4, alpha=0.8)
            self.lcs.append(lc)
            self.ax.add_collection(lc)  # type: ignore

        # points and collision circles
        self.points: list[Any] = []
        self.circles: list[Circle] = []
        for i_agent in range(self.nAgents):
            # px, py = self.getPos(0, i_agent)
            px, py = 0.0, 0.0
            color = self.colors[i_agent % len(self.colors)]
            (pt,) = self.ax.plot(
                [px], [py], marker="o", color=color, markersize=8, label=f"{self.contNames[i_agent]} ({i_agent + 1})"
            )
            self.points.append(pt)
            circ = Circle((px, py), radius=self.envConfig.droneRadius, fill=True, color=color, linestyle="--", alpha=0.3)
            self.circles.append(circ)
            self.ax.add_patch(circ)

        self.agent_value_texts: list[Text] = []
        y_positions = np.linspace(0.7, 0.2, self.nAgents)

        for i in range(self.nAgents):
            self.ax_status.text(
                0.0,
                y_positions[i],
                f"Agent {i + 1}: ",
                fontsize=12,
                ha="left",
                va="top",
                color=self.colors[i % len(self.colors)],
                fontweight="bold",
                transform=self.ax_status.transAxes,
            )

            t_val = self.ax_status.text(
                1.0,
                y_positions[i],
                "",
                fontsize=12,
                ha="right",
                va="top",
                color="black",
                transform=self.ax_status.transAxes,
            )
            self.agent_value_texts.append(t_val)

        self.ax.legend()

    def drawFrame(self, stateLog: list[FullStateInfo[DroneRaceSimState | StratRaceSimState, Any]], iFrame: int) -> None:
        state = stateLog[iFrame].state

        # update trails
        if self.renderTrails:
            for idx, lc in enumerate(self.lcs):
                x_arr = [self.getPos(stateLog[j].state, idx)[0] for j in range(iFrame + 1)]
                y_arr = [self.getPos(stateLog[j].state, idx)[1] for j in range(iFrame + 1)]

                lc.set_segments([[[x_arr[j], y_arr[j]], [x_arr[j + 1], y_arr[j + 1]]] for j in range(iFrame)])
                if iFrame > 1:
                    lc.set_array(np.linspace(0, 1, iFrame))

        # update points and circles
        for idx, pt in enumerate(self.points):
            px, py = self.getPos(state, idx)
            pt.set_data([px], [py])
            self.circles[idx].center = (px, py)

        self.updateStatus(state)

    @abstractmethod
    def updateStatus(self, state: DroneRaceSimState | StratRaceSimState): ...

    def getZoomPos(
        self, stateLog: list[FullStateInfo[DroneRaceSimState | StratRaceSimState, Any]], iFrame: int
    ) -> tuple[float, float]:
        return self.getPos(stateLog[iFrame].state, self.iMppi)

    def coordIndex(self, agent: int, coord: int) -> int:
        return agent * self.envConfig.dim + coord

    def getPos(self, state: DroneRaceSimState | StratRaceSimState, agent: int) -> tuple[float, float]:
        s = state.pos
        return (
            float(s[self.coordIndex(agent, 0)]),
            float(s[self.coordIndex(agent, 1)]),
        )


class DroneRaceEnvRenderer(BaseRaceEnvRenderer[DroneRaceEnvironmentConfig]):
    def postInit(
        self,
        contNames: Optional[list[str]] = None,
        display_raceline: bool | list[bool] = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(display_raceline, bool):
            display_raceline = [display_raceline] * self.envConfig.nRaceLines
        self.display_raceline = display_raceline

        if contNames is None:
            contNames = [
                self.contConfig.getDefaultName()
                if i == self.envConfig.iMppi
                else self.envConfig.opponentPidConfigs[self.envConfig.trueTheta].getDefaultName()
                for i in range(2)
            ]

        super().basePostInit(self.envConfig.iMppi, self.envConfig.nAgents, contNames, **kwargs)

        # only display the racelines which are actually used
        # used = [False] * self.envConfig.nRaceLines
        # for cfg in self.envConfig.opponentPidConfigs:
        #     used[cfg.racelineIndex] = True

    def drawBackground(self) -> None:
        super().drawBackground()

        # draw gates
        gatePts: list[list] = []
        for i in range(self.envConfig.nGates):
            if self.envConfig.dim == 2:
                vec = self.envConfig.gateRadius[i] * np.array(
                    [-self.envConfig.gateVectors[i, 1], self.envConfig.gateVectors[i, 0]]
                )
                gatePts.append([self.envConfig.gateCenters[i] - vec, self.envConfig.gateCenters[i] + vec])

            lc = LineCollection(gatePts, colors=[1.0, 0.0, 0.0, 1.0], linewidth=3)
            self.ax.add_collection(lc)

        # draw race lines
        if self.envConfig.trackPoints is not None:
            for i in range(self.envConfig.nRaceLines):
                if self.display_raceline[i]:
                    pts = self.envConfig.trackPoints[i * self.envConfig.nTrackSamples : (i + 1) * self.envConfig.nTrackSamples, :]
                    self.ax.plot(pts[:, 0], pts[:, 1], color="grey", linestyle="--", alpha=0.3)

        # draw obstacles
        for omin, omax in self.envConfig.obstacles.reshape(self.envConfig.nObstacles, 2, self.envConfig.dim):
            self.drawRectangle(omin, omax)

        for center, radius in zip(
            self.envConfig.roundObsCenters.reshape(self.envConfig.nRoundObstacles, self.envConfig.dim),
            self.envConfig.roundObsRadius,
        ):
            self.drawCircle(tuple(center), radius)

    def updateStatus(self, state: DroneRaceSimState | StratRaceSimState):
        assert isinstance(state, DroneRaceSimState), f"Invalid state class: got {type(state)}, expected DroneRaceSimState"

        for iAgent in range(self.nAgents):
            vel, laps, gates = state.vel, state.laps, state.gates

            speed = np.linalg.norm(vel[iAgent * self.envConfig.dim : (iAgent + 1) * self.envConfig.dim])

            self.agent_value_texts[iAgent].set_text(
                f"Lap {int(laps[iAgent])}/{self.envConfig.nWinLaps} Gate {int(gates[iAgent])}/{self.envConfig.nGates}\nSpeed {speed:.2f}"
            )


class StratRaceEnvRenderer(BaseRaceEnvRenderer[StratRaceEnvironmentConfig]):
    def postInit(
        self,
        contNames: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        if contNames is None:
            contNames = ["MPPI"] + ["Opponent"] * self.envConfig.nOppAgents

        super().basePostInit(0, self.envConfig.nOppAgents + 1, contNames, **kwargs)

    def drawBackground(self) -> None:
        trackPoints = self.envConfig.p_grid

        dpts = np.gradient(trackPoints, axis=0)

        normals = np.stack([-dpts[:, 1], dpts[:, 0]], axis=1)

        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals /= norms

        left = trackPoints + self.envConfig.trackWidth * normals
        right = trackPoints - self.envConfig.trackWidth * normals

        # draw raceline
        self.ax.plot(trackPoints[:, 0], trackPoints[:, 1], "g--", alpha=0.5)

        # draw boundaries
        self.ax.plot(left[:, 0], left[:, 1], "black", linewidth=2)
        self.ax.plot(right[:, 0], right[:, 1], "black", linewidth=2)

        self.ax.plot([left[0, 0], right[0, 0]], [left[0, 1], right[0, 1]], "purple", linewidth=1)

        super().drawBackground()

    def updateStatus(self, state: DroneRaceSimState | StratRaceSimState):
        assert isinstance(state, StratRaceSimState), f"Invalid state class: got {type(state)}, expected StratRaceSimState"

        for iAgent in range(self.nAgents):
            vel, laps, S, latDist = state.vel, state.laps, state.S, state.latDist

            speed = np.linalg.norm(vel[iAgent * self.envConfig.dim : (iAgent + 1) * self.envConfig.dim])

            text = f"Lap {int(laps[iAgent])}/{self.envConfig.nWinLaps} Advance {int(S[iAgent] * 100 / self.envConfig.trackLength)}% \nSpeed {speed:.2f}\nLateral offset {latDist[iAgent]:.2f}"
            if iAgent >= 1:
                text += f"\nGoal lateral offset {state.latDistTarget[iAgent - 1]:.2f}"

            self.agent_value_texts[iAgent].set_text(text)


class HiddenObsEnvRenderer(BaseEnvironmentRenderer[HiddenObsEnvironmentConfig]):
    def postInit(
        self,
        renderTrails: bool = True,
        **kwargs: Any,
    ) -> None:
        self.renderTrails = renderTrails

    def drawBackground(self) -> None:
        # color maps and patch/marker colors for drones
        self.cmap = "Blues"  # "Reds"
        self.color = "blue"  # "red"

        # line collection for trajectory
        self.lc = LineCollection([], cmap=self.cmap, linewidth=4, alpha=0.8)
        self.ax.add_collection(self.lc)  # type: ignore

        # points and collision circles
        self.point = self.ax.plot([0.0], [0.0], marker="o", color=self.color, markersize=8)[0]

        self.circle = Circle(
            (0.0, 0.0), radius=self.envConfig.droneRadius, fill=True, color=self.color, linestyle="--", alpha=0.3
        )
        self.ax.add_patch(self.circle)

        self.agent_value_text = self.ax_status.text(
            1.0,
            0.4,
            "",
            fontsize=12,
            ha="right",
            va="center",
            color="black",
            transform=self.ax_status.transAxes,
        )

        # draw gates
        gatePts: list[list] = []
        for i in range(self.envConfig.nGates):
            if self.envConfig.dim == 2:
                vec = self.envConfig.gateRadius[i] * np.array(
                    [-self.envConfig.gateVectors[i, 1], self.envConfig.gateVectors[i, 0]]
                )
                gatePts.append([self.envConfig.gateCenters[i] - vec, self.envConfig.gateCenters[i] + vec])

            lc = LineCollection(gatePts, colors=[1.0, 0.0, 0.0, 1.0], linewidth=3)
            self.ax.add_collection(lc)

        # draw obstacles
        for omin, omax in self.envConfig.rectObstacles.reshape(self.envConfig.nRectObstacles, 2, self.envConfig.dim):
            self.drawRectangle(omin, omax)

        for i, (center, radius) in enumerate(
            zip(
                self.envConfig.hiddenObsCenters.reshape(self.envConfig.nHiddenObstacles, self.envConfig.dim),
                self.envConfig.hiddenObsRadius,
            )
        ):
            self.drawCircle(tuple(center), radius, fill=self.envConfig.trueTheta == i, dottedEdge=True)

        self.drawHalfPlane(
            self.envConfig.dblHpLimits[0],
            self.envConfig.dblHpLimits[1],
            self.envConfig.dblHpLambdas[0],
            self.envConfig.dblHpLambdas[1],
            False,
        )
        self.drawHalfPlane(
            self.envConfig.dblHpLimits[2],
            self.envConfig.dblHpLimits[3],
            self.envConfig.dblHpLambdas[2],
            self.envConfig.dblHpLambdas[3],
            True,
        )

        self.drawAnnulus(*self.envConfig.annulusCenter, *self.envConfig.annulusRadius)

    def drawFrame(self, stateLog: list[FullStateInfo[HiddenObsSimState, Any]], iFrame: int) -> None:
        state = stateLog[iFrame].state

        # update trails
        if self.renderTrails:
            x_arr = [stateLog[j].state.pos[0] for j in range(iFrame + 1)]
            y_arr = [stateLog[j].state.pos[1] for j in range(iFrame + 1)]

            self.lc.set_segments([[[x_arr[j], y_arr[j]], [x_arr[j + 1], y_arr[j + 1]]] for j in range(iFrame)])
            if iFrame > 1:
                self.lc.set_array(np.linspace(0, 1, iFrame))

        # update points and circles
        self.point.set_data([state.pos[0]], [state.pos[1]])
        self.circle.center = state.pos

        # update status
        vel, laps, gates = state.vel, state.laps, state.gates

        speed = np.linalg.norm(vel)

        self.agent_value_text.set_text(
            f"Lap {int(laps)}/{self.envConfig.nWinLaps} Gate {int(gates)}/{self.envConfig.nGates}\nSpeed {speed:.2f}"
        )

    def getZoomPos(self, stateLog: list[FullStateInfo[HiddenObsSimState, Any]], iFrame: int) -> tuple[float, float]:
        return tuple(stateLog[iFrame].state.pos)

    def coordIndex(self, agent: int, coord: int) -> int:
        return agent * self.envConfig.dim + coord

    def getPos(self, state: DroneRaceSimState, agent: int) -> tuple[float, float]:
        s = state.pos
        return (
            float(s[self.coordIndex(agent, 0)]),
            float(s[self.coordIndex(agent, 1)]),
        )
