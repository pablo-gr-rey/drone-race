import io
import os
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider, TextBox
from PIL import Image
from tqdm import tqdm
from utils import (
    BaseEnvironmentConfig,
    BaseEnvironmentRenderer,
    BaseControllerConfig,
    DroneRaceEnvironmentConfig,
    DroneRaceSimState,
    FullStateInfo,
    HiddenObsEnvironmentConfig,
    HiddenObsSimState,
    MPPIConfig,
    PRMPPIConfig,
)
from controller_renderers import (
    ControllerRenderer,
    MPPIDroneRaceRenderer,
    MPPIHiddenObsRenderer,
    PRMPPIDroneRaceRenderer,
    PRMPPIHiddenObsRenderer,
)


def getRendererClass(
    envConfig: BaseEnvironmentConfig, contConfig: BaseControllerConfig
) -> tuple[type[BaseEnvironmentRenderer], type["ControllerRenderer"]]:
    if isinstance(envConfig, DroneRaceEnvironmentConfig):
        if isinstance(contConfig, MPPIConfig):
            return DroneRaceEnvRenderer, MPPIDroneRaceRenderer
        elif isinstance(contConfig, PRMPPIConfig):
            return DroneRaceEnvRenderer, PRMPPIDroneRaceRenderer

    elif isinstance(envConfig, HiddenObsEnvironmentConfig):
        if isinstance(contConfig, MPPIConfig):
            return HiddenObsEnvRenderer, MPPIHiddenObsRenderer
        elif isinstance(contConfig, PRMPPIConfig):
            return HiddenObsEnvRenderer, PRMPPIHiddenObsRenderer

    raise ValueError(f"Unknown env & cont config classes {type(envConfig)}, {type(contConfig)}")


class DroneRaceEnvRenderer(BaseEnvironmentRenderer[DroneRaceEnvironmentConfig]):
    def postInit(
        self,
        contNames: Optional[list[str]] = None,
        display_raceline: bool | list[bool] = True,
        renderTrails: bool = True,
        **kwargs: Any,
    ) -> None:
        self.iMppi = self.envConfig.iMppi
        self.renderTrails = renderTrails

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
        self.contNames = contNames

        # only display the racelines which are actually used
        used = [False] * self.envConfig.nRaceLines
        for cfg in self.envConfig.opponentPidConfigs:
            used[cfg.racelineIndex] = True

    def drawBackground(self) -> None:
        # color maps and patch/marker colors for drones
        self.cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "BuPu"]
        self.colors = ["blue", "red", "green", "purple", "orange", "gray", "brown", "pink"]

        self.nAgents = self.envConfig.nAgents

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

        self.ax.legend()

    def drawFrame(self, stateLog: list[FullStateInfo[DroneRaceSimState, Any]], iFrame: int) -> None:
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

        # update status
        for iAgent in range(self.nAgents):
            vel, laps, gates = state.vel, state.laps, state.gates

            speed = np.linalg.norm(vel[iAgent * self.envConfig.dim : (iAgent + 1) * self.envConfig.dim])

            self.agent_value_texts[iAgent].set_text(
                f"Lap {int(laps[iAgent])}/{self.envConfig.nWinLaps} Gate {int(gates[iAgent])}/{self.envConfig.nGates}\nSpeed {speed:.2f}"
            )

    def getZoomPos(self, stateLog: list[FullStateInfo[DroneRaceSimState, Any]], iFrame: int) -> tuple[float, float]:
        return self.getPos(stateLog[iFrame].state, self.envConfig.iMppi)

    def coordIndex(self, agent: int, coord: int) -> int:
        return agent * self.envConfig.dim + coord

    def getPos(self, state: DroneRaceSimState, agent: int) -> tuple[float, float]:
        s = state.pos
        return (
            float(s[self.coordIndex(agent, 0)]),
            float(s[self.coordIndex(agent, 1)]),
        )


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


class EnvironmentRenderer:
    """Helper that manages a non-blocking matplotlib window for live rendering.
    If you wish the window to stay open at the end of the dynamics loop, make sure to call renderer.finish() which will block the program.
    """

    def __init__(
        self,
        envConfig: BaseEnvironmentConfig,
        contConfig: BaseControllerConfig,
        oppNames: Optional[list[list[str]]] = None,
        axis: tuple[int, ...] = (0, 1),
        interval: int = 30,
        autoplay: bool = True,
        frameSkipPlayback: int = 2,
        frameSkipWaiting: int = 1,
        startZoomed: bool = False,
        **kwargs: Any,
    ):
        "interval: refresh rate. frameSkipWaiting: how many frames to skip if emitting states faster than we can display (use -1 to always display last frame). use defaultZoomAgent=-1 to start viewing full track, otherwise start zooming on agent"
        self.envConfig = envConfig
        self.axis = axis
        self.interval = interval
        self.frameSkipPlayback = frameSkipPlayback
        self.frameSkipWaiting = frameSkipWaiting

        self.contConfig = contConfig

        if oppNames is None:
            oppNames = [
                [f"Param {k}={i}" for i in range(self.envConfig.modelSizes[k])] for k in range(self.envConfig.nModelFactors)
            ]

        self.stateLog: list[FullStateInfo] = []

        self.collision = False
        self.winner: Optional[int] = None
        self.outside: Optional[int] = None

        plt.rcParams["keymap.back"].remove("left")
        plt.rcParams["keymap.forward"].remove("right")

        plt.ion()
        plt.show()

        self.fig = plt.figure(figsize=(12, 10))

        # try to make picture fullscreen
        # matplotlib can use different backends, so this is not guaranteed to work

        fig_manager = plt.get_current_fig_manager()
        if fig_manager is not None:
            try:
                # default tk backend on X11
                fig_manager.window.attributes("-zoomed", True)  # type: ignore
            except Exception:
                try:
                    # default tk backend (but sometimes works better)
                    fig_manager.window.state("zoomed")  # type: ignore
                except Exception:
                    try:
                        # Qt backend
                        fig_manager.window.showMaximized()  # type: ignore
                    except Exception:
                        # give up
                        pass

        gs = GridSpec(
            2,
            2,
            figure=self.fig,
            width_ratios=[3, 1],
            height_ratios=[15, 1],
            wspace=0.2,
            hspace=0.1,
            left=0.04,
            right=0.96,
            bottom=0.06,
            top=0.96,
        )

        gs_plot = gs[0, 0].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)

        self.ax = self.fig.add_subplot(gs_plot[0])  # track ax
        self.ax.set_aspect("equal", adjustable="box")

        self.axs_action_plot = [self.fig.add_subplot(gs_plot[i + 1]) for i in range(envConfig.actionDim)]

        gs_status_cont = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 8], hspace=0.1)

        self.ax_status = self.fig.add_subplot(gs_status_cont[0])
        self.ax_status.axis("off")

        envRendererCls, contRendererCls = getRendererClass(envConfig, contConfig)

        self.envRenderer = envRendererCls(envConfig, contConfig, self.ax, self.ax_status, oppNames, **kwargs)
        self.contRenderer = contRendererCls(envConfig, contConfig, oppNames, self.ax, self.fig, gs_status_cont[1], axis, **kwargs)

        # slider, empty space, play/pause, save gif, gif name, zoom, +/-, focus
        gs_ui = gs[1, :].subgridspec(1, 6, width_ratios=[7, 1, 1, 1, 1, 1], wspace=0.1)

        # draw static background
        self.envRenderer.drawBackground()

        bmin, bmax = self.envConfig.arenaMin, self.envConfig.arenaMax
        self.ax.set_xlim(xmin=bmin[self.axis[0]], xmax=bmax[self.axis[0]])  # type: ignore
        self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore
        self.ax.set_aspect("equal", adjustable="box")

        self.zoom_radius = 6

        # status text
        self.status_text = self.ax_status.text(
            0.5, 1.0, "Running...", fontsize=14, ha="center", va="top", transform=self.ax_status.transAxes
        )

        # UI: slider, play, save GIF, textbox and zoom
        max_idx = max(1, len(self.stateLog) - 1)
        self.slider = Slider(self.fig.add_subplot(gs_ui[0]), "Time", 0, max_idx, valinit=0, valstep=1)

        self.pause_button = Button(self.fig.add_subplot(gs_ui[2]), "Pause")
        self.pause_button.label.set_fontsize(10)

        self.save_button = Button(self.fig.add_subplot(gs_ui[3]), "Save GIF")
        self.save_button.label.set_fontsize(10)
        self.save_button.set_active(False)

        self.text_box = TextBox(self.fig.add_subplot(gs_ui[4]), "", initial="")
        self.text_box.ax.set_visible(False)  # type: ignore

        if not startZoomed:
            self.zoomed = False
            zoomButtonText = "Zoom"
        else:
            self.zoomed = True
            zoomButtonText = "Full track"

        self.button_zoom = Button(self.fig.add_subplot(gs_ui[5]), zoomButtonText)

        # action plots
        if self.envConfig.actionDim > 4:
            raise ValueError("Must provide dimension names for dim > 4")
        dim_names = ["x", "y", "z", "w"][: self.envConfig.actionDim]

        self.actions_plot: list[tuple[Line2D, Line2D, Line2D]] = []  # plot, plot_dashed (for non-) & vline
        for dim, ax_action in enumerate(self.axs_action_plot):
            self.actions_plot.append(
                (
                    ax_action.plot([], [], color="red", linewidth=2)[0],
                    ax_action.plot([], [], color="orange", linewidth=1, alpha=0.5, zorder=1)[0],
                    ax_action.axvline(x=0, color="green", linewidth=1, alpha=0.5, zorder=1),
                )
            )

            ax_action.set_xlabel("Time")
            ax_action.set_ylabel("Action on " + dim_names[dim])

            maxAccel = self.envConfig.getMaxAccel()
            ax_action.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.3, zorder=0)
            ax_action.axhline(maxAccel, color="blue", linestyle="--", alpha=0.3, zorder=0)
            ax_action.axhline(-maxAccel, color="blue", linestyle="--", alpha=0.3, zorder=0)

            ax_action.autoscale(False)
            ax_action.set_xlim(0, 5)
            ax_action.set_ylim(-maxAccel * 1.1, maxAccel * 1.1)

        # internal state
        self.playing = False
        self.slider_is_updating = False
        self.current_index = 0

        self.isFinished = False

        # timer for playback (non-blocking)
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.timerTick)

        # connect callbacks
        self.slider.on_changed(self.sliderChanged)
        self.pause_button.on_clicked(self.tooglePlay)
        self.save_button.on_clicked(self.askSavePath)
        self.button_zoom.on_clicked(self.toogleZoom)
        self.fig.canvas.mpl_connect("key_press_event", self.onKeyPress)

        self.lastRenderTime = time.perf_counter()

        # initial render
        self.updateDisplay(0)

        if autoplay:
            self.tooglePlay(None, True)

    def updateDisplay(self, i: int, forceZoom: bool = False) -> None:
        n = len(self.stateLog)
        if n == 0:
            return
        i = max(0, min(i, n - 1))

        self.updateStatus(i)

        # update controller-specific plots
        self.envRenderer.drawFrame(self.stateLog, i)
        self.contRenderer.update(self.stateLog[i])

        # update zoom or full view
        if self.zoomed:
            cx, cy = self.envRenderer.getZoomPos(self.stateLog, i)
            self.ax.set_xlim(cx - self.zoom_radius, cx + self.zoom_radius)
            self.ax.set_ylim(cy - self.zoom_radius, cy + self.zoom_radius)
            self.ax.apply_aspect()
        elif forceZoom:  # if the last frame was also full track, no need to change the lims (that avoids re-drawing everything)
            bmin, bmax = self.envConfig.arenaMin, self.envConfig.arenaMax
            self.ax.set_xlim(bmin[self.axis[0]], bmax[self.axis[0]])  # type: ignore
            self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore

        # update slider value without triggering callback
        self.slider_is_updating = True
        self.slider.set_val(i)
        self.slider_is_updating = False

        self.lastRenderTime = time.perf_counter()

        # update action plots
        actionsArr = np.array([state.egoAction for state in self.stateLog])

        scale = np.sqrt(np.sum(actionsArr**2, axis=1))
        coeff = np.maximum(1.0, scale / self.envConfig.getMaxAccel())
        actionNormalized = actionsArr / coeff[:, None]

        for dim, ((plot, lightplot, vline), ax) in enumerate(zip(self.actions_plot, self.axs_action_plot)):
            vline.set_xdata([i, i])
            if len(plot.get_xdata()) != len(self.stateLog):  # type: ignore
                plot.set_xdata(np.arange(len(self.stateLog)))
                lightplot.set_xdata(np.arange(len(self.stateLog)))

                plot.set_ydata(actionNormalized[:, dim])
                lightplot.set_ydata(actionsArr[:, dim])

                ax.set_xlim(-2, len(self.stateLog) + 2)

    def updateStatus(self, i: Optional[int] = None) -> None:
        if i is None:
            i = self.current_index

        # update text status
        if self.collision:
            self.status_text.set_text("Collision")
            self.status_text.set_color("red")
        elif self.winner is not None:
            self.status_text.set_text(f"Winner: {self.winner + 1}")
            self.status_text.set_color("green")
        elif self.outside is not None:
            self.status_text.set_text(f"{self.outside + 1} outside")
            self.status_text.set_color("red")
        elif self.isFinished:
            self.status_text.set_text("Truncated")
            self.status_text.set_color("orange")

    def onNewState(
        self,
        state: FullStateInfo,
        pendingState: bool = False,
    ) -> None:
        "If pendingState is True, it means that there are other states waiting in the queue (ie. they are computed faster than they are rendered); in this case, only 1 frame out of frameSkipWaiting will be shown"
        self.stateLog.append(state)

        # Called by the environment when a new frame is available
        n = len(self.stateLog)
        max_idx = max(n - 1, 1)
        # update slider range
        self.slider.valmax = max_idx  # type: ignore
        self.slider.ax.set_xlim(0, max_idx)  # type: ignore

        # if playing and "waiting" on new states, go to new state
        if (
            self.playing
            and not self.isFinished
            and time.perf_counter() - self.lastRenderTime > self.interval / 1000
            and (not pendingState or self.frameSkipWaiting == -1 or len(self.stateLog) % self.frameSkipWaiting == 0)
        ):
            if self.frameSkipWaiting == -1:
                self.current_index = n - 1
            else:
                self.current_index = min(self.current_index + self.frameSkipWaiting, n - 1)

            # prev = self.lastRenderTime
            self.updateDisplay(self.current_index)
            self.show()
            # print(f"render time: {(self.lastRenderTime - prev) * 1000} ms")

        # if not playing, add small delay to keep UI responsive
        if not self.playing and len(self.stateLog) % 10 == 0:
            self.show()
            # plt.pause(self.interval / 1000)

    def show(self) -> None:
        self.fig.canvas.draw_idle()
        # plt.pause(0.001)
        # print("hey")
        # plt.show()
        if not self.isFinished:
            self.fig.canvas.flush_events()

    def finish(self, tooglePlay: bool = True, jumpToLast: bool = False):
        if jumpToLast:
            self.current_index = len(self.stateLog) - 1

        self.updateDisplay(self.current_index)

        self.isFinished = True
        self.save_button.set_active(True)

        if tooglePlay:
            self.tooglePlay(None)
        elif self.playing:
            self.timer.start()
        plt.show(block=True)

    def sliderChanged(self, val: float) -> None:
        if self.slider_is_updating:
            return
        i = int(val)
        self.current_index = min(len(self.stateLog), i)
        self.updateDisplay(i)
        if not self.isFinished:
            self.playing = False
        self.show()

    def timerTick(self) -> None:
        n = len(self.stateLog)
        if self.current_index >= n - 1 and self.isFinished:
            # stop at the end of available frames
            self.tooglePlay(None)
            return
        elif self.current_index >= n - 1:
            # we caught up and are "waiting" on new frames: stop timer but keep playing
            self.timer.stop()

        if self.playing:
            self.current_index = min(self.current_index + self.frameSkipPlayback, n - 1)

        self.updateDisplay(self.current_index)
        self.show()

    def tooglePlay(self, event: Any, isFirst: bool = False) -> None:
        self.playing = not self.playing
        self.pause_button.label.set_text("Play" if not self.playing else "Pause")  # type: ignore

        if self.playing:
            # if at end, restart
            if self.current_index >= max(0, len(self.stateLog) - 1):
                self.current_index = 0
                self.updateDisplay(0)
            if not isFirst and self.isFinished:
                self.timer.start()
        else:
            self.timer.stop()

    def onKeyPress(self, event: Any) -> None:
        if event.key == "left" and not self.playing:
            self.current_index = max(0, self.current_index - 1)
            self.updateDisplay(self.current_index)
        elif event.key == "right" and not self.playing:
            self.current_index = min(len(self.stateLog) - 1, self.current_index + 1)
            self.updateDisplay(self.current_index)

    def askSavePath(self, event: Any) -> None:
        # show textbox for filename input
        self.text_box.ax.set_visible(True)  # type: ignore
        self.text_box.text_disp.set_text("")  # type: ignore
        self.text_box.on_submit(self.onTextSubmit)
        self.fig.canvas.draw_idle()

    def onTextSubmit(self, text: str) -> None:
        # hide textbox and start saving
        self.text_box.disconnect_events()
        # self.text_box.on_submit(lambda ev: None)
        self.text_box.ax.set_visible(False)  # type: ignore
        self.save_button.set_active(False)
        self.save_button.label.set_text("Saving GIF...")
        self.fig.canvas.draw_idle()
        plt.pause(0.5)
        if text.strip() == "":
            print("No filename provided; aborting GIF save")
            return
        self.saveGif(text.strip())

        self.save_button.label.set_text("GIF saved")

    def saveGif(self, name: str) -> None:
        if not name.endswith(".gif"):
            name = name + ".gif"

        rootdir = Path(__file__).parent.parent
        print("rootdir", rootdir)

        os.makedirs(rootdir / "gifs", exist_ok=True)
        save_path = rootdir / "gifs" / name

        duration = 40

        imgs: list[Image.Image] = []
        n = len(self.stateLog)
        for i in tqdm(range(n), desc=f"Capturing frames for {name}", unit="frame"):
            self.updateDisplay(i)
            self.fig.canvas.draw()

            axes_to_capture = [self.ax, self.ax_status] + self.axs_action_plot + self.contRenderer.getCapturedAxes()
            bboxes = [a.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted()) for a in axes_to_capture]
            full_bbox = Bbox.union(bboxes).padded(0.5)

            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches=full_bbox, pad_inches=0)
            buf.seek(0)

            img = Image.open(buf)
            imgs.append(img.convert("RGB"))
            buf.close()

        try:
            print(f"Saving animation to {save_path}...")
            imgs[0].save(
                save_path, save_all=True, append_images=imgs[1:] + int(1000 / duration) * [imgs[-1]], duration=duration, loop=0
            )
            print(f"Saved animation to {save_path}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")

    def toogleZoom(self, event: Any) -> None:
        self.zoomed = not self.zoomed
        try:
            self.button_zoom.label.set_text("Full track" if self.zoomed else "Zoom")  # type: ignore
        except Exception:
            pass
        # re-run render at current slider position to apply new limits
        self.updateDisplay(int(self.slider.val), forceZoom=True)
        self.fig.canvas.draw_idle()
