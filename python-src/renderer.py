import io
import os
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider, TextBox
from PIL import Image
from tqdm import tqdm
from utils import EVENT_TYPE, FullStateInfo, GateEnvironmentConfig, MPPIConfig, VerifConfig


class EnvironmentRenderer:
    """Helper that manages a non-blocking matplotlib window for live rendering.
    If you wish the window to stay open at the end of the dynamics loop, make sure to call renderer.finish() which will block the program.
    """

    def __init__(
        self,
        envConfig: GateEnvironmentConfig,
        mppiConfig: MPPIConfig,
        contNames: list[str],
        verifConfig: VerifConfig,
        oppNames: list[list[str]],
        axis: tuple[int, ...] = (0, 1),
        interval: int = 30,
        autoplay: bool = True,
        frameSkipPlayback: int = 2,
        frameSkipWaiting: int = 1,
        defaultZoomAgent: int = 0,
        display_raceline: bool | list[bool] = True,
        renderTrails: bool = True,
    ):
        "interval: refresh rate. frameSkipWaiting: how many frames to skip if emitting states faster than we can display (use -1 to always display last frame). use defaultZoomAgent=-1 to start viewing full track, otherwise start zooming on agent"
        self.envConfig = envConfig
        self.verifConfig = verifConfig
        self.axis = axis
        self.interval = interval
        self.frameSkipPlayback = frameSkipPlayback
        self.frameSkipWaiting = frameSkipWaiting
        self.iMppi = self.envConfig.iMppi

        self.mppiConfig = mppiConfig

        if len(oppNames) != envConfig.nModelFactors or any(len(n) != size for n, size in zip(oppNames, envConfig.modelSizes)):
            raise ValueError("Wrong length for oppNames")

        self.oppNames = oppNames
        print(oppNames)

        self.renderTrails = renderTrails

        self.stateLog: list[FullStateInfo] = []

        self.collision = False
        self.winner: Optional[int] = None
        self.outside: Optional[int] = None

        if isinstance(display_raceline, bool):
            display_raceline = [display_raceline] * envConfig.nRaceLines

        plt.rcParams["keymap.back"].remove("left")
        plt.rcParams["keymap.forward"].remove("right")

        plt.ion()
        plt.show()

        self.fig = plt.figure(figsize=(12, 10))

        # try to make picture fullscreen
        # matplotlib can use different backend, so this is not guaranteed to work

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
                    # Qt backend
                    try:
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

        gs_plot = gs[0, 0].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.1)

        self.ax = self.fig.add_subplot(gs_plot[0])  # track ax
        self.ax.set_aspect("equal", adjustable="box")

        self.axs_action_plot = [self.fig.add_subplot(gs_plot[i + 1]) for i in range(envConfig.dim)]

        gs_status = gs[0, 1].subgridspec(4, 1, height_ratios=[1, 1, 3, 3], hspace=0.1)

        self.ax_status = self.fig.add_subplot(gs_status[0])  # for text status
        self.ax_status.axis("off")

        self.ax_failcount = self.fig.add_subplot(gs_status[1])

        gs_marginal = gs_status[2].subgridspec(1, envConfig.nModelFactors)
        self.axs_marg_belief = [self.fig.add_subplot(g) for g in gs_marginal]

        self.ax_joint_belief = self.fig.add_subplot(gs_status[3])

        # slider, empty space, play/pause, save gif, gif name, zoom, +/-, focus
        gs_ui = gs[1, :].subgridspec(1, 8, width_ratios=[7, 1, 1, 1, 1, 1, 0.4, 1], wspace=0.1)

        # draw static background using environment hook
        # self.env.renderBackground(self.ax, display_raceline)
        self.renderBackground(display_raceline)

        # color maps and patch/marker colors for drones
        self.cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "BuPu"]
        self.colors = ["blue", "red", "green", "purple", "orange", "gray", "brown", "pink"]

        # should have shape nModelFactors * (nModelSizes[k]+1)
        self.pred_colors = [["brown", "green", "orange"], ["yellow", "cyan", "purple"]]
        # self.pred_colors = ["brown", "green", "orange"]

        self.nAgents = self.envConfig.nAgents

        # line collections for trajectories
        self.lcs: list[LineCollection] = []

        for i_agent in range(self.nAgents):
            lc = LineCollection([], cmap=self.cmaps[i_agent % self.nAgents], linewidth=4, alpha=0.8)
            self.lcs.append(lc)
            self.ax.add_collection(lc)  # type: ignore

        # line collections for MPPI predictions
        # length: nTrueModels * nModelFactors, with items being (nomMppi, branchMppi, pid). each item is ((nominalStrong, nominalLight), (branchedStrong, branchedLight)). branched color should change based on the real value
        self.lcs_pred: list[list[tuple[tuple[Line2D, Line2D], tuple[Line2D, Line2D], tuple[Line2D, Line2D]]]] = []

        for theta in range(envConfig.nTrueModels):
            preds: list[tuple[tuple[Line2D, Line2D], tuple[Line2D, Line2D], tuple[Line2D, Line2D]]] = []
            for k in range(envConfig.nModelFactors):
                thetaList = envConfig.unflattenTheta(theta)

                nomMppi = (
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=4, alpha=0.8)[0],
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=2, alpha=0.4)[0],
                )

                branchMppi = (
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=4, alpha=0.8)[0],
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=2, alpha=0.4)[0],
                )

                pidTraj = (
                    self.ax.plot(
                        [], color=self.pred_colors[k][thetaList[k] + 1], marker=None, linewidth=3, alpha=0.8, linestyle="-."
                    )[0],
                    self.ax.plot(
                        [], color=self.pred_colors[k][thetaList[k] + 1], marker=None, linewidth=2, alpha=0.4, linestyle="-."
                    )[0],
                )

                preds.append((nomMppi, branchMppi, pidTraj))

            self.lcs_pred.append(preds)

        # points and collision circles
        self.points: list[Any] = []
        self.circles: list[Circle] = []
        for i_agent in range(self.nAgents):
            px, py = self.getPos(0, i_agent)
            color = self.colors[i_agent % len(self.colors)]
            (pt,) = self.ax.plot([px], [py], marker="o", color=color, markersize=8, label=f"{contNames[i_agent]} ({i_agent + 1})")
            self.points.append(pt)
            circ = Circle((px, py), radius=self.envConfig.minDist / 2, fill=True, color=color, linestyle="--", alpha=0.3)
            self.circles.append(circ)
            self.ax.add_patch(circ)

        # crash marker (initially empty)
        maxCollMarkers = self.nAgents * envConfig.nTrueModels
        self.collMarkers = [
            self.ax.plot(
                [], [], marker="*", markersize=20, color="yellow", markeredgecolor="red", markeredgewidth=1, zorder=5, alpha=0.8
            )[0]
            for i in range(maxCollMarkers)
        ]

        bmin, bmax = self.envConfig.arenaMin, self.envConfig.arenaMax
        self.ax.set_xlim(xmin=bmin[self.axis[0]], xmax=bmax[self.axis[0]])  # type: ignore
        self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend()

        self.zoom_radius = 6

        # action plots
        self.actions_plot: list[tuple[Line2D, Line2D]] = []  # plot & vline
        for dim in range(self.envConfig.dim):
            self.actions_plot.append(
                (
                    self.axs_action_plot[dim].plot(
                        [],
                        [],
                        color=self.colors[envConfig.iMppi % len(self.colors)],
                        label=f"{contNames[envConfig.iMppi]} ({envConfig.iMppi + 1})",
                    )[0],
                    self.axs_action_plot[dim].axvline(x=0, color="green", linestyle="--", linewidth=1, alpha=0.5),
                )
            )

            self.axs_action_plot[dim].set_xlabel("Time")
            self.axs_action_plot[dim].set_ylabel("Action on " + ["x", "y"][dim])

        # status text
        self.status_text = self.ax_status.text(
            0.5, 1.0, "Running...", fontsize=14, ha="center", va="top", transform=self.ax_status.transAxes
        )

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

        title = "Model Beliefs & Fail Count"

        self.ax_failcount.text(0.5, 1, title, fontsize=12, ha="center", va="top", fontweight="bold")
        self.verif_text = self.ax_failcount.text(0.5, 0.1, "", fontsize=12, ha="center", va="bottom")
        self.ax_failcount.axis("off")

        # MPPI belief

        # joint belief
        joint_colors: list[list[str]] = []
        names: list[str] = []
        for theta in range(envConfig.nTrueModels):
            thetaList = envConfig.unflattenTheta(theta)
            joint_colors.append([self.pred_colors[k][thetaList[k] + 1] for k in range(envConfig.nModelFactors)])
            names.append("-".join(oppNames[k][thetaList[k]] for k in range(envConfig.nModelFactors)))

        self.joint_belief = self.createBeliefBar(self.ax_joint_belief, names, joint_colors, "Joint belief", None)

        # marginal belief
        self.marginal_belief = [
            self.createBeliefBar(
                ax, names, [[c] for c in colors[1:]], f"Marginal belief for {k}", colors[0], mppiConfig.minConfidence
            )
            for k, (ax, names, colors) in enumerate(zip(self.axs_marg_belief, oppNames, self.pred_colors))
        ]

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

        gs_zoom_buttons = gs_ui[6].subgridspec(2, 1, height_ratios=[1, 1])

        self.zoom_minus_button = Button(self.fig.add_subplot(gs_zoom_buttons[1]), "-")
        self.zoom_minus_button.on_clicked(self.onZoomMinus)

        self.zoom_plus_button = Button(self.fig.add_subplot(gs_zoom_buttons[0]), "+")
        self.zoom_plus_button.on_clicked(self.onZoomPlus)

        if defaultZoomAgent == -1:
            self.zoomAgent = 0
            self.zoomed = False
            zoomButtonText = "Zoom"
        else:
            self.zoomAgent = defaultZoomAgent
            self.zoomed = True
            zoomButtonText = "Full track"

        self.button_zoom = Button(self.fig.add_subplot(gs_ui[5]), zoomButtonText)

        ax_zoom_label = self.fig.add_subplot(gs_ui[7])
        ax_zoom_label.axis("off")
        self.zoom_agent_label = ax_zoom_label.text(
            0.5, 0.5, f"Focus: {self.zoomAgent + 1}", ha="center", va="center", fontsize=10
        )

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

    def createBeliefBar(
        self,
        ax: plt.Axes,  # type: ignore
        names: list[str],
        colors: list[list[str]],
        ylabel: str,
        nomColor: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> tuple[plt.BarContainer, list[Text]]:  # type: ignore
        "Draw a bar with the given arguments. Return the bar itself, and the list of texts above the little bars"
        belief_texts: list[Text] = []

        print(names, colors)

        nVals = len(names)

        x_pos = np.arange(nVals)

        belief_bar = ax.bar(x_pos, np.zeros(nVals), color="#3498db", edgecolor="black", alpha=0.8)

        # threshold line
        if threshold is not None:
            ax.axhline(
                y=threshold,
                color="#e74c3c",
                linestyle="--",
                linewidth=1.5,
            )

        # text for the initial values (initially empty)
        for i in range(nVals):
            belief_texts.append(ax.text(i, 0.02, "", ha="center", va="bottom", fontsize=8, fontweight="bold"))

        # ax.tick_params(axis="x", pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=9, rotation=45)

        # ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 1.05)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))

        ax.grid(axis="y", linestyle=":", alpha=0.4)

        if nomColor is not None:
            ax.text(0.6, 1.12, "Nominal color: ", fontsize=10, fontweight="bold", va="center", ha="right")
            ax.scatter(0.75, 1.12, color=nomColor, marker="s", s=300, edgecolor="black", clip_on=False)

        for i in range(nVals):
            # thetaList = envConfig.unflattenTheta(theta)
            # colors = [self.pred_colors[k][thetaList[k] + 1] for k in range(envConfig.nModelFactors)]

            x_pos = i + 0.1 * np.linspace(-len(colors[i]) + 1, len(colors[i]) - 1, len(colors[i]))

            ax.scatter(x_pos, [-0.12] * len(colors[i]), c=colors[i], marker="s", s=300, edgecolors="black")

        return belief_bar, belief_texts

    def renderBackground(self, display_raceline: list[bool]):
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
                if display_raceline is None or display_raceline[i]:
                    pts = self.envConfig.trackPoints[i * self.envConfig.nTrackSamples : (i + 1) * self.envConfig.nTrackSamples, :]
                    self.ax.plot(pts[:, 0], pts[:, 1], color="grey", linestyle="--", alpha=0.3)

        # draw obstacles
        for omin, omax in self.envConfig.obstacles.reshape(self.envConfig.nObstacles, 2, self.envConfig.dim):
            self.ax.add_patch(
                patches.Rectangle(
                    omin,
                    width=omax[0] - omin[0],
                    height=omax[1] - omin[1],
                    linewidth=3,
                    edgecolor="black",
                    facecolor="gray",
                    hatch="/",
                    fill=True,
                )
            )

        for center, radius in zip(
            self.envConfig.roundObsCenters.reshape(self.envConfig.nRoundObstacles, self.envConfig.dim),
            self.envConfig.roundObsRadius,
        ):
            self.ax.add_patch(
                patches.Circle(tuple(center), radius, linewidth=3, edgecolor="black", facecolor="gray", hatch="/", fill=True)
            )

    def coordIndex(self, agent: int, coord: int) -> int:
        return agent * self.envConfig.dim + coord

    def getPos(self, frame_index: int, agent: int) -> tuple[float, float]:
        if len(self.stateLog) == 0:
            return 0.0, 0.0
        idx = min(frame_index, len(self.stateLog) - 1)
        s = self.stateLog[idx].pos
        return (
            float(s[self.coordIndex(agent, self.axis[0])]),
            float(s[self.coordIndex(agent, self.axis[1])]),
        )

    def updateDisplay(self, i: int, forceZoom: bool = False) -> None:
        n = len(self.stateLog)
        if n == 0:
            return
        i = max(0, min(i, n - 1))

        self.updateStatus(i)

        # update trails
        if self.renderTrails:
            for idx, lc in enumerate(self.lcs):
                x_arr = [self.getPos(j, idx)[0] for j in range(i + 1)]
                y_arr = [self.getPos(j, idx)[1] for j in range(i + 1)]
                lc.set_segments([[[x_arr[j], y_arr[j]], [x_arr[j + 1], y_arr[j + 1]]] for j in range(i)])
                if i > 1:
                    lc.set_array(np.linspace(0, 1, i))

        collMarkers = iter(self.collMarkers)

        # update MPPI predictions
        mppiState = self.stateLog[i].mppiInfo

        if len(mppiState.preds) != self.envConfig.nTrueModels:
            print(f"WARNING: len(mppiState.preds) = {len(mppiState.preds)} is different from {self.envConfig.nTrueModels=}")
            return

        def apply_offset(coords: np.ndarray, side: int, amount: float) -> np.ndarray:
            "Side should be ie. 1 or -1 to shift the trajectory"
            if len(coords) < 2:
                return coords

            # Compute tangent direction
            diff = np.diff(coords, axis=0, append=[coords[-1] + (coords[-1] - coords[-2])])
            # Perpendicular (x, y) -> (-y, x)
            perp = np.stack([-diff[:, 1], diff[:, 0]], axis=1)
            # Normalize
            norm = np.linalg.norm(perp, axis=1, keepdims=True)
            perp = (perp / (norm + 1e-8)) * side * amount

            return coords + perp

        def set_data(
            lineStrong: Line2D,
            lineLight: Line2D,
            arr: np.ndarray | None,
            tOrigin: int,
            side: int = 0,
            amount: float = 0.04,
            color: Optional[str] = None,
        ) -> None:
            if arr is not None:
                ind = max(vHor - tOrigin, 0)

                arr1_offset = apply_offset(arr[: (ind + 1), [self.axis[0], self.axis[1]]], side, amount)
                arr2_offset = apply_offset(arr[ind:, [self.axis[0], self.axis[1]]], side, amount)

                # lineStrong.set_data(arr[:ind, self.axis[0]], arr[:ind, self.axis[1]])
                # lineLight.set_data(arr[max(ind - 1, 0) :, self.axis[0]], arr[max(ind - 1, 0) :, self.axis[1]])

                lineStrong.set_data(arr1_offset[:, 0], arr1_offset[:, 1])
                lineLight.set_data(arr2_offset[:, 0], arr2_offset[:, 1])

                if color is not None:
                    lineStrong.set_color(color)
                    lineLight.set_color(color)

            else:
                lineStrong.set_data([], [])
                lineLight.set_data([], [])

        for theta, (lTrajs, pred) in enumerate(zip(self.lcs_pred, mppiState.preds)):
            thetaTuple = self.envConfig.unflattenTheta(theta)

            curPos = self.stateLog[i].pos.reshape((self.envConfig.nAgents, self.envConfig.dim))
            fullPos = pred.fullPos.reshape((self.mppiConfig.nTimesteps, self.envConfig.nAgents, self.envConfig.dim))

            fullPos = np.concat([[curPos], fullPos])

            vHor = self.verifConfig.horizon

            initPredTheta = [int(round(t)) for t in pred.initPredTheta]
            predTheta = [int(round(t)) for t in pred.predTheta]
            branchTime = [int(round(b)) if pT == 0 else 0 for (b, pT) in zip(pred.branchTime, initPredTheta)]
            # from a rendering point of view, if we're already committed at beginning, then it's as if we committed at time t=0 (but for MPPI computations, it is conceptually different and in this case branchingTime=T)

            sides = np.arange(-self.envConfig.nModelFactors + 1, self.envConfig.nModelFactors, 2)

            # if we branch at time 0, only show the corresponding plot (otherwise, it might get confusing) (skip if we are already committed to a theta, which is different from the current theta)
            compatible = True
            for k in range(self.envConfig.nModelFactors):
                if initPredTheta[k] != 0 and initPredTheta[k] != thetaTuple[k] + 1:
                    compatible = False

            for k, (nomMppi, branchMppi, pid) in enumerate(lTrajs):
                set_data(*nomMppi, fullPos[: (branchTime[k] + 1), self.iMppi, :], 0, sides[k])

                if compatible:
                    color = self.pred_colors[k][predTheta[k]]

                    set_data(*branchMppi, fullPos[branchTime[k] :, self.iMppi, :], branchTime[k], sides[k], color=color)
                    set_data(*pid, fullPos[:, 1 - self.iMppi, :], 0, sides[k])

                    if pred.stopReason == EVENT_TYPE.EVT_COLLISION or pred.stopReason == EVENT_TYPE.EVT_OUTSIDE:
                        marker = next(collMarkers)
                        marker.set_data(
                            [fullPos[pred.stopTime, pred.stopAgent, self.axis[0]]],
                            [fullPos[pred.stopTime, pred.stopAgent, self.axis[1]]],
                        )
                        if (
                            pred.stopTime < self.verifConfig.horizon - 1
                        ):  # the prediction timescale is shifted by one (since it starts from the already actuated state)
                            marker.set_alpha(0.8)
                            marker.set_markersize(20)
                        else:
                            marker.set_alpha(0.4)
                            marker.set_markersize(10)
                else:
                    set_data(*branchMppi, None, 0)
                    set_data(*pid, None, 0)

        # hide remaining coll markers
        for marker in collMarkers:
            marker.set_data([], [])

        # update points and circles
        for idx, pt in enumerate(self.points):
            px, py = self.getPos(i, idx)
            pt.set_data([px], [py])
            self.circles[idx].center = (px, py)

        # update action plots
        for dim, ((plot, vline), ax) in enumerate(zip(self.actions_plot, self.axs_action_plot)):
            vline.set_xdata([i, i])
            if len(plot.get_xdata()) != len(self.stateLog):  # type: ignore
                plot.set_xdata(np.arange(len(self.stateLog)))
                plot.set_ydata(np.array([state.egoAction[dim] for state in self.stateLog]))

                ax.relim()
                ax.autoscale_view()

        # update zoom or full view
        if self.zoomed:
            cx, cy = self.getPos(i, self.zoomAgent)
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

        for iAgent in range(self.nAgents):
            vel, laps, gates = self.stateLog[i].speed, self.stateLog[i].nLaps, self.stateLog[i].currentGates

            speed = np.linalg.norm(vel[iAgent * self.envConfig.dim : (iAgent + 1) * self.envConfig.dim])

            self.agent_value_texts[iAgent].set_text(
                f"Lap {int(laps[iAgent])}/{self.envConfig.nWinLaps} Gate {int(gates[iAgent])}/{self.envConfig.nGates}\nSpeed {speed:.2f}"
            )

        # update MPPI belief
        mppiState = self.stateLog[i].mppiInfo

        if len(mppiState.preds) != self.envConfig.nTrueModels:
            print(f"WARNING: len(mppiState.preds) = {len(mppiState.preds)} is different from {self.envConfig.nTrueModels=}")
            return

        failCount, eps = mppiState.failCount, mppiState.epsilon

        self.verif_text.set_text(
            f"Fail: {sum(failCount) / self.verifConfig.N * 100:.2f}% (coll {failCount[0] / self.verifConfig.N * 100:.2f}%, out {failCount[1] / self.verifConfig.N * 100:.2f}%)\n"
            + (f"Failure rate: {eps:.5f} (partial {mppiState.epsilonPartial:.5f})\n" if i > 0 else "Failure rate: --\n")
            + (
                f"Use new plan: {'YES' if mppiState.useNewPlan else 'NO'} (loss: {mppiState.certifiedLoss:.5f})"
                if i > 0
                else "Use new plan: --\n"
            )
        )

        # update joint belief
        for bar, text, b_val in zip(self.joint_belief[0], self.joint_belief[1], mppiState.belief):
            bar.set_height(b_val)

            # color = "#2ecc71" if b_val >= self.mppiConfig.minConfidence else "#3498db"
            # bar.set_facecolor(color)

            text.set_text(f"{b_val:.2f}")
            text.set_y(b_val + 0.01)

        # update marginal belief
        for k in range(self.envConfig.nModelFactors):
            marg = self.envConfig.computeMarginal(mppiState.belief, k)

            for bar, text, b_val in zip(self.marginal_belief[k][0], self.marginal_belief[k][1], marg):
                bar.set_height(b_val)

                color = "#2ecc71" if b_val >= self.mppiConfig.minConfidence else "#3498db"
                bar.set_facecolor(color)

                text.set_text(f"{b_val:.2f}")
                text.set_y(b_val + 0.01)

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
        self.fig.canvas.draw_idle()
        if text.strip() == "":
            print("No filename provided; aborting GIF save")
            return
        self.saveGif(text.strip())

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

            axes_to_capture = (
                [self.ax, self.ax_status, self.ax_joint_belief, self.ax_failcount] + self.axs_marg_belief + self.axs_action_plot
            )
            bboxes = [a.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted()) for a in axes_to_capture]
            full_bbox = Bbox.union(bboxes).padded(0.5)

            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches=full_bbox, pad_inches=0)
            buf.seek(0)

            img = Image.open(buf)
            imgs.append(img.convert("RGB"))
            buf.close()

            # w, h = self.fig.canvas.get_width_height()
            # buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            # buf = buf.reshape((h, w, 3))
            # imgs.append(Image.fromarray(buf))

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

    def onZoomMinus(self, event: Any) -> None:
        self.zoomAgent = (self.zoomAgent - 1) % self.envConfig.nAgents
        self.zoom_agent_label.set_text(f"Focus: {self.zoomAgent + 1}")
        if not self.playing:
            self.updateDisplay(self.current_index)
            self.show()

    def onZoomPlus(self, event: Any) -> None:
        self.zoomAgent = (self.zoomAgent + 1) % self.envConfig.nAgents
        self.zoom_agent_label.set_text(f"Focus: {self.zoomAgent + 1}")
        if not self.playing:
            self.updateDisplay(self.current_index)
            self.show()
