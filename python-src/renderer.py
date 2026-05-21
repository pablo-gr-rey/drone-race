import io
import os
import time
from typing import Any, Optional

from matplotlib import patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider, TextBox
from PIL import Image
from tqdm import tqdm
from utils import GateEnvironmentConfig, MPPIConfig


class EnvironmentRenderer:
    """Helper that manages a non-blocking matplotlib window for live rendering.
    If you wish the window to stay open at the end of the dynamics loop, make sure to call renderer.finish() which will block the program.
    """

    def __init__(
        self,
        envConfig: GateEnvironmentConfig,
        contNames: list[str],
        oppNames: list[str],
        mppiConfig: MPPIConfig,
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
        self.axis = axis
        self.interval = interval
        self.frameSkipPlayback = frameSkipPlayback
        self.frameSkipWaiting = frameSkipWaiting
        self.oppNames = oppNames
        self.mppiConfig = mppiConfig
        self.renderTrails = renderTrails

        self.posLog: list[np.ndarray] = []
        self.velLog: list[np.ndarray] = []

        self.sLog: list[np.ndarray] = []
        self.nLapsLog: list[np.ndarray] = []
        self.currentGatesLog: list[np.ndarray] = []

        self.beliefLog: list[tuple[int, np.ndarray, list[tuple[int, int, np.ndarray]]]] = []

        self.collision = False
        self.winner: Optional[int] = None
        self.outside: Optional[int] = None

        if isinstance(display_raceline, bool):
            display_raceline = [display_raceline] * envConfig.nRaceLines

        plt.ion()
        plt.show()

        self.fig = plt.figure(figsize=(12, 10))

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

        self.ax = self.fig.add_subplot(gs[0, 0])  # track ax

        gs_status = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.1)

        self.ax_status = self.fig.add_subplot(gs_status[0])  # for text status
        self.ax_status.axis("off")

        self.ax_belief = self.fig.add_subplot(gs_status[1])  # for belief

        # slider, empty space, play/pause, save gif, gif name, zoom, +/-, focus
        gs_ui = gs[1, :].subgridspec(1, 8, width_ratios=[7, 1, 1, 1, 1, 1, 0.4, 1], wspace=0.1)

        # draw static background using environment hook
        # self.env.renderBackground(self.ax, display_raceline)
        self.renderBackground(display_raceline)

        # color maps and patch/marker colors
        self.cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "BuPu"]
        self.colors = ["blue", "red", "green", "purple", "orange", "gray", "brown", "pink"]
        self.pred_colors = ["brown", "green", "orange"]  # for nominal + models

        self.nAgents = self.envConfig.nAgents

        # line collections for trajectories
        self.lcs: list[LineCollection] = []

        for i_agent in range(self.nAgents):
            lc = LineCollection([], cmap=self.cmaps[i_agent % self.nAgents], linewidth=4, alpha=0.8)
            self.lcs.append(lc)
            self.ax.add_collection(lc)  # type: ignore

        # line collections for MPPI predictions
        # length: nModels, with items being (nominalMPPItraj, branchedTraj, PIDtraj)
        self.lcs_pred: list[tuple[Line2D, Line2D, Line2D]] = []
        for iPred in range(len(oppNames)):
            self.lcs_pred.append(
                (
                    self.ax.plot([], color=self.pred_colors[0], marker=None, linewidth=4, alpha=0.8)[0],
                    self.ax.plot([], color=self.pred_colors[iPred + 1], marker=None, linewidth=4, alpha=0.8)[0],
                    self.ax.plot([], color=self.pred_colors[iPred + 1], marker=None, linewidth=4, alpha=0.8, linestyle="-.")[0],
                )
            )

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

        bmin, bmax = self.envConfig.arenaMin, self.envConfig.arenaMax
        self.ax.set_xlim(xmin=bmin[self.axis[0]], xmax=bmax[self.axis[0]])  # type: ignore
        self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend()

        self.zoom_radius = 6

        # status text

        self.status_text = self.ax_status.text(
            0.5, 1.0, "Running...", fontsize=14, ha="center", va="top", transform=self.ax_status.transAxes
        )

        self.agent_value_texts: list[Text] = []
        y_positions = 0.8 - np.arange(self.nAgents) * 0.15

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

        # MPPI belief
        if self.oppNames:
            x_pos = np.arange(len(oppNames))

            self.belief_bars = self.ax_belief.bar(x_pos, np.zeros(len(oppNames)), color="#3498db", edgecolor="black", alpha=0.8)

            # threshold line
            self.ax_belief.axhline(y=self.mppiConfig.minConfidence, color="#e74c3c", linestyle="--", linewidth=1.5)

            # texte for the initial values (initially empty)
            self.belief_texts = []
            for i in range(len(oppNames)):
                t = self.ax_belief.text(
                    i, 0.02, "", ha="center", va="bottom", fontsize=8, fontweight="bold", color=self.pred_colors[1 + i]
                )
                self.belief_texts.append(t)

            self.ax_belief.set_xticks(x_pos)
            self.ax_belief.set_xticklabels(oppNames, fontsize=9, rotation=45)

            # ax_belief.set_xlim(0, 1.05)
            self.ax_belief.set_ylim(-0.5, 1.05)
            self.ax_belief.set_ylabel("Probability", fontsize=8)
            self.ax_belief.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

            self.ax_belief.set_title("Model Beliefs", fontsize=10, fontweight="bold")
            self.ax_belief.spines["top"].set_visible(False)
            self.ax_belief.spines["right"].set_visible(False)
            self.ax_belief.grid(axis="y", linestyle=":", alpha=0.4)

            # we need to draw to get back the labels
            self.fig.canvas.draw_idle()
            x_labels = self.ax_belief.get_xticklabels()
            for label, color in zip(x_labels, self.pred_colors[1:]):
                label.set_color(color)

        # UI: slider, play, save GIF, textbox and zoom
        max_idx = max(1, len(self.posLog) - 1)
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
                    self.ax.plot(pts[:, 0], pts[:, 1], "g--", alpha=0.5)

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

    def coordIndex(self, agent: int, coord: int) -> int:
        return agent * self.envConfig.dim + coord

    def getPos(self, frame_index: int, agent: int) -> tuple[float, float]:
        if len(self.posLog) == 0:
            return 0.0, 0.0
        idx = min(frame_index, len(self.posLog) - 1)
        s = self.posLog[idx]
        return (
            float(s[self.coordIndex(agent, self.axis[0])]),
            float(s[self.coordIndex(agent, self.axis[1])]),
        )

    def updateDisplay(self, i: int, forceZoom: bool = False) -> None:
        n = len(self.posLog)
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

        # update MPPI predictions
        if self.beliefLog and self.oppNames:
            iMppi, belief, preds = self.beliefLog[i]
            for theta, ((lc_nom, lc_branch, lc_opp), (branchingTime, predTheta, fullPos)) in enumerate(zip(self.lcs_pred, preds)):
                fullPos = fullPos.reshape((self.mppiConfig.nTimesteps, self.envConfig.nAgents, self.envConfig.dim))

                if predTheta == -1:
                    branchingTime = self.mppiConfig.nTimesteps

                lc_nom.set_data(fullPos[:branchingTime, iMppi, 0], fullPos[:branchingTime, iMppi, 1])
                if branchingTime != 0 or predTheta == theta:
                    # if we branch at time 0, only show the corresponding plot (otherwise, it might get confuding)
                    lc_branch.set_data(fullPos[branchingTime:, iMppi, 0], fullPos[branchingTime:, iMppi, 1])
                    lc_opp.set_data(fullPos[:, 1 - iMppi, 0], fullPos[:, 1 - iMppi, 1])
                else:
                    lc_branch.set_data([], [])
                    lc_opp.set_data([], [])

        # update points and circles
        for idx, pt in enumerate(self.points):
            px, py = self.getPos(i, idx)
            pt.set_data([px], [py])
            self.circles[idx].center = (px, py)

        # update zoom or full view
        # TODO: do not update ax lims if they did not change (probably way faster)
        if self.zoomed:
            cx, cy = self.getPos(i, self.zoomAgent)
            self.ax.set_xlim(cx - self.zoom_radius, cx + self.zoom_radius)
            self.ax.set_ylim(cy - self.zoom_radius, cy + self.zoom_radius)
            self.ax.apply_aspect()
        elif forceZoom:  # if the last frame was also full track, no need to change the lims (that avoids re-drawing everything)
            bmin, bmax = self.envConfig.arenaMin, self.envConfig.arenaMax
            self.ax.set_xlim(bmin[self.axis[0]], bmax[self.axis[0]])  # type: ignore
            self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore
        self.ax.set_aspect("equal", adjustable="box")

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
            vel, laps, gates = self.velLog[i], self.nLapsLog[i], self.currentGatesLog[i]

            speed = np.linalg.norm(vel[iAgent * self.envConfig.dim : (iAgent + 1) * self.envConfig.dim])

            self.agent_value_texts[iAgent].set_text(
                f"Lap {int(laps[iAgent])}/{self.envConfig.nWinLaps} Gate {int(gates[iAgent])}/{self.envConfig.nGates}\nSpeed {speed:.2f}"
            )

        # update MPPI belief
        if self.beliefLog and self.oppNames:
            iAgent, belief, preds = self.beliefLog[i]
            for i, (bar, b_val) in enumerate(zip(self.belief_bars, belief)):
                bar.set_height(b_val)

                color = "#2ecc71" if b_val >= self.mppiConfig.minConfidence else "#3498db"
                bar.set_facecolor(color)

                # Mise à jour du texte de valeur
                self.belief_texts[i].set_text(f"{b_val:.2f}")
                self.belief_texts[i].set_y(b_val + 0.01)

    def onNewState(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        currentS: np.ndarray,
        nLaps: np.ndarray,
        currentGates: np.ndarray,
        belief: Optional[tuple[int, np.ndarray, list[tuple[int, int, np.ndarray]]]],
        pendingState: bool = False,
    ) -> None:
        "If pendingState is True, it means that there are other states waiting in the queue (ie. they are computed faster than they are rendered); in this case, only 1 frame out of frameSkipWaiting will be shown"
        self.posLog.append(pos)
        self.velLog.append(vel)
        self.sLog.append(currentS)
        self.nLapsLog.append(nLaps)
        self.currentGatesLog.append(currentGates)

        if belief is not None:
            self.beliefLog.append(belief)

        # Called by the environment when a new frame is available
        n = len(self.posLog)
        max_idx = max(n - 1, 1)
        # update slider range
        self.slider.valmax = max_idx  # type: ignore
        self.slider.ax.set_xlim(0, max_idx)  # type: ignore

        # if playing and "waiting" on new states, go to new state
        if (
            self.playing
            and not self.isFinished
            and time.perf_counter() - self.lastRenderTime > self.interval / 1000
            and (self.frameSkipWaiting == -1 or len(self.posLog) % self.frameSkipWaiting == 0)
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
        if not self.playing and len(self.posLog) % 10 == 0:
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
            self.current_index = len(self.posLog) - 1

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
        self.current_index = min(len(self.posLog), i)
        self.updateDisplay(i)
        self.show()

    def timerTick(self) -> None:
        n = len(self.posLog)
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
            if self.current_index >= max(0, len(self.posLog) - 1):
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
            self.current_index = min(len(self.posLog) - 1, self.current_index + 1)
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

        save_path = os.path.join("..", "gifs", name)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        duration = 40

        imgs: list[Image.Image] = []
        n = len(self.posLog)
        for i in tqdm(range(n), desc=f"Capturing frames for {name}", unit="frame"):
            self.updateDisplay(i)
            self.fig.canvas.draw()

            axes_to_capture = [self.ax, self.ax_status, self.ax_belief]
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
