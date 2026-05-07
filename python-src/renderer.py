import os
import time
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider, TextBox
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from environment import BaseEnvironment


class EnvironmentRenderer:
    """Helper that manages a non-blocking matplotlib window for live rendering.
    If you wish the window to stay open at the end of the loop, make sure to call renderer.finish() which will block the program.
    """

    def __init__(
        self,
        env: "BaseEnvironment",
        axis=(0, 1),
        interval: int = 30,
        autoplay: bool = True,
        frameSkipPlayback: int = 2,
        frameSkipWaiting: int = 1,
        defaultZoomAgent: int = 0,
    ):
        "interval: refresh rate. frameSkipWaiting: how many frames to skip if emitting states faster than we can display (use -1 to always display last frame)"
        self.env = env
        self.axis = axis
        self.interval = interval
        self.frameSkipPlayback = frameSkipPlayback
        self.frameSkipWaiting = frameSkipWaiting
        self.zoomAgent = defaultZoomAgent
        self.aspect_ratio = 1.5

        self.collision = False
        self.winner: Optional[int] = None
        self.outside: Optional[int] = None

        plt.ion()
        plt.show()

        self.fig, self.ax = plt.subplots(figsize=(5 * self.aspect_ratio, 6))
        self.fig.tight_layout(pad=0.5)
        plt.subplots_adjust(bottom=0.25)

        # draw static background using environment hook
        self.env.renderBackground(self.ax)

        # color maps and patch/marker colors
        self.cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "BuPu"]
        self.colors = ["blue", "red", "green", "purple", "orange", "gray", "brown", "pink"]

        self.nAgents = self.env.config.nAgents

        # line collections for trajectories
        self.lcs: list[LineCollection] = []

        for i_agent in range(self.nAgents):
            lc = LineCollection([], cmap=self.cmaps[i_agent % self.nAgents])
            lc.set_linewidth(4)
            lc.set_alpha(0.8)
            self.lcs.append(lc)
            self.ax.add_collection(lc)  # type: ignore

        # points and collision circles
        self.points: list[Any] = []
        self.circles: list[Circle] = []
        for i_agent in range(self.nAgents):
            px, py = self.getPos(0, i_agent)
            color = self.colors[i_agent % len(self.colors)]
            (pt,) = self.ax.plot(
                [px], [py], marker="o", color=color, markersize=8, label=f"{self.env.controllers[i_agent].name} ({i_agent + 1})"
            )
            self.points.append(pt)
            circ = Circle((px, py), radius=self.env.config.minDist / 2, fill=True, color=color, linestyle="--", alpha=0.3)
            self.circles.append(circ)
            self.ax.add_patch(circ)

        bmin, bmax = self.env.getBounds()
        self.ax.set_xlim(xmin=bmin[self.axis[0]], xmax=bmax[self.axis[0]])  # type: ignore
        self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore
        self.ax.set_aspect('equal', adjustable='box')
        # self.ax.axis("equal")
        self.ax.legend()

        self.zoom_radius = 6

        # status text
        if self.env.checkCollision():
            text, color = "Collision", "red"
        elif (winner := self.env.checkWinner()) is not None:
            text, color = f"Winner: {winner + 1}", "green"
        elif (outside := self.env.checkOutside()) is not None:
            text, color = f"{outside + 1} outside", "red"
        else:
            text, color = "Running", "blue"

        self.status_text = self.fig.text(0.5, 0.92, text, fontsize=12, ha="center", color=color)

        # UI: slider, play, save GIF, textbox and zoom
        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
        max_idx = max(1, len(self.env.stateLog) - 1)
        self.slider = Slider(ax_slider, "Time", 0, max_idx, valinit=0, valstep=1)

        ax_button = plt.axes((0.8, 0.025, 0.1, 0.04))
        self.button = Button(ax_button, "Pause")

        ax_save = plt.axes((0.35, 0.025, 0.12, 0.04))
        self.save_button = Button(ax_save, "Save GIF")
        self.save_button.set_active(False)

        ax_text = plt.axes((0.15, 0.025, 0.2, 0.04))
        self.text_box = TextBox(ax_text, "GIF name", initial="")
        self.text_box.ax.set_visible(False)  # type: ignore

        ax_zoom_button = plt.axes((0.675, 0.025, 0.12, 0.04))
        self.button_zoom = Button(ax_zoom_button, "Full track")

        ax_zoom_minus = plt.axes((0.5, 0.018, 0.025, 0.025))
        self.zoom_minus_button = Button(ax_zoom_minus, "-")
        self.zoom_minus_button.on_clicked(self.onZoomMinus)

        ax_zoom_plus = plt.axes((0.5, 0.047, 0.025, 0.025))
        self.zoom_plus_button = Button(ax_zoom_plus, "+")
        self.zoom_plus_button.on_clicked(self.onZoomPlus)

        self.zoom_agent_label = self.fig.text(0.6, 0.025, f"Focus: {self.zoomAgent + 1}", ha="center", va="bottom", fontsize=12)

        # internal state
        self.playing = False
        self.zoomed = True
        self.slider_is_updating = False
        self.current_index = 0

        self.isFinished = False

        # timer for playback (non-blocking)
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.timerTick)

        # connect callbacks
        self.slider.on_changed(self.sliderChanged)
        self.button.on_clicked(self.tooglePlay)
        self.save_button.on_clicked(self.askSavePath)
        self.button_zoom.on_clicked(self.toogleZoom)
        self.fig.canvas.mpl_connect("key_press_event", self.onKeyPress)

        self.lastRenderTime = time.perf_counter()

        # initial render
        self.updateDisplay(0)

        if autoplay:
            self.tooglePlay(None, True)

    def coordIndex(self, agent: int, coord: int) -> int:
        return (agent * self.env.config.dim + coord) * 2

    def getPos(self, frame_index: int, agent: int) -> tuple[float, float]:
        if len(self.env.stateLog) == 0:
            return 0.0, 0.0
        idx = min(frame_index, len(self.env.stateLog) - 1)
        s = self.env.stateLog[idx]
        return (
            float(s[self.coordIndex(agent, self.axis[0])]),
            float(s[self.coordIndex(agent, self.axis[1])]),
        )

    def updateDisplay(self, i: int) -> None:
        n = len(self.env.stateLog)
        if n == 0:
            return
        i = max(0, min(i, n - 1))

        for idx, lc in enumerate(self.lcs):
            x_arr = [self.getPos(j, idx)[0] for j in range(i + 1)]
            y_arr = [self.getPos(j, idx)[1] for j in range(i + 1)]
            lc.set_segments([[[x_arr[j], y_arr[j]], [x_arr[j + 1], y_arr[j + 1]]] for j in range(i)])
            if i > 1:
                lc.set_array(np.linspace(0, 1, i))

        # update points and circles
        for idx, pt in enumerate(self.points):
            px, py = self.getPos(i, idx)
            pt.set_data([px], [py])
            self.circles[idx].center = (px, py)

        # update zoom or full view
        if self.zoomed:
            cx, cy = self.getPos(i, self.zoomAgent)
            self.ax.set_xlim(cx - self.zoom_radius * self.aspect_ratio, cx + self.zoom_radius * self.aspect_ratio)
            self.ax.set_ylim(cy - self.zoom_radius, cy + self.zoom_radius)
            self.ax.apply_aspect()
        else:
            bmin, bmax = self.env.getBounds()
            self.ax.set_xlim(bmin[self.axis[0]], bmax[self.axis[0]])  # type: ignore
            self.ax.set_ylim(bmin[self.axis[1]], bmax[self.axis[1]])  # type: ignore
        self.ax.set_aspect('equal', adjustable='box')

        # update slider value without triggering callback
        self.slider_is_updating = True
        self.slider.set_val(i)
        self.slider_is_updating = False

        # update status text
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

        self.lastRenderTime = time.perf_counter()

    def onNewState(self) -> None:
        # Called by the environment when a new frame is available
        n = len(self.env.stateLog)
        if n == 0:
            return
        max_idx = max(0, n - 1)
        # update slider range
        self.slider.valmax = max_idx  # type: ignore
        self.slider.ax.set_xlim(0, max_idx)  # type: ignore

        # if playing and "waiting" on new states, go to new state
        if (
            self.playing
            and not self.isFinished
            and time.perf_counter() - self.lastRenderTime > self.interval / 1000
            and (self.frameSkipWaiting == -1 or len(self.env.stateLog) % self.frameSkipWaiting == 0)
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
        if not self.playing and len(self.env.stateLog) % 10 == 0:
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
            self.current_index = len(self.env.stateLog) - 1

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
        self.current_index = min(len(self.env.stateLog), i)
        self.updateDisplay(i)
        self.show()

    def timerTick(self) -> None:
        n = len(self.env.stateLog)
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
        self.button.label.set_text("Play" if not self.playing else "Pause")  # type: ignore

        if self.playing:
            # if at end, restart
            if self.current_index >= max(0, len(self.env.stateLog) - 1):
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
            self.current_index = min(len(self.env.stateLog) - 1, self.current_index + 1)
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

        save_path = os.path.join("gifs", name)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        imgs: list[Image.Image] = []
        n = len(self.env.stateLog)
        for i in tqdm(range(n), desc=f"Capturing frames for {name}", unit="frame"):
            self.updateDisplay(i)
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            buf = buf.reshape((h, w, 3))
            imgs.append(Image.fromarray(buf))

        try:
            print(f"Saving animation to {save_path}...")
            imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=20, loop=0)
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
        self.updateDisplay(int(self.slider.val))
        self.fig.canvas.draw_idle()

    def onZoomMinus(self, event: Any) -> None:
        self.zoomAgent = (self.zoomAgent - 1) % self.env.config.nAgents
        self.zoom_agent_label.set_text(f"Focus: {self.zoomAgent + 1}")
        if not self.playing:
            self.updateDisplay(self.current_index)
            self.show()

    def onZoomPlus(self, event: Any) -> None:
        self.zoomAgent = (self.zoomAgent + 1) % self.env.config.nAgents
        self.zoom_agent_label.set_text(f"Focus: {self.zoomAgent + 1}")
        if not self.playing:
            self.updateDisplay(self.current_index)
            self.show()
