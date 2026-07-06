from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from utils import (
    EVENT_TYPE,
    BaseEnvironmentConfig,
    BaseControllerConfig,
    BaseSimState,
    DroneRaceEnvironmentConfig,
    DroneRaceSimState,
    FullStateInfo,
    HiddenObsEnvironmentConfig,
    HiddenObsSimState,
    MPPIConfig,
    MPPIStateInfo,
    PRMPPIConfig,
    PRMPPIStateInfo,
)


class ControllerRenderer[EnvConfigT: BaseEnvironmentConfig, ContConfigT: BaseControllerConfig](ABC):
    def __init__(
        self,
        envConfig: EnvConfigT,
        contConfig: ContConfigT,
        oppNames: list[list[str]],
        ax: plt.Axes,  # type: ignore
        fig: plt.Figure,  # type: ignore
        gs: plt.SubplotSpec,  # type: ignore
        axis: tuple[int, ...],
        **kwargs: Any,
    ):
        self.config = contConfig
        self.envConfig = envConfig
        self.oppNames = oppNames

        self.ax = ax
        self.fig = fig
        self.base_gs = gs

        self.axis = axis

        self.init(**kwargs)

    @abstractmethod
    def init(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def update(self, state: FullStateInfo) -> None: ...

    @abstractmethod
    def getCapturedAxes(self) -> list[plt.Axes]: ...  # type: ignore

    # general utility methods

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
            x_pos = i + 0.1 * np.linspace(-len(colors[i]) + 1, len(colors[i]) - 1, len(colors[i]))

            ax.scatter(x_pos, [-0.12] * len(colors[i]), c=colors[i], marker="s", s=300, edgecolors="black")

        return belief_bar, belief_texts

    def apply_offset(self, coords: np.ndarray, side: int, amount: float = 0.04) -> np.ndarray:
        "Shift the coords array to side*amount, in the direction perpendicular to its tangent"
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


class BaseMPPIRenderer[EnvConfigT: BaseEnvironmentConfig, SimStateT: BaseSimState](ControllerRenderer[EnvConfigT, MPPIConfig]):
    def baseInit(self, nAdditionalTrajs: int = 0, **kwargs: Any) -> None:
        "nAdditionalTrajs should be the number of lines (other than MPPI's) we should create, for example 1 if there's another agent/moving stuff and 0 otherwise"
        self.severalModels = self.envConfig.nModelFactors > 1
        self.nAddTrajs = nAdditionalTrajs

        gs_mppi = self.base_gs.subgridspec(
            2 + int(self.severalModels), 1, height_ratios=[1, 3, 3] if self.severalModels else [1, 3], hspace=0.1
        )
        self.ax_failcount = self.fig.add_subplot(gs_mppi[0])

        gs_marginal = gs_mppi[1].subgridspec(1, self.envConfig.nModelFactors)
        self.axs_marg_belief = [self.fig.add_subplot(g) for g in gs_marginal]

        self.ax_joint_belief = self.fig.add_subplot(gs_mppi[2]) if self.severalModels else None

        # should have shape nModelFactors * (nModelSizes[k]+1)
        self.pred_colors = [["brown", "green", "orange"], ["yellow", "cyan", "purple"]]

        # line collections for MPPI predictions
        # length: nTrueModels * nModelFactors, with items being (nomMppi, branchMppi, <nAdditionalTrajs> item). each item is ((nominalStrong, nominalLight), (branchedStrong, branchedLight)). branched color should change based on the real value
        self.lcs_pred: list[list[tuple[tuple[Line2D, Line2D], ...]]] = []

        for theta in range(self.envConfig.nTrueModels):
            preds: list[tuple[tuple[Line2D, Line2D], ...]] = []
            for k in range(self.envConfig.nModelFactors):
                thetaList = self.envConfig.unflattenTheta(theta)

                nomMppi = (
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=4, alpha=0.8)[0],
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=2, alpha=0.4)[0],
                )

                branchMppi = (
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=4, alpha=0.8)[0],
                    self.ax.plot([], color=self.pred_colors[k][0], marker=None, linewidth=2, alpha=0.4)[0],
                )

                pidTrajs = tuple(
                    (
                        self.ax.plot(
                            [], color=self.pred_colors[k][thetaList[k] + 1], marker=None, linewidth=3, alpha=0.8, linestyle="-."
                        )[0],
                        self.ax.plot(
                            [], color=self.pred_colors[k][thetaList[k] + 1], marker=None, linewidth=2, alpha=0.4, linestyle="-."
                        )[0],
                    )
                    for i in range(self.nAddTrajs)
                )

                preds.append((nomMppi, branchMppi) + pidTrajs)

            self.lcs_pred.append(preds)

        # crash marker (initially empty)
        maxCollMarkers = (1 + self.nAddTrajs) * self.envConfig.nTrueModels
        self.collMarkers = [
            self.ax.plot(
                [], [], marker="*", markersize=20, color="yellow", markeredgecolor="red", markeredgewidth=1, zorder=5, alpha=0.8
            )[0]
            for i in range(maxCollMarkers)
        ]

        # MPPI failcount status
        self.ax_failcount.text(
            0.5,
            0.95,
            "Controller Type: Branching-MPPI (" + ["without", "with"][self.config.useSplines] + " splines)",
            fontsize=15,
            ha="center",
            va="top",
            fontweight="bold",
        )
        self.verif_text = self.ax_failcount.text(0.5, 0.1, "", fontsize=12, ha="center", va="bottom")
        self.ax_failcount.axis("off")

        # MPPI belief

        # joint belief
        joint_colors: list[list[str]] = []
        names: list[str] = []
        for theta in range(self.envConfig.nTrueModels):
            thetaList = self.envConfig.unflattenTheta(theta)
            joint_colors.append([self.pred_colors[k][thetaList[k] + 1] for k in range(self.envConfig.nModelFactors)])
            names.append("-".join(self.oppNames[k][thetaList[k]] for k in range(self.envConfig.nModelFactors)))

        self.joint_belief = (
            self.createBeliefBar(self.ax_joint_belief, names, joint_colors, "Joint belief", None)
            if self.ax_joint_belief is not None
            else None
        )

        # marginal belief
        self.marginal_belief = [
            self.createBeliefBar(
                ax, names, [[c] for c in colors[1:]], f"Marginal belief for {k}", colors[0], self.config.minConfidence
            )
            for k, (ax, names, colors) in enumerate(zip(self.axs_marg_belief, self.oppNames, self.pred_colors))
        ]

    def set_data(
        self,
        lineStrong: Line2D,
        lineLight: Line2D,
        arr: np.ndarray | None,
        effVerifHorizon: int,  # should be verifHorizon - tOrigin
        side: int = 0,
        amount: float = 0.04,
        color: Optional[str] = None,
    ) -> None:
        if arr is not None:
            ind = max(effVerifHorizon, 0)

            arr1_offset = self.apply_offset(arr[: (ind + 1), [self.axis[0], self.axis[1]]], side, amount)
            arr2_offset = self.apply_offset(arr[ind:, [self.axis[0], self.axis[1]]], side, amount)

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

    def baseUpdate(self, state: FullStateInfo[SimStateT, MPPIStateInfo], curPos: np.ndarray, fullPoss: list[np.ndarray]) -> None:
        "fullPoss should have length len(state.contInfo.preds), and contain the full poss per state (1 + nAdditionalTrajs), DIM coords. MPPI is always considered to be the first coords, other ones are the additional trajs"
        collMarkers = iter(self.collMarkers)

        mppiState = state.contInfo

        if len(state.contInfo.preds) != self.envConfig.nTrueModels:
            print(f"WARNING: len(mppiState.preds) = {len(mppiState.preds)} is different from {self.envConfig.nTrueModels=}")
            return

        for theta, (lTrajs, pred, fullPos) in enumerate(zip(self.lcs_pred, mppiState.preds, fullPoss)):
            thetaTuple = self.envConfig.unflattenTheta(theta)

            fullPos = np.concat([[curPos], fullPos])

            vHor = self.config.verifHorizon

            initPredTheta = [int(round(t)) for t in pred.initPredTheta]
            predTheta = [int(round(t)) for t in pred.predTheta]
            branchTime = [
                int(round(b)) if pT != 0 else self.config.nTimesteps + 1 for (b, pT) in zip(pred.branchTime, pred.predTheta)
            ]
            # from a rendering point of view, if we're already committed at beginning, then it's as if we committed at time t=0; if we never commit, then it's as if we committed at time T+1 (but it is stored as 0)

            sides = np.arange(-self.envConfig.nModelFactors + 1, self.envConfig.nModelFactors, 2)

            # if we branch at time 0, only show the corresponding plot (otherwise, it might get confusing) (skip if we are already committed to a theta, which is different from the current theta)
            compatible = True
            for k in range(self.envConfig.nModelFactors):
                if initPredTheta[k] != 0 and initPredTheta[k] != thetaTuple[k] + 1:
                    compatible = False

            for k, (nomMppi, branchMppi, *addTrajs) in enumerate(lTrajs):
                self.set_data(*nomMppi, fullPos[: (branchTime[k] + 1), 0, :], vHor, sides[k])

                if compatible:
                    color = self.pred_colors[k][predTheta[k]]

                    self.set_data(
                        *branchMppi,
                        fullPos[branchTime[k] :, 0, :],
                        vHor - branchTime[k],
                        sides[k],
                        color=color,
                    )

                    for i, addTraj in enumerate(addTrajs):
                        self.set_data(*addTraj, fullPos[:, i + 1, :], vHor, sides[k])

                    if pred.stopReason == EVENT_TYPE.EVT_OUTSIDE:
                        marker = next(collMarkers)
                        marker.set_data(
                            [fullPos[pred.stopTime + 1, 0, self.axis[0]]],
                            [fullPos[pred.stopTime + 1, 0, self.axis[1]]],
                        )
                        if (
                            pred.stopTime < self.config.verifHorizon - 1
                        ):  # the prediction timescale is shifted by one (since it starts from the already actuated state)
                            marker.set_alpha(0.8)
                            marker.set_markersize(20)
                        else:
                            marker.set_alpha(0.4)
                            marker.set_markersize(10)
                else:
                    self.set_data(*branchMppi, None, 0)
                    for addTraj in addTrajs:
                        self.set_data(*addTraj, None, 0)

        # hide remaining coll markers
        for marker in collMarkers:
            marker.set_data([], [])

        # Update fail count
        failCount, eps = mppiState.failCount, mppiState.epsilon

        self.verif_text.set_text(
            f"Fail: {failCount / self.config.nVerifSamples * 100:.3f}%\n"
            + (f"Failure rate: {eps:.5f} (partial {mppiState.epsilonPartial:.5f})\n")  # if i > 0 else "Failure rate: --\n")
            + (
                f"Use new plan: {'YES' if mppiState.useNewPlan else 'NO'} (loss: {mppiState.certifiedLoss:.5f})"
                # if i > 0
                # else "Use new plan: --\n"
            )
        )

        # update joint belief
        if self.joint_belief is not None:
            for bar, text, b_val in zip(self.joint_belief[0], self.joint_belief[1], mppiState.belief):
                bar.set_height(b_val)

                text.set_text(f"{b_val:.2f}")
                text.set_y(b_val + 0.01)

        # update marginal belief
        for k in range(self.envConfig.nModelFactors):
            marg = self.envConfig.computeMarginal(mppiState.belief, k)

            for bar, text, b_val in zip(self.marginal_belief[k][0], self.marginal_belief[k][1], marg):
                bar.set_height(b_val)

                color = "#2ecc71" if b_val >= self.config.minConfidence else "#3498db"
                bar.set_facecolor(color)

                text.set_text(f"{b_val:.2f}")
                text.set_y(b_val + 0.01)

    def getCapturedAxes(self) -> list[plt.Axes]:  # type: ignore
        ans = [self.ax_failcount] + self.axs_marg_belief
        if self.ax_joint_belief is not None:
            ans.append(self.ax_joint_belief)
        return ans


class MPPIDroneRaceRenderer(BaseMPPIRenderer[DroneRaceEnvironmentConfig, DroneRaceSimState]):
    def init(self, **kwargs: Any):
        super().baseInit(1, **kwargs)  # one additional trajectory for PID

    def update(self, state: FullStateInfo[DroneRaceSimState, MPPIStateInfo]) -> None:
        assert isinstance(state.contInfo, MPPIStateInfo), f"got {type(state.contInfo)} controller info, expected MPPIStateInfo"

        curPos = state.state.pos.copy().reshape((self.envConfig.nAgents, self.envConfig.dim))

        fullPoss = [
            pred.fullPos.copy().reshape((self.config.nTimesteps, self.envConfig.nAgents, self.envConfig.dim))
            for pred in state.contInfo.preds
        ]

        # we need to make sure that mppi is at spot 0
        iMppi = self.envConfig.iMppi
        if iMppi != 0:
            curPos[[0, iMppi], :] = curPos[[iMppi, 0], :]
            for i in range(len(state.contInfo.preds)):
                fullPoss[i][:, [0, iMppi], :] = fullPoss[i][:, [iMppi, 0], :]

        super().baseUpdate(state, curPos, fullPoss)


class MPPIHiddenObsRenderer(BaseMPPIRenderer[HiddenObsEnvironmentConfig, HiddenObsSimState]):
    def init(self, **kwargs: Any):
        super().baseInit(0, **kwargs)  # no additional trajectory

    def update(self, state: FullStateInfo[HiddenObsSimState, MPPIStateInfo]) -> None:
        assert isinstance(state.contInfo, MPPIStateInfo), f"got {type(state.contInfo)} controller info, expected MPPIStateInfo"

        curPos = state.state.pos[None, :]
        fullPoss = [pred.fullPos.reshape((self.config.nTimesteps, 1, self.envConfig.dim)) for pred in state.contInfo.preds]

        super().baseUpdate(state, curPos, fullPoss)


class BasePRMPPIRenderer[EnvConfigT: BaseEnvironmentConfig, SimStateT: BaseSimState](
    ControllerRenderer[EnvConfigT, PRMPPIConfig]
):
    def baseInit(self, nAdditionalTrajs: int = 0, **kwargs: Any) -> None:
        self.min_show_confidence = 0.01  # if confidence is less than this amount, do not show the trajectories
        self.severalModels = self.envConfig.nModelFactors > 1
        self.nAddTrajs = nAdditionalTrajs

        gs_mppi = self.base_gs.subgridspec(
            2 + int(self.severalModels), 1, height_ratios=[1, 3, 3] if self.severalModels else [1, 3], hspace=0.1
        )

        self.ax_failcount = self.fig.add_subplot(gs_mppi[0])

        gs_marginal = gs_mppi[1].subgridspec(1, self.envConfig.nModelFactors)
        self.axs_marg_belief = [self.fig.add_subplot(g) for g in gs_marginal]

        self.ax_joint_belief = self.fig.add_subplot(gs_mppi[2]) if self.severalModels else None

        # should have shape nModelFactors * (nModelSizes[k])
        self.pred_colors = [["green", "orange"], ["cyan", "purple"]]
        self.nom_color = "violet"
        self.rob_color = "brown"

        # line collections for MPPI predictions
        # length: nTrueModels * nTrajs * nModelFactors (each list of size nModelFactors should represent one global theta by parallel lines, one line color represent that specific parameter value)
        # since in this case the nominal action is not reactive to the environment, PRMPPI's trajectory is the same for all branches
        self.lcs_pred: list[list[list[Line2D]]] = []
        self.nom_pred = self.ax.plot([], color=self.nom_color, marker=None, linewidth=4, alpha=0.8)[0]
        self.rob_pred = self.ax.plot([], color=self.rob_color, marker=None, linewidth=4, alpha=0.8)[0]

        for theta in range(self.envConfig.nTrueModels):
            a_preds: list[list[Line2D]] = []
            for iAdd in range(self.nAddTrajs):
                preds: list[Line2D] = []
                for k in range(self.envConfig.nModelFactors):
                    thetaList = self.envConfig.unflattenTheta(theta)

                    preds.append(
                        self.ax.plot(
                            [], color=self.pred_colors[k][thetaList[k]], marker=None, linewidth=3, alpha=0.8, linestyle="-."
                        )[0]
                    )
                a_preds.append(preds)

            self.lcs_pred.append(a_preds)

        # crash marker (initially empty)
        maxCollMarkers = (1 + self.nAddTrajs) * self.envConfig.nTrueModels
        self.collMarkers = [
            self.ax.plot(
                [], [], marker="*", markersize=20, color="yellow", markeredgecolor="red", markeredgewidth=1, zorder=5, alpha=0.8
            )[0]
            for i in range(maxCollMarkers)
        ]

        # MPPI failcount status
        self.ax_failcount.text(
            0.5,
            0.95,
            "Controller Type: Parameter-robust-MPPI",
            fontsize=15,
            ha="center",
            va="top",
            fontweight="bold",
            transform=self.ax_failcount.transAxes,
        )
        self.verif_text = self.ax_failcount.text(
            0.5, 0.7, "", fontsize=12, ha="center", va="center", transform=self.ax_failcount.transAxes
        )

        self.reset_text = self.ax_failcount.text(
            0.5,
            0.5,
            "Nominal plan was reset",
            fontsize=12,
            ha="center",
            va="center",
            transform=self.ax_failcount.transAxes,
            color="red",
            fontweight="bold",
            visible=False,
        )

        self.ax_failcount.text(
            0.6, 0.3, "Nominal plan:", fontsize=12, ha="right", va="center", transform=self.ax_failcount.transAxes
        )
        self.ax_failcount.text(
            0.6, 0.1, "Robust plan:", fontsize=12, ha="right", va="center", transform=self.ax_failcount.transAxes
        )
        self.ax_failcount.scatter(
            [0.7, 0.7], [0.3, 0.1], marker="s", s=200, edgecolors="black", color=[self.nom_color, self.rob_color]
        )

        self.ax_failcount.set_xlim(0, 1)
        self.ax_failcount.set_ylim(0, 1)
        self.ax_failcount.axis("off")

        # MPPI belief

        # joint belief
        joint_colors: list[list[str]] = []
        names: list[str] = []
        for theta in range(self.envConfig.nTrueModels):
            thetaList = self.envConfig.unflattenTheta(theta)
            joint_colors.append([self.pred_colors[k][thetaList[k]] for k in range(self.envConfig.nModelFactors)])
            names.append("-".join(self.oppNames[k][thetaList[k]] for k in range(self.envConfig.nModelFactors)))

        self.joint_belief = (
            self.createBeliefBar(self.ax_joint_belief, names, joint_colors, "Joint belief", None)
            if self.ax_joint_belief is not None
            else None
        )

        # marginal belief
        self.marginal_belief = [
            self.createBeliefBar(ax, names, [[c] for c in colors], f"Marginal belief for {k}", None, None)
            for k, (ax, names, colors) in enumerate(zip(self.axs_marg_belief, self.oppNames, self.pred_colors))
        ]

    def baseUpdate(self, state: FullStateInfo[SimStateT, PRMPPIStateInfo], fullPoss: list[np.ndarray]):
        collMarkers = iter(self.collMarkers)

        mppiState = state.contInfo
        assert isinstance(mppiState, PRMPPIStateInfo), (
            f"Expected state.contInfo to be of type PRMPPIStateInfo, but received {type(mppiState)}"
        )

        if len(mppiState.preds) != self.envConfig.nTrueModels + 1:
            print(f"WARNING: len(mppiState.preds) = {len(mppiState.preds)} is different from {(self.envConfig.nTrueModels+1)=}")
            return

        # update MPPI predictions

        for theta, (lTrajs, pred, fullPos) in enumerate(zip(self.lcs_pred, mppiState.preds[:-1], fullPoss[:-1])):
            thetaTuple = self.envConfig.unflattenTheta(theta)

            # curPos = state.pos.reshape((self.envConfig.nAgents, self.envConfig.dim))
            # fullPos = pred.fullPos.reshape((self.config.nTimesteps, self.envConfig.nAgents, self.envConfig.dim))
            # fullPosConc = np.concat([[mppiState.prevPos.reshape((self.envConfig.nAgents, self.envConfig.dim))], fullPos])

            sides = np.arange(-self.envConfig.nModelFactors + 1, self.envConfig.nModelFactors, 2)

            if mppiState.belief[theta] > self.min_show_confidence:
                # TODO: set_data is useless here (we only should keep one trajectory)
                # self.set_data(*self.nom_pred, fullPos[:, self.envConfig.iMppi, :], vHor, sides[k])
                # self.nom_pred.set_data(
                #     fullPos[:, self.envConfig.iMppi, self.axis[0]], fullPos[:, self.envConfig.iMppi, self.axis[1]]
                # )
                self.nom_pred.set_data(fullPos[:, 0, self.axis[0]], fullPos[:, 0, self.axis[1]])

                for iAddTraj, trajs in enumerate(lTrajs):
                    for k, pid in enumerate(trajs):
                        color = self.pred_colors[k][thetaTuple[k]]

                        # self.set_data(*pid, fullPos[:, 1 - self.envConfig.iMppi, :], vHor, sides[k])
                        arr_offset = self.apply_offset(fullPos[:, iAddTraj + 1, self.axis], sides[k])
                        pid.set_data(arr_offset[:, 0], arr_offset[:, 1])

                        print(f"drawing for {theta=} {iAddTraj=} {k=}")

                        if pred.stopReason == EVENT_TYPE.EVT_OUTSIDE:
                            marker = next(collMarkers)
                            marker.set_data(
                                [fullPos[pred.stopTime, 0, self.axis[0]]],
                                [fullPos[pred.stopTime, 0, self.axis[1]]],
                            )

                            marker.set_alpha(0.8)
                            marker.set_markersize(20)
            else:
                for trajs in lTrajs:
                    for pid in trajs:
                        # self.set_data(*pid, None, 0)
                        pid.set_data([[], []])

        # show robust nominal
        # fullPos = mppiState.preds[-1].fullPos.reshape((self.config.nTimesteps, 1 + self.nAddTrajs, self.envConfig.dim))
        fullPos = fullPoss[-1]
        self.rob_pred.set_data(fullPos[:, 0, self.axis[0]], fullPos[:, 0, self.axis[1]])

        # hide remaining coll markers
        for marker in collMarkers:
            marker.set_data([], [])

        # update joint belief
        if self.joint_belief is not None:
            for bar, text, b_val in zip(self.joint_belief[0], self.joint_belief[1], mppiState.belief):
                bar.set_height(b_val)

                text.set_text(f"{b_val:.2f}")
                text.set_y(b_val + 0.01)

        # update marginal belief
        for k in range(self.envConfig.nModelFactors):
            marg = self.envConfig.computeMarginal(mppiState.belief, k)

            for bar, text, b_val in zip(self.marginal_belief[k][0], self.marginal_belief[k][1], marg):
                bar.set_height(b_val)

                # color = "#2ecc71" if b_val >= self.config.minConfidence else "#3498db"
                color = "#3498db"
                bar.set_facecolor(color)

                text.set_text(f"{b_val:.2f}")
                text.set_y(b_val + 0.01)

        # update text status

        self.verif_text.set_text(f"Plan used: {('Robust', 'Nominal')[mppiState.useNomPlan]}")
        self.reset_text.set_visible(mppiState.resetNom)

    def getCapturedAxes(self) -> list[plt.Axes]:  # type: ignore
        ans = [self.ax_failcount] + self.axs_marg_belief
        if self.ax_joint_belief is not None:
            ans.append(self.ax_joint_belief)
        return ans


class PRMPPIDroneRaceRenderer(BasePRMPPIRenderer[DroneRaceEnvironmentConfig, DroneRaceSimState]):
    def init(self, **kwargs: Any):
        super().baseInit(1, **kwargs)  # one additional trajectory for PID

    def update(self, state: FullStateInfo[DroneRaceSimState, PRMPPIStateInfo]) -> None:
        assert isinstance(state.contInfo, PRMPPIStateInfo), (
            f"got {type(state.contInfo)} controller info, expected PRMPPIStateInfo"
        )

        fullPoss = [
            pred.fullPos.copy().reshape((self.config.nTimesteps, self.envConfig.nAgents, self.envConfig.dim))
            for pred in state.contInfo.preds
        ]

        # we need to make sure that mppi is at spot 0
        iMppi = self.envConfig.iMppi
        if iMppi != 0:
            for i in range(len(state.contInfo.preds)):
                fullPoss[i][:, [0, iMppi], :] = fullPoss[i][:, [iMppi, 0], :]

        super().baseUpdate(state, fullPoss)


class PRMPPIHiddenObsRenderer(BasePRMPPIRenderer[HiddenObsEnvironmentConfig, HiddenObsSimState]):
    def init(self, **kwargs: Any):
        super().baseInit(0, **kwargs)  # no additional trajectory

    def update(self, state: FullStateInfo[HiddenObsSimState, PRMPPIStateInfo]) -> None:
        assert isinstance(state.contInfo, PRMPPIStateInfo), (
            f"got {type(state.contInfo)} controller info, expected PRMPPIStateInfo"
        )

        fullPoss = [pred.fullPos.reshape((self.config.nTimesteps, 1, self.envConfig.dim)) for pred in state.contInfo.preds]

        super().baseUpdate(state, fullPoss)
