import numpy as np
from utils import AddStateType, ConfigType, Controller, MPPIConfig, PIDConfig


class DummyController(Controller[ConfigType, AddStateType]):
    "Does not do anything"

    def __init__(self, config: ConfigType, name: str = "dummy"):
        super().__init__(name, config)

    def getControl(self, agent: int, state: np.ndarray, addState: AddStateType) -> np.ndarray:
        return np.zeros(self.envConfig.dim)


class PIDController(Controller[ConfigType, AddStateType]):
    "PID controller with optional repulsion (factor / dist^power) (default target is [0,0,0])"

    def __init__(
        self,
        envConfig: ConfigType,
        config: PIDConfig,
        name: str = "PID",
    ):
        """
        kp, kd: PID parameters
        """
        super().__init__(name, envConfig)

        self.config = config

    def getControl(self, agent: int, state: np.ndarray, addState: AddStateType) -> np.ndarray:
        if self.environment is None:
            raise ValueError("Environment has not been set for controller")

        dim = self.envConfig.dim

        base = agent * dim * 2
        pos = state[base : base + dim * 2 : 2]
        vel = state[base + 1 : base + dim * 2 : 2]

        error = self.environment.getTarget(agent, state, addState) - pos

        # if agent == 1:
        #     print(f"pos: {pos}, target: {self.environment.getTarget(agent, state, addState)}, current s: {addState[0][1]}")

        action = self.config.kp * error + self.config.kd * (-vel)

        for new_agent in range(self.envConfig.nAgents):
            if agent != new_agent:
                diff = state[new_agent * dim * 2 : (new_agent + 1) * dim * 2 : 2] - pos
                # print(f"difference: norm {np.linalg.norm(diff)} diff {diff}")
                if np.linalg.norm(diff) < self.config.repulsionDistFactor * self.envConfig.minDist:
                    action += -self.config.repulsionFactor * diff / np.linalg.norm(diff) ** (self.config.repulsionPower + 1)
                    # print(f"hi? added {self.repulsionFactor * diff / np.linalg.norm(diff) ** (self.repulsionPower + 1)}")

        return action


class MPPIController(Controller[ConfigType, AddStateType]):
    def __init__(self, envConfig: ConfigType, config: MPPIConfig, name="MPPI"):
        super().__init__(name, envConfig)

        self.config = config

        self.nominalAction = np.zeros((self.config.nTimesteps, self.envConfig.dim))

    def stateCost(self, agent: int, state: np.ndarray, addState: AddStateType, time: int) -> float:
        # so far, this is only dependant on the (projected) distance towards the opponents
        # todo: add cost for going outside the track

        cost = 0.0

        if self.environment is None:
            raise ValueError("Should specify environment in controller")

        pos = state[agent * self.envConfig.dim * 2 : (agent + 1) * self.envConfig.dim * 2 : 2]

        for newAgent in range(self.envConfig.nAgents):
            if newAgent != agent:
                dist = np.linalg.norm(
                    state[newAgent * self.envConfig.dim * 2 : (newAgent + 1) * self.envConfig.dim * 2 : 2] - pos
                )
                if dist < self.config.oppDistThresholdFactor * self.envConfig.minDist:
                    cost += self.config.oppDistWeight / (dist / self.envConfig.minDist) ** (self.config.oppDistPower)
                if dist < self.envConfig.minDist:
                    cost += self.config.collisionCost * 0.9**time

                if self.environment.closestBoundaryDist(newAgent, state, addState) < 0:
                    cost -= self.config.oppOutsideCost
            else:
                dist = self.environment.closestBoundaryDist(agent, state, addState)
                cost -= self.config.boundaryCost * dist
                dangerousDist = self.config.boundaryThresholdFactor * self.envConfig.minDist
                if dist < dangerousDist:
                    cost += self.config.boundaryCost * (dangerousDist - dist) / dangerousDist
                if dist < 0:
                    cost += (
                        self.config.outsideCost * 0.9**time
                    )  # decaying cost for being outside to keep some trajectories that go outside only at the end of the horizon
                    # print("careful!!")

        if self.environment.checkWinner(state, addState) == agent:
            cost -= self.config.winCost * 0.9**time

        return cost  # type: ignore # np.linalg.norm might return an array of floats

    def finalCost(self, agent: int, state: np.ndarray, addState: AddStateType) -> float:
        cost = 0.0
        maxOppAdv = -np.inf

        if self.environment is None:
            raise ValueError("Should specify environment in controller")

        for newAgent in range(self.envConfig.nAgents):
            diffToTarget = (
                self.environment.getTarget(newAgent, state, addState)
                - state[newAgent * self.envConfig.dim * 2 : (newAgent + 1) * self.envConfig.dim * 2 : 2]
            )
            advance = self.environment.getAdvance(newAgent, state, addState)

            if newAgent == agent:
                speed = state[newAgent * self.envConfig.dim * 2 + 1 : (newAgent + 1) * self.envConfig.dim * 2 : 2]
                cost -= self.config.finalAdvWeight * advance + self.config.finalSpeedWeight * np.dot(
                    speed, diffToTarget / np.linalg.norm(diffToTarget)
                )
            else:
                maxOppAdv = max(maxOppAdv, advance)

        return cost + self.config.finalOppAdvWeight * maxOppAdv  # type: ignore # same as above

    def getControl(self, agent: int, state: np.ndarray, addState: AddStateType) -> np.ndarray:
        conf = self.config

        if conf.opponentPredictors is None:
            raise ValueError("Should specify MPPIConfig.opponentPredictors")
        if self.environment is None:
            raise ValueError("Should specify environment in controller")

        totalCosts = np.zeros(conf.nSamples)
        noise = np.random.normal(0.0, conf.samplingNoise, (conf.nTimesteps, conf.nSamples, self.envConfig.dim))

        for sample in range(conf.nSamples):
            currentState = state.copy()
            currentAddState = addState

            for time in range(conf.nTimesteps):
                action = [
                    self.nominalAction[time] + noise[time, sample, :]
                    if i == agent
                    else conf.opponentPredictors[i](i, currentState, currentAddState)
                    for i in range(self.envConfig.nAgents)
                ]

                currentState, currentAddState = self.environment.dynStep(action, currentState, currentAddState)
                totalCosts[sample] += self.stateCost(agent, currentState, currentAddState, time)

            totalCosts[sample] += self.finalCost(agent, currentState, currentAddState)

        minCost = np.min(totalCosts)

        weights = np.exp(-(totalCosts - minCost) / conf.inv_temperature)

        nu = np.sum(weights)

        self.nominalAction += (noise * weights[None, :, None]).sum(axis=1) / nu

        finalAction = self.nominalAction[0]

        self.nominalAction[:-1] = self.nominalAction[1:]
        self.nominalAction[-1] = np.zeros(self.envConfig.dim)

        return finalAction
