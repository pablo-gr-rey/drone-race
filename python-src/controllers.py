import numpy as np
from utils import AddStateType, ConfigType, Controller, MPPIConfig, PIDConfig


class DummyController(Controller[ConfigType, AddStateType]):
    "Does not do anything"

    def __init__(self, config: ConfigType, name: str = "dummy"):
        super().__init__(name, config)

    def getControl(self, agent: int, pos: np.ndarray, vel: np.ndarray, addState: AddStateType) -> np.ndarray:
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

    def getControl(self, agent: int, pos: np.ndarray, vel: np.ndarray, addState: AddStateType) -> np.ndarray:
        if self.environment is None:
            raise ValueError("Environment has not been set for controller")

        dim = self.envConfig.dim
        base_pos = agent * dim
        pos = pos[base_pos : base_pos + dim]
        vel = vel[base_pos : base_pos + dim]

        error = self.environment.getTarget(agent, pos, vel, addState) - pos

        action = self.config.kp * error + self.config.kd * (-vel)

        for new_agent in range(self.envConfig.nAgents):
            if agent != new_agent:
                diff = pos[new_agent * dim : (new_agent + 1) * dim] - pos
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

    def stateCost(self, agent: int, pos: np.ndarray, vel: np.ndarray, addState: AddStateType, time: int) -> float:
        # so far, this is only dependant on the (projected) distance towards the opponents
        # todo: add cost for going outside the track

        cost = 0.0

        if self.environment is None:
            raise ValueError("Should specify environment in controller")

        pos_agent = pos[agent * self.envConfig.dim : (agent + 1) * self.envConfig.dim]

        for newAgent in range(self.envConfig.nAgents):
            if newAgent != agent:
                pos_other = pos[newAgent * self.envConfig.dim : (newAgent + 1) * self.envConfig.dim]
                dist = np.linalg.norm(pos_other - pos_agent)
                if dist < self.config.oppDistThresholdFactor * self.envConfig.minDist:
                    cost += self.config.oppDistWeight / (dist / self.envConfig.minDist) ** (self.config.oppDistPower)
                if dist < self.envConfig.minDist:
                    cost += self.config.collisionCost * 0.9**time

                if self.environment.closestBoundaryDist(newAgent, pos, vel, addState) < 0:
                    cost -= self.config.oppOutsideCost
            else:
                dist = self.environment.closestBoundaryDist(agent, pos, vel, addState)
                cost -= self.config.boundaryCost * dist
                dangerousDist = self.config.boundaryThresholdFactor * self.envConfig.minDist
                if dist < dangerousDist:
                    cost += self.config.boundaryCost * (dangerousDist - dist) / dangerousDist
                if dist < 0:
                    cost += (
                        self.config.outsideCost * 0.9**time
                    )  # decaying cost for being outside to keep some trajectories that go outside only at the end of the horizon
                    # print("careful!!")

        if self.environment.checkWinner(pos, vel, addState) == agent:
            cost -= self.config.winCost * 0.9**time

        return cost  # type: ignore # np.linalg.norm might return an array of floats

    def finalCost(self, agent: int, pos: np.ndarray, vel: np.ndarray, addState: AddStateType) -> float:
        cost = 0.0
        maxOppAdv = -np.inf

        if self.environment is None:
            raise ValueError("Should specify environment in controller")

        for newAgent in range(self.envConfig.nAgents):
            target = self.environment.getTarget(newAgent, pos, vel, addState)
            pos_other = pos[newAgent * self.envConfig.dim : (newAgent + 1) * self.envConfig.dim]
            diffToTarget = target - pos_other
            advance = self.environment.getAdvance(newAgent, pos, vel, addState)

            if newAgent == agent:
                speed = vel[newAgent * self.envConfig.dim : (newAgent + 1) * self.envConfig.dim]
                cost -= self.config.finalAdvWeight * advance + self.config.finalSpeedWeight * np.dot(
                    speed, diffToTarget / np.linalg.norm(diffToTarget)
                )
            else:
                maxOppAdv = max(maxOppAdv, advance)

        return cost + self.config.finalOppAdvWeight * maxOppAdv  # type: ignore # same as above

    def getControl(self, agent: int, pos: np.ndarray, vel: np.ndarray, addState: AddStateType) -> np.ndarray:
        conf = self.config

        if conf.opponentPredictors is None:
            raise ValueError("Should specify MPPIConfig.opponentPredictors")
        if self.environment is None:
            raise ValueError("Should specify environment in controller")

        totalCosts = np.zeros(conf.nSamples)
        noise = np.random.normal(0.0, conf.samplingNoise, (conf.nTimesteps, conf.nSamples, self.envConfig.dim))

        for sample in range(conf.nSamples):
            currentPos = pos.copy()
            currentVel = vel.copy()
            currentAddState = addState

            for time in range(conf.nTimesteps):
                action = [
                    self.nominalAction[time] + noise[time, sample, :]
                    if i == agent
                    else conf.opponentPredictors[i](i, currentPos, currentVel, currentAddState)
                    for i in range(self.envConfig.nAgents)
                ]

                currentPos, currentVel, currentAddState = self.environment.dynStep(
                    action, currentPos, currentVel, currentAddState
                )
                totalCosts[sample] += self.stateCost(agent, currentPos, currentVel, currentAddState, time)

            totalCosts[sample] += self.finalCost(agent, currentPos, currentVel, currentAddState)

        minCost = np.min(totalCosts)

        weights = np.exp(-(totalCosts - minCost) / conf.inv_temperature)

        nu = np.sum(weights)

        self.nominalAction += (noise * weights[None, :, None]).sum(axis=1) / nu

        finalAction = self.nominalAction[0]

        self.nominalAction[:-1] = self.nominalAction[1:]
        self.nominalAction[-1] = np.zeros(self.envConfig.dim)

        return finalAction
