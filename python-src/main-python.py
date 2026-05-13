import numpy as np
from controllers import DummyController, MPPIConfig, MPPIController, PIDController
from environment import SimpleEnvironment, TrackEnvironment
from tqdm import tqdm
from utils import PIDConfig, SimpleEnvironmentConfig, TrackEnvironmentConfig


def roundTrack(s: float) -> np.ndarray:
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)]) * 10


def lissajous(s):
    R = 10 + 2 * np.sin(8 * np.pi * s)
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)]) * R


def main():
    init_state = [0, 0.5, -4, 0, 0, 0, -3, 0]

    config = SimpleEnvironmentConfig(
        init_state=init_state,
        posNoiseLevel=0.01,
        speedNoiseLevel=0.01,
        actionNoiseLevel=0.05,
        maxSpeed=np.array([1.5, 0.5]),
        maxAccel=np.array([1.5, 0.5]),
        minDist=0.3,
    )

    mppiconfig = MPPIConfig(
        nSamples=50,
        finalAdvWeight=100,
        finalSpeedWeight=50,
        oppDistWeight=50,
        finalOppAdvWeight=5,
        samplingNoise=2,
        nTimesteps=10,
        outsideCost=0,
        boundaryCost=0.1,
        oppOutsideCost=0,
    )
    mppicont = MPPIController(config, mppiconfig)

    pidconfig = PIDConfig(kp=10, kd=5, repulsionFactor=20)

    dumcont = DummyController(config)
    pidcont = PIDController(config, pidconfig)

    e = SimpleEnvironment(config, [mppicont, pidcont])

    mppiconfig.opponentPredictors = [pidcont.getControl, pidcont.getControl]

    done = False
    while not done and e.nSteps < 1000:
        collide, outside, win = e.step()
        done = collide or outside is not None or win is not None

    e.render()


def mainTrack():
    # startS = [0, 0.1]
    startS = [0.1, 0]

    init_state = [
        roundTrack(startS[0])[0],
        0,
        roundTrack(startS[0])[1],
        0,
        roundTrack(startS[1])[0],
        0,
        roundTrack(startS[1])[1],
        0,
    ]

    config = TrackEnvironmentConfig(
        # centerline=roundTrack,
        init_state=init_state,
        add_state=(np.array(startS), np.array([0.0, 0.0])),
        centerline=lissajous,
        posNoiseLevel=0,
        speedNoiseLevel=0.0,
        actionNoiseLevel=0.05,
        trackWidth=3,
        maxAccel=np.array([5, 5]),
        # maxSpeed=np.array([2, 1.5]),
        maxSpeed=np.array([1.5, 2]),
        # maxSpeed=np.array([2, 2]),
        targetDistance=0.02,
        nWinLaps=1,
        minDist=0.6,
        dt=0.1,
    )

    mppiconfig = MPPIConfig(
        nSamples=15,
        nTimesteps=15,
        samplingNoise=2,
        finalAdvWeight=100,
        finalSpeedWeight=50,
        # oppDistWeight=20,
        oppDistWeight=0,
        oppDistPower=2,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=200,
        boundaryCost=0.01,
        boundaryThresholdFactor=1,
        oppOutsideCost=1000,
        outsideCost=1000,
        collisionCost=1000,
        winCost=1000,
    )

    mppiconfig2 = MPPIConfig(
        nSamples=15,
        nTimesteps=15,
        samplingNoise=2,
        finalAdvWeight=100,
        finalSpeedWeight=50,
        oppDistWeight=50,
        # oppDistWeight=0,
        oppDistPower=2,
        oppDistThresholdFactor=2,
        finalOppAdvWeight=100,
        boundaryCost=0.01,
        boundaryThresholdFactor=1,
        oppOutsideCost=1000,
        outsideCost=1000,
        collisionCost=1000,
        winCost=1000,
    )

    pidconfig = PIDConfig(kp=10, kd=5, repulsionFactor=20)
    cautiousconfig = PIDConfig(kp=10, kd=5, repulsionFactor=20)

    mppicont = MPPIController(config, mppiconfig)
    mppicont2 = MPPIController(config, mppiconfig2)

    dumcont = DummyController(config)
    cautiouscont = PIDController(config, cautiousconfig)
    pidcont = PIDController(config, pidconfig)

    e = TrackEnvironment(
        config,
        [mppicont, pidcont],
        # [mppicont, mppicont2],
        # [pidcont, pidcont],
        live_render=True,
    )

    pidcont.setEnvironment(e)
    cautiouscont.setEnvironment(e)

    mppiconfig.opponentPredictors = [pidcont.getControl, pidcont.getControl]
    mppiconfig2.opponentPredictors = [pidcont.getControl, pidcont.getControl]
    # mppiconfig.opponentPredictors = [cautiouscont.getControl, cautiouscont.getControl]

    done = False
    with tqdm(desc="Agent 2 in front", total=100) as pbar:
        lastVal = 0
        collide, outside, win = None, None, None
        while not done and e.nSteps < 1000:
            collide, outside, win = e.step()
            done = collide or outside is not None or win is not None
            # print(np.max(e.lastS), e.lastS)
            pbar.update(int(np.max(e.addState[0] + e.addState[1]) / config.nWinLaps * 100 - lastVal))
            lastVal += int(np.max(e.addState[0] + e.addState[1]) / config.nWinLaps * 100 - lastVal)
            pbar.set_description(f"Agent {np.argmax(e.addState[0] + e.addState[1]) + 1} in front")

        if collide:
            pbar.set_description("Collision!")
        elif outside is not None:
            pbar.set_description(f"Agent {outside + 1} is outside")
        elif win is not None:
            pbar.set_description(f"Agent {win + 1} is the winner")
        pbar.update(100 - lastVal)

    print("final pos", e.pos, "final vel", e.vel, "final additional state", e.addState)
    if e.live_render:
        e._renderer.finish()  # type: ignore
    else:
        e.render()


if __name__ == "__main__":
    # main()
    mainTrack()
