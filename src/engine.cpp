#include "engine.h"
#include "dynamics.h"
#include "state.h"
#include "protocol.h"

#include <zmq.h>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <variant>
#include <type_traits>

SimulationEngine::SimulationEngine(
    const EnvironmentConfig& config,
    const std::vector<ControllerSpec>& specs)
    : envConfig(config), rng(42)
{
    envConfig.recompute();

    hasCollision = false;
    isWinner = -1;
    isOutside = -1;

    phys_state = config.initState;
    currentS = config.initS;
    nLaps = config.initLaps;

    for (int i = 0; i < envConfig.nAgents; i++)
    {
        std::unique_ptr<Controller> ctrl = makeController(specs[i]);
        controllerNames.push_back(ctrl->name);
        controllers.push_back(std::move(ctrl));
    }
}

std::unique_ptr<Controller> SimulationEngine::makeController(const ControllerSpec& sp)
{
    std::unique_ptr<Controller> ctrl;

    std::visit([&](auto&& contConfig)
        {
            using T = std::decay_t<decltype(contConfig)>;

            if constexpr (std::is_same_v<T, DummyConfig>)
                ctrl = std::make_unique<DummyController>(envConfig);
            else if constexpr (std::is_same_v<T, PIDConfig>)
                ctrl = std::make_unique<PIDController>(envConfig, contConfig);
            else if constexpr (std::is_same_v<T, MPPIConfig>)
            {
                ctrl = std::make_unique<MPPIController>(envConfig, contConfig);
                // OpponentModelType oppModel = OpponentModelType::Dummy; // default
                // PIDConfig oppPid{}; // meaningful only for PID opponent

                // std::visit([&](auto&& opp)
                //     {
                //         using O = std::decay_t<decltype(opp)>;

                //         if constexpr (std::is_same_v<O, DummyConfig>)
                //             oppModel = OpponentModelType::Dummy;
                //         else if constexpr (std::is_same_v<O, PIDConfig>)
                //         {
                //             oppModel = OpponentModelType::PID;
                //             oppPid = opp;
                //         }
                //     }, contConfig.opponent);

                // ctrl = std::make_unique<MPPIController>(envConfig, contConfig, oppModel, oppPid);
            }
        }, sp.config);

    if (!ctrl)
        throw std::runtime_error("Unknown/invalid controller config for '" + sp.name + "'");

    ctrl->name = sp.name;
    ctrl->engine = this;
    return ctrl;
}

void SimulationEngine::sendState(zmq::socket_t& sock, int step)
{
    if (envConfig.sendStates)
    {
        Writer writer;

        writer.pushInt32(MSG_STATE);

        writer.pushInt32(step);
        writer.pushFloatArray(phys_state);
        writer.pushFloatArray(currentS);
        writer.pushFloatArray(nLaps);

        sock.send(zmq::buffer(writer.data));
    }
}

void SimulationEngine::sendEvent(zmq::socket_t& sock, EventType type, int info)
{
    Writer writer;

    writer.pushInt32(MSG_EVENT);
    writer.pushInt32(type);
    writer.pushInt32(info);

    sock.send(zmq::buffer(writer.data));
}

void SimulationEngine::sendDone(zmq::socket_t& sock)
{
    Writer writer;

    writer.pushInt32(MSG_DONE);

    sock.send(zmq::buffer(writer.data));
}

void SimulationEngine::dynStep(const std::vector<float>& actions)
{
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // std::memcpy(newPhys, phys, cfg.physDim * sizeof(float));
    // std::memcpy(newS, S, cfg.nAgents * sizeof(float));
    // std::memcpy(newLaps, laps, cfg.nAgents * sizeof(float));

    // clamp + noise actions

    std::vector<float> act = actions;

    for (int a = 0; a < envConfig.nAgents; a++)
        for (int d = 0; d < envConfig.dim; d++)
        {
            float& v = act[a * envConfig.dim + d];
            v = std::clamp(v, -envConfig.maxAccel[a], envConfig.maxAccel[a]);
            v += nd(rng) * envConfig.actionNoiseLevel;
        }

    for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    {
        // integrate position
        for (int d = 0; d < envConfig.dim; d++)
            phys_state[iAgent * envConfig.dim * 2 + d * 2] += envConfig.dt * phys_state[iAgent * envConfig.dim * 2 + d * 2 + 1];
        // setPos(newPhys, a, d, envConfig.dim,
        //     getPos(newPhys, a, d, envConfig.dim)
        //     + envConfig.dt * getVel(newPhys, a, d, envConfig.dim));

        // integrate velocity
        float sqSpeedNorm = 0.f;
        for (int d = 0; d < envConfig.dim; d++)
        {
            float& speed = phys_state[iAgent * envConfig.dim * 2 + d * 2 + 1];
            speed += envConfig.dt * act[iAgent * envConfig.dim + d];
            sqSpeedNorm += speed * speed;
        }
        // setVel(newPhys, a, d, envConfig.dim,
        //     getVel(newPhys, a, d, envConfig.dim)
        //     + envConfig.dt * act[a * envConfig.dim + d]);

        // cap speed
        if (sqSpeedNorm > envConfig.maxSpeed[iAgent] * envConfig.maxSpeed[iAgent])
        {
            float sc = envConfig.maxSpeed[iAgent] / std::sqrt(sqSpeedNorm);
            for (int d = 0; d < envConfig.dim; d++)
                phys_state[iAgent * envConfig.dim * 2 + d * 2 + 1] *= sc;
            // setVel(newPhys, iAgent, d, envConfig.dim,
            //     getVel(newPhys, iAgent, d, envConfig.dim) * sc);
        }

        // noise
        for (int d = 0; d < envConfig.dim; d++)
        {
            phys_state[iAgent * envConfig.dim * 2 + d * 2] += nd(rng) * envConfig.posNoiseLevel;
            phys_state[iAgent * envConfig.dim * 2 + d * 2 + 1] += nd(rng) * envConfig.speedNoiseLevel;
            // setPos(newPhys, iAgent, d, envConfig.dim,
            //     getPos(newPhys, iAgent, d, envConfig.dim) + nd(rng) * envConfig.posNoiseLevel);
            // setVel(newPhys, iAgent, d, envConfig.dim,
            //     getVel(newPhys, iAgent, d, envConfig.dim) + nd(rng) * envConfig.speedNoiseLevel);
        }
    }

    // update S and laps
    for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    {
        // float pos[MAX_DIM];
        // for (int d = 0; d < envConfig.dim; d++)
        //     // pos[d] = getPos(newPhys, a, d, envConfig.dim);
        //     pos[d] = phys_state[iAgent * envConfig.dim * 2 + d * 2];

        float s = cpuProjectOnTrack(envConfig.trackPoints, envConfig.nTrackSamples, envConfig.dim, phys_state.begin() + 2 * iAgent * envConfig.dim, 2).first;

        if (s > currentS[iAgent] + 0.5f)
            nLaps[iAgent] -= 1.0f;
        if (s < currentS[iAgent] - 0.5f)
            nLaps[iAgent] += 1.0f;

        currentS[iAgent] = s;
    }
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    sendState(sock, 0);

    int step;

    // ── simulation loop ──────────────────────────────────────────────
    for (step = 1; step <= maxSteps; step++)
    {
        // compute actions
        std::vector<float> actions(envConfig.actionDim, 0.0f);
        for (int i = 0; i < envConfig.nAgents; i++)
            controllers[i]->getControl(i, phys_state.data(), currentS.data(),
                nLaps.data(),
                actions.data() + i * envConfig.dim);

        // step dynamics
        std::vector<float> np(envConfig.physDim), ns(envConfig.nAgents), nl(envConfig.nAgents);

        dynStep(actions);

        sendState(sock, step);

        int best = 0;
        for (int k = 1; k < envConfig.nAgents; k++)
            if (currentS[k] + nLaps[k] > currentS[best] + currentS[k])
                best = k;

        // printf("Current state at step %d: agent %d in front\n", step, best + 1);
        // for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        // {
        //     printf("\tAgent %d: position ", iAgent + 1);
        //     for (int d = 0; d < envConfig.dim; d++)
        //         printf("%.2f ", phys_state[iAgent * envConfig.dim * 2 + d * 2]);
        //     printf("\tspeed: ");
        //     for (int d = 0; d < envConfig.dim; d++)
        //         printf("%.2f ", phys_state[iAgent * envConfig.dim * 2 + d * 2 + 1]);
        //     printf("\tcurrentS: %.2f\tnLaps: %d\n", currentS[iAgent], (int) nLaps[iAgent]);
        // }

        // termination checks
        hasCollision = checkCollision();
        isOutside = checkOutside();
        isWinner = checkWinner();

        if (hasCollision || isOutside >= 0 || isWinner >= 0)
        {
            if (isOutside >= 0)
                sendEvent(sock, EVT_OUTSIDE, isOutside);
            else if (isWinner >= 0)
                sendEvent(sock, EVT_WINNER, isWinner);
            else
                sendEvent(sock, EVT_COLLISION, -1);

            break;
        }

        if (step == maxSteps)
            sendEvent(sock, EVT_TRUNCATED, -1);
    }

    sendDone(sock);

    std::cout << "Simulation finished. " << envConfig.nAgents << " agents, " << envConfig.nTrackSamples << " track samples after " << step << " steps.\nStop reason: ";
    if (hasCollision)
        std::cout << "Collision\n";
    else if (isOutside >= 0)
        std::cout << "Agent " << (isOutside + 1) << " is outside\n";
    else if (isWinner >= 0)
        std::cout << "Agent " << (isWinner + 1) << "wins\n";
    else
        std::cout << "Simulation truncated (maxSteps reached)\n";
}


std::vector<float> SimulationEngine::getTarget(int agent, const float* S, int racelineIndex) const
{
    // TODO: handle racelineIndex!
    return cpuSampleCenterline(envConfig.trackPoints, envConfig.nTrackSamples, envConfig.dim, std::fmod(S[agent] + envConfig.targetDistance, 1.0f));
}

float SimulationEngine::getAdvance(int agent, const float* S, const float* laps) const
{
    return S[agent] + laps[agent];
}

float SimulationEngine::closestBoundaryDist(int agent, const float* phys) const
{
    float dist = cpuProjectOnTrack(envConfig.trackPoints, envConfig.nTrackSamples, envConfig.dim, phys + agent * envConfig.dim * 2, 2).second;
    return envConfig.trackWidth * 0.5f - dist;
}

bool SimulationEngine::checkCollision() const
{
    // TODO: this sucks
    for (int a = 0; a < envConfig.nAgents; a++)
        for (int b = a + 1; b < envConfig.nAgents; b++)
            if (agentDist(phys_state.data(), a, b, envConfig.dim) < envConfig.minDist)
                return true;
    return false;
}

int SimulationEngine::checkOutside() const
{
    for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    {
        float pos[MAX_DIM];
        for (int d = 0; d < envConfig.dim; d++)
            pos[d] = getPos(phys_state.data(), iAgent, d, envConfig.dim);
        float dist = cpuProjectOnTrack(envConfig.trackPoints, envConfig.nTrackSamples, envConfig.dim, phys_state.begin() + iAgent * envConfig.dim * 2, 2).second;
        if (dist > envConfig.trackWidth * 0.5f)
        {
            std::cout << "for agent " << (iAgent + 1) << " dist " << dist << " half-width " << envConfig.trackWidth << " pos " << pos[0] << ' ' << pos[1] << '\n';
            return iAgent;
        }
    }
    return -1;
}

int SimulationEngine::checkWinner() const
{
    for (int a = 0; a < envConfig.nAgents; a++)
        if (nLaps[a] >= (float) envConfig.nWinLaps) return a;
    return -1;
}
