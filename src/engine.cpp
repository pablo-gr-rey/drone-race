#include "engine.h"
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
#include <iomanip>

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
    currentGates = config.initGates;

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
                ctrl = std::make_unique<MPPIController>(envConfig, contConfig);
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
        writer.pushIntArray(nLaps);
        writer.pushIntArray(currentGates);

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

    // clamp + noise actions

    std::vector<float> act = actions;
    std::vector<float> old_phys = phys_state;

    int dim = envConfig.dim;

    for (int a = 0; a < envConfig.nAgents; a++)
    {
        float sqNorm = 0.0f;
        for (int d = 0; d < dim; d++)
            sqNorm += act[a * dim + d] * act[a * dim + d];

        float factor = 1.0f;
        if (sqNorm > envConfig.maxAccel[a] * envConfig.maxAccel[a])
            factor = envConfig.maxAccel[a] / fsqrt(sqNorm);

        for (int d = 0; d < dim; d++)
        {
            float& v = act[a * dim + d];
            v *= factor;
            v += nd(rng) * envConfig.actionNoiseLevel;
        }
    }

    for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    {
        // integrate position
        for (int d = 0; d < dim; d++)
            phys_state[iAgent * dim * 2 + d * 2] += envConfig.dt * phys_state[iAgent * dim * 2 + d * 2 + 1];

        // integrate velocity
        float sqSpeedNorm = 0.f;
        for (int d = 0; d < dim; d++)
        {
            float& speed = phys_state[iAgent * dim * 2 + d * 2 + 1];
            speed += envConfig.dt * act[iAgent * dim + d];
            sqSpeedNorm += speed * speed;
        }

        // cap speed
        if (sqSpeedNorm > envConfig.maxSpeed[iAgent] * envConfig.maxSpeed[iAgent])
        {
            float sc = envConfig.maxSpeed[iAgent] / std::sqrt(sqSpeedNorm);
            for (int d = 0; d < dim; d++)
                phys_state[iAgent * dim * 2 + d * 2 + 1] *= sc;
        }

        // noise
        for (int d = 0; d < dim; d++)
        {
            phys_state[iAgent * dim * 2 + d * 2] += nd(rng) * envConfig.posNoiseLevel;
            phys_state[iAgent * dim * 2 + d * 2 + 1] += nd(rng) * envConfig.speedNoiseLevel;
        }
    }

    // update S, gates and laps
    for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    {
        // float s = cpuProjectOnTrack(envConfig.trackPoints, envConfig.nTrackSamples, dim, phys_state.begin() + 2 * iAgent * dim, 2).first;
        float s = cpuProjectOnTrack(phys_state, iAgent).first;
        currentS[iAgent] = s;

        // if (s > currentS[iAgent] + 0.5f)
        //     nLaps[iAgent] -= 1.0f;
        // if (s < currentS[iAgent] - 0.5f)
        //     nLaps[iAgent] += 1.0f;

        // check if we passed through next gate: compute lambda = dot(vec, center - x_t) / dot(vec, x_{t+1} - x_t)
        int nextGate = (currentGates[iAgent] + 1) % envConfig.nGates;
        float num = 0., denom = 0.;
        for (int d = 0; d < dim; d++)
        {
            num += envConfig.gateVectors[nextGate * dim + d] * (envConfig.gateCenters[nextGate * dim + d] - old_phys[iAgent * dim * 2 + d * 2]);
            denom += envConfig.gateVectors[nextGate * dim + d] * (phys_state[iAgent * dim * 2 + d * 2] - old_phys[iAgent * dim * 2 + d * 2]);
        }

        // direction is inside the gate plan: cannot cross
        if (std::fabs(denom) < 1e-10)
            continue;

        float lambda = num / denom;
        // we cross if 0 <= lambda <= 1 and if the projection of the segment (x_t, x_t+1) on the gate plan (ie. (1 - lambda) * x_t + lambda * x_t+1) is at distance <= radius from the center
        // if we want to make sure we cross the gate in the right direction, we have to check num >= 0 (<=> denom > 0)

        // std::cout << "\nnum = " << num << " denom = " << denom << " went from " << old_phys[0] << "; " << old_phys[2] << " to " << phys_state[0] << "; " << phys_state[2] << "\n";

        if (lambda < 0. || lambda > 1.)
            continue;

        float sqDist = 0.;
        for (int d = 0; d < dim; d++)
        {
            float dx = (1. - lambda) * old_phys[iAgent * dim * 2 + d * 2] + lambda * phys_state[iAgent * dim * 2 + d * 2] - envConfig.gateCenters[nextGate * dim + d];
            sqDist += dx * dx;
        }

        // std::cout.precision(5);
        // std::cout << std::fixed << "\tsqDist = " << sqDist << " sq radius " << envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate] << "\n";

        if (sqDist <= envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate])
        {
            currentGates[iAgent]++;
            if (currentGates[iAgent] == envConfig.nGates)
            {
                currentGates[iAgent] = 0;
                nLaps[iAgent]++;
            }
        }
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
                currentGates.data(),
                actions.data() + i * envConfig.dim);

        // step dynamics
        // std::vector<float> np(envConfig.physDim), ns(envConfig.nAgents), nl(envConfig.nAgents);

        dynStep(actions);

        sendState(sock, step);

        // int best = -1;
        // float bestAdvance = 0.;
        // for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        // {
        //     float advance = getAdvance(nLaps.data(), currentGates.data(), iAgent, envConfig.nGates, phys_state.data() + 2 * envConfig.dim * iAgent, envConfig.gateCenters.data(), envConfig.dim);
        //     if (iAgent == 0 || advance > bestAdvance)
        //     {
        //         best = iAgent;
        //         bestAdvance = advance;
        //     }
        // }

        // std::cout << std::fixed << std::setprecision(2);

        // std::cout << "Current state at step " << step << ": agent " << (best + 1) << " in front (advance " << bestAdvance << ")" << std::endl;

        // for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        // {
        //     std::cout << "\tAgent " << (iAgent + 1)
        //         << ": currentS: " << currentS[iAgent]
        //         << "\tcurrentGates: " << currentGates[iAgent]
        //         << "\tnLaps: " << nLaps[iAgent]
        //         << "\tposition ";

        //     for (int d = 0; d < envConfig.dim; d++)
        //         std::cout << phys_state[iAgent * envConfig.dim * 2 + d * 2] << " ";

        //     std::cout << "\tspeed: ";
        //     for (int d = 0; d < envConfig.dim; d++)
        //         std::cout << phys_state[iAgent * envConfig.dim * 2 + d * 2 + 1] << " ";

        //     std::cout << std::endl;
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
    if (racelineIndex == -1)    // just return the center of the next gate
        return std::vector<float>(envConfig.gateCenters.begin() + (currentGates[agent] + 1) % envConfig.nGates * envConfig.dim, envConfig.gateCenters.begin() + ((currentGates[agent] + 1) % envConfig.nGates + 1) * envConfig.dim);

    // std::vector<float> ans = cpuSampleCenterline(envConfig.trackPoints, envConfig.nTrackSamples, envConfig.dim, std::fmod(S[agent] + envConfig.targetDistance, 1.0f), racelineIndex);
    std::vector<float> ans = cpuSampleCenterline(std::fmod(S[agent] + envConfig.targetDistance, 1.0f), racelineIndex);

    // if (racelineIndex == 1)
    // {
    //     std::cout << "for S = " << S[agent] << ": sampled ";
    //     for (int i = 0; i < envConfig.dim; i++)
    //         std::cout << ans[i] << ' ';
    //     int ind = racelineIndex * envConfig.nTrackSamples + (int) (S[agent] * envConfig.nTrackSamples);
    //     std::cout << "sampling on this single S (from ind " << ind << ") ";
    //     for (int i = 0; i < envConfig.dim; i++)
    //         std::cout << envConfig.trackPoints[ind * envConfig.dim + i] << ' ';
    //     std::cout << '\n';
    // }

    return ans;
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
        for (int d = 0; d < envConfig.dim; d++)
        {
            float pos = getPos(phys_state.data(), iAgent, d, envConfig.dim);
            if (pos < envConfig.arenaMin[d] || pos > envConfig.arenaMax[d])
            {
                std::cout << "agent " << (iAgent + 1) << " outside (dimension " << d << ": position " << pos << " outside of arena\n";
                return iAgent;
            }
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


// return the centerline sampled at given s
std::vector<float> SimulationEngine::cpuSampleCenterline(float s, int racelineIndex) const
{
    std::vector<float> out(envConfig.dim);

    s = s - std::floor(s);
    float idx_f = s * envConfig.nTrackSamples;
    int idx0 = (int) idx_f;
    int idx1 = (idx0 + 1) % envConfig.nTrackSamples;
    float t = idx_f - idx0;

    for (int d = 0; d < envConfig.dim; d++)
        out[d] = (1.0f - t) * envConfig.trackPoints[(envConfig.nTrackSamples * racelineIndex + idx0) * envConfig.dim + d] + t * envConfig.trackPoints[(envConfig.nTrackSamples * racelineIndex + idx1) * envConfig.dim + d];

    return out;
}

// return the closest S and the distance to it for the given pos. allows specifying a stride on reading (useful for phys state)
std::pair<float, float> SimulationEngine::cpuProjectOnTrack(const std::vector<float>& state, int iAgent) const
{
    float bestS = 0.0f, bestdSq = 1e30f;
    const float sStep = 1.0f / envConfig.nTrackSamples;

    int iTrack = 0;

    for (int i = 0; i < envConfig.nTrackSamples; ++i)
    {
        float dSq = 0.0f;

        for (int d = 0; d < envConfig.dim; ++d)
        {
            float dx = envConfig.trackPoints[iTrack] - state[iAgent * envConfig.dim * 2 + d * 2];
            dSq += dx * dx;
            iTrack++;
        }

        if (dSq < bestdSq) { bestdSq = dSq; bestS = i * sStep; }
    }

    return std::make_pair(bestS, std::sqrt(bestdSq));
}
