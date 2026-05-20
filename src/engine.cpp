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
#include <cuda_runtime.h>
#include <optional>


SimulationEngine::SimulationEngine(
    const EnvironmentConfig& config,
    std::vector<float> trackPts,
    const std::vector<ControllerSpec>& specs)
    : rng(42)
{
    hasCollision = false;
    anyWinner = -1;
    anyOutside = -1;
    d_trackPoints = nullptr;

    trackPoints = trackPts;
    envConfig = config;
    // pos = config.initPos;
    // speed = config.initSpeed;
    // currentS = config.initS;
    // nLaps = config.initLaps;
    // currentGates = config.initGates;

    pos.assign(config.initPos, config.initPos + config.nAgents * config.dim);
    speed.assign(config.initSpeed, config.initSpeed + config.nAgents * config.dim);

    currentS.assign(config.initS, config.initS + config.nAgents * config.nRacelines);
    nLaps.assign(config.initLaps, config.initLaps + config.nAgents);
    currentGates.assign(config.initGates, config.initGates + config.nAgents);

    for (int i = 0; i < envConfig.nAgents; i++)
    {
        std::unique_ptr<Controller> ctrl = makeController(specs[i], i);
        controllerNames.push_back(ctrl->name);
        controllers.push_back(std::move(ctrl));
    }
}

SimulationEngine::~SimulationEngine()
{
    if (d_trackPoints != nullptr)
    {
        if (cudaFree(d_trackPoints) != cudaSuccess)
            std::cerr << "Failed to free d_trackPoints during cleanup!";
        d_trackPoints = nullptr;
    }
}

std::unique_ptr<Controller> SimulationEngine::makeController(const ControllerSpec& sp, int iCont)
{
    std::unique_ptr<Controller> ctrl;

    std::visit([&](auto&& contConfig)
        {
            using T = std::decay_t<decltype(contConfig)>;

            bool useS = false;

            if constexpr (std::is_same_v<T, DummyConfig>)
                ctrl = std::make_unique<DummyController>(envConfig);
            else if constexpr (std::is_same_v<T, PIDConfig>)
            {
                ctrl = std::make_unique<PIDController>(envConfig, contConfig);
                useS = true;
            }
            else if constexpr (std::is_same_v<T, MPPIConfig>)
            {
                if (d_trackPoints == nullptr)
                    allocTrack();

                ctrl = std::make_unique<MPPIController>(envConfig, contConfig, d_trackPoints);
            }

            if (!useS)       // only PID should update its S (otherwise, it is useless for MPPI or dummy)
                for (int iRaceline = 0; iRaceline < envConfig.nRacelines; iRaceline++)
                    currentS[iCont * envConfig.nRacelines + iRaceline] = -1.0f;

        }, sp.config);

    if (!ctrl)
        throw std::runtime_error("Unknown/invalid controller config for '" + sp.name + "'");

    ctrl->name = sp.name;
    ctrl->engine = this;
    return ctrl;
}

void SimulationEngine::allocTrack()
{
    if (d_trackPoints != nullptr)
        return;

    // upload track
    size_t bytes = envConfig.nRacelines * envConfig.nTrackSamples * envConfig.dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_trackPoints, bytes));
    CUDA_CHECK(cudaMemcpy(d_trackPoints, trackPoints.data(), bytes, cudaMemcpyHostToDevice));
}

void SimulationEngine::sendState(zmq::socket_t& sock, int step)
{
    if (envConfig.sendStates)
    {
        Writer writer;

        writer.pushInt32(MSG_STATE);

        writer.pushInt32(step);
        writer.pushFloatArray(pos);
        writer.pushFloatArray(speed);
        writer.pushFloatArray(currentS);
        writer.pushIntArray(nLaps);
        writer.pushIntArray(currentGates);

        // send MPPI info
        for (int iMppi = 0; iMppi < envConfig.nAgents; iMppi++)
            if (MPPIController* cont = dynamic_cast<MPPIController*>(controllers[iMppi].get()))
            {
                std::cout << "controller " << iMppi << " is MPPI controller" << std::endl;
                // send belief
                writer.pushInt32(iMppi);
                writer.pushFloatArray(cont->h_belief);

                // build the predicted probabilities
                // we simulate it for nominal and every theta

                // we keep copies of the initial state, since they are modified by dynStep

                std::vector<float> initPos = pos, initSpeed = speed, initS = currentS;
                std::vector<int> initGates = currentGates, initLaps = nLaps;
                MPPIConfig mppiConfig = cont->mppiConfig;

                std::vector<float> fullPos(mppiConfig.nTimesteps * envConfig.nAgents * envConfig.dim);
                std::vector<float> actions(envConfig.nAgents * envConfig.dim);
                std::vector<float> nomPIDactions(mppiConfig.nModels * envConfig.dim);

                for (int theta = 0; theta < mppiConfig.nModels; theta++)
                {
                    std::vector<float> belief = cont->h_belief;
                    int predTheta = findConfident(belief.data(), mppiConfig.nModels, mppiConfig.minConfidence);
                    int branchingTime = 0;

                    for (int t = 0; t < mppiConfig.nTimesteps; t++)
                    {
                        // build actions (we do not add PID noise for reproductibility) TODO: should we use noise?
                        for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
                        {
                            if (iAgent == iMppi)
                            {
                                int startInd = ((predTheta + 1) * mppiConfig.nTimesteps + t - branchingTime) * envConfig.dim;
                                std::copy(cont->h_nominal.begin() + startInd, cont->h_nominal.begin() + startInd + envConfig.dim, actions.begin() + iAgent * envConfig.dim);
                            }
                            else
                            {
                                for (int thetaT = 0; thetaT < mppiConfig.nModels; thetaT++)
                                    computePIDAction(iAgent, pos.data(), speed.data(), currentS.data(), envConfig, mppiConfig.oppPid[thetaT], trackPoints.data(), nomPIDactions.data() + thetaT * envConfig.dim);

                                std::copy(nomPIDactions.begin() + theta * envConfig.dim, nomPIDactions.begin() + (theta + 1) * envConfig.dim, actions.begin() + iAgent * envConfig.dim);
                            }
                        }

                        // compute step, and update positions
                        dynStep(actions);
                        std::copy(pos.begin(), pos.end(), fullPos.begin() + t * envConfig.nAgents * envConfig.dim);

                        // update belief
                        updateBelief(belief.data(), actions.data() + (1 - iMppi) * envConfig.dim, nomPIDactions.data(), mppiConfig.oppPid, mppiConfig.nModels, envConfig.dim, envConfig.maxAccel[1 - iMppi]);
                        if (predTheta == -1 && (predTheta = findConfident(belief.data(), mppiConfig.nModels, mppiConfig.minConfidence)) != -1)
                            branchingTime = t + 1;
                    }

                    // copy back original state
                    pos = initPos;
                    speed = initSpeed;
                    currentS = initS;
                    currentGates = initGates;
                    nLaps = initLaps;

                    std::cout << "final belief for theta = " << theta << ": ";
                    for (float v : belief)
                        std::cout << v << " ";
                    std::cout << "\n";

                    // send branching time, last predTheta, and full pos
                    writer.pushInt32(branchingTime);
                    writer.pushInt32(predTheta);
                    writer.pushFloatArray(fullPos);
                }

                sock.send(zmq::buffer(writer.data));
            }
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
    // clamp + noise actions

    std::vector<float> act = actions;
    std::vector<float> old_pos = pos;
    std::vector<float> old_speed = speed;

    int dim = envConfig.dim;

    for (int a = 0; a < envConfig.nAgents; a++)
    {
        float sqNorm = 0.0f;
        for (int d = 0; d < dim; d++)
            sqNorm += act[a * dim + d] * act[a * dim + d];

        float factor = 1.0f;
        if (sqNorm > envConfig.maxAccel[a] * envConfig.maxAccel[a])
            factor = envConfig.maxAccel[a] / sqrtf(sqNorm);

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
            pos[iAgent * dim + d] += envConfig.dt * speed[iAgent * dim + d];

        // integrate velocity
        float sqSpeedNorm = 0.f;
        for (int d = 0; d < dim; d++)
        {
            float& sp = speed[iAgent * dim + d];
            sp += envConfig.dt * act[iAgent * dim + d];
            sqSpeedNorm += sp * sp;
        }

        // cap speed
        if (sqSpeedNorm > envConfig.maxSpeed[iAgent] * envConfig.maxSpeed[iAgent])
        {
            float sc = envConfig.maxSpeed[iAgent] / std::sqrt(sqSpeedNorm);
            for (int d = 0; d < dim; d++)
                speed[iAgent * dim + d] *= sc;
        }

        // noise
        for (int d = 0; d < dim; d++)
        {
            pos[iAgent * dim + d] += nd(rng) * envConfig.posNoiseLevel;
            speed[iAgent * dim + d] += nd(rng) * envConfig.speedNoiseLevel;
        }
    }

    // update S, gates and laps
    // for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    // {
    //     float dist;
    //     float s = fastProjectOnTrack(trackPoints.data(), envConfig.nTrackSamples, envConfig.dim, pos.data() + iAgent * envConfig.dim, nullptr, dist, -1.0f);
    //     // float s = projectOnTrack(trackPoints.data(), envConfig.nTrackSamples, envConfig.dim, pos.data() + iAgent * envConfig.dim, nullptr, dist);
    //     currentS[iAgent] = s;

    //     // if (s > currentS[iAgent] + 0.5f)
    //     //     nLaps[iAgent] -= 1.0f;
    //     // if (s < currentS[iAgent] - 0.5f)
    //     //     nLaps[iAgent] += 1.0f;

    //     // check if we passed through next gate: compute lambda = dot(vec, center - x_t) / dot(vec, x_{t+1} - x_t)
    //     int nextGate = (currentGates[iAgent] + 1) % envConfig.nGates;
    //     float num = 0., denom = 0.;
    //     for (int d = 0; d < dim; d++)
    //     {
    //         num += envConfig.gateVectors[nextGate * dim + d] * (envConfig.gateCenters[nextGate * dim + d] - old_pos[iAgent * dim + d]);
    //         denom += envConfig.gateVectors[nextGate * dim + d] * (pos[iAgent * dim + d] - old_pos[iAgent * dim + d]);
    //     }

    //     // direction is inside the gate plan: cannot cross
    //     if (std::fabs(denom) < 1e-10)
    //         continue;

    //     float lambda = num / denom;
    //     // we cross if 0 <= lambda <= 1 and if the projection of the segment (x_t, x_t+1) on the gate plan (ie. (1 - lambda) * x_t + lambda * x_t+1) is at distance <= radius from the center
    //     // if we want to make sure we cross the gate in the right direction, we have to check num >= 0 (<=> denom > 0)

    //     // std::cout << "\nnum = " << num << " denom = " << denom << " went from " << old_phys[0] << "; " << old_phys[2] << " to " << phys_state[0] << "; " << phys_state[2] << "\n";

    //     if (lambda < 0. || lambda > 1.)
    //         continue;

    //     float sqDist = 0.;
    //     for (int d = 0; d < dim; d++)
    //     {
    //         float dx = (1. - lambda) * old_pos[iAgent * dim + d] + lambda * pos[iAgent * dim + d] - envConfig.gateCenters[nextGate * dim + d];
    //         sqDist += dx * dx;
    //     }

    //     // std::cout.precision(5);
    //     // std::cout << std::fixed << "\tsqDist = " << sqDist << " sq radius " << envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate] << "\n";

    //     if (sqDist <= envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate])
    //     {
    //         currentGates[iAgent]++;
    //         if (currentGates[iAgent] == envConfig.nGates)
    //         {
    //             currentGates[iAgent] = 0;
    //             nLaps[iAgent]++;
    //         }
    //     }
    // }

    updateGates(envConfig, pos.data(), old_pos.data(), currentS.data(), currentGates.data(), nLaps.data(), trackPoints.data());
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    sendState(sock, 0);

    int step;

    std::vector<float> prevPos;
    std::vector<float> prevSpeed;
    std::vector<float> prevS;
    std::vector<float> prevAction;

    // ── simulation loop ──────────────────────────────────────────────
    for (step = 1; step <= maxSteps; step++)
    {
        std::cout << "\nSTEP " << step << "\n";
        // compute actions
        std::vector<float> actions(envConfig.nAgents * envConfig.dim, 0.0f);
        for (int i = 0; i < envConfig.nAgents; i++)
            controllers[i]->getControl(i, pos.data(), speed.data(), currentS.data(),
                nLaps.data(),
                currentGates.data(),
                actions.data() + i * envConfig.dim,
                nd,
                rng,
                step == 1 ? std::nullopt : std::make_optional(prevAction),
                step == 1 ? std::nullopt : std::make_optional(prevPos),
                step == 1 ? std::nullopt : std::make_optional(prevSpeed),
                step == 1 ? std::nullopt : std::make_optional(prevS)
            );

        prevAction = actions;
        prevPos = pos;
        prevSpeed = speed;
        prevS = currentS;

        dynStep(actions);

        // float sq = 0.f;
        // int iPid = 0;
        // for (int d = 0; d < envConfig.dim; d++)
        //     sq += pos[iPid * envConfig.dim + d] * pos[iPid * envConfig.dim + d];
        // std::cout << "position of PID: radius " << sqrtf(sq) << " pos " << pos[iPid * envConfig.dim] << " " << pos[iPid * envConfig.dim + 1] << "\n";

        sendState(sock, step);

        int best = -1;
        float bestAdvance = 0.;
        for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        {
            float advance = getAdvance(nLaps.data(), currentGates.data(), iAgent, envConfig.nGates, pos.data() + envConfig.dim * iAgent, envConfig.gateCenters, envConfig.dim);
            if (iAgent == 0 || advance > bestAdvance)
            {
                best = iAgent;
                bestAdvance = advance;
            }
        }

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "Current state at step " << step << ": agent " << (best + 1) << " in front (advance " << bestAdvance << ")" << std::endl;

        for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
        {
            std::cout << "\tAgent " << (iAgent + 1)
                << ": currentS:";
            for (int i = 0; i < envConfig.nRacelines; i++)
                std::cout << " " << currentS[iAgent * envConfig.nRacelines + i];

            std::cout << "\tcurrentGates: " << currentGates[iAgent]
                << "\tnLaps: " << nLaps[iAgent]
                << "\tposition";

            for (int d = 0; d < envConfig.dim; d++)
                std::cout << " " << pos[iAgent * envConfig.dim + d];

            std::cout << "\tspeed:";
            for (int d = 0; d < envConfig.dim; d++)
                std::cout << " " << speed[iAgent * envConfig.dim + d];

            std::cout << std::endl;
        }

        // termination checks
        hasCollision = checkCollision();
        anyOutside = checkOutside();
        anyWinner = checkWinner();

        if (hasCollision || anyOutside >= 0 || anyWinner >= 0)
        {
            if (anyOutside >= 0)
                sendEvent(sock, EVT_OUTSIDE, anyOutside);
            else if (anyWinner >= 0)
                sendEvent(sock, EVT_WINNER, anyWinner);
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
    else if (anyOutside >= 0)
        std::cout << "Agent " << (anyOutside + 1) << " is outside\n";
    else if (anyWinner >= 0)
        std::cout << "Agent " << (anyWinner + 1) << "wins\n";
    else
        std::cout << "Simulation truncated (maxSteps reached)\n";
}

bool SimulationEngine::checkCollision() const
{
    for (int a = 0; a < envConfig.nAgents; a++)
        for (int b = a + 1; b < envConfig.nAgents; b++)
            if (agentDist(pos.data(), a, b, envConfig.dim) < envConfig.minDist)
                return true;
    return false;
}

int SimulationEngine::checkOutside() const
{
    for (int iAgent = 0; iAgent < envConfig.nAgents; iAgent++)
    {
        if (isOutside(envConfig, pos.data() + iAgent * envConfig.dim, envConfig.minDist / 2.0f))
            return iAgent;

        // for (int d = 0; d < envConfig.dim; d++)
        // {
        //     float p = pos[iAgent * envConfig.dim + d];
        //     if (p < envConfig.arenaMin[d] || p > envConfig.arenaMax[d])
        //     {
        //         std::cout << "agent " << (iAgent + 1) << " outside (dimension " << d << ": position " << p << " outside of arena\n";
        //         return iAgent;
        //     }
        // }
    }

    return -1;
}

int SimulationEngine::checkWinner() const
{
    for (int a = 0; a < envConfig.nAgents; a++)
        if (nLaps[a] >= (float) envConfig.nWinLaps) return a;
    return -1;
}
