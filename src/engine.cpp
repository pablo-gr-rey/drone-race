#include "engine.h"
#include "state.h"
#include "protocol.h"
#include "environment.h"

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
    const VerifConfig& vConfig,
    const std::vector<ControllerSpec>& specs,
    int s)
    : rng(s)
{
    d_trackPoints = nullptr;
    seed = s;

    trackPoints = std::move(trackPts);
    envConfig = config;
    verifConfig = vConfig;

    // initialize current state from config
    for (int a = 0; a < N_AGENTS; a++)
    {
        for (int d = 0; d < DIM; d++)
        {
            state.pos[a * DIM + d] = config.initPos[a * DIM + d];
            state.vel[a * DIM + d] = config.initSpeed[a * DIM + d];
        }

        for (int r = 0; r < N_RACELINES; r++)
            state.S[a * N_RACELINES + r] = config.initS[a * N_RACELINES + r];

        state.laps[a] = config.initLaps[a];
        state.gates[a] = config.initGates[a];
    }

    nMppiCont = 0;
    int lastMppiIndex = -1;

    for (int i = 0; i < N_AGENTS; i++)
    {
        auto [ctrl, isMPPI] = makeController(specs[i]);
        controllerNames.push_back(ctrl->name);
        controllers.push_back(std::move(ctrl));

        if (isMPPI)
        {
            lastMppiIndex = i;
            nMppiCont++;
        }
    }

    // if there is exactly one MPPI controller, then we can skip updating its S
    // (otherwise, all PID need their S, and if someone else is MPPI, it needs to predict us using our S)
    if (nMppiCont == 1)
    {
        for (int r = 0; r < N_RACELINES; r++)
            state.S[lastMppiIndex * N_RACELINES + r] = -1.0f;
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

std::pair<std::unique_ptr<Controller>, bool> SimulationEngine::makeController(const ControllerSpec& sp)
{
    std::unique_ptr<Controller> ctrl;
    bool isMPPI = false;

    std::visit([&](auto&& contConfig)
        {
            using T = std::decay_t<decltype(contConfig)>;

            if constexpr (std::is_same_v<T, DummyConfig>)
                ctrl = std::make_unique<DummyController>(envConfig);
            else if constexpr (std::is_same_v<T, PIDConfig>)
                ctrl = std::make_unique<PIDController>(envConfig, contConfig);
            else if constexpr (std::is_same_v<T, MPPIConfig>)
            {
                if (d_trackPoints == nullptr)
                    allocTrack();

                ctrl = std::make_unique<MPPIController>(envConfig, contConfig, verifConfig, d_trackPoints, seed);
                isMPPI = true;
            }
        }, sp.config);

    if (!ctrl)
        throw std::runtime_error("Unknown/invalid controller config for '" + sp.name + "'");

    ctrl->name = sp.name;
    ctrl->engine = this;
    return { std::move(ctrl), isMPPI };
}

void SimulationEngine::allocTrack()
{
    if (d_trackPoints != nullptr)
        return;

    size_t bytes = N_RACELINES * N_TRACK_SAMPLES * DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_trackPoints, bytes));
    CUDA_CHECK(cudaMemcpy(d_trackPoints, trackPoints.data(), bytes, cudaMemcpyHostToDevice));
}

void SimulationEngine::sendState(zmq::socket_t& sock, int step)
{
    if (!envConfig.sendStates)
        return;

    Writer writer;

    writer.pushInt32(MSG_STATE);
    writer.pushInt32(step);

    writer.pushFloatArray(state.pos, N_AGENTS * DIM);
    writer.pushFloatArray(state.vel, N_AGENTS * DIM);
    writer.pushFloatArray(state.S, N_AGENTS * N_RACELINES);

    writer.pushIntArray(state.laps, N_AGENTS);
    writer.pushIntArray(state.gates, N_AGENTS);

    writer.pushInt32(nMppiCont);

    for (int iMppi = 0; iMppi < N_AGENTS; iMppi++)
    {
        if (MPPIController* cont = dynamic_cast<MPPIController*>(controllers[iMppi].get()))
        {
            std::cout << "controller " << iMppi << " is MPPI controller" << std::endl;

            writer.pushInt32(iMppi);
            writer.pushFloatArray(cont->h_belief);

            writer.pushIntArray(cont->failCount);
            writer.pushFloat(cont->epsilon);
            writer.pushFloat(cont->epsilonPartial);
            writer.pushInt32((int) cont->useNewPlan);
            writer.pushFloat(cont->certifiedLoss);

            writer.pushInt32(N_MODELS);

            SimState initState = state;
            MPPIConfig mppiConfig = cont->mppiConfig;

            std::vector<float> fullPos(mppiConfig.nTimesteps * N_AGENTS * DIM);
            std::vector<float> fullActions(N_AGENTS * DIM);
            std::vector<float> beliefVec = cont->h_belief;

            HostRNG hrng{ &nd, &rng };

            for (int theta = 0; theta < N_MODELS; theta++)
            {
                SimState predState = initState;

                BranchState bstate;
                initBranchState(bstate, beliefVec.data(), mppiConfig.minConfidence);

                int stopTime = -1;
                EventType stopReason = EVT_TRUNCATED;
                int stopAgent = -1;

                float scratchActions[N_AGENTS * DIM];
                float nomPidAction[N_MODELS * DIM];
                float egoAction[DIM];

                for (int t = 0; t < mppiConfig.nTimesteps; t++)
                {
                    int startInd = ((bstate.predTheta + 1) * mppiConfig.nTimesteps + t - bstate.branchingTime) * DIM;
                    for (int d = 0; d < DIM; d++)
                        egoAction[d] = cont->h_nominal[startInd + d];

                    TerminalType term = simulateContingentStep(
                        t,
                        iMppi,
                        theta,
                        envConfig,
                        mppiConfig,
                        trackPoints.data(),
                        egoAction,
                        false,                  // no PID noise for reproducible display
                        predState,
                        bstate,
                        scratchActions,
                        nomPidAction,
                        hrng);

                    for (int i = 0; i < N_AGENTS * DIM; i++)
                        fullPos[t * N_AGENTS * DIM + i] = predState.pos[i];

                    if (term != TERM_NONE)
                    {
                        for (int tt = t + 1; tt < mppiConfig.nTimesteps; tt++)
                        {
                            for (int i = 0; i < N_AGENTS * DIM; i++)
                                fullPos[tt * N_AGENTS * DIM + i] = predState.pos[i];
                        }

                        std::cout << "STOPPING SIMULATION at step " << t
                            << " term " << (int) term << std::endl;

                        stopTime = t;
                        std::optional<std::pair<EventType, int>> parsed = parseTerm(term, iMppi);
                        stopReason = parsed->first;
                        stopAgent = parsed->second;

                        break;
                    }
                }

                std::cout << "final belief for theta = " << theta << ": ";
                for (int k = 0; k < N_MODELS; k++)
                    std::cout << bstate.belief[k] << " ";
                std::cout << "\n";

                writer.pushInt32(bstate.branchingTime);
                writer.pushInt32(bstate.predTheta);
                writer.pushFloatArray(fullPos);
                writer.pushInt32(stopReason);
                writer.pushInt32(stopTime);
                writer.pushInt32(stopAgent);
            }
        }
    }

    sock.send(zmq::buffer(writer.data));
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

std::optional<std::pair<EventType, int>> SimulationEngine::dynStep(const std::vector<float>& actions)
{
    float actBuf[N_AGENTS * DIM] = {};
    for (int i = 0; i < N_AGENTS * DIM; i++)
        actBuf[i] = actions[i];

    HostRNG hrng{ &nd, &rng };

    // controlAgent only matters for ego/opp terminal labeling, so we use 0 and ignore the return value
    // TODO: we should check it ourselves
    TerminalType term = applyEnvironmentDynamics(
        0,
        envConfig,
        trackPoints.data(),
        state,
        actBuf,
        hrng);

    return parseTerm(term, 0);
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    sendState(sock, 0);

    int step;

    std::optional<SimState> prevState = std::nullopt;
    std::optional<std::vector<float>> prevAction = std::nullopt;

    std::optional<std::pair<EventType, int>> stopInfo = std::nullopt;

    for (step = 1; step <= maxSteps; step++)
    {
        std::cout << "\nSTEP " << step << "\n";

        std::vector<float> actions(N_AGENTS * DIM, 0.0f);

        for (int i = 0; i < N_AGENTS; i++)
        {
            controllers[i]->getControl(
                i,
                state,
                actions.data() + i * DIM,
                nd,
                rng,
                prevAction,
                prevState);
        }

        prevAction = actions;
        prevState = state;

        stopInfo = dynStep(actions);

        sendState(sock, step);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Current state at step " << step << ":\n";

        for (int a = 0; a < N_AGENTS; a++)
        {
            std::cout << "\tAgent " << (a + 1) << ": currentS:";
            for (int r = 0; r < N_RACELINES; r++)
                std::cout << " " << state.S[a * N_RACELINES + r];

            std::cout << "\tcurrentGates: " << state.gates[a]
                << "\tnLaps: " << state.laps[a]
                << "\tposition";

            for (int d = 0; d < DIM; d++)
                std::cout << " " << state.pos[a * DIM + d];

            std::cout << "\tspeed:";
            for (int d = 0; d < DIM; d++)
                std::cout << " " << state.vel[a * DIM + d];

            std::cout << std::endl;
        }

        if (stopInfo)
            std::cout << "Event " << stopInfo->first << " (agent " << stopInfo->second << ")\n";

        // termination checks
        // hasCollision = (bool) checkCollision();
        // anyOutside = checkOutside();
        // anyWinner = checkWinner();

        // if (hasCollision || anyOutside >= 0 || anyWinner >= 0)
        // {
        //     if (anyOutside >= 0)
        //         sendEvent(sock, EVT_OUTSIDE, anyOutside);
        //     else if (anyWinner >= 0)
        //         sendEvent(sock, EVT_WINNER, anyWinner);
        //     else
        //         sendEvent(sock, EVT_COLLISION, -1);

        //     break;
        // }

        if (stopInfo)
        {
            sendEvent(sock, stopInfo->first, stopInfo->second);
            break;
        }

        if (step == maxSteps)
            sendEvent(sock, EVT_TRUNCATED, -1);
    }

    sendDone(sock);

    std::cout << "Simulation finished. "
        << N_AGENTS << " agents, "
        << N_TRACK_SAMPLES << " track samples after "
        << step << " steps.\nStop reason: ";

    if (!stopInfo)
        std::cout << "Simulation truncated (maxSteps reached)\n";
    else if (stopInfo->first == EVT_COLLISION)
        std::cout << "Collision\n";
    else if (stopInfo->first == EVT_OUTSIDE)
        std::cout << "Agent " << (stopInfo->second + 1) << " is outside\n";
    else if (stopInfo->first == EVT_WINNER)
        std::cout << "Agent " << (stopInfo->second + 1) << " wins\n";
}

std::optional<std::pair<EventType, int>> SimulationEngine::parseTerm(TerminalType term, int egoAgent)
{
    switch (term)
    {
    case TERM_NONE:
        return std::nullopt;
    case TERM_COLLISION:
        return { {EVT_COLLISION, -1} };
    case TERM_EGO_OUTSIDE:
        return { {EVT_OUTSIDE, egoAgent} };
    case TERM_OPP_OUTSIDE:
        return { {EVT_OUTSIDE, 1 - egoAgent} };
    case TERM_WIN:
        return { {EVT_WINNER, egoAgent} };
    case TERM_OPP_WIN:
        return { {EVT_WINNER, 1 - egoAgent} };
    default:
        throw std::runtime_error("Unexpected TerminalType value");
    }
}
