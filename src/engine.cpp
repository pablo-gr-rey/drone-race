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
    const VerifConfig& vConfig,
    const MPPIConfig& mppiConfig,
    int s)
    : rng(s)
{
    d_trackPoints = nullptr;
    seed = s;

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

    allocTrack();
    mppiCont = MPPIController(envConfig, mppiConfig, verifConfig, d_trackPoints, seed);
    mppiCont.engine = this;

    // we only need the S of the opponent
    // TODO: this could be faster (since for now we only assume 1 opponent)
    for (int r = 0; r < N_RACELINES; r++)
        state.S[envConfig.iMppi * N_RACELINES + r] = -1.0f;
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

void SimulationEngine::allocTrack()
{
    if (d_trackPoints != nullptr)
        return;

    size_t bytes = N_RACELINES * N_TRACK_SAMPLES * DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_trackPoints, bytes));
    CUDA_CHECK(cudaMemcpy(d_trackPoints, envConfig.trackPoints, bytes, cudaMemcpyHostToDevice));
}

void SimulationEngine::sendState(zmq::socket_t& sock, int step)
{
    if (!envConfig.sendStates)
        return;

    Writer writer;

    writer.pushInt32(MSG_STATE);
    writer.pushInt32(step);

    writer.pushFloatArray(state.pos);
    writer.pushFloatArray(state.vel);
    writer.pushFloatArray(state.S);

    writer.pushIntArray<int>(state.laps);
    writer.pushIntArray<int>(state.gates);

    std::cout << "controller " << envConfig.iMppi << " is MPPI controller" << std::endl;

    writer.pushFloatArray(mppiCont.h_belief);

    writer.pushIntArray<uint>(mppiCont.failCount);
    writer.pushFloat(mppiCont.epsilon);
    writer.pushFloat(mppiCont.epsilonPartial);
    writer.pushInt32((int) mppiCont.useNewPlan);
    writer.pushFloat(mppiCont.certifiedLoss);

    writer.pushInt32(N_MODELS);

    SimState initState = state;
    MPPIConfig mppiConfig = mppiCont.mppiConfig;

    std::vector<float> fullPos(mppiConfig.nTimesteps * N_AGENTS * DIM);
    std::vector<float> fullActions(N_AGENTS * DIM);

    HostRNG hrng{ &nd, &rng };

    for (int theta = 0; theta < N_MODELS; theta++)
    {
        SimState predState = initState;

        BranchState bstate;
        initBranchState(bstate, mppiCont.h_belief.data(), mppiConfig.minConfidence);

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
                egoAction[d] = mppiCont.h_nominal[startInd + d];

            TerminalType term = environmentStep(
                t,
                envConfig.iMppi,
                theta,
                envConfig,
                mppiConfig,
                envConfig.trackPoints,
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
                std::optional<std::pair<EventType, int>> parsed = parseTerm(term, envConfig.iMppi);
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

// simulate one state for the given action. belief is updated in-place if given
std::optional<std::pair<EventType, int>> SimulationEngine::dynStep(const std::array<float, DIM>& action, int t, std::array<float, N_MODELS>& belief)
{
    static std::array<float, N_AGENTS* DIM> actBuf;
    static std::array<float, N_MODELS* DIM> nomPidBuf;

    HostRNG hrng{ &nd, &rng };

    BranchState branchState;
    initBranchState(branchState, belief.data(), 2.0f);       // here, we only care about belief, branching time/theta is unused anyway (it is recomputed by MPPI) (TODO: change that?)

    TerminalType term = environmentStep(
        t,
        envConfig.iMppi,
        envConfig.trueTheta,
        envConfig,
        mppiCont.mppiConfig,
        envConfig.trackPoints,
        action.data(),
        true,
        state,
        branchState,
        actBuf.data(),
        nomPidBuf.data(),
        hrng
    );

    std::copy(std::begin(branchState.belief), std::end(branchState.belief), belief.begin());

    return parseTerm(term, 0);
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    sendState(sock, 0);

    int step;

    std::optional<std::pair<EventType, int>> stopInfo = std::nullopt;

    for (step = 1; step <= maxSteps; step++)
    {
        std::cout << "\nSTEP " << step << "\n";

        std::array<float, DIM> action;
        mppiCont.getControl(envConfig.iMppi, state, action.data());

        stopInfo = dynStep(action, step, mppiCont.h_belief);

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
