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
    const AnyControllerConfig& contConfig,
    int s)
    : rng(s)
{
    d_trackPoints = nullptr;
    seed = s;

    envConfig = config;

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

    // initialize controller
    controller = std::visit([&](auto&& concreteConfig) -> std::unique_ptr<Controller> {
        using T = std::decay_t<decltype(concreteConfig)>;

        if constexpr (std::is_same_v<T, MPPIConfig>)
        {
            contKind = CONT_MPPI;
            return std::make_unique<MPPIController>(envConfig, concreteConfig, d_trackPoints, s);
        }
        else if constexpr (std::is_same_v<T, PRMPPIConfig>)
        {
            contKind = CONT_PRMPPI;
            return std::make_unique<PRMPPIController>(envConfig, concreteConfig, d_trackPoints, s);
        }
        }, contConfig);

    controller->engine = this;

    // we only need the S of the opponent
    // this could be faster if we only assume 1 opponent)
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

void SimulationEngine::sendState(zmq::socket_t& sock, int step, const std::array<float, DIM>& egoAction, const SimState& prevState)
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

    writer.pushFloatArray(egoAction);

    if (auto mppiCont = dynamic_cast<MPPIController*>(controller.get()))
    {
        std::cout << "controller " << envConfig.iMppi << " is MPPI controller" << std::endl;

        writer.pushFloatArray(mppiCont->h_belief);

        writer.pushIntArray<uint>(mppiCont->failCount);
        writer.pushFloat(mppiCont->epsilon);
        writer.pushFloat(mppiCont->epsilonPartial);
        writer.pushInt32((int) mppiCont->useNewPlan);
        writer.pushFloat(mppiCont->certifiedLoss);

        writer.pushInt32(N_TRUE_MODELS);

        SimState initState = state;
        MPPIConfig mppiConfig = mppiCont->mppiConfig;

        std::vector<float> fullPos(mppiConfig.nTimesteps * N_AGENTS * DIM);

        HostRNG hrng{ &nd, &rng };

        for (int theta = 0; theta < N_TRUE_MODELS; theta++)
        {
            SimState predState = initState;

            BranchState bstate;
            initBranchState(bstate, mppiCont->h_belief.data(), mppiConfig.minConfidence);

            int stopTime = -1;
            EventType stopReason = EVT_TRUNCATED;
            int stopAgent = -1;

            float scratchActions[N_AGENTS * DIM];
            float nomPidAction[N_TRUE_MODELS * DIM];
            std::vector<float> egoActions(mppiConfig.nTimesteps * DIM);

            writer.pushIntArray<int>(bstate.predTheta);

            // std::cout << "sending initPredTheta: " << bstate.predTheta[0] << "\n";

            int tOrigin = 0;

            for (int t = 0; t < mppiConfig.nTimesteps; t++)
            {
                // int startInd = ((bstate.predTheta + 1) * mppiConfig.nTimesteps + t - bstate.branchingTime) * DIM;
                // int startInd = (flattenBranchIndex(bstate.predTheta) * mppiConfig.nTimesteps + t - localBranchTimeOrigin(bstate.predTheta, bstate.branchingTime)) * DIM;
                int startInd = (flattenBranchIndex(bstate.predTheta) * mppiConfig.nTimesteps + t - tOrigin) * DIM;

                for (int d = 0; d < DIM; d++)
                    egoActions[t * DIM + d] = mppiCont->h_nominal[startInd + d];

                bool branched = false;

                TerminalType term = environmentStep<true, true>(
                    t,
                    envConfig.iMppi,
                    theta,
                    envConfig,
                    egoActions.data() + t * DIM,
                    false,                  // no PID noise for reproducible display
                    predState,
                    bstate,
                    branched,
                    mppiConfig.minConfidence,
                    scratchActions,
                    nomPidAction,
                    hrng);

                if (branched)
                    tOrigin = t + 1;

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

            std::cout << "Final belief for theta = " << theta << ": ";
            for (int k = 0; k < N_TRUE_MODELS; k++)
                std::cout << bstate.belief[k] << " ";
            std::cout << "\n";

            std::cout << "Final branching time: ";
            for (int t : bstate.branchingTime)
                std::cout << t << ' ';
            std::cout << "\nFinal pred theta: ";
            for (int t : bstate.predTheta)
                std::cout << t << ' ';
            std::cout << "\n";

            writer.pushIntArray<int>(bstate.branchingTime);
            writer.pushIntArray<int>(bstate.predTheta);
            writer.pushFloatArray(fullPos);
            writer.pushFloatArray(egoActions);
            writer.pushInt32(stopReason);
            writer.pushInt32(stopTime);
            writer.pushInt32(stopAgent);
        }
    }

    else if (auto prmppiCont = dynamic_cast<PRMPPIController*>(controller.get()))
    {
        std::cout << "controller " << envConfig.iMppi << " is MPPI controller" << std::endl;

        writer.pushFloatArray(prevState.pos);
        writer.pushFloatArray(prmppiCont->h_belief);

        writer.pushInt32((int) prmppiCont->useNomPlan);
        writer.pushInt32((int) prmppiCont->resetNom);

        // we do an additional run with last model and rob_nominal
        writer.pushInt32(N_TRUE_MODELS + 1);

        SimState initState = prevState;
        PRMPPIConfig prmppiConfig = prmppiCont->mppiConfig;

        std::vector<float> fullPos(prmppiConfig.nTimesteps * N_AGENTS * DIM);

        HostRNG hrng{ &nd, &rng };

        // one additional pass for rob_nominal on the last model

        for (int j = 0; j <= N_TRUE_MODELS; j++)
        {
            int theta = std::min(j, N_TRUE_MODELS - 1);

            SimState predState = initState;

            BranchState bstate;

            int stopTime = -1;
            EventType stopReason = EVT_TRUNCATED;
            int stopAgent = -1;

            float scratchActions[N_AGENTS * DIM];
            float nomPidAction[N_TRUE_MODELS * DIM];
            std::vector<float> egoActions(prmppiConfig.nTimesteps * DIM);

            for (int t = 0; t < prmppiConfig.nTimesteps; t++)
            {
                for (int d = 0; d < DIM; d++)
                {
                    if (j < N_TRUE_MODELS)
                        egoActions[t * DIM + d] = prmppiCont->h_nom_nominal[t * DIM + d];
                    else
                        egoActions[t * DIM + d] = prmppiCont->h_rob_nominal[t * DIM + d];
                }

                bool branched = false;

                TerminalType term = environmentStep<false, false>(
                    t,
                    envConfig.iMppi,
                    theta,
                    envConfig,
                    egoActions.data() + t * DIM,
                    false,                  // no PID noise for reproducible display
                    predState,
                    bstate,
                    branched,
                    0.0f,
                    scratchActions,
                    nomPidAction,
                    hrng);

                for (int i = 0; i < N_AGENTS * DIM; i++)
                    fullPos[t * N_AGENTS * DIM + i] = predState.pos[i];

                if (term != TERM_NONE)
                {
                    for (int tt = t + 1; tt < prmppiConfig.nTimesteps; tt++)
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

            writer.pushFloatArray(fullPos);
            writer.pushFloatArray(egoActions);
            writer.pushInt32(stopReason);
            writer.pushInt32(stopTime);
            writer.pushInt32(stopAgent);
        }
    }

    else
        throw std::runtime_error("Ill-formed controller type in sendState");

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
std::optional<std::pair<EventType, int>> SimulationEngine::dynStep(const std::array<float, DIM>& action, int t, std::array<float, N_TRUE_MODELS>& belief)
{
    static std::array<float, N_AGENTS* DIM> actBuf;
    static std::array<float, N_TRUE_MODELS* DIM> nomPidBuf;

    HostRNG hrng{ &nd, &rng };

    BranchState branchState;
    initBranchState(branchState, belief.data(), 2.0f);       // here, we only care about belief, branching time/theta is unused anyway (it is recomputed by MPPI)

    bool branched = false;

    TerminalType term;

    if (auto mppiCont = dynamic_cast<MPPIController*>(controller.get()))
        // MPPI needs belief & branching update
        term = environmentStep<true, true>(
            t,
            envConfig.iMppi,
            envConfig.trueTheta,
            envConfig,
            action.data(),
            true,
            state,
            branchState,
            branched,
            mppiCont->mppiConfig.minConfidence,
            actBuf.data(),
            nomPidBuf.data(),
            hrng
        );
    else
        // other controllers may only use belief update
        term = environmentStep<true, false>(
            t,
            envConfig.iMppi,
            envConfig.trueTheta,
            envConfig,
            action.data(),
            true,
            state,
            branchState,
            branched,
            0.0f,
            actBuf.data(),
            nomPidBuf.data(),
            hrng
        );

    std::copy(std::begin(branchState.belief), std::end(branchState.belief), belief.begin());

    return parseTerm(term, envConfig.iMppi);
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    std::array<float, DIM> dummyAction{};
    sendState(sock, 0, dummyAction, state);

    int step;

    std::optional<std::pair<EventType, int>> stopInfo = std::nullopt;

    for (step = 1; step <= maxSteps; step++)
    {
        std::cout << "\nSTEP " << step << "\n";

        std::array<float, DIM> action;
        controller->getControl(envConfig.iMppi, state, action.data());

        SimState prevState = state;

        stopInfo = dynStep(action, step, controller->h_belief);

        sendState(sock, step, action, prevState);

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
        {
            std::cout << "Event " << stopInfo->first << " (agent " << stopInfo->second << ")\n";
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
