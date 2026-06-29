#include "engine.h"
#include "state.h"
#include "protocol.h"
#include "costs.cuh"

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
    const SimState& initSimState,
    int s)
    : rng(s)
{
    seed = s;

    envConfig = config;

    state = initSimState;

    d_envConfig = Env::allocDeviceMemory(envConfig);

    // initialize controller
    controller = std::visit([&](auto&& concreteConfig) -> std::unique_ptr<Controller> {
        using T = std::decay_t<decltype(concreteConfig)>;

        if constexpr (std::is_same_v<T, MPPIConfig>)
        {
            contKind = CONT_MPPI;
            return std::make_unique<MPPIController>(d_envConfig, concreteConfig, s);
        }
        else if constexpr (std::is_same_v<T, PRMPPIConfig>)
        {
            contKind = CONT_PRMPPI;
            return std::make_unique<PRMPPIController>(d_envConfig, concreteConfig, s);
        }
        }, contConfig);

    controller->engine = this;
}

SimulationEngine::~SimulationEngine()
{
    Env::freeDeviceConfig(d_envConfig);
}

void SimulationEngine::sendState(zmq::socket_t& sock, int step, const std::array<float, ACTION_DIM>& egoAction, const SimState& prevState)
{
    if (!envConfig.sendStates)
        return;

    Writer writer;

    writer.pushInt32(MSG_STATE);
    writer.pushInt32(step);

    Env::pushSimState(writer, state);

    writer.pushFloatArray(egoAction);

    ScratchEnvBuffer buffer;

    if (auto mppiCont = dynamic_cast<MPPIController*>(controller.get()))
    {
        std::cout << "controller " << envConfig.iMppi << " is MPPI controller" << std::endl;

        writer.pushFloatArray(mppiCont->h_belief);

        writer.pushInt32(std::accumulate(mppiCont->failCount.begin(), mppiCont->failCount.end(), 0u));
        writer.pushFloat(mppiCont->epsilon);
        writer.pushFloat(mppiCont->epsilonPartial);
        writer.pushInt32((int) mppiCont->useNewPlan);
        writer.pushFloat(mppiCont->certifiedLoss);

        writer.pushInt32(N_TRUE_MODELS);

        SimState initState = state;
        MPPIConfig mppiConfig = mppiCont->mppiConfig;

        // std::vector<float> fullPos(mppiConfig.nTimesteps * N_AGENTS * ACTION_DIM);

        HostRNG hrng{ &nd, &rng };

        for (int theta = 0; theta < N_TRUE_MODELS; theta++)
        {
            float trajCost = 0.0f;
            float decay = 1.0f;

            SimState predState = initState;

            BranchState bstate;
            initBranchState(bstate, mppiCont->h_belief.data(), mppiConfig.minConfidence);

            int stopTime = -1;
            EventType stopReason = EVT_TRUNCATED;
            int stopAgent = -1;

            std::vector<float> egoActions(mppiConfig.nTimesteps * ACTION_DIM);
            std::vector<SimState> states;

            writer.pushIntArray<int>(bstate.predTheta);

            int tOrigin = 0;

            for (int t = 0; t < mppiConfig.nTimesteps; t++)
            {
                int startInd = (flattenBranchIndex(bstate.predTheta) * mppiConfig.nTimesteps + t - tOrigin) * ACTION_DIM;

                for (int d = 0; d < ACTION_DIM; d++)
                    egoActions[t * ACTION_DIM + d] = mppiCont->h_nominal[startInd + d];

                bool branched = false;

                TerminalType term = environmentStep<true, true>(
                    t,
                    theta,
                    envConfig,
                    egoActions.data() + t * ACTION_DIM,
                    false,                  // no PID noise for reproducible display
                    predState,
                    bstate,
                    branched,
                    mppiConfig.minConfidence,
                    buffer,
                    hrng);

                if (branched)
                    tOrigin = t + 1;

                states.push_back(predState);

                trajCost += stateCost(predState, t, envConfig, mppiConfig) * decay;
                decay *= DECAY;

                if (term != TERM_NONE)
                {
                    states.insert(states.end(), mppiConfig.nTimesteps - t - 1, predState);

                    std::cout << "STOPPING SIMULATION at step " << t << " term " << (int) term << std::endl;

                    stopTime = t;
                    std::optional<std::pair<EventType, int>> parsed = parseTerm(term, envConfig.iMppi);
                    stopReason = parsed->first;
                    stopAgent = parsed->second;

                    break;
                }
            }

            trajCost += finalCost(predState, envConfig, mppiConfig);

            std::cout << "Final cost for theta = " << theta << ": " << trajCost << std::endl;
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
            Env::pushPredictions(writer, states);
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

            std::vector<float> egoActions(prmppiConfig.nTimesteps * ACTION_DIM);
            std::vector<SimState> states;

            for (int t = 0; t < prmppiConfig.nTimesteps; t++)
            {
                for (int d = 0; d < ACTION_DIM; d++)
                {
                    if (j < N_TRUE_MODELS)
                        egoActions[t * ACTION_DIM + d] = prmppiCont->h_nom_nominal[t * ACTION_DIM + d];
                    else
                        egoActions[t * ACTION_DIM + d] = prmppiCont->h_rob_nominal[t * ACTION_DIM + d];
                }

                bool branched = false;

                TerminalType term = Env::environmentStep<false, false>(
                    t,
                    theta,
                    envConfig,
                    egoActions.data() + t * ACTION_DIM,
                    false,                  // no PID noise for reproducible display
                    predState,
                    bstate,
                    branched,
                    0.0f,
                    buffer,
                    hrng);

                states.push_back(predState);

                if (term != TERM_NONE)
                {
                    states.insert(states.end(), prmppiConfig.nTimesteps - t - 1, predState);

                    std::cout << "STOPPING SIMULATION at step " << t << " term " << (int) term << std::endl;

                    stopTime = t;
                    std::optional<std::pair<EventType, int>> parsed = parseTerm(term, envConfig.iMppi);
                    stopReason = parsed->first;
                    stopAgent = parsed->second;

                    break;
                }
            }

            // writer.pushFloatArray(fullPos);
            Env::pushPredictions(writer, states);

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
std::optional<std::pair<EventType, int>> SimulationEngine::dynStep(const std::array<float, ACTION_DIM>& action, int t, std::array<float, N_TRUE_MODELS>& belief)
{
    static ScratchEnvBuffer buffer;

    HostRNG hrng{ &nd, &rng };

    BranchState branchState;
    initBranchState(branchState, belief.data(), 2.0f);       // here, we only care about belief, branching time/theta is unused anyway (it is recomputed by MPPI)

    bool branched = false;

    TerminalType term;

    if (auto mppiCont = dynamic_cast<MPPIController*>(controller.get()))
        // MPPI needs belief & branching update
        term = environmentStep<true, true>(
            t,
            envConfig.trueTheta,
            envConfig,
            action.data(),
            true,
            state,
            branchState,
            branched,
            mppiCont->mppiConfig.minConfidence,
            buffer,
            hrng
        );
    else
        // other controllers may only use belief update
        term = environmentStep<true, false>(
            t,
            envConfig.trueTheta,
            envConfig,
            action.data(),
            true,
            state,
            branchState,
            branched,
            0.0f,
            buffer,
            hrng
        );

    std::copy(std::begin(branchState.belief), std::end(branchState.belief), belief.begin());

    return parseTerm(term, envConfig.iMppi);
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    std::array<float, ACTION_DIM> dummyAction{};
    sendState(sock, 0, dummyAction, state);

    int step;

    std::optional<std::pair<EventType, int>> stopInfo = std::nullopt;

    for (step = 1; step <= maxSteps; step++)
    {
        std::cout << "\nSTEP " << step << "\n";

        std::array<float, ACTION_DIM> action;
        controller->getControl(envConfig.iMppi, state, action.data());

        SimState prevState = state;

        stopInfo = dynStep(action, step, controller->h_belief);

        sendState(sock, step, action, prevState);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Current state at step " << step << ":\n";
        Env::printState(state);

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

    std::cout << "Simulation finished after " << step << " steps.\nStop reason: ";

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
    case TERM_LOSE:
        return { {EVT_OUTSIDE, egoAgent} };
    case TERM_WIN:
        return { {EVT_WINNER, 1 - egoAgent} };
    default:
        throw std::runtime_error("Unexpected TerminalType value");
    }
}
