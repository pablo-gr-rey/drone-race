#include "engine.h"
#include "costs.cuh"
#include "protocol.h"
#include "state.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <optional>
#include <type_traits>
#include <variant>
#include <zmq.h>

SimulationEngine::SimulationEngine(const EnvironmentConfig& config, const AnyControllerConfig& contConfig, const SimState& initSimState, int s,
                                   int tTheta)
    : rng(s)
{
    seed = s;
    trueTheta = tTheta;

    envConfig = config;

    state = initSimState;

    d_envConfig = Env::allocDeviceMemory(envConfig);

    // initialize controller
    controller = std::visit(
        [&](auto&& concreteConfig) -> std::unique_ptr<Controller>
        {
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
        },
        contConfig);

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

    Env::pushAddInfo(writer, state, prevState, envConfig, trueTheta);

    writer.pushFloatArray(egoAction);

    ScratchEnvBuffer buffer;

    if (auto mppiCont = dynamic_cast<MPPIController*>(controller.get()))
    {
        writer.pushFloatArray(mppiCont->h_belief);

        writer.pushInt32(std::accumulate(mppiCont->failCount.begin(), mppiCont->failCount.end(), 0u));
        writer.pushFloat(mppiCont->epsilon);
        writer.pushFloat(mppiCont->epsilonPartial);
        writer.pushInt32((int)mppiCont->useNewPlan);
        writer.pushFloat(mppiCont->certifiedLoss);

        writer.pushInt32(N_TRUE_MODELS);

        SimState initState = state;
        MPPIConfig mppiConfig = mppiCont->mppiConfig;

        // std::vector<float> fullPos(mppiConfig.nTimesteps * N_AGENTS * ACTION_DIM);

        HostRNG hrng{&nd, &rng};

        for (int theta = 0; theta < N_TRUE_MODELS; theta++)
        {
            float trajCost = 0.0f;
            float decay = 1.0f;

            SimState predState = initState;

            BranchState bstate;
            initBranchState(bstate, mppiCont->h_belief.data(), mppiConfig.minConfidence);

            int stopTime = -1;
            EventType stopReason = EVT_TRUNCATED;

            std::vector<float> egoActions(mppiConfig.nTimesteps * ACTION_DIM);
            std::vector<SimState> states;

            writer.pushIntArray<int>(bstate.predTheta);

            int tOrigin = 0;

            for (int t = 0; t < mppiConfig.nTimesteps; t++)
            {
                int startInd = (flattenBranchIndex(bstate.predTheta.data()) * mppiConfig.nTimesteps + t - tOrigin) * ACTION_DIM;

                for (int d = 0; d < ACTION_DIM; d++)
                    egoActions[t * ACTION_DIM + d] = mppiCont->h_nominal[startInd + d];

                bool branched = false;

                TerminalType term = environmentStep<true, true, false, false>(t, theta, envConfig, egoActions.data() + t * ACTION_DIM,
                                                                              false, // no PID noise for reproducible display
                                                                              predState, bstate, branched, mppiConfig.minConfidence, buffer, hrng);

                if (branched)
                    tOrigin = t + 1;

                states.push_back(predState);

                trajCost += stateCost(predState, t, envConfig, mppiConfig, trueTheta) * decay;
                decay *= DECAY;

                if (term != TERM_NONE)
                {
                    states.insert(states.end(), mppiConfig.nTimesteps - t - 1, predState);

                    std::cout << "STOPPING SIMULATION at step " << t << " term " << (int)term << std::endl;

                    stopTime = t;
                    std::optional<EventType> parsed = parseTerm(term);
                    stopReason = *parsed;

                    break;
                }
            }

            trajCost += finalCost(predState, envConfig, mppiConfig, trueTheta);

            std::cout << "Final cost for theta = " << theta << ": " << trajCost << "\tfinal belief for theta = " << theta << ": ";
            for (int k = 0; k < N_TRUE_MODELS; k++)
                std::cout << bstate.belief[k] << " ";
            std::cout << "\tFinal branching time: ";
            for (int t : bstate.branchingTime)
                std::cout << t << ' ';
            std::cout << "\tFinal pred theta: ";
            for (int t : bstate.predTheta)
                std::cout << t << ' ';
            std::cout << "\n";

            writer.pushIntArray<int>(bstate.branchingTime);
            writer.pushIntArray<int>(bstate.predTheta);
            Env::pushPredictions(writer, states);
            writer.pushFloatArray(egoActions);
            writer.pushInt32(stopReason);
            writer.pushInt32(stopTime);
        }
    }

    else if (auto prmppiCont = dynamic_cast<PRMPPIController*>(controller.get()))
    {
        writer.pushFloatArray(prevState.pos);
        writer.pushFloatArray(prmppiCont->h_belief);

        writer.pushInt32((int)prmppiCont->useNomPlan);
        writer.pushInt32((int)prmppiCont->resetNom);

        // we do an additional run with last model and rob_nominal
        writer.pushInt32(N_TRUE_MODELS + 1);

        SimState initState = prevState;
        PRMPPIConfig prmppiConfig = prmppiCont->mppiConfig;

        HostRNG hrng{&nd, &rng};

        // one additional pass for rob_nominal on the last model

        for (int j = 0; j <= N_TRUE_MODELS; j++)
        {
            float cost = 0.0f;
            float safetyCost = -INFINITY;
            float decay = 1.0f;

            int theta = std::min(j, N_TRUE_MODELS - 1);

            SimState predState = initState;

            BranchState bstate;

            int stopTime = -1;
            EventType stopReason = EVT_TRUNCATED;

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

                TerminalType term = Env::environmentStep<false, false, false, false>(t, theta, envConfig, egoActions.data() + t * ACTION_DIM,
                                                                                     false, // no PID noise for reproducible display
                                                                                     predState, bstate, branched, 0.0f, buffer, hrng);

                safetyCost = fmaxf(safetyCost, PRMPPIsafetyCost(state, t, envConfig, prmppiConfig, theta));
                cost += PRMPPIstateCost(state, t, envConfig, prmppiConfig, theta) * decay;
                decay *= DECAY;

                states.push_back(predState);

                if (term != TERM_NONE)
                {
                    states.insert(states.end(), prmppiConfig.nTimesteps - t - 1, predState);

                    std::cout << "STOPPING SIMULATION at step " << t << " term " << (int)term << std::endl;
                    std::cout << "safety cost " << PRMPPIsafetyCost(state, t, envConfig, prmppiConfig, theta) << " isOutside "
                              << EnvStratRace::isOutside(state, envConfig, envConfig.droneRadius, theta) << " BD "
                              << EnvStratRace::trackBoundaryDist(state, envConfig, theta) << " bli " << envConfig.droneRadius << " * "
                              << prmppiConfig.collDistFactor << "\n";

                    stopTime = t;
                    std::optional<EventType> parsed = parseTerm(term);
                    stopReason = *parsed;

                    break;
                }
            }

            cost += PRMPPIfinalCost(state, envConfig, prmppiConfig, theta);
            float fullCost = cost + (safetyCost > 0.0f ? prmppiConfig.safetyWeight : 0.0f);

            std::cout << "for theta=" << theta << " controller " << (j == N_TRUE_MODELS ? "robust" : "nominal") << " PRMPPI full cost is " << fullCost
                      << "\t(performance cost " << cost << ")\tsafety cost " << safetyCost << "\n";

            // writer.pushFloatArray(fullPos);
            Env::pushPredictions(writer, states);

            writer.pushFloatArray(egoActions);
            writer.pushInt32(stopReason);
            writer.pushInt32(stopTime);
        }
    }

    else
        throw std::runtime_error("Ill-formed controller type in sendState");

    sock.send(zmq::buffer(writer.data));
}

void SimulationEngine::sendEvent(zmq::socket_t& sock, EventType type)
{
    Writer writer;

    writer.pushInt32(MSG_EVENT);
    writer.pushInt32(type);

    sock.send(zmq::buffer(writer.data));
}

void SimulationEngine::sendDone(zmq::socket_t& sock)
{
    Writer writer;

    writer.pushInt32(MSG_DONE);

    sock.send(zmq::buffer(writer.data));
}

// simulate one state for the given action. belief is updated in-place if given
std::optional<EventType> SimulationEngine::dynStep(const std::array<float, ACTION_DIM>& action, int t, std::array<float, N_TRUE_MODELS>& belief)
{
    static ScratchEnvBuffer buffer;

    HostRNG hrng{&nd, &rng};

    BranchState branchState;
    initBranchState(branchState, belief.data(),
                    2.0f); // here, we only care about belief, branching time/theta is unused anyway (it is recomputed by MPPI)

    bool branched = false;

    TerminalType term;

    if (auto mppiCont = dynamic_cast<MPPIController*>(controller.get()))
        // MPPI needs belief & branching update
        term = environmentStep<true, true, false, false>(t, trueTheta, envConfig, action.data(), true, state, branchState, branched,
                                                         mppiCont->mppiConfig.minConfidence, buffer, hrng);
    else
        // other controllers may only use belief update
        term = environmentStep<true, false, false, false>(t, trueTheta, envConfig, action.data(), true, state, branchState, branched, 0.0f, buffer,
                                                          hrng);

    std::copy(std::begin(branchState.belief), std::end(branchState.belief), belief.begin());

    return parseTerm(term);
}

void SimulationEngine::run(int maxSteps, zmq::socket_t& sock)
{
    std::array<float, ACTION_DIM> dummyAction{};
    sendState(sock, 0, dummyAction, state);

    int step;

    std::optional<EventType> stopInfo = std::nullopt;

    for (step = 1; step <= maxSteps; step++)
    {
        std::cout << "\nSTEP " << step << "\n";

        std::array<float, ACTION_DIM> action;
        controller->getControl(state, action.data());

        SimState prevState = state;

        stopInfo = dynStep(action, step, controller->h_belief);

        sendState(sock, step, action, prevState);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Current state at step " << step << ":\n";
        Env::printState(state);

        if (stopInfo)
        {
            std::cout << "Event " << *stopInfo << "\n";
            sendEvent(sock, *stopInfo);
            break;
        }

        if (step == maxSteps)
            sendEvent(sock, EVT_TRUNCATED);
    }

    sendDone(sock);

    std::cout << "Simulation finished after " << step << " steps.\nStop reason: ";

    if (!stopInfo)
        std::cout << "Simulation truncated (maxSteps reached)\n";
    else if (*stopInfo == EVT_OUTSIDE)
        std::cout << "Agent is outside\n";
    else if (*stopInfo == EVT_WINNER)
        std::cout << "Agent wins\n";
    else if (*stopInfo == EVT_OPP_WINNER)
        std::cout << "Opponent wins\n";
}

std::optional<EventType> SimulationEngine::parseTerm(TerminalType term)
{
    switch (term)
    {
    case TERM_NONE:
        return std::nullopt;
    case TERM_LOSE:
        return {EVT_OUTSIDE};
    case TERM_WIN:
        return {EVT_WINNER};
    case TERM_OPP_WIN:
        return {EVT_OPP_WINNER};
    default:
        throw std::runtime_error("Unexpected TerminalType value");
    }
}
