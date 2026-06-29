#pragma once

#include "config.h"
#include "controllers.h"
#include "state.h"

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <optional>

class SimulationEngine
{
public:
    SimulationEngine(
        const EnvironmentConfig& config,
        const AnyControllerConfig& contConfig,
        const SimState& initSimState,
        int seed);

    ~SimulationEngine();

    void sendState(zmq::socket_t& sock, int step, const std::array<float, ACTION_DIM>& egoAction, const SimState& prevState);
    void sendEvent(zmq::socket_t& sock, EventType type, int info);
    void sendDone(zmq::socket_t& sock);

    // Run until termination or maxSteps. Publishes over ZMQ.
    void run(int maxSteps, zmq::socket_t& sock);

    // public config & track data
    EnvironmentConfig envConfig;
    EnvironmentConfig d_envConfig;

    // current simulation state
    SimState state;

private:
    std::normal_distribution<float> nd{ 0.0f, 1.0f };

    int seed;
    CONTROLLER_KIND contKind;

    // controllers
    std::unique_ptr<Controller> controller;

    std::mt19937 rng;

    std::optional<std::pair<EventType, int>> dynStep(const std::array<float, ACTION_DIM>& action, int t, std::array<float, N_TRUE_MODELS>& belief);

    std::optional<std::pair<EventType, int>> parseTerm(TerminalType term, int egoAgent);
};
 