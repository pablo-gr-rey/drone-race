#pragma once

#include "config.h"
#include "controllers.h"

#include <memory>
#include <optional>
#include <random>

class SimulationEngine
{
  public:
    SimulationEngine(const EnvironmentConfig& config, const AnyControllerConfig& contConfig, const SimState& initSimState, int seed, int tTheta);

    ~SimulationEngine();

    void sendState(zmq::socket_t& sock, int step, const std::array<float, ACTION_DIM>& egoAction, const SimState& prevState);
    void sendEvent(zmq::socket_t& sock, EventType type);
    void sendDone(zmq::socket_t& sock);

    // Run until termination or maxSteps. Publishes over ZMQ.
    void run(int maxSteps, zmq::socket_t& sock);

    // public config & track data
    EnvironmentConfig envConfig;
    EnvironmentConfig d_envConfig;

    // current simulation state
    SimState state;

  private:
    std::normal_distribution<float> nd{0.0f, 1.0f};

    int seed;
    int trueTheta;
    CONTROLLER_KIND contKind;

    // controllers
    std::unique_ptr<Controller> controller;

    std::mt19937 rng;

    std::optional<EventType> dynStep(const std::array<float, ACTION_DIM>& action, int t, std::array<float, N_TRUE_MODELS>& belief);

    std::optional<EventType> parseTerm(TerminalType term);
};
