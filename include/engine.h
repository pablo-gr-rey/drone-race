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
        const EnvironmentConfig& cfg,
        std::vector<float> trackPoints,
        const VerifConfig& verifConfig,
        const std::vector<ControllerSpec>& specs,
        int seed);

    ~SimulationEngine();

    void sendState(zmq::socket_t& sock, int step);
    void sendEvent(zmq::socket_t& sock, EventType type, int info);
    void sendDone(zmq::socket_t& sock);

    // Run until termination or maxSteps. Publishes over ZMQ.
    void run(int maxSteps, zmq::socket_t& sock);

    // public config & track data
    EnvironmentConfig envConfig;
    std::vector<float> trackPoints;
    VerifConfig verifConfig;

    // current simulation state
    SimState state;

private:
    int nMppiCont = 0;  // count of MPPI controllers (used to send states)

    std::normal_distribution<float> nd{ 0.0f, 1.0f };

    float* d_trackPoints = nullptr;   // shared across all MPPI controllers
    int seed;

    // controllers
    std::vector<std::unique_ptr<Controller>> controllers;
    std::vector<std::string> controllerNames;

    std::mt19937 rng;

    // build actual controller from spec, return it + true iff controller is MPPI
    std::pair<std::unique_ptr<Controller>, bool> makeController(const ControllerSpec& sp);
    void allocTrack();

    std::optional<std::pair<EventType, int>> dynStep(const std::vector<float>& actions);

    std::optional<std::pair<EventType, int>> parseTerm(TerminalType term, int egoAgent);
};
 