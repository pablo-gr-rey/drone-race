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
        const VerifConfig& verifConfig,
        const MPPIConfig& mppiConfig,
        int seed);

    ~SimulationEngine();

    void sendState(zmq::socket_t& sock, int step);
    void sendEvent(zmq::socket_t& sock, EventType type, int info);
    void sendDone(zmq::socket_t& sock);

    // Run until termination or maxSteps. Publishes over ZMQ.
    void run(int maxSteps, zmq::socket_t& sock);

    // public config & track data
    EnvironmentConfig envConfig;
    VerifConfig verifConfig;

    // current simulation state
    SimState state;

private:
    std::normal_distribution<float> nd{ 0.0f, 1.0f };

    float* d_trackPoints = nullptr;   // sent to MPPI controller
    int seed;

    // controllers
    MPPIController mppiCont;

    std::mt19937 rng;

    void allocTrack();

    std::optional<std::pair<EventType, int>> dynStep(const std::array<float, DIM>& action, int t, std::array<float, N_TRUE_MODELS>& belief);

    std::optional<std::pair<EventType, int>> parseTerm(TerminalType term, int egoAgent);
};
 