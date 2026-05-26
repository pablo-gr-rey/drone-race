#pragma once

#include "config.h"
#include "controllers.h"

#include <vector>
#include <memory>
#include <random>
#include <string>

class SimulationEngine
{
public:
    SimulationEngine(const EnvironmentConfig& cfg, std::vector<float> trackPoints, const VerifConfig& verifConfig, const std::vector<ControllerSpec>& specs, int seed);
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

private:
    std::normal_distribution<float> nd{ 0.0f, 1.0f };

    float* d_trackPoints;   // shared across all MPPI controllers
    int seed;

    // state
    std::vector<float> pos;   // (nAgents * dim) positions (x1 y1 .. x2 y2 ..)
    std::vector<float> speed; // (nAgents * dim) speeds  (vx1 vy1 .. vx2 vy2 ..)
    std::vector<float> currentS;    // (nAgents * nRacelines) current S relative to the given raceline, -1.0f means it is not used and should not be updated (since it's only useful for PID) 
    std::vector<int> nLaps;       // (nAgents) current number of laps
    std::vector<int> currentGates;  // (nAgents) current number of gates passed (1 = we already went through gates[0], now we aim at gates[1])

    // stop reason
    bool hasCollision;
    int anyOutside;      // -1 = none
    int anyWinner;       // -1 = none

    // controllers
    std::vector<std::unique_ptr<Controller>> controllers;
    std::vector<std::string> controllerNames;

    std::mt19937 rng;

    // build actual controller from spec, and updates currentS (-1.0f if controller is dummy or MPPI, since it won't use S)
    std::unique_ptr<Controller> makeController(const ControllerSpec& sp, int i);
    void allocTrack();

    void dynStep(const std::vector<float>& actions);

    // checks
    bool checkCollision() const;
    int  checkOutside()   const;  // -1 = none
    int  checkWinner()    const;  // -1 = none
};
