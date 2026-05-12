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
    SimulationEngine(const EnvironmentConfig& cfg, const std::vector<ControllerSpec>& specs);

    void sendState(zmq::socket_t& sock, int step);
    void sendEvent(zmq::socket_t& sock, EventType type, int info);
    void sendDone(zmq::socket_t& sock);

    // Run until termination or maxSteps. Publishes over ZMQ.
    void run(int maxSteps, zmq::socket_t& sock);

    std::vector<float> getTarget(int agent, const float* S, int racelineIndex) const;

    // public config & track data
    EnvironmentConfig envConfig;

private:
    // state
    std::vector<float> pos;   // (nAgents * dim) positions (x1 y1 .. x2 y2 ..)
    std::vector<float> speed; // (nAgents * dim) speeds  (vx1 vy1 .. vx2 vy2 ..)
    std::vector<float> currentS;    // (nAgents) current S 
    std::vector<int> nLaps;       // (nAgents) current number of laps
    std::vector<int> currentGates;  // (nAgents) current number of gates passed (1 = we already went through gates[0], now we aim at gates[1])

    // stop reason
    bool hasCollision;
    int isOutside;      // -1 = none
    int isWinner;       // -1 = none

    // controllers
    std::vector<std::unique_ptr<Controller>> controllers;
    std::vector<std::string> controllerNames;

    std::mt19937 rng;

    // build actual controller from spec
    std::unique_ptr<Controller> makeController(const ControllerSpec& sp);

    void dynStep(const std::vector<float>& actions);

    // checks
    bool checkCollision() const;
    int  checkOutside()   const;  // -1 = none
    int  checkWinner()    const;  // -1 = none

    // track utils

    // return the centerline sampled at given s
    std::vector<float> cpuSampleCenterline(float s, int racelineIndex) const;

    // return the closest S and the distance to it for the given pos.
    std::pair<float, float> cpuProjectOnTrack(const std::vector<float>& pos, int iAgent) const;
};
