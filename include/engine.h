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
    float getAdvance(int agent, const float* S, const float* laps) const;
    float closestBoundaryDist(int agent, const float* phys) const;

    // public config & track data
    EnvironmentConfig envConfig;

private:
    // state
    std::vector<float> phys_state;   // (nAgents * dim * 2) physical state (x1 vx1 y1 vy1 .. x2 vx2 ...)
    std::vector<float> currentS;    // (nAgents) current S 
    std::vector<float> nLaps;       // (nAgents) current number of laps

    // new state (for the computation)
    std::vector<float> new_phys_state;   // (nAgents * dim * 2) new physical state (x1 vx1 y1 vy1 .. x2 vx2 ...)
    std::vector<float> new_currentS;    // (nAgents) new current S 
    std::vector<float> new_nLaps;       // (nAgents) new current number of laps

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

    // stop reason
    bool hasCollision;
    int isOutside;      // -1 = none
    int isWinner;       // -1 = none
};
