#include "config.h"
#include "engine.h"
#include <cmath>
#include <cstdio>
#include <zmq.hpp>
#include <vector>
#include <chrono>
#include <iostream>

void runEngine(zmq::socket_t& sock)
{
    int max_steps = 500;

    std::cout << "Waiting for track configuration header...\n";
    zmq::message_t msg;
    auto res = sock.recv(msg);
    if (!res)
        throw std::runtime_error("Track configuration zmq recv failed");

    EnvironmentConfig envConfig;
    envConfig.unpackHeader(msg.data(), msg.size());

    std::cout << "Header unpacked, starting simulation\n";

    std::vector<ControllerSpec> specs(envConfig.nAgents);
    for (int i = 0; i < envConfig.nAgents; i++)
    {
        res = sock.recv(msg);
        if (!res)
            throw std::runtime_error("Failed to receive agent configuration");

        specs[i].unpackHeader(msg.data(), msg.size());
    }

    SimulationEngine engine(envConfig, specs);

    auto begin = std::chrono::steady_clock::now();

    engine.run(max_steps, sock);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Simulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n\n";
}

int main(int argc, char** argv)
{
    std::string zmqAddr = "tcp://*:5555";
    if (argc > 1) zmqAddr = argv[1];

    zmq::context_t ctx{ 1 };
    zmq::socket_t sock{ ctx, zmq::socket_type::pair };

    std::cout << "Binding ZMQ to addr " << zmqAddr << "...\n";
    sock.bind(zmqAddr);

    while (1)
        runEngine(sock);
}
