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

    auto [envConfig, seed] = Env::unpackEnvConfig(msg.data(), msg.size());

    res = sock.recv(msg);
    if (!res)
        throw std::runtime_error("Failed to receive MPPI configuration");

    AnyControllerConfig contConfig = loadControllerConfig(msg.data(), msg.size());

    res = sock.recv(msg);
    if (!res)
        throw std::runtime_error("Failed to receive initial state");

    SimState initSimState = Env::unpackSimState(msg.data(), msg.size(), envConfig);

    std::cout << "Header unpacked, starting simulation\n";

    SimulationEngine engine(envConfig, contConfig, initSimState, seed);

    auto begin = std::chrono::steady_clock::now();

    engine.run(max_steps, sock);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Simulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n\n";
}

int main(int argc, char** argv)
{
#ifdef DEBUG
    std::cout << "RUNNING IN DEBUG MODE. Warning: added synchronization will make code VERY slow" << std::endl;
#else
    std::cout << "RUNNING IN RELEASE MODE" << std::endl;
#endif
    std::cout << "EnvironmentConfig size: " << sizeof(EnvironmentConfig) << " bytes; MPPIConfig size: " << sizeof(MPPIConfig) << " bytes\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("\tMax threads per block:            %d\n", prop.maxThreadsPerBlock);
        printf("\tMax threads per multiprocessor:   %d\n", prop.maxThreadsPerMultiProcessor);
        printf("\tNumber of multiprocessors (SM):   %d\n", prop.multiProcessorCount);
        printf("\tTotal resident threads:           %d\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
        printf("\tTotal global memory:              %lu\n", prop.totalGlobalMem);
        printf("\tMaximum shared mem. / block:      %lu\n", prop.sharedMemPerBlock);
        printf("\tRegisters per block:              %d\n", prop.regsPerBlock);
        printf("\tWarp size:                        %d\n", prop.warpSize);
    }

    std::string zmqAddr = "tcp://*:5555";
    if (argc > 1) zmqAddr = argv[1];

    zmq::context_t ctx{ 1 };
    zmq::socket_t sock{ ctx, zmq::socket_type::pair };
    sock.set(zmq::sockopt::linger, 0);

    std::cout << "Binding ZMQ to addr " << zmqAddr << "...\n";
    sock.bind(zmqAddr);

    // while (1)
    runEngine(sock);

    sock.close();
    ctx.close();
}
