#include "config.h"
#include "engine.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <zmq.hpp>

void runEngine(zmq::socket_t& sock, std::optional<zmq::socket_t>& rosSock)
{
    int max_steps = 500;

    std::cout << "Waiting for track configuration header...\n";
    zmq::message_t msg;
    auto res = sock.recv(msg);
    if (!res)
        throw std::runtime_error("Track configuration zmq recv failed");

    auto [envConfig, seed, trueTheta] = Env::unpackEnvConfig(msg.data(), msg.size());

    res = sock.recv(msg);
    if (!res)
        throw std::runtime_error("Failed to receive MPPI configuration");

    AnyControllerConfig contConfig = loadControllerConfig(msg.data(), msg.size());

    res = sock.recv(msg);
    if (!res)
        throw std::runtime_error("Failed to receive initial state");

    SimState initSimState = Env::unpackSimState(msg.data(), msg.size(), envConfig);

    bool errorMode = false;

    if (rosSock)
    {
        std::cout << "Waiting for ROS handshake...\n";
        if (!rosSock->recv(msg))
            throw std::runtime_error("Failed to receive ROS handshake!");
        std::cout << "Sending ROS handshake...\n";

        Writer writer;
        writer.pushInt32(MSG_HEADER);
        if (!rosSock->send(zmq::buffer(writer.data)))
            throw std::runtime_error("Failed to send ROS handshake!");

        Reader reader(msg.data(), msg.size());
        if (reader.readInt32() != MSG_HEADER)
            throw std::runtime_error("Invalid ROS handshake");
        reader.assertFinished();

        std::cout << "Sending first state position...\n";
        writer = {};
        writer.pushInt32(MSG_HEADER);
        writer.pushFloatArray(initSimState.pos);

        if (!rosSock->send(zmq::buffer(writer.data)))
            throw std::runtime_error("Failed to send init state!");

        std::cout << "Waiting for ROS first state...\n";
        if (!rosSock->recv(msg))
            throw std::runtime_error("Failed to receive ROS first state!");

        auto [physSimState, step] = Env::unpackSimStateRaw(msg.data(), msg.size(), envConfig, initSimState);

        std::cout << "Received physical state:\n";
        Env::printState(physSimState);
        std::cout << "\nExpected state:\n";
        Env::printState(initSimState);

        if (step != 0)
        {
            std::cout << "ERROR: received first step " << step << ", expected 0\n. Sending emergency stop\n";
            errorMode = true;
        }

        initSimState = physSimState;
    }

    std::cout << "Header unpacked, starting simulation\n";

    SimulationEngine engine(envConfig, contConfig, initSimState, seed, trueTheta);

    if (errorMode)
        engine.sendEvent(*rosSock, EVT_EMERGENCY_STOP);

    auto begin = std::chrono::steady_clock::now();

    engine.run(max_steps, sock, rosSock);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Simulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n\n";
}

int main(int argc, char** argv)
{
#ifdef DEBUG
    std::cout << "RUNNING IN DEBUG MODE. Warning: added synchronization will "
                 "make code VERY slow"
              << std::endl;
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
    if (argc > 1)
        zmqAddr = argv[1];

    zmq::context_t ctx{1};
    zmq::socket_t sock{ctx, zmq::socket_type::pair};
    sock.set(zmq::sockopt::linger, 0);

    std::cout << "Binding ZMQ to Python frontend: addr " << zmqAddr << "...\n";
    sock.bind(zmqAddr);

    std::optional<zmq::context_t> ros_ctx = std::nullopt;
    std::optional<zmq::socket_t> ros_sock = std::nullopt;
    if (REAL_EXPERIMENT)
    {
        std::string rosAddr = "tcp://*:5556";
        if (argc > 2)
            rosAddr = argv[2];

        ros_ctx = zmq::context_t(1);
        ros_sock = zmq::socket_t(ctx, zmq::socket_type::pair);
        ros_sock->set(zmq::sockopt::linger, 0);

        std::cout << "Binding ZMQ to ROS on addr " << rosAddr << "...\n";
        ros_sock->bind(rosAddr);
    }
    else
        std::cout << "Running on pure simulation\n";

    // while (1)
    runEngine(sock, ros_sock);

    sock.close();
    ctx.close();

    if (ros_ctx)
    {
        ros_sock->close();
        ros_ctx->close();
    }
}
