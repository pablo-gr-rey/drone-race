#include "config.h"
#include "engine.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <zmq.hpp>

void runEngine(zmq::socket_t& sock, std::optional<zmq::socket_t>& rosStateSock, std::optional<zmq::socket_t>& rosAccSock)
{
    int max_steps = REAL_EXPERIMENT ? 10000 : 500;

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

    if (rosStateSock)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // make sure bridge has time to subscribe

        for (int i = 0; i < 2; i++)
        {
            std::cout << "Waiting for ROS handshake (" << (i + 1) << "/2)...\n";
            if (!rosStateSock->recv(msg))
                throw std::runtime_error("Failed to receive ROS handshake!");

            Reader reader(msg.data(), msg.size());
            if (reader.readInt32() != MSG_HEADER)
                throw std::runtime_error("Invalid ROS handshake");
            reader.assertFinished();

            std::cout << "Sending ROS handshake...\n";

            Writer writer;
            writer.pushInt32(MSG_HEADER);
            if (!rosAccSock->send(zmq::buffer(writer.data)))
                throw std::runtime_error("Failed to send ROS handshake!");
        }

        std::cout << "Sending first state position...\n";
        Writer writer;
        writer.pushInt32(MSG_HEADER);
        writer.pushFloatArray(initSimState.pos);

        if (!rosAccSock->send(zmq::buffer(writer.data)))
            throw std::runtime_error("Failed to send init state!");

        std::cout << "Waiting for ROS first state...\n";
        if (!rosStateSock->recv(msg))
            throw std::runtime_error("Failed to receive ROS first state!");

        SimState physSimState = Env::unpackSimStateRaw(msg.data(), msg.size(), envConfig, initSimState);

        std::cout << "Received physical state:\n";
        Env::printState(physSimState);
        std::cout << "\nExpected state:\n";
        Env::printState(initSimState);

        initSimState = physSimState;
    }

    std::cout << "Header unpacked, starting simulation\n";

    SimulationEngine engine(envConfig, contConfig, initSimState, seed, trueTheta);

    if (errorMode)
        engine.sendEvent(*rosAccSock, EVT_EMERGENCY_STOP);

    auto begin = std::chrono::steady_clock::now();

    engine.run(max_steps, sock, rosStateSock, rosAccSock);

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

#ifndef USE_ENV_STRATRACE
    if constexpr (REAL_EXPERIMENT)
        throw std::runtime_error("ROS bridge only implemented on env stratrace");
#endif

    std::string zmqAddr = "tcp://*:5555";
    if (argc > 1)
        zmqAddr = argv[1];

    zmq::context_t ctx{1};
    zmq::socket_t sock{ctx, zmq::socket_type::pair};
    sock.set(zmq::sockopt::linger, 0);

    std::cout << "Binding ZMQ to Python frontend: addr " << zmqAddr << "...\n";
    sock.bind(zmqAddr);

    std::optional<zmq::socket_t> ros_state_sock = std::nullopt;
    std::optional<zmq::socket_t> ros_acc_sock = std::nullopt;
    if (REAL_EXPERIMENT)
    {
        std::string rosStateAddr = "tcp://*:5556", rosAccAddr = "tcp://*:5557";
        if (argc > 3)
        {
            rosStateAddr = argv[2];
            rosAccAddr = argv[3];
        }

        ros_state_sock = zmq::socket_t(ctx, zmq::socket_type::sub);
        ros_acc_sock = zmq::socket_t(ctx, zmq::socket_type::pub);

        ros_state_sock->set(zmq::sockopt::conflate, 1); // only keep latest state
        ros_state_sock->set(zmq::sockopt::subscribe, "");

        std::cout << "Binding ZMQ to ROS on addrs " << rosStateAddr << " and " << rosAccAddr << "...\n";
        ros_state_sock->bind(rosStateAddr);
        ros_acc_sock->bind(rosAccAddr);
    }
    else
        std::cout << "Running on pure simulation\n";

    // while (1)
    runEngine(sock, ros_state_sock, ros_acc_sock);

    sock.set(zmq::sockopt::linger, 0);
    sock.close();

    if (ros_state_sock)
    {
        ros_state_sock->set(zmq::sockopt::linger, 0);
        ros_acc_sock->set(zmq::sockopt::linger, 0);
        ros_state_sock->close();
        ros_acc_sock->close();
    }

    ctx.close();
}
