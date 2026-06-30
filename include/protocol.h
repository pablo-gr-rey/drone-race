#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <zmq.hpp>

enum MsgType : uint32_t
{
    MSG_HEADER = 0,
    MSG_STATE = 1,
    MSG_EVENT = 2,
    MSG_DONE = 3,
};

enum EventType : uint32_t
{
    EVT_COLLISION = 0,
    EVT_OUTSIDE = 1,
    EVT_WINNER = 2,
    EVT_TRUNCATED = 3,
};

// byte reader util
struct Reader
{
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t offset = 0;

    Reader(const void* buffer, size_t s);

    template <typename T> T readPod();

    int32_t readInt32();
    float readFloat();

    std::vector<float> readFloatArray();
    std::vector<int> readIntArray();

    int readFloatArray(float* arr); // returns length. assumes array is allocated and has the right size
    int readIntArray(int* arr);     // returns length. assumes array is allocated and has the right size

    void assertFinished();
};

// byte writer util
struct Writer
{
    std::vector<uint8_t> data;

    template <typename T> void pushPod(const T& v);

    void pushInt32(int32_t v);
    void pushFloat(float v);

    void pushFloatArray(std::span<const float> arr);

    template <typename T> void pushIntArray(std::span<const T> arr);

    const uint8_t* bytes() const { return data.data(); }
    size_t size() const { return data.size(); }
};

template <typename T> void Writer::pushIntArray(std::span<const T> arr)
{
    std::vector<float> n(arr.size());

    for (size_t i = 0; i < arr.size(); i++)
        n[i] = arr[i];

    pushFloatArray(n);
}
