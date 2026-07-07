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
    EVT_OUTSIDE = 0,
    EVT_WINNER = 1,
    EVT_TRUNCATED = 2,
};

// byte reader util
struct Reader
{
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t offset = 0;

    Reader(const void* buffer, size_t s);

    template <typename T> T readPod()
    {
        if (offset + sizeof(T) > size)
            throw std::runtime_error("failed to read scalar from buffer (underflow)");

        T v;
        std::memcpy(&v, data + offset, sizeof(T));
        offset += sizeof(T);
        return v;
    }

    int32_t readInt32();
    float readFloat();

    // necessary to be able to call readArray(arr); with float arr[2] (otherwise, it cannot be automatically converted to std::span)
    template <typename T, size_t N> size_t readArray(T (&arr)[N])
    {
        return readArray(std::span<T, N>(arr));
    }

    template <typename T, size_t S> size_t readArray(std::span<T, S> arr)
    {
        size_t length = readInt32();

        if (length != arr.size())
            throw std::runtime_error(std::format("Wrong size when reading array: read length {}, allocated length {}", length, arr.size()));

        for (size_t i = 0; i < length; i++)
            arr[i] = readPod<T>();

        return length;
    }

    template <typename T> std::vector<T> readArray()
    {
        size_t length = readInt32();

        std::vector<T> arr(length);

        for (size_t i = 0; i < length; i++)
            arr[i] = readPod<T>();

        return arr;
    }

    template <typename T> size_t allocReadArray(T*& arr, size_t length)
    {
        size_t r_length = readInt32();
        if (r_length != length)
            throw std::runtime_error(std::format("Failed to alloc & read array: expected length {}, read length {}", length, r_length));

        arr = (T*)malloc(length * sizeof(T));

        for (size_t i = 0; i < length; i++)
            arr[i] = readPod<T>();

        return length;
    }

    // TODO: this should be removed, readArray is enough
    size_t readFloatArray(std::span<float> arr)
    {
        return readArray(arr);
    }

    size_t readIntArray(std::span<int> arr)
    {
        return readArray(arr);
    }

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

    const uint8_t* bytes() const
    {
        return data.data();
    }
    size_t size() const
    {
        return data.size();
    }
};

template <typename T> void Writer::pushIntArray(std::span<const T> arr)
{
    std::vector<float> n(arr.size());

    for (size_t i = 0; i < arr.size(); i++)
        n[i] = arr[i];

    pushFloatArray(n);
}
