#include "protocol.h"
#include <cstring>
#include <stdexcept>

Reader::Reader(const void* buffer, size_t s) : data((const uint8_t*)buffer), size(s)
{
}

int32_t Reader::readInt32()
{
    return readPod<int32_t>();
}
float Reader::readFloat()
{
    return readPod<float>();
}

void Reader::assertFinished()
{
    if (offset != size)
        throw std::runtime_error("Header not fully parsed");
}

template <typename T> void Writer::pushPod(const T& v)
{
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    data.insert(data.end(), p, p + sizeof(T));
}

void Writer::pushInt32(int32_t v)
{
    pushPod<int32_t>(v);
}

void Writer::pushFloat(float v)
{
    pushPod<float>(v);
}

void Writer::pushFloatArray(std::span<const float> arr)
{
    pushInt32(static_cast<int32_t>(arr.size()));
    if (!arr.empty())
    {
        const auto* p = reinterpret_cast<const uint8_t*>(arr.data());
        data.insert(data.end(), p, p + arr.size() * sizeof(float));
    }
}
