#include "protocol.h"
#include <stdexcept>
#include <cstring>
#include <cmath>

Reader::Reader(const void* buffer, size_t s) : data((const uint8_t*) buffer), size(s) {}

template <typename T>
T Reader::readPod()
{
    if (offset + sizeof(T) > size)
        throw std::runtime_error("failed to read scalar from buffer (underflow)");
    T v;
    std::memcpy(&v, data + offset, sizeof(T));
    offset += sizeof(T);
    return v;
}

int32_t Reader::readInt32() { return readPod<int32_t>(); }
float Reader::readFloat() { return readPod<float>(); }

std::vector<float> Reader::readFloatArray()
{
    int32_t n = readInt32();
    if (n < 0)
        throw std::runtime_error("read negative size for array");

    size_t nBytes = size_t(n) * sizeof(float);
    if (offset + nBytes > size)
        throw std::runtime_error("failed to read array from buffer (underflow)");

    std::vector<float> out((const float*) (data + offset), (const float*) (data + offset) + n);
    offset += nBytes;

    return out;
}

std::vector<int> Reader::readIntArray()
{
    std::vector<float> arr = readFloatArray();
    std::vector<int> ans(arr.size());

    for (int i = 0; i < arr.size(); i++)
        ans[i] = std::round(arr[i]);

    return ans;
}

void Reader::assertFinished()
{
    if (offset != size)
        throw std::runtime_error("Header not fully parsed");
}

template <typename T>
void Writer::pushPod(const T& v)
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

void Writer::pushFloatArray(const std::vector<float>& arr)
{
    pushInt32(static_cast<int32_t>(arr.size()));
    if (!arr.empty())
    {
        const auto* p = reinterpret_cast<const uint8_t*>(arr.data());
        data.insert(data.end(), p, p + arr.size() * sizeof(float));
    }
}

void Writer::pushIntArray(const std::vector<int>& arr)
{
    std::vector<float> n(arr.size());

    for (int i = 0; i < arr.size(); i++)
        n[i] = arr[i];

    pushFloatArray(n);
}
