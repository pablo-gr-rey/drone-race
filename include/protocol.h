#pragma once

#include <cstdint>
#include <vector>
#include <zmq.hpp>

/*
 * ZMQ message protocol (all little-endian floats/ints):
 *
 * ── HEADER (sent once at start) ───────────────────────────
 *   uint32_t  msgType     = 0  (HEADER)
 *   uint32_t  nAgents
 *   uint32_t  dim
 *   uint32_t  nTrackSamples
 *   float     trackWidth
 *   float     minDist
 *   float     dt
 *   int32_t   nLaps
 *   float[]   trackPoints     // nTrackSamples * dim floats
 *   char[]    controllerNames // nAgents null-terminated strings, back-to-back
 *
 * ── STATE (sent every step) ───────────────────────────────
 *   uint32_t  msgType     = 1  (STATE)
 *   uint32_t  stepIndex
 *   float[]   physState       // nAgents * dim * 2 floats
 *   float[]   S               // nAgents floats
 *   float[]   laps            // nAgents floats
 *
 * ── EVENT (sent on termination) ───────────────────────────
 *   uint32_t  msgType     = 2  (EVENT)
 *   uint32_t  eventType       // 0=collision, 1=outside(agent), 2=winner(agent), 3=truncated
 *   int32_t   agentId         // -1 if N/A
 *
 * ── DONE (sent after EVENT, signals no more data) ─────────
 *   uint32_t  msgType     = 3  (DONE)
 */

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

enum ControllerTypeProto : uint32_t
{
    CONT_DUMMY = 0,
    CONT_PID = 1,
    CONT_MPPI = 2,
};

// byte reader util
struct Reader
{
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t offset = 0;

    Reader(const void* buffer, size_t s);

    template <typename T>
    T readPod();

    int32_t readInt32();
    float readFloat();

    std::vector<float> readFloatArray();

    void assertFinished();
};

// byte writer util
struct Writer
{
    std::vector<uint8_t> data;

    template <typename T>
    void pushPod(const T& v);

    void pushInt32(int32_t v);
    void pushFloat(float v);
    void pushFloatArray(const std::vector<float>& arr);

    const uint8_t* bytes() const { return data.data(); }
    size_t size() const { return data.size(); }
};
