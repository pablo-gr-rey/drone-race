#pragma once

#include "config.h"
#include <random>

// return the closest S and the distance to it for the given pos. allows specifying a stride on reading (useful for phys state)
// float cpuProjectOnTrack(const std::vector<float>& trackPoints, int nSamples, int dim, const float* pos, float& dist);
template <class PosIt>
std::pair<float, float> cpuProjectOnTrack(const std::vector<float>& trackPoints, int nTrackSamples, int dim, PosIt posBegin, int posStride = 1)
{
    float bestS = 0.0f, bestdSq = 1e30f;
    const float sStep = 1.0f / nTrackSamples;

    int iTrack = 0;

    for (int i = 0; i < nTrackSamples; ++i)
    {
        float dSq = 0.0f;

        for (int d = 0; d < dim; ++d)
        {
            float dx = trackPoints[iTrack] - *(posBegin + d * posStride);
            dSq += dx * dx;
            iTrack++;
        }

        if (dSq < bestdSq) { bestdSq = dSq; bestS = i * sStep; }
    }

    return std::make_pair(bestS, std::sqrt(bestdSq));
}

// return the centerline sampled at given s
std::vector<float> cpuSampleCenterline(const std::vector<float>& trackPoints, int nSamples, int dim, float s, int racelineIndex);
