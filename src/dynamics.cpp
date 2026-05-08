#include "dynamics.h"
#include "state.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// ═════════════════════════════════════════════════════════════════════
// CPU track helpers
// ═════════════════════════════════════════════════════════════════════

// float cpuProjectOnTrack(const std::vector<float>& tp, int nSamples, int dim, const float* pos, float& dist)
// {
//     float bestS = 0.0f, bestD2 = 1e30f;
//     float sStep = 1.0f / nSamples;
//     for (int i = 0; i < nSamples; i++)
//     {
//         float d2 = 0.0f;
//         for (int d = 0; d < dim; d++)
//         {
//             float dx = tp[i * dim + d] - pos[d];
//             d2 += dx * dx;
//         }
//         if (d2 < bestD2) { bestD2 = d2; bestS = i * sStep; }
//     }
//     dist = std::sqrt(bestD2);

//     return bestS;
// }

// sample the centerline at the given s (linear interpolation)
std::vector<float> cpuSampleCenterline(const std::vector<float>& trackPoints, int nSamples, int dim, float s, int racelineIndex)
{
    std::vector<float> out(dim);

    s = s - std::floor(s);
    float idx_f = s * nSamples;
    int idx0 = (int) idx_f;
    int idx1 = (idx0 + 1) % nSamples;
    float t = idx_f - idx0;

    for (int d = 0; d < dim; d++)
        out[d] = (1.0f - t) * trackPoints[(nSamples * racelineIndex + idx0) * dim + d] + t * trackPoints[(nSamples * racelineIndex + idx1) * dim + d];

    return out;
}
