#pragma once

#include "config.h"

// ── Find min cost (reduction) ────────────────────────────────────────
// Writes a single float to `outMin`.
__global__ void minReduceKernel(const float* __restrict__ costs,
    float* __restrict__ outMin,
    int n);

// ── Weighted average of noise ────────────────────────────────────────
// One block per (timestep * dim) entry.
// Updates nominalAction in-place: nominalAction[t*dim+d] += weightedAvg
__global__ void weightedAverageKernel(
    const float* __restrict__ costs,
    const float* __restrict__ noise,    // (nTimesteps, nSamples, dim)
    float* __restrict__ nominalAction,  // (nTimesteps, dim)
    float  minCost,
    float  invTemperature,
    int    nSamples,
    int    nTimesteps,
    int    dim);