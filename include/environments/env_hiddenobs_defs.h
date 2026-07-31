#pragma once

#include "common.h"

#include <cuda/std/array>

/*
Assumptions (ugly, but makes configuration somewhat simple and allows for more compile-time optimizations):
- uncertain obstacles are circles
- TODO: if UNCERTAIN_MODE is 0, then there are 2 possible theta values and either the first or second one is active. if
UNCERTAIN_MODE is 1, then there are 2 model factors, and each obstacle can be active or not
- there is one annulus obstacles, top-left quarter circles. distance to it is ill-defined outside of its quarter, so it must be properly bounded by
other primitives
- there are 2 double-half-plane obstacles, configured by minx1, maxx1, lambda1top, lambda1bot and same for 2nd one.
    1st one is defined by minx1 <= x <= maxx1 AND (y - x > lambda1top OR -y - x > lambda1bot)
    2nd one is defined by minx2 <= x <= maxx2 AND (x + y > lambda2top OR x - y > lambda2bot)

- the drone can see the hidden obstacles iff it can see its center (TODO?)
- the hidden obstacles are outside of the curved corridor quadrant
- the line of sight can only be blocked by the rectangular obstacles, the outer arc of the annulus or the inner arc of the annulus
*/

namespace EnvHiddenObs
{
inline constexpr int DIM = 2;
inline constexpr int N_GATES = 3;
inline constexpr int N_RECT_OBSTACLES = 3;

inline constexpr int ACTION_DIM = DIM;
inline constexpr int N_MODEL_FACTORS = 1;

inline constexpr int N_UNCERTAIN_OBSTACLES = 2;

// CUDA does not like constexpr arrays, so we use constexpr inline functions

HD INLINE constexpr int MODEL_SIZE(int /* k */)
{
    // for several models with different sizes, tests are fine: return (k == 0) ? 2 : 3 or a switch, but here we can just fold it
    return 2;
}

HD INLINE constexpr int BRANCH_SIZE(int /* k */)
{
    // same comment as above
    return 3;
}

inline constexpr int N_TRUE_MODELS = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : MODEL_SIZE(0) * MODEL_SIZE(1);
inline constexpr int N_BRANCH_PLANS = N_MODEL_FACTORS == 1 ? BRANCH_SIZE(0) : BRANCH_SIZE(0) * BRANCH_SIZE(1);
inline constexpr int MAX_MODEL_SIZE = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : (MODEL_SIZE(0) > MODEL_SIZE(1) ? MODEL_SIZE(0) : MODEL_SIZE(1));

// Environment configuration
// this should have the same layout as HiddenObsEnvironmentConfig in the Python side
struct EnvironmentConfig
{
    cuda::std::array<float, DIM> arenaMin; // (dim)
    cuda::std::array<float, DIM> arenaMax; // (dim)

    float dt;

    bool sendStates;

    float droneRadius;

    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed;
    float maxAccel;

    int nWinLaps;

    cuda::std::array<float, N_GATES * DIM> gateCenters; // (nGates * dim)
    cuda::std::array<float, N_GATES * DIM> gateVectors; // (nGates * dim)
    cuda::std::array<float, N_GATES> gateRadius;        // (nGates)

    cuda::std::array<float, N_RECT_OBSTACLES * DIM * 2>
        rectObstacles; // (nRectObstacles * dim * 2): rectangle obstacles, i.e. [xmin, ymin, xmax, ymax]
    cuda::std::array<float, N_UNCERTAIN_OBSTACLES * DIM> hiddenObsCenters; // (nUncertainObstacles * dim): center of obstacles
    cuda::std::array<float, N_UNCERTAIN_OBSTACLES> hiddenObsRadius;        // (nUncertainObstacles): radius of obstacles

    cuda::std::array<float, 4> dblHpLimits;  // (4): (minx1, maxx1, minx2, maxx2). see top of file
    cuda::std::array<float, 4> dblHpLambdas; // (4): (lambda1top, lambda1bot, lambda2top, lambda2bot). see top of file

    cuda::std::array<float, DIM> annulusCenter; // (dim): center of annulus. annulus is always top-left slice (x < cx, y > cy)
    cuda::std::array<float, 2> annulusRadius;   // (2): inner & outer radius. annulus is always top-left slice (x < cx, y > cy)

    // TODO: move elsewhere (not useful in kernels)
    cuda::std::array<float, N_TRUE_MODELS> initBelief;
};

struct SimState
{
    cuda::std::array<float, DIM> pos;
    cuda::std::array<float, DIM> vel;

    int laps;
    int gates;
};

struct ScratchEnvBuffer
{
    cuda::std::array<float, DIM> prevPos;
    cuda::std::array<float, DIM> action;
};
} // namespace EnvHiddenObs
