#pragma once

/*
Assumptions (ugly, but makes configuration somewhat simple and allows for more compile-time optimizations):
- uncertain obstacles are circles. there are always 2
- TODO: if UNCERTAIN_MODE is 0, then there are 2 possible theta values and either the first or second one is active. if
UNCERTAIN_MODE is 1, then there are 2 model factors, and each obstacle can be active or not
- all annulus obstacles are top-left quarter circles (usually, just one)
- there are 2 double-half-plane obstacles, configured by minx1, maxx1, lambda1top, lambda1bot and same for 2nd one.
    1st one is defined by minx1 <= x <= maxx1 AND (y - x > lambda1top OR -y - x > lambda1bot)
    2nd one is defined by minx2 <= x <= maxx2 AND (x + y > lambda2top OR x - y > lambda2bot)
- there is one rectangular obstacle
*/

namespace EnvHiddenObs
{
constexpr int DIM = 2;
constexpr int N_GATES = 3;
constexpr int N_RECT_OBSTACLES = 1;

constexpr int ACTION_DIM = DIM;
constexpr int N_MODEL_FACTORS = 1;

constexpr int N_UNCERTAIN_OBSTACLES = 2;

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

constexpr int N_TRUE_MODELS = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : MODEL_SIZE(0) * MODEL_SIZE(1);
constexpr int N_BRANCH_PLANS = N_MODEL_FACTORS == 1 ? BRANCH_SIZE(0) : BRANCH_SIZE(0) * BRANCH_SIZE(1);
constexpr int MAX_MODEL_SIZE = N_MODEL_FACTORS == 1 ? MODEL_SIZE(0) : (MODEL_SIZE(0) > MODEL_SIZE(1) ? MODEL_SIZE(0) : MODEL_SIZE(1));

// Environment configuration
// this should have the same layout as HiddenObsEnvironmentConfig in the Python side
struct EnvironmentConfig
{
    float dt;

    bool sendStates;

    float droneRadius;

    float posNoiseLevel;
    float speedNoiseLevel;
    float actionNoiseLevel;

    float maxSpeed;
    float maxAccel;

    // track
    int nWinLaps;
    float targetDistance;

    float gateCenters[N_GATES * DIM]; // (nGates * dim)
    float gateVectors[N_GATES * DIM]; // (nGates * dim)
    float gateRadius[N_GATES];        // (nGates)

    float arenaMin[DIM]; // (dim)
    float arenaMax[DIM]; // (dim)

    float rectObstacles[N_RECT_OBSTACLES * DIM * 2];     // (nRectObstacles * dim * 2): rectangle obstacles, i.e. [xmin, ymin, xmax, ymax]
    float hiddenObsCenters[N_UNCERTAIN_OBSTACLES * DIM]; // (nUncertainObstacles * dim): center of obstacles
    float hiddenObsRadius[N_UNCERTAIN_OBSTACLES];        // (nUncertainObstacles): radius of obstacles

    float dblHpLimits[4];  // (4): (minx1, maxx1, minx2, maxx2). see top of file
    float dblHpLambdas[4]; // (4): (lambda1top, lambda1bot, lambda2top, lambda2bot). see top of file

    float annulusCenter[DIM]; // (dim): center of annulus. annulus is always top-left slice (x < cx, y > cy)
    float annulusRadius[2];   // (2): inner & outer radius. annulus is always top-left slice (x < cx, y > cy)

    // TODO: move elsewhere (not useful in kernels)
    float initBelief[N_TRUE_MODELS];

    int trueTheta;
};

struct SimState
{
    float pos[DIM];
    float vel[DIM];

    int laps;
    int gates;
};

struct ScratchEnvBuffer
{
};
} // namespace EnvHiddenObs
