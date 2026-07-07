#pragma once

#include <math.h>

#include "env_hiddenobs_defs.h"
#include "protocol.h"
#include "state.h"

namespace EnvHiddenObs
{
EnvironmentConfig allocDeviceMemory(const EnvironmentConfig& hostConfig);

void freeDeviceConfig(EnvironmentConfig& d_config);

void pushSimState(Writer& writer, const SimState& state);

// only push full pos
void pushPredictions(Writer& writer, const std::vector<SimState>& preds);

void printState(const SimState& state);

SimState unpackSimState(const void* buf, size_t len, const EnvironmentConfig& envConfig);

std::tuple<EnvironmentConfig, int, int> unpackEnvConfig(const void* buf, size_t len); // return seed, trueTheta

// square distance to rectangle (0 if inside)
HD INLINE float sqDistToRectangle(float x, float y, float xmin, float ymin, float xmax, float ymax)
{
    // dist to closest points
    float dx = x - fminf(fmaxf(x, xmin), xmax);
    float dy = y - fminf(fmaxf(y, ymin), ymax);

    return dx * dx + dy * dy;
}

// square distance from x,y to the region {xmin <= x <= xmax && x + y >= lambda}
HD INLINE float sqDistToHalfPlane(float x, float y, float xmin, float xmax, float lambda)
{
    if (xmin <= x && x <= xmax && x + y >= lambda)
        return 0.0f;

    // if we're outside, minDist is the min of distances to the 3 segments/half-lines on the boundary

    // projection on diagonal
    float projx = fmaxf(fminf((x - y + lambda) * 0.5f, xmax), xmin);
    float projy = lambda - projx;

    float sqMinDist = (x - projx) * (x - projx) + (y - projy) * (y - projy);

    // projection on the two boundary half-lines
    float y_left = fmaxf(y, lambda - xmin);
    sqMinDist = fminf(sqMinDist, (x - xmin) * (x - xmin) + (y - y_left) * (y - y_left));

    float y_right = fmaxf(y, lambda - xmax);
    sqMinDist = fminf(sqMinDist, (x - xmax) * (x - xmax) + (y - y_right) * (y - y_right));

    return sqMinDist;
}

// distance from x,y to the region {x <= xcenter, y >= ycenter, dist((x,y), center) <= rmin or >= rmax}. if we are outside the quadrant, return
// INFINITY (other primitives should handle it)
HD INLINE float distToAnnulus(float x, float y, float xcenter, float ycenter, float rmin, float rmax)
{
    // shift to center coordinates
    x -= xcenter;
    y -= ycenter;

    // distance to center
    float rSq = x * x + y * y;
    float rminSq = rmin * rmin, rmaxSq = rmax * rmax;

    // inside obstacle?
    if (x <= 0 && y >= 0 && (rSq <= rminSq || rSq >= rmaxSq))
        return 0.0f;

    // inside quadrant?
    if (x <= 0 && y >= 0)
    {
        float r = sqrtf(rSq);
        float d = fminf(r - rmin, rmax - r);
        return d;
    }

    return INFINITY;
}

HD INLINE float distToCircle(float x, float y, float xcenter, float ycenter, float radius)
{
    return sqrtf((x - xcenter) * (x - xcenter) + (y - ycenter) * (y - ycenter)) - radius;
}

// Boundary distance = distance of point to closest boundary or opponent (<= 0 if outside), does not take into account agent's radius
HD INLINE float trackBoundaryDist(const SimState& state, const EnvironmentConfig& envConfig, int trueTheta)
{
    float sqMinDist = INFINITY;
    float x = state.pos[0], y = state.pos[1];

    // min distance to rectangle
    for (int iRect = 0; iRect < N_RECT_OBSTACLES; iRect++)
        sqMinDist = fminf(sqMinDist, sqDistToRectangle(x, y, envConfig.rectObstacles[4 * iRect], envConfig.rectObstacles[4 * iRect + 1],
                                                       envConfig.rectObstacles[4 * iRect + 2], envConfig.rectObstacles[4 * iRect + 3]));

    // min distance to the half-planes
    // there are 2 double-half-plane obstacles, configured by minx1, maxx1, lambda1top, lambda1bot and same for 2nd one.
    // 1st one is defined by minx1 <= x <= maxx1 AND (y - x > lambda1top OR -y - x > lambda1bot)
    // 2nd one is defined by minx2 <= x <= maxx2 AND (x + y > lambda2top OR x - y > lambda2bot)

    sqMinDist =
        fminf(sqMinDist, fminf(fminf(sqDistToHalfPlane(-x, y, -envConfig.dblHpLimits[1], -envConfig.dblHpLimits[0], envConfig.dblHpLambdas[0]),
                                     sqDistToHalfPlane(-x, -y, -envConfig.dblHpLimits[1], -envConfig.dblHpLimits[0], envConfig.dblHpLambdas[1])),
                               fminf(sqDistToHalfPlane(x, y, envConfig.dblHpLimits[2], envConfig.dblHpLimits[3], envConfig.dblHpLambdas[2]),
                                     sqDistToHalfPlane(x, -y, envConfig.dblHpLimits[2], envConfig.dblHpLimits[3], envConfig.dblHpLambdas[3]))));

    float minDist = sqrtf(sqMinDist);

    minDist = fminf(minDist,
                    fminf(fminf(x - envConfig.arenaMin[0], envConfig.arenaMax[0] - x), fminf(y - envConfig.arenaMin[1], envConfig.arenaMax[1] - y)));

    minDist = fminf(
        minDist, distToAnnulus(x, y, envConfig.annulusCenter[0], envConfig.annulusCenter[1], envConfig.annulusRadius[0], envConfig.annulusRadius[1]));

    // TODO: update for 4-thetas mode

    minDist = fminf(minDist, distToCircle(x, y, envConfig.hiddenObsCenters[trueTheta * 2], envConfig.hiddenObsCenters[trueTheta * 2 + 1],
                                          envConfig.hiddenObsRadius[trueTheta]));

    return minDist;
}

// radius should be e.g. droneRadius in the actual dynamics and droneRadius*factor for MPPI
HD INLINE bool isOutside(const SimState& state, const EnvironmentConfig& envConfig, float radius, int trueTheta)
{
    return trackBoundaryDist(state, envConfig, trueTheta) < radius;
}

HD INLINE bool isWinner(const SimState& state, const EnvironmentConfig& envConfig)
{
    return state.laps >= envConfig.nWinLaps;
}

// return true if line from (px, py) to (qx, qy) intersects the rectangle
HD INLINE bool segmentIntersectsRectangle(float px, float py, float qx, float qy, float xmin, float ymin, float xmax, float ymax)
{
    // Liang-Barsky algorithm
    float dx = qx - px;
    float dy = qy - py;

    float tmin = 0.0f;
    float tmax = 1.0f;

    // x slabs
    if (fabsf(dx) < 1e-12f)
    {
        // Segment is vertical
        if (px < xmin || px > xmax)
            return false;
    }
    else
    {
        float t1 = (xmin - px) / dx;
        float t2 = (xmax - px) / dx;
        if (t1 > t2)
        {
            float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax)
            return false;
    }

    // y slabs
    if (fabsf(dy) < 1e-12f)
    {
        if (py < ymin || py > ymax)
            return false;
    }
    else
    {
        float t1 = (ymin - py) / dy;
        float t2 = (ymax - py) / dy;
        if (t1 > t2)
        {
            float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax)
            return false;
    }

    return true;
}

// return true if line from (px, py) to (qx, qy) intersects annulus of given center and radius in its part (x <= xcenter, y >= ycenter)
HD INLINE bool segmentIntersectsAnnulus(float px, float py, float qx, float qy, float xcenter, float ycenter, float rmin, float rmax)
{
    // shift to center-oriented coordinates
    px -= xcenter;
    py -= ycenter;
    qx -= xcenter;
    qy -= ycenter;

    // we parametrize the segment: x(t) = px + t(qx-px), y(t) = py + t(qy-py)

    // first, compute tmin, tmax such that the intersection of the segment with the quadrant (x <= 0, y >= 0) is [tmin, tmax]
    float tmin = 0.0f, tmax = 1.0f;

    // intersection with quadrant x <= 0
    if (fabsf(px - qx) > 1e-12f)
    {
        float threshold = px / (px - qx);
        if (qx > px)
            tmax = fminf(tmax, threshold);
        else
            tmin = fmaxf(tmin, threshold);
    }
    else if (px > 0)
        return false;

    if (tmax < tmin)
        return false;

    // intersection with quadrant y >= 0
    if (fabsf(py - qy) > 1e-12f)
    {
        float threshold = py / (py - qy);
        if (qy > py)
            tmin = fmaxf(tmin, threshold);
        else
            tmax = fminf(tmax, threshold);
    }
    else if (py > 0)
        return false;

    if (tmax < tmin)
        return false;

    // intersection with the inner arc
    // we compute t such that x^2(t) + y^2(t) is minimal, clamp it to the boundaries to get the min of the interval (since this function of t is
    // convex), and check if this value is less than rmin x^2(t) + y^2(t) = t^2 ((px-qx)^2 + (py-qy)^2) + 2t(px(qx-px) + py(qy-py)) + constant

    float sqDistp2q = (px - qx) * (px - qx) + (py - qy) * (py - qy);
    if (sqDistp2q > 1e-12f) // if p and q are close, then they can see each other (if they are both inside the obstacle, then the drone will be marked
                            // as outside the arena and the simulation stops anyway)
    {
        float t_mindist = (px * (px - qx) + py * (py - qy)) / sqDistp2q;
        t_mindist = fmaxf(tmin, fminf(tmax, t_mindist));

        float x_mindist = px + t_mindist * (qx - px);
        float y_mindist = py + t_mindist * (qy - py);
        if (x_mindist * x_mindist + y_mindist * y_mindist < rmin * rmin)
            return true;
    }

    // intersection with the outer arc
    // since x^2(t) + y^2(t) is convex, we can just check on the boundaries

    float xt = px + tmin * (qx - px);
    float yt = py + tmin * (qy - py);
    if (xt * xt + yt * yt > rmax * rmax)
        return true;

    xt = px + tmax * (qx - px);
    yt = py + tmax * (qy - py);
    return xt * xt + yt * yt > rmax * rmax;
}

HD INLINE bool canSeeHiddenObs(const SimState& state, const EnvironmentConfig& envConfig, int hiddenObsId)
{
    float px = state.pos[0], py = state.pos[1];
    float qx = envConfig.hiddenObsCenters[hiddenObsId * 2], qy = envConfig.hiddenObsCenters[hiddenObsId * 2 + 1];

    for (int iRect = 0; iRect < N_RECT_OBSTACLES; iRect++)
        if (segmentIntersectsRectangle(px, py, qx, qy, envConfig.rectObstacles[4 * iRect], envConfig.rectObstacles[4 * iRect + 1],
                                       envConfig.rectObstacles[4 * iRect + 2], envConfig.rectObstacles[4 * iRect + 3]))
            return false;

    if (segmentIntersectsAnnulus(px, py, qx, qy, envConfig.annulusCenter[0], envConfig.annulusCenter[1], envConfig.annulusRadius[0],
                                 envConfig.annulusRadius[1]))
        return false;

    return true;
}

// Advance = currentGate + nGates * laps + scale * (1 - normalizedDistToGate)// (it is much better to pass through a gate than to just be close to it)
// (this is a rough measure, it doesn't include speed for example)
HD INLINE float getAdvance(const SimState& state, const EnvironmentConfig& envConfig)
{
    float sqGateDist = 0.0f;     // sq dist between pos and next gate
    float sqConsGateDist = 0.0f; // sq dist between current gate and next gate
    int nextGate = (state.gates + 1) % N_GATES;

    float scale = 0.8f; // 1.0f means reward is continuous when going through a gate; 0.5f
                        // for example means that reward will be between 0.0 and 0.5
                        // before the first gate, 1 and 1.5 between 1st and 2nd, etc

    const float* __restrict__ prevGateCenter = envConfig.gateCenters.data() + state.gates * DIM;
    const float* __restrict__ nextGateCenter = envConfig.gateCenters.data() + nextGate * DIM;

    for (int d = 0; d < DIM; d++)
    {
        sqGateDist += (nextGateCenter[d] - state.pos[d]) * (nextGateCenter[d] - state.pos[d]);
        sqConsGateDist += (nextGateCenter[d] - prevGateCenter[d]) * (nextGateCenter[d] - prevGateCenter[d]);
    }

    return state.gates + N_GATES * state.laps + scale * (1.0f - sqrtf(sqGateDist / sqConsGateDist));
}

// for this environment, always return 0.0f
HD INLINE float getOppAdvance(const SimState& /* state */, const EnvironmentConfig& /* envConfig */)
{
    return 0.0f;
}

HD INLINE float getMaxAccel(const EnvironmentConfig& envConfig)
{
    return envConfig.maxAccel;
}

template <bool addMargin>
HD INLINE void updateGates(const EnvironmentConfig& envConfig, SimState& state, const float* __restrict__ prevPos, float margin = 1.0f)
{
    // check if we passed through next gate: compute lambda = dot(vec, center - x_t) / dot(vec, x_{t+1} - x_t)
    int nextGate = (state.gates + 1) % N_GATES;
    float num = 0.f, denom = 0.f;
    for (int d = 0; d < DIM; d++)
    {
        num += envConfig.gateVectors[nextGate * DIM + d] * (envConfig.gateCenters[nextGate * DIM + d] - prevPos[d]);
        denom += envConfig.gateVectors[nextGate * DIM + d] * (state.pos[d] - prevPos[d]);
    }

    // direction is inside the gate plan: cannot cross
    if (fabsf(denom) < 1e-10f)
        return;

    float lambda = num / denom;
    // we cross if 0 <= lambda <= 1 and if the projection of the segment (x_t, x_t+1) on the gate plan (ie. (1 - lambda) * x_t + lambda * x_t+1) is at
    // distance <= radius from the center if we want to make sure we cross the gate in the right direction, we have to check num >= 0 (<=> denom > 0).
    // Here, we allows passing through in both directions

    if (lambda < 0.f || lambda > 1.f)
        return;

    float sqDist = 0.f;
    for (int d = 0; d < DIM; d++)
    {
        float dx = (1.f - lambda) * prevPos[d] + lambda * state.pos[d] - envConfig.gateCenters[nextGate * DIM + d];
        sqDist += dx * dx;
    }

    float rad = envConfig.gateRadius[nextGate] * envConfig.gateRadius[nextGate];
    if constexpr (addMargin)
        rad *= margin * margin;

    if (sqDist <= rad)
    {
        state.gates++;
        if (state.gates == N_GATES)
        {
            state.gates = 0;
            state.laps++;
        }
    }
}

// compute env dynamics, belief update (only if shouldUpdateBelief) and branch update (only if considerBranching is true; if we become specialized,
// set corresponding branching time to t+1)
template <bool shouldUpdateBelief, bool considerBranching, bool addMargin, typename RNG>
HD INLINE TerminalType environmentStep(
    int t, int trueTheta, const EnvironmentConfig& envConfig,
    const float* __restrict__ egoAction, // (dim)
    bool applyNoise,                     // TODO: this could be a template (but probably doesn't matter if we're inlined anyway)
    SimState& state,
    BranchState&
        branchState,     // belief is updated in-place if shouldUpdateBelief. branchingTime and branchUsed are updated if considerBranching is true
    bool& branched,      // set to true if we branched at this step. must be previously initialized to false. unused if considerBranching is false
    float minConfidence, // for branching. unused if considerBranching is false
    ScratchEnvBuffer& scratch, RNG& rng, float gateMargin = 0.0f)
{
    // 1. Copy current pos (for gate update) and action
    for (int i = 0; i < DIM; i++)
    {
        scratch.prevPos[i] = state.pos[i];
        scratch.action[i] = egoAction[i];
    }

    // 2. Clamp action and add env action noise
    float sqNorm = 0.0f;
    for (int d = 0; d < DIM; d++)
        sqNorm += scratch.action[d] * scratch.action[d];

    float factor = 1.0f;
    float maxSq = envConfig.maxAccel * envConfig.maxAccel;
    if (sqNorm > maxSq)
        factor = envConfig.maxAccel / sqrtf(sqNorm);

    for (int d = 0; d < DIM; d++)
    {
        scratch.action[d] *= factor;

        if (envConfig.actionNoiseLevel != 0.0f && applyNoise)
            scratch.action[d] += sampleNormal(rng) * envConfig.actionNoiseLevel;
    }

    // 3. Integrate position
    for (int d = 0; d < DIM; d++)
        state.pos[d] += envConfig.dt * state.vel[d];

    // 4. Integrate speed
    for (int d = 0; d < DIM; d++)
        state.vel[d] += envConfig.dt * scratch.action[d];

    // 5. Cap speed
    float sqSpeed = 0.0f;
    for (int d = 0; d < DIM; d++)
        sqSpeed += state.vel[d] * state.vel[d];

    if (sqSpeed > envConfig.maxSpeed * envConfig.maxSpeed)
    {
        float sc = envConfig.maxSpeed / sqrtf(sqSpeed);
        for (int d = 0; d < DIM; d++)
            state.vel[d] *= sc;
    }

    // 7. State noise (pos + speed)
    if (applyNoise && (envConfig.posNoiseLevel != 0.0f || envConfig.speedNoiseLevel != 0.0f))
        for (int d = 0; d < DIM; d++)
        {
            if (envConfig.posNoiseLevel != 0.0f)
                state.pos[d] += sampleNormal(rng) * envConfig.posNoiseLevel;
            if (envConfig.speedNoiseLevel != 0.0f)
                state.vel[d] += sampleNormal(rng) * envConfig.speedNoiseLevel;
        }

    // 8. Update gates with prev pos
    updateGates<addMargin>(envConfig, state, scratch.prevPos.data(), gateMargin);

    // 9. Update belief & branching time
    if constexpr (shouldUpdateBelief)
    {
        // TODO: will change if 2 independant obstacles

        // if we are not sure yet, then update. here, since we know that 1 of them is blocked, if we can see either then we will know
        if (branchState.belief[0] < 0.9f && branchState.belief[1] < 0.9f &&
            (canSeeHiddenObs(state, envConfig, trueTheta) || canSeeHiddenObs(state, envConfig, 1 - trueTheta)))
        {
            branchState.belief[trueTheta] = 1.0f;
            branchState.belief[1 - trueTheta] = 0.0f;
        }
    }

    if constexpr (considerBranching)
    {
        int newPredTheta[N_MODEL_FACTORS];
        findConfident(branchState.belief.data(), minConfidence, newPredTheta);

        for (int k = 0; k < N_MODEL_FACTORS; k++)
            if (branchState.predTheta[k] == 0 && newPredTheta[k] != 0)
            {
                branchState.predTheta[k] = newPredTheta[k];
                branchState.branchingTime[k] = t + 1;
                branched = true;
            }
    }

    if (isOutside(state, envConfig, envConfig.droneRadius, trueTheta))
        return TERM_LOSE;

    if (isWinner(state, envConfig))
        return TERM_WIN;

    return TERM_NONE;
}
} // namespace EnvHiddenObs
