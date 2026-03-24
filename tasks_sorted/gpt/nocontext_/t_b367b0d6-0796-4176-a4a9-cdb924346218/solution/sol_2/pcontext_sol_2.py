import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid strategy:
    - Latin-hypercube-like seeding (stratified random per dimension)
    - Local improvement with adaptive step (pattern search / hill-climb)
    - Occasional random restarts to escape local minima

    Returns:
        best (float): best (minimum) fitness found within max_time seconds
    """

    # ------------------------ helpers ------------------------
    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def sample_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_point(x):
        # func expects an "array-like"; list is acceptable per prompt
        try:
            return float(func(x))
        except TypeError:
            # some funcs might require tuple
            return float(func(tuple(x)))

    def make_stratified_points(n):
        # Simple LHS-style: for each dimension, create n strata, shuffle, pick one per point
        per_dim = []
        for i in range(dim):
            lo, hi = bounds[i]
            if hi == lo:
                per_dim.append([lo] * n)
                continue
            step = (hi - lo) / float(n)
            strata = [lo + (k + random.random()) * step for k in range(n)]
            random.shuffle(strata)
            per_dim.append(strata)
        pts = []
        for k in range(n):
            pts.append([per_dim[i][k] for i in range(dim)])
        return pts

    # ------------------------ initialization ------------------------
    start = time.time()
    deadline = start + max_time

    best = float("inf")
    best_x = None

    # Try to spend a small part of time on stratified exploration
    # Choose n such that evaluation cost is unknown; keep it modest
    n_seed = max(8, min(64, 8 * dim))
    seed_points = make_stratified_points(n_seed)

    for x in seed_points:
        if time.time() >= deadline:
            return best
        f = eval_point(x)
        if f < best:
            best, best_x = f, x[:]

    # Fallback if something weird happened
    if best_x is None:
        best_x = sample_uniform()
        best = eval_point(best_x)

    # Initial step sizes relative to range
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_steps = [r * 0.2 if r > 0 else 0.0 for r in ranges]
    min_steps = [max(1e-12, r * 1e-6) if r > 0 else 0.0 for r in ranges]

    # ------------------------ main loop ------------------------
    # Strategy:
    # - Perform local search around a "current" point with coordinate perturbations.
    # - If stuck, shrink steps; if very stuck, restart from a new random point.
    current = best_x[:]
    current_f = best

    steps = base_steps[:]
    no_improve = 0
    restart_after = 40 + 10 * dim  # heuristic

    while time.time() < deadline:
        improved = False

        # Coordinate pattern search: try +/- step in each dimension (random order)
        order = list(range(dim))
        random.shuffle(order)

        for i in order:
            if time.time() >= deadline:
                return best
            if steps[i] <= 0:
                continue

            lo, hi = bounds[i]

            # Try positive move
            cand = current[:]
            cand[i] = clip(cand[i] + steps[i], lo, hi)
            if cand[i] != current[i]:
                f = eval_point(cand)
                if f < current_f:
                    current, current_f = cand, f
                    improved = True
                    if f < best:
                        best, best_x = f, cand[:]
                    continue  # continue exploring from improved point

            # Try negative move
            cand = current[:]
            cand[i] = clip(cand[i] - steps[i], lo, hi)
            if cand[i] != current[i]:
                f = eval_point(cand)
                if f < current_f:
                    current, current_f = cand, f
                    improved = True
                    if f < best:
                        best, best_x = f, cand[:]
                    continue

        if improved:
            no_improve = 0
            # Mildly increase steps on success (adaptive)
            for i in range(dim):
                steps[i] = steps[i] * 1.05
        else:
            no_improve += 1
            # Shrink steps when not improving
            for i in range(dim):
                steps[i] = max(min_steps[i], steps[i] * 0.5)

        # If steps are tiny or we're stuck for long: random restart
        tiny = True
        for i in range(dim):
            if steps[i] > min_steps[i] * 10:
                tiny = False
                break

        if tiny or no_improve >= restart_after:
            # Random restart + slight bias toward best_x (mixing)
            x_rand = sample_uniform()
            mix = 0.7  # bias towards best
            current = [
                clip(mix * best_x[i] + (1.0 - mix) * x_rand[i], bounds[i][0], bounds[i][1])
                for i in range(dim)
            ]
            current_f = eval_point(current)
            if current_f < best:
                best, best_x = current_f, current[:]

            # Reset steps
            steps = base_steps[:]
            no_improve = 0

        # Occasional pure random sample to keep global exploration alive
        if time.time() < deadline and random.random() < 0.05:
            x = sample_uniform()
            f = eval_point(x)
            if f < best:
                best, best_x = f, x[:]
                current, current_f = x[:], f
                steps = base_steps[:]
                no_improve = 0

    return best
