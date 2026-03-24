import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded global optimization (minimization) using:
      - Latin-hypercube-like initial sampling
      - Adaptive local search around current best (step-size annealing)
      - Occasional random restarts / global samples
      - Bound handling by clipping

    Returns:
        best (float): fitness value of the best found solution
    """

    # ----------------- helpers -----------------
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_vec(x):
        # func expects an "array-like"; we provide a plain list to keep it self-contained.
        return float(func(x))

    def make_lhs_like(n):
        # Simple LHS-style: per dimension, use one sample from each stratum then shuffle.
        # Produces n points in [0,1]^dim then scales to bounds.
        strata = []
        for _ in range(dim):
            s = [(i + random.random()) / n for i in range(n)]
            random.shuffle(s)
            strata.append(s)

        pts = []
        for k in range(n):
            x = []
            for d in range(dim):
                u = strata[d][k]
                lo, hi = bounds[d]
                x.append(lo + u * (hi - lo))
            pts.append(x)
        return pts

    # ----------------- initialization -----------------
    t0 = time.time()
    deadline = t0 + max_time

    # If bounds are degenerate, still handle.
    span = [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    best = float("inf")
    best_x = None

    # Initial design size depends on dimension and time budget (kept small & safe).
    init_n = max(10, min(60, 10 * dim))
    for x in make_lhs_like(init_n):
        if time.time() >= deadline:
            return best
        fx = eval_vec(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # Should not happen, but safe fallback.
        x = rand_vec()
        return eval_vec(x)

    # ----------------- main loop (adaptive local + global) -----------------
    # Initial step sizes: a fraction of each dimension range.
    step = [0.25 * s if s > 0 else 0.0 for s in span]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in span]

    it = 0
    no_improve = 0

    while time.time() < deadline:
        it += 1

        # Mix strategies: mostly local perturbations, sometimes global samples.
        # Probability of global move increases when stagnating.
        p_global = 0.05 + min(0.45, 0.01 * no_improve)
        do_global = (random.random() < p_global)

        if do_global:
            # Global exploration
            x = rand_vec()
        else:
            # Local exploration: Gaussian-like perturbation via sum of uniforms (no libs)
            x = best_x[:]
            for d in range(dim):
                if step[d] <= 0:
                    continue
                # Approximate N(0,1): Irwin–Hall (sum of 12 uniforms - 6)
                z = sum(random.random() for _ in range(12)) - 6.0
                x[d] = clip(x[d] + z * step[d], bounds[d][0], bounds[d][1])

            # Occasional coordinate search "kick"
            if dim > 0 and random.random() < 0.15:
                d = random.randrange(dim)
                direction = -1.0 if random.random() < 0.5 else 1.0
                x[d] = clip(best_x[d] + direction * step[d] * (0.5 + random.random()),
                            bounds[d][0], bounds[d][1])

        fx = eval_vec(x)

        if fx < best:
            best, best_x = fx, x
            no_improve = 0
            # Slightly expand steps on success (helps traverse valleys)
            for d in range(dim):
                step[d] = max(min_step[d], step[d] * 1.05)
        else:
            no_improve += 1
            # Anneal steps on failures
            if no_improve % (10 + dim) == 0:
                for d in range(dim):
                    step[d] = max(min_step[d], step[d] * 0.7)

        # Random restart if heavily stuck (but keep best)
        if no_improve > (80 + 10 * dim):
            no_improve = 0
            # Reset step sizes somewhat and jump near random point or best
            for d in range(dim):
                step[d] = max(min_step[d], 0.25 * span[d])
            if random.random() < 0.7:
                best_x = rand_vec()  # move search focus (best value is preserved in 'best')
            # else keep best_x and continue

    # return fitness of the best found solution
    return best
