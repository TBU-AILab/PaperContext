import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
      - Latin-hypercube-like stratified sampling to seed points
      - Local coordinate search (pattern search) from best points
      - Occasional random restarts and step-size adaptation

    Returns: best (float) = best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clamp(x, i):
        if x < lo[i]:
            return lo[i]
        if x > hi[i]:
            return hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # Box-Muller normal without external libs
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def eval_point(x):
        # func expects an array-like; list is fine in typical setups
        v = func(x)
        # Guard against NaN / inf
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return float("inf")
        return float(v)

    # --- initial best ---
    best = float("inf")
    best_x = None

    # --- initial design: stratified sampling per dimension ---
    # Creates n points by permuting bins in each dimension (LHS-style).
    def lhs_sample(n):
        # For each dimension, create a random permutation of n strata
        perms = []
        for _ in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)

        pts = []
        for k in range(n):
            x = []
            for i in range(dim):
                # Sample uniformly within stratum perms[i][k]
                a = (perms[i][k] + random.random()) / n
                x.append(lo[i] + a * span[i])
            pts.append(x)
        return pts

    # Budget a little time for seeding
    # Choose n based on dim, but keep moderate
    n_seed = max(10, min(80, 10 * dim))
    for x in lhs_sample(n_seed):
        if time.time() >= deadline:
            return best
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, x

    # If somehow not evaluated:
    if best_x is None:
        x = rand_point()
        best = eval_point(x)
        best_x = x

    # --- local search parameters ---
    # Initial step as fraction of span; minimum step threshold
    step = [0.2 * s if s > 0 else 1.0 for s in span]
    min_step = [1e-8 * (s if s > 0 else 1.0) for s in span]

    # Keep a small pool of good points for restarts
    elite = [(best, best_x[:])]
    elite_max = 6

    # --- main loop ---
    while time.time() < deadline:
        # Select a start point: mostly best, sometimes another elite, sometimes random restart
        r = random.random()
        if r < 0.70:
            x = best_x[:]
            fx = best
        elif r < 0.90 and elite:
            fx, x = random.choice(elite)
            x = x[:]
        else:
            x = rand_point()
            fx = eval_point(x)

        # Local coordinate/pattern search from x
        improved_any = False
        # number of coordinate sweeps depends on dim but capped
        sweeps = min(10, 2 + dim // 2)

        for _ in range(sweeps):
            if time.time() >= deadline:
                return best
            improved = False

            # Randomize coordinate order
            coords = list(range(dim))
            random.shuffle(coords)

            for i in coords:
                if time.time() >= deadline:
                    return best

                si = step[i]
                if si <= min_step[i]:
                    continue

                # Try +/- step along coordinate i
                cand_best_x = None
                cand_best_fx = fx

                for direction in (-1.0, 1.0):
                    xi = x[:]
                    xi[i] = clamp(xi[i] + direction * si, i)
                    fxi = eval_point(xi)
                    if fxi < cand_best_fx:
                        cand_best_fx = fxi
                        cand_best_x = xi

                if cand_best_x is not None:
                    x, fx = cand_best_x, cand_best_fx
                    improved = True
                    improved_any = True

                    # Update global best and elite pool
                    if fx < best:
                        best, best_x = fx, x[:]
                        elite.append((fx, x[:]))
                        elite.sort(key=lambda t: t[0])
                        if len(elite) > elite_max:
                            elite = elite[:elite_max]

                else:
                    # No improvement along this coordinate: try a small Gaussian perturbation
                    # (helps escape small plateaus)
                    xi = x[:]
                    # small noise relative to step
                    xi[i] = clamp(xi[i] + 0.25 * si * randn(), i)
                    fxi = eval_point(xi)
                    if fxi < fx:
                        x, fx = xi, fxi
                        improved = True
                        improved_any = True
                        if fx < best:
                            best, best_x = fx, x[:]
                            elite.append((fx, x[:]))
                            elite.sort(key=lambda t: t[0])
                            if len(elite) > elite_max:
                                elite = elite[:elite_max]

            # Step-size adaptation per sweep
            if improved:
                # gently increase steps if progress is being made
                for i in range(dim):
                    step[i] = min(step[i] * 1.2, span[i] if span[i] > 0 else step[i] * 1.2)
            else:
                # shrink if stuck
                for i in range(dim):
                    step[i] *= 0.5

        # If no improvement for this attempt, do a broader random move from best
        if not improved_any and time.time() < deadline:
            # random "kick" around best_x scaled by span
            x = best_x[:]
            for i in range(dim):
                # 10% of span kick
                x[i] = clamp(x[i] + (random.random() * 2.0 - 1.0) * 0.1 * span[i], i)
            fx = eval_point(x)
            if fx < best:
                best, best_x = fx, x[:]
                elite.append((fx, x[:]))
                elite.sort(key=lambda t: t[0])
                if len(elite) > elite_max:
                    elite = elite[:elite_max]

        # If steps have become tiny everywhere, reset them (restart intensification)
        if all(step[i] <= min_step[i] for i in range(dim)):
            step = [0.2 * s if s > 0 else 1.0 for s in span]

    return best
