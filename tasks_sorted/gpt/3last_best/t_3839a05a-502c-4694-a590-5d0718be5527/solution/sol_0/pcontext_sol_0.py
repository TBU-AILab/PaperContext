import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - initial Latin-hypercube-like stratified sampling
      - local coordinate search around the incumbent
      - adaptive step-size (shrinks on stagnation, occasional expansion)
      - random restarts to escape local minima

    Returns:
        best (float): fitness of the best found solution
    """
    t0 = time.time()

    def time_left():
        return (time.time() - t0) < max_time

    # --- basic helpers (no numpy) ---
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clamp(x):
        # clamp to bounds
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    def evaluate(x):
        # func expects an "array-like"; list is acceptable per template/example.
        return float(func(x))

    # --- initialization: stratified sampling per dimension (cheap LHS variant) ---
    # number of initial samples: modest, depends on dimension
    init_n = max(10, min(60, 8 * dim))
    # prepare strata permutations for each dimension
    perms = []
    for i in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    best_x = None
    best = float("inf")

    for s in range(init_n):
        if not time_left():
            return best
        x = []
        for i in range(dim):
            # sample uniformly inside stratum
            u = (perms[i][s] + random.random()) / init_n
            x.append(lo[i] + u * span[i])
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # in case max_time is extremely small
        return best

    # --- main loop: coordinate pattern search with adaptive step and restarts ---
    # initial step sizes: fraction of range
    step = [0.15 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]

    # stagnation / restart control
    no_improve = 0
    best_seen_for_restart = best
    restart_after = 40 + 10 * dim  # attempts without improvement before restart

    # exploration probability (small, to add occasional jumps)
    p_jump = 0.08

    while time_left():
        improved = False

        # occasional global jump around best (helps escape local basins)
        if random.random() < p_jump:
            x = best_x[:]
            for i in range(dim):
                # gaussian-like perturbation using sum of uniforms (approx normal)
                z = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random() - 3.0)
                x[i] += z * step[i]
            clamp(x)
            fx = evaluate(x)
            if fx < best:
                best = fx
                best_x = x
                improved = True

        # coordinate search: try +/- along dimensions in random order
        order = list(range(dim))
        random.shuffle(order)
        for i in order:
            if not time_left():
                return best

            if step[i] <= min_step[i]:
                continue

            # try positive direction
            x1 = best_x[:]
            x1[i] += step[i]
            clamp(x1)
            f1 = evaluate(x1)
            if f1 < best:
                best = f1
                best_x = x1
                improved = True
                continue  # keep moving with updated incumbent

            if not time_left():
                return best

            # try negative direction
            x2 = best_x[:]
            x2[i] -= step[i]
            clamp(x2)
            f2 = evaluate(x2)
            if f2 < best:
                best = f2
                best_x = x2
                improved = True
                continue

        if improved:
            no_improve = 0
            # very mild expansion (can speed up if we are making progress)
            for i in range(dim):
                step[i] *= 1.05
                # cap step to range
                if step[i] > span[i]:
                    step[i] = span[i]
        else:
            no_improve += 1
            # shrink steps on no progress
            for i in range(dim):
                step[i] *= 0.5
                if step[i] < min_step[i]:
                    step[i] = min_step[i]

        # restart if stagnating: sample new points biased around best and globally
        if no_improve >= restart_after:
            no_improve = 0

            # If we haven't improved at all since last restart, broaden search
            if best >= best_seen_for_restart - 1e-15:
                # global restart point
                cand = rand_point()
            else:
                # local-biased restart around current best
                cand = best_x[:]
                for i in range(dim):
                    # perturb within a decent fraction of range
                    cand[i] += (random.random() * 2.0 - 1.0) * (0.25 * span[i])
                clamp(cand)

            best_seen_for_restart = best

            if time_left():
                fc = evaluate(cand)
                if fc < best:
                    best = fc
                    best_x = cand

            # reset step sizes moderately
            step = [0.15 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    # return fitness of the best found solution
    return best
