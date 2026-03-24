import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      1) Latin-hypercube-like initialization (stratified random per dimension)
      2) Adaptive local search around the incumbent (coordinate + random directions)
      3) Occasional global exploration restarts
      4) Step-size adaptation based on success rate
    Requires no external libraries.

    func: callable(params_list)->float
    dim: int
    bounds: list of (low, high) pairs, length=dim
    max_time: seconds (int/float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # Guard against degenerate bounds
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    def clip(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_uniform():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # Standard normal using Box-Muller (no external libs)
    have_spare = [False]
    spare = [0.0]
    def randn():
        if have_spare[0]:
            have_spare[0] = False
            return spare[0]
        u1 = 1e-12 + random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        spare[0] = z1
        have_spare[0] = True
        return z0

    def random_unit_vector():
        v = [randn() for _ in range(dim)]
        norm = math.sqrt(sum(vi * vi for vi in v)) or 1.0
        return [vi / norm for vi in v]

    # --- initialization: stratified samples per dimension (LHS-style) ---
    # Pick a modest initial budget based on dimension; keep it time-safe.
    init_n = max(8, min(60, 10 * dim))
    # Create per-dimension strata permutations
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    best_x = None
    best = float("inf")

    # Evaluate initial design quickly
    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # sample uniformly within stratum
            a = perms[i][j]
            u = (a + random.random()) / init_n
            x.append(lows[i] + u * spans[i])
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # fallback (shouldn't happen)
        best_x = rand_uniform()
        best = evaluate(best_x)

    # --- adaptive local search ---
    # step sizes (per-dimension) start at 20% of span, but not zero
    steps = [0.2 * s if s > 0 else 0.0 for s in spans]
    min_steps = [1e-12 + 1e-9 * (s if s > 0 else 1.0) for s in spans]
    max_steps = [0.5 * s if s > 0 else 0.0 for s in spans]

    # parameters
    shrink = 0.82
    grow = 1.18
    stall_limit = 30 + 5 * dim
    global_restart_every = 60 + 10 * dim

    it = 0
    stall = 0

    # Maintain a small pool of good points to restart from
    pool = [(best, best_x[:])]
    pool_max = 8

    while time.time() < deadline:
        it += 1

        # Decide exploration mode
        do_restart = (it % global_restart_every == 0) or (stall >= stall_limit)

        if do_restart:
            stall = 0
            # 50%: restart near one of the pool elites, 50%: fully random
            if pool and random.random() < 0.5:
                _, base = random.choice(pool)
                x0 = base[:]
                # moderate perturbation
                udir = random_unit_vector()
                scale = 0.3 + 0.7 * random.random()
                x = [x0[i] + udir[i] * scale * (spans[i] * 0.25) for i in range(dim)]
                x = clip(x)
            else:
                x = rand_uniform()

            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x
                pool.append((best, best_x[:]))
                pool.sort(key=lambda t: t[0])
                pool = pool[:pool_max]
            continue

        x0 = best_x

        improved = False

        # Mix of coordinate moves and random-direction moves
        # Coordinate probing
        order = list(range(dim))
        random.shuffle(order)

        for idx in order:
            if time.time() >= deadline:
                return best

            if spans[idx] == 0:
                continue

            s = steps[idx]
            if s <= 0:
                continue

            # Try +s and -s
            cand1 = x0[:]
            cand1[idx] = cand1[idx] + s
            cand1 = clip(cand1)
            f1 = evaluate(cand1)

            cand2 = x0[:]
            cand2[idx] = cand2[idx] - s
            cand2 = clip(cand2)
            f2 = evaluate(cand2)

            if f1 < best or f2 < best:
                if f1 <= f2:
                    best, best_x = f1, cand1
                else:
                    best, best_x = f2, cand2
                improved = True

                # enlarge step a bit on success
                steps[idx] = min(max_steps[idx], steps[idx] * grow)
                # update pool
                pool.append((best, best_x[:]))
                pool.sort(key=lambda t: t[0])
                pool = pool[:pool_max]
                x0 = best_x  # continue from new point
            else:
                # shrink step on failure
                steps[idx] = max(min_steps[idx], steps[idx] * shrink)

        # Random-direction move (helps in rotated landscapes)
        if time.time() < deadline:
            udir = random_unit_vector()
            # overall scale based on average step
            avg_step = sum(steps) / max(1, dim)
            alpha = (0.5 + random.random()) * avg_step

            cand = [x0[i] + alpha * udir[i] for i in range(dim)]
            cand = clip(cand)
            fc = evaluate(cand)
            if fc < best:
                best, best_x = fc, cand
                improved = True
                # mild expansion of all steps when random move succeeds
                for i in range(dim):
                    steps[i] = min(max_steps[i], steps[i] * 1.05)
                pool.append((best, best_x[:]))
                pool.sort(key=lambda t: t[0])
                pool = pool[:pool_max]
            else:
                # mild contraction when it fails
                for i in range(dim):
                    steps[i] = max(min_steps[i], steps[i] * 0.99)

        stall = 0 if improved else (stall + 1)

    return best
