import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (self-contained, no external libs).

    Improvements vs prior version:
      - Switch to a compact Differential Evolution (DE/rand/1/bin) global search
        with time-adaptive parameters (more exploration early, more exploitation late).
      - Strong local refinement on the current best using a (1+1)-ES style Gaussian search
        with the 1/5 success rule (adaptive step-size).
      - Occasional coordinate-wise line search (coarse-to-fine) to quickly exploit separable structure.
      - Robust boundary handling via reflection (better than clamping for local moves).
      - Keeps everything within the time limit; returns best fitness found.

    Returns:
        best (float): best (minimum) function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    if max_time <= 0 or dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # Fix possibly swapped bounds; handle degenerate dimensions
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = highs[i] - lows[i]
        if spans[i] == 0.0:
            lows[i] = highs[i] = (lows[i] + highs[i]) * 0.5

    # ---------- RNG (fast, deterministic-ish) ----------
    rng_state = random.getrandbits(64) ^ (int(time.time() * 1e9) & ((1 << 64) - 1))
    def u01():
        nonlocal rng_state
        rng_state = (6364136223846793005 * rng_state + 1442695040888963407) & ((1 << 64) - 1)
        return ((rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def randn():
        # Box-Muller
        a = max(1e-300, u01())
        b = u01()
        return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)

    # ---------- Helpers ----------
    def reflect(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # Reflect repeatedly until inside [lo, hi]
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        # numerical safety
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def make_random_point():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] == 0.0:
                x[i] = lows[i]
            else:
                x[i] = lows[i] + u01() * spans[i]
        return x

    def eval_f(x):
        return float(func(list(x)))

    # ---------- Initialize population ----------
    # Keep pop modest for speed; scale mildly with dimension.
    pop_size = max(12, min(60, 10 + 5 * int(math.sqrt(dim))))
    pop = [make_random_point() for _ in range(pop_size)]
    fit = [eval_f(x) for x in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # ---------- Local refinement: (1+1)-ES with 1/5 success rule ----------
    # step sizes in absolute units
    es_sigma = [0.2 * s if s > 0 else 0.0 for s in spans]
    es_sigma_min = [1e-12 * (s if s > 0 else 1.0) for s in spans]
    es_sigma_max = [0.5 * s if s > 0 else 0.0 for s in spans]

    es_success = 0
    es_trials = 0

    def es_step(x, fx, intensity=1.0):
        nonlocal es_success, es_trials
        y = x[:]
        for i in range(dim):
            if spans[i] == 0.0:
                continue
            step = intensity * es_sigma[i] * randn()
            y[i] = reflect(y[i] + step, i)
        fy = eval_f(y)
        es_trials += 1
        if fy < fx:
            es_success += 1
            return y, fy, True
        return x, fx, False

    def es_adapt():
        nonlocal es_success, es_trials
        if es_trials < 20:
            return
        rate = es_success / float(es_trials)
        # 1/5 success rule
        if rate > 0.2:
            mult = 1.25
        else:
            mult = 0.82
        for i in range(dim):
            if spans[i] == 0.0:
                continue
            es_sigma[i] = max(es_sigma_min[i], min(es_sigma_max[i], es_sigma[i] * mult))
        es_success = 0
        es_trials = 0

    # ---------- Coordinate line search (coarse-to-fine) ----------
    def coord_search(x, fx, budget_steps=2):
        # Try a couple of coarse-to-fine passes on a random subset of coords
        coords = list(range(dim))
        # shuffle
        for j in range(dim - 1, 0, -1):
            r = int(u01() * (j + 1))
            coords[j], coords[r] = coords[r], coords[j]

        # limit number of coords touched for big dim
        touch = dim if dim <= 16 else max(8, int(0.4 * dim))
        coords = coords[:touch]

        curx, curf = x[:], fx
        for i in coords:
            if time.time() >= deadline:
                break
            if spans[i] == 0.0:
                continue

            # start from a step based on current ES sigma (more informed)
            step0 = max(1e-12, min(0.25 * spans[i], max(es_sigma[i], 0.01 * spans[i])))
            step = step0

            # a few refinement steps
            for _ in range(budget_steps):
                xp = curx[:]; xm = curx[:]
                xp[i] = reflect(xp[i] + step, i)
                xm[i] = reflect(xm[i] - step, i)
                fp = eval_f(xp)
                fm = eval_f(xm)

                if fp < curf or fm < curf:
                    if fp <= fm:
                        curx, curf = xp, fp
                    else:
                        curx, curf = xm, fm
                    step *= 1.6
                else:
                    step *= 0.5

                if step < 1e-15 * (spans[i] if spans[i] > 0 else 1.0):
                    break
        return curx, curf

    # ---------- Main loop: DE global + ES local ----------
    gen = 0
    while time.time() < deadline:
        gen += 1
        now = time.time()
        t = (now - t0) / max(1e-9, (deadline - t0))
        if t > 1.0:
            break

        # Time-adaptive DE parameters:
        #  - early: larger mutation, higher CR
        #  - late: smaller mutation, moderate CR
        F = 0.9 - 0.5 * t          # 0.9 -> 0.4
        CR = 0.95 - 0.35 * t       # 0.95 -> 0.60

        # A couple of local improvements per generation (focus on best)
        # More local near the end.
        local_steps = 1 if t < 0.5 else 2
        for _ in range(local_steps):
            if time.time() >= deadline:
                return best
            best_x, best, _ = es_step(best_x, best, intensity=1.0)
            es_adapt()

        # Occasional coordinate search, especially mid/late
        if time.time() < deadline and (t > 0.35) and (u01() < 0.15 + 0.25 * t):
            bx, bf = coord_search(best_x, best, budget_steps=2 if dim <= 20 else 1)
            if bf < best:
                best_x, best = bx, bf

        # DE iteration over population
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # Choose r1, r2, r3 distinct and != i
            r1 = i
            while r1 == i:
                r1 = int(u01() * pop_size)
            r2 = r1
            while r2 == i or r2 == r1:
                r2 = int(u01() * pop_size)
            r3 = r2
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = int(u01() * pop_size)

            x1, x2, x3 = pop[r1], pop[r2], pop[r3]

            # Mutation: v = x1 + F*(x2-x3)
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = x1[d] + F * (x2[d] - x3[d])

            # Binomial crossover with guarantee at least one mutant dimension
            jrand = int(u01() * dim)
            u = pop[i][:]  # trial
            for d in range(dim):
                if d == jrand or u01() < CR:
                    u[d] = reflect(v[d], d)

            fu = eval_f(u)

            # Selection
            if fu < fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # Mild restart/diversification if stuck: replace worst with jittered best
        if u01() < 0.20:
            worst_idx = max(range(pop_size), key=lambda k: fit[k])
            y = best_x[:]
            # jitter scale decreases over time
            jitter = (0.25 * (1.0 - t) + 0.02)  # fraction of span
            for d in range(dim):
                if spans[d] == 0.0:
                    continue
                y[d] = reflect(y[d] + (u01() * 2.0 - 1.0) * jitter * spans[d], d)
            fy = eval_f(y)
            pop[worst_idx] = y
            fit[worst_idx] = fy
            if fy < best:
                best, best_x = fy, y[:]

    return best
