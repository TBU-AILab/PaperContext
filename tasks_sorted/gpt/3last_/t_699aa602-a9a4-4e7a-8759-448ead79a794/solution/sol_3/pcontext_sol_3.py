import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-budgeted box-constrained minimizer.

    Improvements over previous versions:
      - Uses a strong global+local hybrid: Differential Evolution (DE) + periodic
        local refinement by coordinate/line search (pattern search).
      - Works without numpy / external libs; strict time checks.
      - Handles bound constraints robustly via reflection + clamp.
      - Maintains an archive of best points and injects them to speed convergence.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------- helpers ----------
    def now():
        return time.time()

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_bounds(x):
        # reflect into bounds (better than clamp for DE steps)
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                # reflect with period 2w
                y = (xi - a) % (2.0 * w)
                if y <= w:
                    xi = a + y
                else:
                    xi = b - (y - w)
                x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # cheap normal(0,1) for local refinement
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-300)
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare = z1
        _has_spare = True
        return z0

    # ---------- initialization / seeding ----------
    best = float("inf")
    best_x = None

    # Seed points: center + a few corners + randoms
    def try_update(x):
        nonlocal best, best_x
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    if now() >= deadline:
        return best

    # center
    x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    try_update(x0)

    # corners (limited)
    corner_bits = min(dim, 6)  # at most 64 corners
    max_corners = min(12, 1 << corner_bits)
    for mask in range(max_corners):
        if now() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if i < corner_bits:
                x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
            else:
                x[i] = 0.5 * (lo[i] + hi[i])
        try_update(x)

    # random seeds
    seed_count = 10 + 5 * dim
    for _ in range(seed_count):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------- Differential Evolution (DE) ----------
    # population sizing: moderate but not too big (time-limited)
    # keep it at least 12, scale mildly with dim.
    NP = max(12, min(40, 8 + 2 * dim))
    if dim >= 30:
        NP = max(16, min(60, 6 + 2 * int(math.sqrt(dim) * 3)))

    # build initial population with some bias to best_x
    pop = []
    fit = []

    # include best and a couple near-best perturbations
    if best_x is not None:
        pop.append(best_x[:])
        fit.append(best)

        for _ in range(min(3, NP - 1)):
            if now() >= deadline:
                return best
            x = best_x[:]
            for i in range(dim):
                x[i] += (2.0 * random.random() - 1.0) * 0.05 * span_safe[i]
            clamp_inplace(x)
            fit.append(try_update(x))
            pop.append(x)

    while len(pop) < NP:
        if now() >= deadline:
            return best
        x = rand_point()
        fx = try_update(x)
        pop.append(x)
        fit.append(fx)

    # archive of a few best indices to help mutation
    def best_indices(k):
        idx = list(range(NP))
        idx.sort(key=lambda i: fit[i])
        return idx[:k]

    # DE control parameters (adaptive-ish)
    F_base = 0.55
    CR_base = 0.85

    # local refinement: coordinate/pattern search around best every so often
    def local_refine(x_start, f_start, time_limit):
        # small coordinate search with shrinking step; stops early on time.
        x = x_start[:]
        fx = f_start
        # initial step sizes relative to span
        step = [0.12 * s for s in span_safe]
        min_step = [1e-12 * s for s in span_safe]

        # a few passes; each pass tries +/- on each dimension
        # keep it cheap
        passes = 2
        for _ in range(passes):
            improved_any = False
            for j in range(dim):
                if now() >= time_limit:
                    return x, fx

                sj = step[j]
                if sj <= min_step[j]:
                    continue

                # try + and -
                for direction in (1.0, -1.0):
                    if now() >= time_limit:
                        return x, fx
                    y = x[:]
                    y[j] += direction * sj
                    # reflect for smoother behavior near bounds
                    reflect_bounds(y)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved_any = True
                        break  # move to next coordinate
            # shrink steps if no progress
            if not improved_any:
                for j in range(dim):
                    step[j] *= 0.5
            else:
                for j in range(dim):
                    step[j] *= 0.9
        return x, fx

    # main DE loop
    gen = 0
    last_local = now()
    local_period = 0.12 * max_time  # do a local refine periodically
    local_period = max(0.15, min(local_period, 1.2))

    # stagnation tracking for occasional restart injection
    no_improve_gens = 0
    best_seen = best

    while now() < deadline:
        gen += 1

        # Slightly randomize parameters each generation (jitter helps)
        F = max(0.15, min(0.95, F_base + 0.15 * (random.random() - 0.5)))
        CR = max(0.05, min(0.98, CR_base + 0.20 * (random.random() - 0.5)))

        elite = best_indices(max(2, NP // 6))

        for i in range(NP):
            if now() >= deadline:
                return best

            # choose distinct indices for mutation
            # "current-to-best/1" with elite best target: v = xi + F*(x_best-xi) + F*(xr1-xr2)
            a = i
            # pick a "best" target from elite
            b = elite[random.randrange(len(elite))]

            r1 = random.randrange(NP)
            while r1 == i or r1 == b:
                r1 = random.randrange(NP)
            r2 = random.randrange(NP)
            while r2 == i or r2 == b or r2 == r1:
                r2 = random.randrange(NP)

            xi = pop[i]
            xbest = pop[b]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # binomial crossover
            jrand = random.randrange(dim)
            trial = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = xi[j] + F * (xbest[j] - xi[j]) + F * (xr1[j] - xr2[j])
                else:
                    trial[j] = xi[j]

            reflect_bounds(trial)
            ftrial = evaluate(trial)

            # selection
            if ftrial <= fit[i]:
                pop[i] = trial
                fit[i] = ftrial
                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

        # stagnation / diversification
        if best < best_seen - 1e-15 * (1.0 + abs(best_seen)):
            best_seen = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens > (10 + dim // 2):
            # inject a few fresh points and some perturbed best to escape
            no_improve_gens = 0
            inject = max(1, NP // 6)
            for _ in range(inject):
                if now() >= deadline:
                    return best
                idx = random.randrange(NP)
                if best_x is not None and random.random() < 0.6:
                    x = best_x[:]
                    for j in range(dim):
                        x[j] += 0.15 * span_safe[j] * randn()
                    reflect_bounds(x)
                else:
                    x = rand_point()
                fx = evaluate(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

        # periodic local refinement around best (cheap, but high impact)
        if best_x is not None and (now() - last_local) >= local_period and now() < deadline:
            last_local = now()
            # only spend a small fraction of remaining time
            remain = deadline - now()
            tl = now() + min(0.08 * max_time, 0.25 * remain, 0.6)
            xr, fr = local_refine(best_x, best, tl)
            if fr < best:
                best = fr
                best_x = xr[:]
                # also push refined best into the population replacing worst
                worst = max(range(NP), key=lambda k: fit[k])
                pop[worst] = xr[:]
                fit[worst] = fr

    return best
