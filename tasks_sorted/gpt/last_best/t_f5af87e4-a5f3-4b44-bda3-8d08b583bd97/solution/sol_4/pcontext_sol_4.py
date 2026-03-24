import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (stdlib only).

    Main improvements:
      - Better global coverage: Halton + Latin-like stratified random + opposition + center
      - Differential Evolution (DE/rand-to-best/1/bin) as the main global optimizer
      - Lightweight local refinement on the current best: adaptive coordinate/pattern search
      - Stagnation handling: partial reinitialization + parameter auto-tuning
    Returns: best (float)
    """
    t0 = time.perf_counter()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def clamp_inplace(x):
        for i in range(dim):
            v = x[i]
            lo = lows[i]; hi = highs[i]
            if v < lo:
                x[i] = lo
            elif v > hi:
                x[i] = hi

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def center_point():
        return [0.5 * (lows[i] + highs[i]) for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # approx N(0,1) quickly
    def randn():
        return (random.random() + random.random() + random.random() + random.random() - 2.0)

    # ---------------- Halton (quasi-random) ----------------
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
        109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173
    ]

    def radical_inverse(n, base):
        inv = 1.0 / base
        f = inv
        r = 0.0
        while n > 0:
            n, mod = divmod(n, base)
            r += mod * f
            f *= inv
        return r

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            base = primes[i % len(primes)]
            u = radical_inverse(index, base)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- Initialization: diverse population ----------------
    # Population size: moderate (time-bounded). DE needs a pool.
    pop_size = max(18, min(90, 10 + 6 * dim))

    pop = []
    fit = []

    best = float("inf")
    best_x = None

    def push(x, fx):
        nonlocal best, best_x
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = list(x)

    # Always try center and its opposite (often helps on bounded tasks)
    if time.perf_counter() < deadline:
        x = center_point()
        fx = eval_f(x)
        push(list(x), fx)
    if time.perf_counter() < deadline:
        xo = opposite_point(center_point())
        clamp_inplace(xo)
        push(list(xo), eval_f(xo))

    # Fill remaining with a mix: Halton, stratified, random, and opposition pairs.
    # Stratified: for each dimension, sample from one of pop_size bins to reduce clustering.
    halton_start = 1 + random.randrange(1, 10000)
    bins = pop_size

    k = 0
    while len(pop) < pop_size and time.perf_counter() < deadline:
        r = random.random()
        if r < 0.45:
            x = halton_point(halton_start + k)
        elif r < 0.80:
            # stratified-like point
            x = [0.0] * dim
            for i in range(dim):
                if spans[i] == 0.0:
                    x[i] = lows[i]
                    continue
                b = (k + 13 * i) % bins
                u = (b + random.random()) / float(bins)
                x[i] = lows[i] + u * spans[i]
        else:
            x = rand_point()

        k += 1
        fx = eval_f(x)
        push(list(x), fx)

        if len(pop) < pop_size and time.perf_counter() < deadline and random.random() < 0.75:
            xo = opposite_point(x)
            clamp_inplace(xo)
            fxo = eval_f(xo)
            push(list(xo), fxo)

    if not pop:
        return float("inf")

    # ---------------- DE parameters (self-tuned ranges) ----------------
    # We'll randomize F and CR per trial (jDE-ish) for robustness.
    F_min, F_max = 0.35, 0.95
    CR_min, CR_max = 0.05, 0.95

    # "rand-to-best/1" mix factor controls how strongly we pull toward best.
    # Start modest; increase if stagnating to exploit.
    pull = 0.35

    # Stagnation tracking
    last_best = best
    last_improve_time = time.perf_counter()
    stall_seconds = max(0.15, 0.20 * float(max_time))  # after this, diversify

    # ---------------- Local refinement (pattern / coord search) ----------------
    def local_refine(x_start, fx_start, budget_evals=25):
        """
        Cheap local search: coordinate/pattern steps with shrinking radius.
        Good near the end and after DE finds a basin.
        """
        nonlocal best, best_x
        x = list(x_start)
        fx = fx_start
        # initial step: a few percent of span, bounded
        step = 0.06
        evals = 0
        # random coordinate order each pass
        while evals < budget_evals and time.perf_counter() < deadline:
            improved = False
            coords = list(range(dim))
            random.shuffle(coords)
            for i in coords:
                if evals >= budget_evals or time.perf_counter() >= deadline:
                    break
                if spans[i] == 0.0:
                    continue
                delta = step * safe_spans[i]

                xp = list(x); xp[i] += delta
                clamp_inplace(xp)
                f1 = eval_f(xp); evals += 1
                if f1 < fx:
                    x, fx = xp, f1
                    improved = True
                    if f1 < best:
                        best, best_x = f1, list(x)
                    continue

                xm = list(x); xm[i] -= delta
                clamp_inplace(xm)
                f2 = eval_f(xm); evals += 1
                if f2 < fx:
                    x, fx = xm, f2
                    improved = True
                    if f2 < best:
                        best, best_x = f2, list(x)

            if not improved:
                step *= 0.55
                if step < 1e-7:
                    break
        return x, fx

    # ---------------- Main DE loop ----------------
    idxs = list(range(len(pop)))

    # For mutation we need at least 4 distinct individuals; if not, pad with randoms
    while len(pop) < 4 and time.perf_counter() < deadline:
        x = rand_point()
        push(x, eval_f(x))

    # Main evolution
    gen = 0
    while time.perf_counter() < deadline:
        gen += 1

        # mild schedule: more exploitation near the end
        remaining = deadline - time.perf_counter()
        frac_left = remaining / max(1e-12, float(max_time))
        if frac_left < 0.25:
            pull = 0.55
        elif frac_left < 0.10:
            pull = 0.70

        # Iterate individuals
        n = len(pop)
        if n < 4:
            break
        for i in range(n):
            if time.perf_counter() >= deadline:
                return best

            # choose r1,r2,r3 distinct and != i
            # small loop is fine for moderate pop sizes
            r1 = i
            while r1 == i:
                r1 = random.randrange(n)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(n)
            r3 = i
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(n)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]
            xr3 = pop[r3]
            xb = best_x if best_x is not None else xi

            # sample trial parameters
            F = F_min + (F_max - F_min) * random.random()
            CR = CR_min + (CR_max - CR_min) * random.random()

            # mutation: v = xr1 + pull*(xb - xr1) + F*(xr2 - xr3)
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xr1[d] + pull * (xb[d] - xr1[d]) + F * (xr2[d] - xr3[d])

            # binomial crossover
            u = list(xi)
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
            clamp_inplace(u)

            fu = eval_f(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_best = best
                    last_improve_time = time.perf_counter()

        # Stagnation handling: diversify a slice of population
        now = time.perf_counter()
        if (now - last_improve_time) > stall_seconds:
            last_improve_time = now

            # diversify worst fraction; keep best few
            # also broaden parameter sampling slightly
            F_min, F_max = 0.30, 1.00
            CR_min, CR_max = 0.02, 0.98

            # rank indices by fitness
            order = sorted(range(len(pop)), key=lambda k: fit[k])
            keep = max(3, len(pop) // 4)
            # reinit the rest with mixture around best and uniform
            for j in order[keep:]:
                if time.perf_counter() >= deadline:
                    return best
                if best_x is not None and random.random() < 0.7:
                    # sample around best with moderate gaussian
                    x = list(best_x)
                    for d in range(dim):
                        if spans[d] == 0.0:
                            continue
                        x[d] += randn() * (0.18 * safe_spans[d])
                    clamp_inplace(x)
                else:
                    x = rand_point()
                pop[j] = x
                fit[j] = eval_f(x)
                if fit[j] < best:
                    best = fit[j]
                    best_x = list(x)
                    last_best = best

            # quick local refine after restart (often converts to real gains)
            if best_x is not None and time.perf_counter() < deadline:
                _, _ = local_refine(best_x, best, budget_evals=18)

        # Near end: do a couple of local refine passes on current best
        if (deadline - time.perf_counter()) < 0.18 * float(max_time):
            if best_x is not None and time.perf_counter() < deadline:
                _, _ = local_refine(best_x, best, budget_evals=22)

    return best
