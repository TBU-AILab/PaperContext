import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over previous version:
      - Keeps a small elite population (CMA-ES-like but simpler): estimate mean + per-dim sigma
      - Uses multiple operators: gaussian sampling, differential evolution (DE) mutation, local coordinate search
      - Adapts step sizes from success rate and shrinks when stuck
      - Better bound handling (reflection)
      - Strict time checks; always returns best-so-far
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def reflect_into_bounds(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if hi <= lo:
                y[i] = lo
                continue
            span = hi - lo
            # reflect using modulus into [0, 2*span)
            v = (v - lo) % (2.0 * span)
            if v > span:
                v = 2.0 * span - v
            y[i] = lo + v
        return y

    def evaluate(x):
        try:
            v = float(func(x))
            if not is_finite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def avg_span():
        s = 0.0
        for i in range(dim):
            lo, hi = bounds[i]
            s += (hi - lo)
        return s / max(1, dim)

    def clamp01(u):
        return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)

    # Box-Muller gaussian
    def gauss():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Robust per-dimension median
    def median(vals):
        a = sorted(vals)
        n = len(a)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return a[mid]
        return 0.5 * (a[mid - 1] + a[mid])

    # ---------------- initialization ----------------
    base = avg_span()
    if base <= 0:
        base = 1.0

    # population sizing (small, time-friendly)
    pop = max(8, min(28, 4 + 2 * dim))
    elite_n = max(3, pop // 3)

    # initial sigma per dim = fraction of domain
    sig = []
    for i in range(dim):
        lo, hi = bounds[i]
        sig.append(0.22 * (hi - lo) if hi > lo else 1e-9)

    # Start with random population
    P = []
    best = float("inf")
    best_x = None

    # Try to spend a small part of time on initial spread
    init_evals = pop
    for _ in range(init_evals):
        if time.time() >= deadline:
            return best
        x = rand_vec()
        fx = evaluate(x)
        P.append([x, fx])
        if fx < best:
            best, best_x = fx, x

    # If everything failed (pathological), ensure best_x exists
    if best_x is None:
        best_x = rand_vec()
        best = evaluate(best_x)

    # Stagnation bookkeeping
    last_improve = time.time()
    no_improve_iters = 0

    # operator success stats for adaptive mixing
    # 0: gaussian around mean, 1: DE/rand/1, 2: coordinate local
    succ = [1.0, 1.0, 1.0]
    tries = [3.0, 3.0, 3.0]

    # ---------------- main loop ----------------
    while time.time() < deadline:
        # Sort by fitness
        P.sort(key=lambda t: t[1])
        if P[0][1] < best:
            best, best_x = P[0][1], P[0][0][:]
            last_improve = time.time()
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        # Build elite set
        elite = P[:elite_n]

        # Compute robust center: per-dimension median of elites
        mu = [0.0] * dim
        for d in range(dim):
            mu[d] = median([elite[j][0][d] for j in range(elite_n)])

        # Adapt sigma: use MAD-like spread from elites, then blend with current
        # If elites collapse, keep a floor.
        for d in range(dim):
            vals = [elite[j][0][d] for j in range(elite_n)]
            m = mu[d]
            absdev = [abs(v - m) for v in vals]
            mad = median(absdev)  # median abs deviation
            # convert to a "sigma-like" scale; keep within [floor, cap]
            lo, hi = bounds[d]
            span = (hi - lo) if hi > lo else 1.0
            target = max(1e-12 * span, min(0.35 * span, 1.4826 * mad))
            # Blend: when stagnant, inflate a bit; otherwise slowly follow target
            stagnant = (time.time() - last_improve) > 0.35 * max_time
            blend = 0.25 if not stagnant else 0.45
            sig[d] = (1.0 - blend) * sig[d] + blend * (target * (1.25 if stagnant else 1.0))
            # ensure nonzero
            sig[d] = max(sig[d], 1e-15 * (span if span > 0 else 1.0))

        # Choose operator probabilistically by past success
        w = [succ[i] / max(1e-9, tries[i]) for i in range(3)]
        sw = w[0] + w[1] + w[2]
        r = random.random() * sw
        if r < w[0]:
            op = 0
        elif r < w[0] + w[1]:
            op = 1
        else:
            op = 2

        # Create one candidate and replace worst if improved
        worst_x, worst_f = P[-1][0], P[-1][1]

        if time.time() >= deadline:
            break

        cand = None

        if op == 0:
            # Gaussian around mu with correlated "pull" to best
            tries[0] += 1.0
            cand = [0.0] * dim
            pull = 0.2 + 0.6 * random.random()  # mix mu and best
            for d in range(dim):
                center = pull * best_x[d] + (1.0 - pull) * mu[d]
                cand[d] = center + gauss() * sig[d]
            cand = reflect_into_bounds(cand)

        elif op == 1:
            # Differential Evolution mutation using random individuals + crossover
            tries[1] += 1.0
            a, b, c = random.sample(P[:max(pop, 4)], 3)
            xa, xb, xc = a[0], b[0], c[0]
            F = 0.4 + 0.6 * random.random()
            CR = 0.2 + 0.7 * random.random()

            # base vector slightly biased toward best
            base_vec = xa
            if random.random() < 0.5:
                base_vec = best_x

            cand = [0.0] * dim
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    cand[d] = base_vec[d] + F * (xb[d] - xc[d])
                else:
                    cand[d] = P[random.randrange(pop)][0][d]
                # small gaussian jitter to avoid degeneracy
                cand[d] += 0.05 * gauss() * sig[d]
            cand = reflect_into_bounds(cand)

        else:
            # Coordinate/local search around current best (first-improvement sweep)
            tries[2] += 1.0
            cand = best_x[:]
            # step size depends on sigma; slightly larger when stagnant
            stagnant = (time.time() - last_improve) > 0.30 * max_time
            for d in range(dim):
                if time.time() >= deadline:
                    break
                step = (1.0 if not stagnant else 1.6) * (0.9 * sig[d] + 0.02 * base)
                if step <= 0:
                    continue
                improved = False
                for sgn in (-1.0, 1.0):
                    trial = cand[:]
                    trial[d] += sgn * step
                    trial = reflect_into_bounds(trial)
                    ftrial = evaluate(trial)
                    if ftrial < best:
                        best, best_x = ftrial, trial[:]
                        last_improve = time.time()
                        no_improve_iters = 0
                        cand = trial[:]  # keep moving
                        improved = True
                        break
                if improved:
                    # keep going with updated cand
                    pass
            # If no coordinate improvement, still perturb slightly
            if cand == best_x:
                for d in range(dim):
                    cand[d] = best_x[d] + gauss() * (0.5 * sig[d])
                cand = reflect_into_bounds(cand)

        fc = evaluate(cand)

        # Replacement strategy: replace worst if better, else occasionally inject
        replaced = False
        if fc < worst_f:
            P[-1] = [cand, fc]
            replaced = True
        else:
            # occasional random re-injection when stuck
            if no_improve_iters > 8 and random.random() < 0.25:
                rx = rand_vec()
                rfx = evaluate(rx)
                if rfx < worst_f:
                    P[-1] = [rx, rfx]
                    replaced = True

        # Update success stats for chosen operator
        if op in (0, 1):
            if replaced and fc < P[-2][1]:
                succ[op] += 1.0
        else:
            # local op: consider success if best improved recently
            if (time.time() - last_improve) < 0.02 * max_time:
                succ[2] += 0.5

        # If very stagnant, expand exploration a bit (sigma inflation) but bounded
        if (time.time() - last_improve) > 0.55 * max_time:
            for d in range(dim):
                lo, hi = bounds[d]
                span = (hi - lo) if hi > lo else 1.0
                sig[d] = min(0.5 * span, sig[d] * 1.15)

    return best
