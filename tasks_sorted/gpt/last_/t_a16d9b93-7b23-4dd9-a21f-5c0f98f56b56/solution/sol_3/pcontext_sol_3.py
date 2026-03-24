import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Main changes vs last version:
      - Proper "ask/tell" loop with multiple candidates per generation (better use of time)
      - Explicit exploration/exploitation schedule + restarts
      - (mu, lambda) evolution strategy with rank-based update and per-dim sigma adaptation
      - Lightweight quadratic-like local refinement (2-point parabolic step per coordinate) near best
      - Cache to avoid re-evaluating identical/near-identical points
      - Stronger DE/current-to-best operator and occasional global samples

    Returns:
      best fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def reflect_into_bounds(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
                continue
            span = hi - lo
            v = x[i]
            v = (v - lo) % (2.0 * span)
            if v > span:
                v = 2.0 * span - v
            y[i] = lo + v
        return y

    # Box-Muller gaussian
    _has_spare = False
    _spare = 0.0
    def gauss():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _spare = z1
        _has_spare = True
        return z0

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def avg_span():
        s = 0.0
        for i in range(dim):
            lo, hi = bounds[i]
            s += (hi - lo) if hi > lo else 0.0
        return s / max(1, dim)

    # --- evaluation cache (quantize point to reduce duplicates) ---
    cache = {}
    def key_of(x):
        # Quantize relative to bounds span; robust even if span=0
        k = []
        for i in range(dim):
            lo, hi = bounds[i]
            span = hi - lo
            if span <= 0:
                k.append(0)
            else:
                # 1e-4 of span buckets (coarse enough to help, fine enough to not ruin)
                q = int((x[i] - lo) / span * 10000.0)
                k.append(q)
        return tuple(k)

    def evaluate(x):
        x = reflect_into_bounds(x)
        k = key_of(x)
        if k in cache:
            return cache[k], x
        try:
            v = float(func(x))
            if not is_finite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v, x

    def median(vals):
        a = sorted(vals)
        n = len(a)
        if n == 0:
            return 0.0
        m = n // 2
        return a[m] if (n % 2 == 1) else 0.5 * (a[m - 1] + a[m])

    # ---------- setup ----------
    base = avg_span()
    if base <= 0:
        base = 1.0

    # population and selection sizes
    # keep modest to fit most time budgets, but generate batches for efficiency
    lam = max(12, min(64, 8 + 4 * dim))     # offspring per generation
    mu_n = max(4, lam // 4)                 # parents

    # initial center = random; sigma = fraction of span
    x_best = None
    f_best = float("inf")

    # initialize mean from multiple random samples (better than single)
    init_n = max(lam, 2 * mu_n)
    P = []
    for _ in range(init_n):
        if time.time() >= deadline:
            return f_best
        x = rand_vec()
        fx, x = evaluate(x)
        P.append((fx, x))
        if fx < f_best:
            f_best, x_best = fx, x[:]
    P.sort(key=lambda t: t[0])
    mean = [0.0] * dim
    for d in range(dim):
        mean[d] = median([P[i][1][d] for i in range(min(mu_n, len(P)))])
    # per-dim sigma
    sigma = []
    for i in range(dim):
        lo, hi = bounds[i]
        span = (hi - lo) if hi > lo else 1.0
        sigma.append(0.25 * span)

    last_improve_t = time.time()
    stall_gens = 0

    # rank weights for recombination (positive, sum=1)
    # simple linear weights
    w = [float(mu_n - i) for i in range(mu_n)]
    sw = sum(w)
    w = [wi / sw for wi in w]

    # ---------- operators ----------
    def sample_es(mean, sigma, explore_scale):
        # Diagonal Gaussian sampling around mean, with mild "toward best" pull.
        pull = 0.2 + 0.6 * random.random()
        x = [0.0] * dim
        for d in range(dim):
            center = mean[d]
            if x_best is not None and random.random() < 0.7:
                center = (1.0 - pull) * center + pull * x_best[d]
            x[d] = center + gauss() * sigma[d] * explore_scale
        return reflect_into_bounds(x)

    def sample_de_current_to_best():
        # current-to-best/1 with binomial crossover; uses a few cached "good" points via Pbest pool
        if len(P) < 5:
            return rand_vec()
        # p-best from top fraction
        topk = max(3, len(P) // 4)
        x_pbest = P[random.randrange(topk)][1]
        x_r1 = P[random.randrange(len(P))][1]
        x_r2 = P[random.randrange(len(P))][1]
        x_base = P[random.randrange(topk)][1] if random.random() < 0.5 else x_pbest

        F = 0.4 + 0.5 * random.random()
        CR = 0.2 + 0.75 * random.random()

        jrand = random.randrange(dim)
        out = [0.0] * dim
        for d in range(dim):
            if random.random() < CR or d == jrand:
                bestterm = 0.0
                if x_best is not None:
                    bestterm = (x_pbest[d] - x_base[d])
                out[d] = x_base[d] + F * bestterm + F * (x_r1[d] - x_r2[d])
            else:
                out[d] = x_base[d]
            # light jitter
            out[d] += 0.02 * gauss() * sigma[d]
        return reflect_into_bounds(out)

    def local_parabolic_refine(x0, f0, budget_evals=2*50):
        # coordinate-wise 1D parabolic step using f(x-h), f(x), f(x+h)
        # budget is soft-limited by time checks.
        x = x0[:]
        fx = f0
        # step based on sigma; shrink during refine
        for d in range(dim):
            if time.time() >= deadline:
                break
            lo, hi = bounds[d]
            span = (hi - lo) if hi > lo else 1.0
            h = max(1e-12 * span, 0.35 * sigma[d])
            h = min(h, 0.2 * span)

            # Try a couple decreasing h if needed
            for _ in range(2):
                if time.time() >= deadline:
                    break
                x_m = x[:]
                x_p = x[:]
                x_m[d] -= h
                x_p[d] += h
                f_m, x_m = evaluate(x_m)
                f_p, x_p = evaluate(x_p)

                # Fit parabola through (-h, f_m), (0, fx), (+h, f_p)
                denom = (f_m - 2.0 * fx + f_p)
                if denom <= 0 or not is_finite(denom):
                    # if not convex-ish, just take best among three
                    if f_m < fx or f_p < fx:
                        if f_m <= f_p and f_m < fx:
                            x, fx = x_m[:], f_m
                        elif f_p < fx:
                            x, fx = x_p[:], f_p
                    h *= 0.5
                    continue

                # vertex location t* in [-h, +h]
                t = 0.5 * h * (f_m - f_p) / denom
                if abs(t) > 2.0 * h:
                    t = max(-2.0 * h, min(2.0 * h, t))

                x_t = x[:]
                x_t[d] += t
                f_t, x_t = evaluate(x_t)

                # accept best
                if f_t < fx or f_m < fx or f_p < fx:
                    if f_t <= f_m and f_t <= f_p and f_t < fx:
                        x, fx = x_t[:], f_t
                    elif f_m <= f_p and f_m < fx:
                        x, fx = x_m[:], f_m
                    elif f_p < fx:
                        x, fx = x_p[:], f_p
                h *= 0.5
        return fx, x

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # exploration schedule: gradually reduce, but bump up when stalled
        elapsed = (time.time() - t0) / max(1e-9, max_time)
        explore = 1.5 - 1.1 * min(1.0, max(0.0, elapsed))  # from ~1.5 -> ~0.4
        if (time.time() - last_improve_t) > 0.35 * max_time:
            explore *= 1.35

        # produce offspring batch
        offspring = []
        # mix operators
        for k in range(lam):
            if time.time() >= deadline:
                break
            r = random.random()
            if r < 0.60:
                x = sample_es(mean, sigma, explore)
            elif r < 0.90:
                x = sample_de_current_to_best()
            else:
                x = rand_vec()  # global injection

            fx, x = evaluate(x)
            offspring.append((fx, x))

            if fx < f_best:
                f_best, x_best = fx, x[:]
                last_improve_t = time.time()
                stall_gens = 0

        if not offspring:
            break

        offspring.sort(key=lambda t: t[0])
        P = offspring  # keep last generation only (time/memory friendly)

        # recombination update of mean
        new_mean = [0.0] * dim
        for d in range(dim):
            s = 0.0
            for i in range(min(mu_n, len(P))):
                s += w[i] * P[i][1][d]
            new_mean[d] = s
        mean = reflect_into_bounds(new_mean)

        # sigma adaptation based on elite spread (robust)
        elite = P[:min(mu_n, len(P))]
        for d in range(dim):
            lo, hi = bounds[d]
            span = (hi - lo) if hi > lo else 1.0
            m = median([e[1][d] for e in elite])
            mad = median([abs(e[1][d] - m) for e in elite])
            target = max(1e-12 * span, min(0.45 * span, 1.4826 * mad))
            # if stalled, inflate a bit to escape; else follow target
            stalled = (time.time() - last_improve_t) > 0.20 * max_time
            a = 0.25 if not stalled else 0.45
            sigma[d] = (1.0 - a) * sigma[d] + a * (target * (1.25 if stalled else 1.0))
            sigma[d] = max(sigma[d], 1e-15 * span)

        # detect stall and restart around best / global
        if (time.time() - last_improve_t) > 0.55 * max_time:
            # heavy restart once late and stuck: reset mean and broaden sigma
            if x_best is not None and random.random() < 0.7:
                mean = x_best[:]
            else:
                mean = rand_vec()
            for d in range(dim):
                lo, hi = bounds[d]
                span = (hi - lo) if hi > lo else 1.0
                sigma[d] = max(sigma[d], 0.35 * span)
            last_improve_t = time.time()  # avoid repeated immediate restarts

        # occasional local refinement near best (only if time allows)
        if x_best is not None and (gen % 7 == 0) and (time.time() + 0.01 < deadline):
            f_loc, x_loc = local_parabolic_refine(x_best, f_best)
            if f_loc < f_best:
                f_best, x_best = f_loc, x_loc[:]
                last_improve_t = time.time()
                stall_gens = 0
            else:
                stall_gens += 1

    return f_best
