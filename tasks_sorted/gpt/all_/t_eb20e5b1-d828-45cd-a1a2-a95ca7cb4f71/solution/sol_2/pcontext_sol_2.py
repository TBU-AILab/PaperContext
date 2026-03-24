import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Strategy (hybrid, time-aware):
    1) Multi-start: spend early time sampling + keep an elite set.
    2) Main engine: CMA-ES-like diagonal Gaussian search (mean, per-dim sigma) with
       weighted recombination + 1/5 success step-size control.
    3) Secondary engine: small DE/current-to-best/1 moves to escape occasional traps.
    4) Local polish: short coordinate/gaussian perturbations near the incumbent.
    5) Restarts on stagnation with sigma reset + elite injection.

    Returns best fitness (float).
    Only uses Python stdlib.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- helpers ----------------
    if dim <= 0:
        try:
            v = float(func([]))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    def clip01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def ensure_bounds(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            # reflect repeatedly if outside
            while y[i] < lo or y[i] > hi:
                if y[i] < lo:
                    y[i] = lo + (lo - y[i])
                if y[i] > hi:
                    y[i] = hi - (y[i] - hi)
            if y[i] < lo: y[i] = lo
            if y[i] > hi: y[i] = hi
        return y

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def weighted_choice(weights):
        s = 0.0
        for w in weights:
            s += w
        if s <= 0.0:
            return random.randrange(len(weights))
        r = random.random() * s
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                return i
        return len(weights) - 1

    # ---------------- initial sampling / elites ----------------
    # small elite pool helps for restarts + recombination
    elite_cap = max(6, min(30, 2 * dim + 6))
    elites = []  # list of (fit, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # time split: quick sampling (helps a lot on nasty landscapes)
    # but don't overspend on large dim
    sample_budget = min(0.18, 0.06 + 0.003 * dim)
    sample_deadline = t0 + sample_budget * float(max_time)

    best = float("inf")
    best_x = rand_vec()
    # at least some samples, but bounded by time
    while time.time() < sample_deadline:
        x = rand_vec()
        f = safe_eval(x)
        if f < best:
            best = f
            best_x = x[:]
        push_elite(f, x)

    # ---------------- main loop (CMA-ES-like diagonal) ----------------
    # population sizes
    lam = max(16, min(70, 8 + int(3.5 * math.sqrt(dim)) + 2 * dim))  # offspring
    mu = max(4, lam // 3)                                            # parents
    mu = min(mu, lam - 1)
    # recombination weights (log)
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    w_sum = sum(w)
    w = [wi / w_sum for wi in w]
    # Effective mu
    mueff = 1.0 / sum(wi * wi for wi in w)

    # state
    mean = best_x[:]
    # initial sigma per dimension as a fraction of span
    sigmas = [max(1e-12, 0.35 * spans[i] if spans[i] > 0 else 1e-12) for i in range(dim)]
    # global scaling (adapts via success rule)
    g_sigma = 1.0

    # success-based control (1/5th rule variant)
    succ_ema = 0.2
    succ_rate = 0.2

    # stagnation / restart
    last_improve_t = time.time()
    stall_restart_seconds = max(0.20, 0.22 * float(max_time))
    restart_count = 0

    # a tiny DE pool near mean/best for occasional jumps
    de_pop_n = max(10, min(28, 6 + 2 * dim))
    de_pop = [ensure_bounds([mean[d] + random.gauss(0.0, sigmas[d]) for d in range(dim)])
              for _ in range(de_pop_n)]
    de_fit = [safe_eval(x) for x in de_pop]
    for f, x in zip(de_fit, de_pop):
        if f < best:
            best = f
            best_x = x[:]
            last_improve_t = time.time()
        push_elite(f, x)

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        elapsed = now - t0
        frac = min(1.0, elapsed / max(1e-12, float(max_time)))

        # time-aware: later -> smaller steps
        # but keep some minimum exploration
        target_scale = 0.25 + 0.75 * (1.0 - frac)  # from ~1.0 early down to 0.25 late
        # incorporate success-based adaptation
        # if succ_rate > 0.2 -> slightly increase; else decrease
        if succ_rate > 0.22:
            g_sigma *= 1.06
        elif succ_rate < 0.18:
            g_sigma *= 0.94
        g_sigma = max(0.05, min(3.0, g_sigma))
        step_scale = target_scale * g_sigma

        # ----- generate offspring around mean (diagonal gaussian) -----
        offspring = []
        for _ in range(lam):
            if time.time() >= deadline:
                return best
            # sample
            x = [mean[d] + random.gauss(0.0, sigmas[d] * step_scale) for d in range(dim)]
            x = ensure_bounds(x)
            f = safe_eval(x)
            offspring.append((f, x))
            if f < best:
                best = f
                best_x = x[:]
                last_improve_t = time.time()
            push_elite(f, x)

        offspring.sort(key=lambda t: t[0])
        parents = offspring[:mu]

        # ----- update mean (weighted) -----
        old_mean = mean[:]
        mean = [0.0] * dim
        for i in range(mu):
            _, x = parents[i]
            wi = w[i]
            for d in range(dim):
                mean[d] += wi * x[d]
        mean = ensure_bounds(mean)

        # ----- update sigmas (diagonal) based on weighted std of selected steps -----
        # conservative learning to avoid collapse on noisy functions
        cs = min(0.35, (mueff + 2.0) / (dim + mueff + 5.0))
        for d in range(dim):
            # weighted variance of selected around new mean (or old_mean)
            var = 0.0
            md = mean[d]
            for i in range(mu):
                _, x = parents[i]
                diff = (x[d] - md)
                var += w[i] * diff * diff
            sd = math.sqrt(max(1e-24, var))
            # mix with current sigma; keep lower/upper bounds
            # also prevent sigma going below a tiny fraction of span (unless span==0)
            min_sd = 1e-12 if spans[d] == 0 else (1e-9 * spans[d] + 1e-12)
            max_sd = 2.0 * spans[d] + 1e-12 if spans[d] > 0 else sigmas[d]
            sigmas[d] = (1.0 - cs) * sigmas[d] + cs * max(min_sd, min(max_sd, sd))

        # ----- success-rate EMA: fraction of offspring beating current best_x? -----
        # Use improvement over median parent to stabilize
        ref = parents[-1][0]
        succ = 0
        for f, _ in offspring:
            if f < ref:
                succ += 1
        succ = succ / float(lam)
        succ_rate = (1.0 - succ_ema) * succ_rate + succ_ema * succ

        # ----- occasional DE "kick" (current-to-best + rand diff) -----
        # helps escaping when mean collapses around local basin
        if time.time() < deadline and (gen % 3 == 0):
            # update DE pool slightly toward best
            # pick a few iterations only
            iters = 1 if dim > 25 else 2
            for _ in range(iters):
                if time.time() >= deadline:
                    return best
                # ensure best is in pool sometimes
                if random.random() < 0.25:
                    j = random.randrange(de_pop_n)
                    de_pop[j] = best_x[:]
                    de_fit[j] = best
                # one DE sweep
                for i in range(de_pop_n):
                    if time.time() >= deadline:
                        return best
                    # pick r1,r2 != i
                    idxs = list(range(de_pop_n))
                    idxs.remove(i)
                    r1, r2 = random.sample(idxs, 2)
                    xi = de_pop[i]
                    xr1 = de_pop[r1]
                    xr2 = de_pop[r2]
                    # time-aware F, CR
                    F = 0.25 + 0.65 * (1.0 - frac)  # 0.9 -> 0.25
                    CR = 0.15 + 0.75 * (1.0 - frac) # 0.9 -> 0.15

                    # current-to-best/1
                    v = [xi[d] + F * (best_x[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
                    v = ensure_bounds(v)
                    jrand = random.randrange(dim)
                    u = [v[d] if (random.random() < CR or d == jrand) else xi[d] for d in range(dim)]
                    u = ensure_bounds(u)
                    fu = safe_eval(u)
                    if fu <= de_fit[i]:
                        de_pop[i] = u
                        de_fit[i] = fu
                        push_elite(fu, u)
                        if fu < best:
                            best = fu
                            best_x = u[:]
                            last_improve_t = time.time()

        # ----- local polish near best (cheap) -----
        if time.time() < deadline and (gen % 4 == 0):
            # a few tries with shrinking step
            tries = 5 if dim <= 12 else 3
            base = 0.12 * (1.0 - frac) ** 2 + 0.008
            for _ in range(tries):
                if time.time() >= deadline:
                    return best
                cand = best_x[:]
                # perturb 1..k coords
                k = 1 if dim == 1 else random.randint(1, max(1, dim // 5))
                for __ in range(k):
                    d = random.randrange(dim)
                    if spans[d] > 0:
                        cand[d] += random.gauss(0.0, base * spans[d])
                cand = ensure_bounds(cand)
                fc = safe_eval(cand)
                if fc < best:
                    best = fc
                    best_x = cand[:]
                    last_improve_t = time.time()
                    push_elite(fc, cand)

        # ----- restart if stalled -----
        if time.time() - last_improve_t > stall_restart_seconds:
            restart_count += 1
            last_improve_t = time.time()

            # pick restart center: best or random elite-biased
            if elites and random.random() < 0.85:
                # bias toward better elites
                # weights ~ 1/(rank+1)
                weights = [1.0 / (i + 1.0) for i in range(len(elites))]
                idx = weighted_choice(weights)
                mean = elites[idx][1][:]
            else:
                mean = rand_vec()

            # reset sigmas broader with some decay over restarts
            decay = 0.85 ** min(10, restart_count)
            sigmas = [max(1e-12, (0.55 * decay + 0.18) * spans[i] if spans[i] > 0 else 1e-12)
                      for i in range(dim)]
            g_sigma = 1.0
            succ_rate = 0.2

            # refresh DE pool around new mean + inject best
            de_pop = []
            for i in range(de_pop_n):
                if i == 0:
                    x = best_x[:]
                elif elites and random.random() < 0.30:
                    x = elites[random.randrange(min(len(elites), 8))][1][:]
                else:
                    x = [mean[d] + random.gauss(0.0, sigmas[d]) for d in range(dim)]
                    x = ensure_bounds(x)
                de_pop.append(x)
            de_fit = [safe_eval(x) for x in de_pop]
            for f, x in zip(de_fit, de_pop):
                if f < best:
                    best = f
                    best_x = x[:]
                push_elite(f, x)
