import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (stdlib only).

    Hybrid algorithm tuned for strong performance under tight time limits:
      1) Sobol-ish (LHS-like) seeding + opposition points for good initial coverage
      2) Adaptive DE/current-to-pbest/1 + archive (JADE/SHADE-inspired, simplified)
      3) Interleaved local trust-region search around best (axis + random subspace)
      4) Stagnation-triggered partial restarts using an elite set

    Returns: best fitness (float).  (Matches template requirement.)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ----------------- basic guards -----------------
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
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def ensure_bounds(x):
        # Reflect into bounds; handles far-out values robustly
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            # reflect repeatedly if out of range
            while y[i] < lo or y[i] > hi:
                if y[i] < lo:
                    y[i] = lo + (lo - y[i])
                if y[i] > hi:
                    y[i] = hi - (y[i] - hi)
            # numeric safety
            if y[i] < lo: y[i] = lo
            if y[i] > hi: y[i] = hi
        return y

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def opposite(x):
        # Opposition point (mirror within bounds)
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            y[i] = (lo + hi) - x[i]
        return ensure_bounds(y)

    # ----------------- elite handling -----------------
    elite_cap = max(8, min(40, 2 * dim + 10))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # ----------------- initial seeding (fast, diverse) -----------------
    # LHS-like: for each dim, use shuffled bins. This is cheap and improves coverage.
    # Also evaluate opposition points to accelerate early best discovery.
    # Seed budget uses a small portion of time.
    seed_frac = min(0.18, 0.08 + 0.0025 * dim)
    seed_deadline = t0 + seed_frac * float(max_time)

    # Decide pop size (kept moderate for time-bounded eval)
    pop_size = max(18, min(90, 10 + 4 * int(math.sqrt(dim)) + 2 * dim))
    # For very expensive funcs, smaller pop is better; we adapt a bit to time:
    # If max_time is tiny, shrink pop size.
    if float(max_time) <= 1.0:
        pop_size = max(14, min(pop_size, 28))
    elif float(max_time) <= 3.0:
        pop_size = max(16, min(pop_size, 48))

    # Prepare LHS bins
    bins = []
    for d in range(dim):
        n = pop_size
        if spans[d] <= 0:
            bins.append([lows[d]] * n)
            continue
        edges = [(k + random.random()) / n for k in range(n)]
        random.shuffle(edges)
        bins.append([lows[d] + e * spans[d] for e in edges])

    pop = []
    fit = []

    best = float("inf")
    best_x = rand_vec()

    i = 0
    while i < pop_size and time.time() < seed_deadline:
        x = [bins[d][i] for d in range(dim)]
        x = ensure_bounds(x)
        f = safe_eval(x)
        pop.append(x); fit.append(f)
        push_elite(f, x)
        if f < best:
            best, best_x = f, x[:]

        # opposition eval (often helps on many benchmarks)
        xo = opposite(x)
        fo = safe_eval(xo)
        pop.append(xo); fit.append(fo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

        i += 1

    # If seeding time was too short, fill remaining randomly
    while len(pop) < pop_size and time.time() < deadline:
        x = rand_vec()
        f = safe_eval(x)
        pop.append(x); fit.append(f)
        push_elite(f, x)
        if f < best:
            best, best_x = f, x[:]

    # If we overshot due to opposition, truncate to pop_size
    if len(pop) > pop_size:
        order = sorted(range(len(pop)), key=lambda k: fit[k])[:pop_size]
        pop = [pop[k] for k in order]
        fit = [fit[k] for k in order]

    # ----------------- adaptive DE (JADE/SHADE-lite) -----------------
    archive = []
    archive_max = pop_size

    mu_F = 0.6
    mu_CR = 0.55
    c = 0.12  # learning rate

    def clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    last_improve_t = time.time()
    stall_restart_seconds = max(0.25, 0.22 * float(max_time))

    # local search trust region (relative to span)
    tr = [0.25 * s if s > 0 else 0.0 for s in spans]  # per-dim radius

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        frac = min(1.0, (now - t0) / max(1e-12, float(max_time)))

        # p-best fraction increases over time -> more exploitation later
        p_min, p_max = 0.05, 0.35
        p_frac = p_min + (p_max - p_min) * (0.20 + 0.80 * frac)
        p_cnt = max(2, int(math.ceil(p_frac * pop_size)))

        # rank order
        order = sorted(range(pop_size), key=lambda k: fit[k])

        S_F, S_CR, dF = [], [], []

        # a small number of forced best-centered trials each generation
        # (helps when population drifts away)
        forced_best_trials = 1 if dim > 25 else 2

        for _ in range(forced_best_trials):
            if time.time() >= deadline:
                return best
            # random perturb around best with shrinking radius
            scale = (0.35 * (1.0 - frac) + 0.06)
            x = best_x[:]
            for d in range(dim):
                if spans[d] > 0:
                    x[d] += random.gauss(0.0, scale * spans[d])
            x = ensure_bounds(x)
            fx = safe_eval(x)
            push_elite(fx, x)
            if fx < best:
                best, best_x = fx, x[:]
                last_improve_t = time.time()

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # --- sample Fi from cauchy around mu_F; CRi from normal around mu_CR ---
            Fi = None
            for _try in range(10):
                u = random.random()
                val = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < val <= 1.0:
                    Fi = val
                    break
            if Fi is None:
                Fi = max(0.05, min(1.0, mu_F))
            CRi = clip01(random.gauss(mu_CR, 0.1))

            # time-aware tweak: later slightly lower F to refine
            Fi *= (0.90 + 0.20 * (1.0 - frac))
            Fi = max(0.05, min(1.0, Fi))

            # pick pbest from top p_cnt
            pbest_idx = order[random.randrange(p_cnt)]
            xpb = pop[pbest_idx]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from pop ∪ archive != (i,r1)
            union = pop + archive
            union_n = len(union)
            x2 = None
            for _try in range(25):
                j = random.randrange(union_n)
                if j < pop_size and (j == i or j == r1):
                    continue
                x2 = union[j]
                break
            if x2 is None:
                # fallback
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                x2 = pop[r2]

            xr1 = pop[r1]
            xr2 = x2

            # mutation: current-to-pbest/1 with archive
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            v = ensure_bounds(v)

            # crossover
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    uvec[d] = v[d]
                else:
                    uvec[d] = xi[d]
            uvec = ensure_bounds(uvec)

            fu = safe_eval(uvec)
            push_elite(fu, uvec)

            if fu <= fi:
                # archive parent
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = uvec
                fit[i] = fu

                if fu < fi:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    dF.append(fi - fu)

                if fu < best:
                    best, best_x = fu, uvec[:]
                    last_improve_t = time.time()

        # update parameter memories
        if dF:
            s = sum(dF)
            if s <= 0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / s for di in dF]

            # weighted arithmetic mean for CR
            new_mu_CR = 0.0
            for w, cr in zip(weights, S_CR):
                new_mu_CR += w * cr

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            new_mu_F = mu_F if den <= 1e-12 else (num / den)

            mu_CR = (1.0 - c) * mu_CR + c * clip01(new_mu_CR)
            mu_F = (1.0 - c) * mu_F + c * max(0.05, min(1.0, new_mu_F))

        # ----------------- local trust-region search around best -----------------
        # This is the main improvement vs typical DE-only: it can quickly exploit
        # smooth basins and finish strong near the time limit.
        if time.time() < deadline and (gen % 2 == 0):
            # shrink trust region over time, but not to zero
            shrink = 0.85 + 0.10 * (1.0 - frac)
            floor = 0.004
            for d in range(dim):
                if spans[d] > 0:
                    tr[d] = max(floor * spans[d], tr[d] * shrink)

            # do a few inexpensive probes
            probes = 6 if dim <= 12 else (4 if dim <= 30 else 3)

            # (a) axis-aligned +/- steps
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                d = random.randrange(dim)
                if spans[d] <= 0:
                    continue
                step = tr[d]
                cand1 = best_x[:]
                cand2 = best_x[:]
                cand1[d] += step
                cand2[d] -= step
                cand1 = ensure_bounds(cand1)
                cand2 = ensure_bounds(cand2)
                f1 = safe_eval(cand1)
                f2 = safe_eval(cand2)
                push_elite(f1, cand1)
                push_elite(f2, cand2)
                if f1 < best:
                    best, best_x = f1, cand1[:]
                    last_improve_t = time.time()
                if f2 < best:
                    best, best_x = f2, cand2[:]
                    last_improve_t = time.time()

            # (b) random subspace gaussian
            for _ in range(max(1, probes // 2)):
                if time.time() >= deadline:
                    return best
                cand = best_x[:]
                k = 1 if dim == 1 else random.randint(1, max(1, dim // 5))
                for __ in range(k):
                    d = random.randrange(dim)
                    if spans[d] > 0:
                        cand[d] += random.gauss(0.0, tr[d] * 0.75)
                cand = ensure_bounds(cand)
                fc = safe_eval(cand)
                push_elite(fc, cand)
                if fc < best:
                    best, best_x = fc, cand[:]
                    last_improve_t = time.time()

        # ----------------- stagnation handling: partial restart -----------------
        if time.time() - last_improve_t > stall_restart_seconds:
            # keep a few elites, refill rest by mixing:
            # - random points
            # - around best
            # - around other elites (helps multimodal)
            keep = max(3, pop_size // 6)
            # build from global elite list if available
            base = elites[:min(len(elites), keep)]
            if not base:
                base = [(best, best_x[:])]

            new_pop = [x[:] for (f, x) in base]
            new_fit = [safe_eval(x) for x in new_pop]

            # reset trust region a bit wider
            for d in range(dim):
                if spans[d] > 0:
                    tr[d] = max(tr[d], 0.18 * spans[d])

            archive = []
            mu_F = 0.65
            mu_CR = 0.50

            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.50:
                    x = rand_vec()
                elif r < 0.80:
                    # around best
                    x = best_x[:]
                    s = (0.35 * (1.0 - frac) + 0.10)
                    for d in range(dim):
                        if spans[d] > 0:
                            x[d] += random.gauss(0.0, s * spans[d])
                    x = ensure_bounds(x)
                else:
                    # around a random elite
                    ex = base[random.randrange(len(base))][1]
                    x = ex[:]
                    s = (0.25 * (1.0 - frac) + 0.08)
                    for d in range(dim):
                        if spans[d] > 0:
                            x[d] += random.gauss(0.0, s * spans[d])
                    x = ensure_bounds(x)

                fx = safe_eval(x)
                new_pop.append(x)
                new_fit.append(fx)
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x[:]

            pop, fit = new_pop, new_fit
            pop_size = len(pop)  # safety
            last_improve_t = time.time()
