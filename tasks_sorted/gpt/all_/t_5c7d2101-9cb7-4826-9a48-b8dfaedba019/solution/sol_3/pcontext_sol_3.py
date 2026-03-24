import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libraries).

    Improvement over previous versions:
      - Uses CMA-ES style sampling (diagonal covariance) for fast local convergence.
      - Interleaves with a strong, adaptive DE/current-to-pbest/1 (JADE-lite) global search.
      - Uses tight time accounting and an evaluation cache (quantized) to avoid repeats.
      - Has restarts with increasing population for robustness.
      - Returns best fitness found within max_time.

    Notes:
      - Designed for continuous bounded domains.
      - `func` is called with a Python list (array-like); typically accepted.

    Returns:
        best (float): best objective value found.
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ------------------------- helpers -------------------------
    def now():
        return time.time()

    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def clip_vec(x):
        for j in range(dim):
            lo, hi = bounds[j]
            v = x[j]
            if v < lo:
                # reflection
                v = lo + (lo - v)
                if v > hi:
                    v = lo
            elif v > hi:
                v = hi - (v - hi)
                if v < lo:
                    v = hi
            x[j] = v
        return x

    # approx N(0,1) without random.gauss (faster in some runtimes)
    def gauss01():
        # CLT: sum 12 U - 6 ~ N(0,1)
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span_j(j):
        return bounds[j][1] - bounds[j][0]

    # quantized cache key to reduce duplicate evals in local searches
    # (keeps cache smallish and helps with expensive funcs)
    cache = {}
    def eval_f(x):
        # quantize by relative position to bounds (10k bins each dim)
        # and keep only a limited cache to avoid memory blow-ups.
        if dim <= 12:
            key = []
            for j in range(dim):
                lo, hi = bounds[j]
                s = hi - lo
                if s <= 0:
                    q = 0
                else:
                    q = int(10000.0 * (x[j] - lo) / s + 0.5)
                    if q < 0: q = 0
                    if q > 10000: q = 10000
                key.append(q)
            key = tuple(key)
            if key in cache:
                return cache[key]
            fx = float(func(x))
            if len(cache) < 20000:
                cache[key] = fx
            return fx
        else:
            return float(func(x))

    # ------------------------- edge cases -------------------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # precompute spans and a numeric floor for step sizes
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # handle degenerate bounds
    for s in spans:
        if not (s > 0.0):
            # if any span is 0 or invalid, just evaluate once at bounds mid
            x = []
            for j in range(dim):
                lo, hi = bounds[j]
                x.append(0.5 * (lo + hi))
            return eval_f(x)

    # ------------------------- global best -------------------------
    best = float("inf")
    best_x = None

    # ------------------------- Hybrid loop with restarts -------------------------
    # Each restart runs:
    #   - DE exploration for a while
    #   - CMA-diagonal exploitation around best
    # with occasional reseeding.
    restart = 0

    while now() < deadline - eps_time:
        restart += 1

        # ------------ population sizes / budgets (time-bounded) ------------
        # Start moderate and increase a bit with restarts.
        de_pop = max(18, min(90, 8 * dim + 10 + 4 * (restart - 1)))
        cma_lam = max(10, min(60, 4 + int(3.0 * math.log(dim + 1.0)) * 4 + 2 * (restart - 1)))
        cma_mu = max(2, cma_lam // 2)

        # ------------ initialize DE population (mix best-centered + random) ------------
        pop = []
        fit = []

        # if we already have a best, seed some around it
        near_frac = 0.35 if best_x is not None else 0.0
        for i in range(de_pop):
            if now() >= deadline - eps_time:
                return best
            if best_x is not None and random.random() < near_frac:
                x = best_x[:]
                # moderate radius early; smaller later restarts
                rad = 0.25 / (1.0 + 0.3 * (restart - 1))
                for j in range(dim):
                    x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * spans[j] * rad,
                                bounds[j][0], bounds[j][1])
            else:
                x = rand_vec()
            fx = eval_f(x)
            pop.append(x)
            fit.append(fx)
            if fx < best:
                best = fx
                best_x = x[:]

        # ------------ JADE-lite parameters ------------
        H = 5
        M_F = [0.6] * H
        M_CR = [0.7] * H
        mem_k = 0
        archive = []
        archive_max = de_pop
        pmin = 2.0 / de_pop
        pmax = 0.2

        def pick_top_pool(pfrac):
            k = max(2, int(de_pop * pfrac))
            # partial selection by repeated scan (k is small)
            chosen = []
            used = set()
            for _ in range(k):
                bi = None
                bf = float("inf")
                for idx in range(de_pop):
                    if idx in used:
                        continue
                    f = fit[idx]
                    if f < bf:
                        bf = f
                        bi = idx
                used.add(bi)
                chosen.append(bi)
            return chosen

        # ------------ CMA-ES diagonal state ------------
        # Mean starts at current global best (or best in pop).
        mean = best_x[:] if best_x is not None else pop[0][:]
        # sigma starts as fraction of range, decreases with restarts
        sigma = 0.20 / (1.0 + 0.5 * (restart - 1))
        # diagonal covariance in normalized coordinates
        C = [1.0] * dim

        # log weights for CMA
        # weights ~ log(mu+0.5) - log(i)
        w = []
        for i in range(1, cma_mu + 1):
            w.append(math.log(cma_mu + 0.5) - math.log(i))
        wsum = sum(w)
        w = [wi / wsum for wi in w]

        # evolution path-ish scalars (very simplified)
        damp = 0.92
        c_up = 0.15  # covariance learning rate
        min_C = 1e-10
        min_sigma = 1e-12

        # ------------ schedule within this restart ------------
        # Alternate blocks of DE and CMA.
        # Make blocks small to react quickly under time limits.
        de_block = max(1, 2 + 20 // max(1, dim))      # ~ few generations
        cma_block = max(1, 1 + 30 // max(1, dim))     # ~ few iterations
        max_outer = 10**9  # time-bounded anyway

        stagnant = 0
        last_best = best

        outer = 0
        while now() < deadline - eps_time and outer < max_outer:
            outer += 1

            # ===================== DE block =====================
            for _ in range(de_block):
                if now() >= deadline - eps_time:
                    return best

                p = random.uniform(pmin, pmax)
                pbest_pool = pick_top_pool(p)

                S_F, S_CR, dF = [], [], []

                for i in range(de_pop):
                    if now() >= deadline - eps_time:
                        return best

                    xi = pop[i]
                    fi = fit[i]

                    r = random.randrange(H)
                    muF = M_F[r]
                    muCR = M_CR[r]

                    # CR ~ N(muCR,0.1) clipped
                    CRi = muCR + 0.1 * gauss01()
                    if CRi < 0.0: CRi = 0.0
                    if CRi > 1.0: CRi = 1.0

                    # F ~ cauchy(muF,0.1) clipped/resampled
                    Fi = muF + 0.1 * math.tan(math.pi * (random.random() - 0.5))
                    tries = 0
                    while (Fi <= 0.0 or Fi > 1.0) and tries < 5:
                        Fi = muF + 0.1 * math.tan(math.pi * (random.random() - 0.5))
                        tries += 1
                    if Fi <= 0.0: Fi = 0.1
                    if Fi > 1.0: Fi = 1.0

                    pbest = pop[random.choice(pbest_pool)]

                    r1 = i
                    while r1 == i:
                        r1 = random.randrange(de_pop)

                    # pick r2 from pop or archive
                    if archive and random.random() < 0.5:
                        r2_vec = archive[random.randrange(len(archive))]
                    else:
                        r2 = r1
                        while r2 == i or r2 == r1:
                            r2 = random.randrange(de_pop)
                        r2_vec = pop[r2]

                    xr1 = pop[r1]

                    # current-to-pbest/1
                    v = [0.0] * dim
                    for j in range(dim):
                        v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - r2_vec[j])
                    clip_vec(v)

                    # binomial crossover
                    jrand = random.randrange(dim)
                    u = [0.0] * dim
                    for j in range(dim):
                        u[j] = v[j] if (random.random() < CRi or j == jrand) else xi[j]

                    fu = eval_f(u)

                    if fu <= fi:
                        # archive
                        archive.append(xi[:])
                        if len(archive) > archive_max:
                            # random delete
                            k = random.randrange(len(archive))
                            archive[k] = archive[-1]
                            archive.pop()

                        pop[i] = u
                        fit[i] = fu

                        if fu < best:
                            best = fu
                            best_x = u[:]
                            mean = best_x[:]  # move CMA mean to new best quickly

                        S_F.append(Fi)
                        S_CR.append(CRi)
                        dF.append(max(0.0, fi - fu))

                # update memories
                if S_F:
                    wsum2 = sum(dF)
                    if wsum2 <= 0.0:
                        weights = [1.0 / len(dF)] * len(dF)
                    else:
                        weights = [di / wsum2 for di in dF]

                    num = 0.0
                    den = 0.0
                    for ww, fval in zip(weights, S_F):
                        num += ww * fval * fval
                        den += ww * fval
                    new_MF = (num / den) if den > 1e-12 else M_F[mem_k]

                    new_MCR = 0.0
                    for ww, crv in zip(weights, S_CR):
                        new_MCR += ww * crv

                    M_F[mem_k] = clip(new_MF, 0.05, 1.0)
                    M_CR[mem_k] = clip(new_MCR, 0.0, 1.0)
                    mem_k = (mem_k + 1) % H

            # ===================== CMA diagonal block =====================
            for _ in range(cma_block):
                if now() >= deadline - eps_time:
                    return best

                # sample lambda candidates
                cand = []
                for _k in range(cma_lam):
                    if now() >= deadline - eps_time:
                        return best
                    x = [0.0] * dim
                    for j in range(dim):
                        # sample in normalized coords then scale by span
                        z = gauss01() * math.sqrt(C[j])
                        xj = mean[j] + (sigma * spans[j]) * z
                        lo, hi = bounds[j]
                        # reflect
                        if xj < lo:
                            xj = lo + (lo - xj)
                            if xj > hi: xj = lo
                        elif xj > hi:
                            xj = hi - (xj - hi)
                            if xj < lo: xj = hi
                        x[j] = xj
                    fx = eval_f(x)
                    cand.append((fx, x))

                cand.sort(key=lambda t: t[0])

                if cand[0][0] < best:
                    best = cand[0][0]
                    best_x = cand[0][1][:]
                # update mean
                new_mean = mean[:]
                for j in range(dim):
                    s = 0.0
                    for i in range(cma_mu):
                        s += w[i] * cand[i][1][j]
                    new_mean[j] = s

                # update diagonal covariance based on selected steps
                # compute normalized step per dimension
                for j in range(dim):
                    # weighted variance around old mean (in normalized units)
                    v = 0.0
                    denom = (sigma * spans[j])
                    if denom < 1e-18:
                        denom = 1e-18
                    m0 = mean[j]
                    for i in range(cma_mu):
                        d = (cand[i][1][j] - m0) / denom
                        v += w[i] * (d * d)
                    # exponential smoothing
                    C[j] = max(min_C, (1.0 - c_up) * C[j] + c_up * v)

                mean = new_mean

                # sigma adaptation: 1/5 success rule-ish using best improvement frequency
                # Compare median of selected to current best to decide shrink/expand
                med_fx = cand[min(cma_mu - 1, len(cand) - 1)][0]
                if med_fx < best + 1e-12:
                    sigma *= 1.03
                else:
                    sigma *= 0.90
                if sigma < min_sigma:
                    sigma = min_sigma
                if sigma > 0.6:
                    sigma = 0.6

            # ===================== stagnation & reseed =====================
            if best < last_best - 1e-12:
                last_best = best
                stagnant = 0
            else:
                stagnant += 1

            if stagnant >= 4 and now() < deadline - eps_time:
                # partial reseed: replace worst 20% in DE pop with (near best) + random
                k = max(2, de_pop // 5)
                # find k worst indices
                worst = []
                used = set()
                for _ in range(k):
                    wi = None
                    wf = -float("inf")
                    for idx in range(de_pop):
                        if idx in used:
                            continue
                        f = fit[idx]
                        if f > wf:
                            wf = f
                            wi = idx
                    used.add(wi)
                    worst.append(wi)

                for t, idx in enumerate(worst):
                    if now() >= deadline - eps_time:
                        return best
                    if best_x is not None and t < k // 2:
                        x = best_x[:]
                        rad = 0.18 / (1.0 + 0.3 * (restart - 1))
                        for j in range(dim):
                            x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * spans[j] * rad,
                                        bounds[j][0], bounds[j][1])
                    else:
                        x = rand_vec()
                    fx = eval_f(x)
                    pop[idx] = x
                    fit[idx] = fx
                    if fx < best:
                        best = fx
                        best_x = x[:]
                        mean = best_x[:]
                stagnant = 0

        # end restart loop; continue if time remains
    return best
