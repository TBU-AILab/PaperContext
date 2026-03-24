import random, time, math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libraries).

    Improvements vs the provided ones (practical, not just “more code”):
      - Keeps a single strong core: L-SHADE/JADE-style DE (current-to-pbest/1 + archive + success-history)
      - Much faster p-best selection via rank sampling (no repeated O(N^2) scans)
      - Population size reduction over time (focuses exploitation later)
      - A compact, effective local search: adaptive coordinate search + quadratic 1D interpolation
      - Restarts/injections based on *time-based* stagnation (robust across max_time settings)
      - Low overhead time checking and bound handling (reflection)

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ----------------- helpers -----------------
    def now():
        return time.time()

    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect(v, lo, hi):
        # reflection keeps step direction information better than hard clip
        if v < lo:
            v = lo + (lo - v)
            if v > hi:
                v = lo
        elif v > hi:
            v = hi - (v - hi)
            if v < lo:
                v = hi
        return v

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # approx N(0,1) fast (CLT)
    def gauss01():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    # quantized cache: helps mostly for local search; kept small
    cache = {}
    CACHE_MAX = 20000
    def eval_f(x):
        if dim <= 14:
            key = []
            for j in range(dim):
                lo, hi = bounds[j]
                s = hi - lo
                if s <= 0.0:
                    q = 0
                else:
                    q = int(4096.0 * (x[j] - lo) / s + 0.5)
                    if q < 0: q = 0
                    if q > 4096: q = 4096
                key.append(q)
            key = tuple(key)
            v = cache.get(key)
            if v is not None:
                return v
            fx = float(func(x))
            if len(cache) < CACHE_MAX:
                cache[key] = fx
            return fx
        return float(func(x))

    # ----------------- edge cases -----------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    spans = []
    for j in range(dim):
        lo, hi = bounds[j]
        s = hi - lo
        if not (s > 0.0):
            x = [0.5 * (b[0] + b[1]) for b in bounds]
            return eval_f(x)
        spans.append(s)

    # ----------------- initialization -----------------
    # population sizing (time-bounded): moderate initial, reduced later
    pop0 = max(24, min(120, 10 * dim + 20))
    pop_min = max(10, min(45, 4 * dim + 6))
    pop_size = pop0

    pop = []
    fit = []
    best = float("inf")
    best_x = None

    # mixed init: stratified-ish + uniform
    half = max(1, pop_size // 2)
    for i in range(pop_size):
        if now() >= deadline - eps_time:
            return best
        x = [0.0] * dim
        if i < half:
            # cheap per-dim stratification (wrap) to improve early coverage
            u0 = (i + random.random()) / float(half)
            for j in range(dim):
                lo, hi = bounds[j]
                u = u0 + 0.13 * (random.random() - 0.5)
                u -= math.floor(u)
                x[j] = lo + (hi - lo) * u
        else:
            x = rand_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    # DE archive
    archive = []
    archive_max = pop_size

    # SHADE memories
    H = 6
    M_F = [0.55] * H
    M_CR = [0.75] * H
    mem_k = 0

    # pbest fraction range
    pmax = 0.20

    # -------- fast p-best selection by rank sampling --------
    # Build ranks occasionally: get indices sorted by fitness once, then sample from prefix.
    sorted_idx = list(range(pop_size))
    sorted_stamp_gen = -1
    gen = 0

    def ensure_sorted():
        nonlocal sorted_idx, sorted_stamp_gen
        if sorted_stamp_gen == gen:
            return
        sorted_idx = list(range(pop_size))
        # Python Timsort is efficient; do it once per generation
        sorted_idx.sort(key=lambda i: fit[i])
        sorted_stamp_gen = gen

    def sample_pbest_index():
        # choose p in [2/N, pmax], then random index from top-k
        ensure_sorted()
        pmin = 2.0 / pop_size
        p = pmin + (pmax - pmin) * random.random()
        k = max(2, int(p * pop_size))
        return sorted_idx[random.randrange(k)]

    # ----------------- local search (adaptive coordinate + 1D quadratic) -----------------
    ls_sigma = 0.12  # fraction of span
    ls_succ = 0
    ls_tr = 0

    def local_improve(bx, bf, budget):
        nonlocal best, best_x, ls_sigma, ls_succ, ls_tr
        xbest = bx[:]
        fbest = bf

        for _ in range(budget):
            if now() >= deadline - eps_time:
                break
            ls_tr += 1

            x = xbest[:]
            if random.random() < 0.75:
                j = random.randrange(dim)
                step = gauss01() * spans[j] * ls_sigma
                x[j] = reflect(x[j] + step, bounds[j][0], bounds[j][1])
            else:
                # small multi-dim jitter
                for j in range(dim):
                    step = gauss01() * spans[j] * (ls_sigma * 0.25)
                    x[j] = reflect(x[j] + step, bounds[j][0], bounds[j][1])

            fx = eval_f(x)
            if fx < fbest:
                xbest, fbest = x, fx
                ls_succ += 1
                if fx < best:
                    best, best_x = fx, x[:]

            # occasional 1D quadratic interpolation around current best
            if now() >= deadline - eps_time:
                break
            if random.random() < 0.12:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                a = xbest[j]
                delta = spans[j] * max(1e-8, ls_sigma * 0.50)

                x1 = xbest[:]; x1[j] = reflect(a - delta, lo, hi)
                x2 = xbest[:]; x2[j] = reflect(a + delta, lo, hi)
                f1 = eval_f(x1)
                if now() >= deadline - eps_time:
                    break
                f2 = eval_f(x2)

                denom = (f1 - 2.0 * fbest + f2)
                if abs(denom) > 1e-18:
                    t = 0.5 * (f1 - f2) / denom
                    if t < -1.5: t = -1.5
                    if t >  1.5: t =  1.5
                    xq = xbest[:]
                    xq[j] = reflect(a + t * delta, lo, hi)
                    fq = eval_f(xq)
                    if fq < fbest:
                        xbest, fbest = xq, fq
                        if fq < best:
                            best, best_x = fq, xq[:]

        # adapt local step (simple success rule)
        if ls_tr >= 30:
            rate = ls_succ / float(ls_tr)
            ls_sigma *= (1.15 if rate > 0.22 else 0.85)
            if ls_sigma < 1e-10: ls_sigma = 1e-10
            if ls_sigma > 0.35:  ls_sigma = 0.35
            ls_succ = 0
            ls_tr = 0

        return xbest, fbest

    # ----------------- main loop -----------------
    last_best = best
    last_improve_t = now()
    # time-based stagnation trigger is more robust than "gens" under varying cost funcs
    stagnation_seconds = max(0.15, 0.08 * max_time)

    while now() < deadline - eps_time:
        gen += 1

        # population size reduction (linear in time)
        if pop_size > pop_min:
            frac = (now() - t0) / max(1e-12, max_time)
            target = int(round(pop0 - (pop0 - pop_min) * min(1.0, frac)))
            if target < pop_size:
                # shrink by removing worst individuals (use sorted indices)
                ensure_sorted()
                # remove from end (worst)
                remove_n = pop_size - target
                # remove_n is small; do it by marking
                kill = set(sorted_idx[-remove_n:])
                new_pop, new_fit = [], []
                for i in range(pop_size):
                    if i not in kill:
                        new_pop.append(pop[i])
                        new_fit.append(fit[i])
                pop, fit = new_pop, new_fit
                pop_size = len(pop)
                archive_max = pop_size
                if len(archive) > archive_max:
                    archive = archive[:archive_max]
                # force recompute sorted
                sorted_stamp_gen = -1

        # one DE generation (current-to-pbest/1 + archive + SHADE memories)
        ensure_sorted()
        S_F, S_CR, dF = [], [], []

        for i in range(pop_size):
            if now() >= deadline - eps_time:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            CRi = muCR + 0.1 * gauss01()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            Fi = muF + 0.1 * math.tan(math.pi * (random.random() - 0.5))
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 6:
                Fi = muF + 0.1 * math.tan(math.pi * (random.random() - 0.5))
                tries += 1
            if Fi <= 0.0: Fi = 0.08
            if Fi > 1.0: Fi = 1.0

            pbest = pop[sample_pbest_index()]

            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # r2 from archive with prob, else from pop
            if archive and random.random() < 0.5:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = r1
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                xr2 = pop[r2]
            xr1 = pop[r1]

            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = reflect(vj, bounds[j][0], bounds[j][1])

            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                u[j] = v[j] if (random.random() < CRi or j == jrand) else xi[j]

            fu = eval_f(u)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > archive_max:
                    k = random.randrange(len(archive))
                    archive[k] = archive[-1]
                    archive.pop()

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_improve_t = now()

                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(max(0.0, fi - fu))

        # update SHADE memories
        if S_F:
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                inv = 1.0 / wsum
                weights = [di * inv for di in dF]

            num = 0.0
            den = 0.0
            for ww, ff in zip(weights, S_F):
                num += ww * ff * ff
                den += ww * ff
            new_MF = (num / den) if den > 1e-12 else M_F[mem_k]

            new_MCR = 0.0
            for ww, cc in zip(weights, S_CR):
                new_MCR += ww * cc

            M_F[mem_k] = clip(new_MF, 0.05, 1.0)
            M_CR[mem_k] = clip(new_MCR, 0.0, 1.0)
            mem_k = (mem_k + 1) % H

        # exploitation: local improvement (small but frequent)
        if best_x is not None and now() < deadline - eps_time:
            budget = 2 + dim // 6
            best_x, best = local_improve(best_x, best, budget)

        # time-based stagnation injection
        if best < last_best - 1e-12:
            last_best = best
            last_improve_t = now()

        if (now() - last_improve_t) >= stagnation_seconds and now() < deadline - eps_time:
            # inject: replace worst 25% with half near-best and half random; clear archive a bit
            ensure_sorted()
            k = max(2, pop_size // 4)
            worst_idx = sorted_idx[-k:]

            for t, idx in enumerate(worst_idx):
                if now() >= deadline - eps_time:
                    return best
                if best_x is not None and t < k // 2:
                    x = best_x[:]
                    rad = 0.18 + 0.10 * random.random()
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
                    last_best = best
            # reduce archive to keep diversity but not bloat
            if len(archive) > archive_max // 2:
                archive = archive[:archive_max // 2]
            last_improve_t = now()
            # also loosen local step a bit
            ls_sigma = min(0.25, ls_sigma * 1.2)
            sorted_stamp_gen = -1

    return best
