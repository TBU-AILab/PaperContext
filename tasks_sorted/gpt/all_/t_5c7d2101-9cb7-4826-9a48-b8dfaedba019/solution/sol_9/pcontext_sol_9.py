import random, time, math

def run(func, dim, bounds, max_time):
    """
    Hybrid time-bounded minimizer (self-contained, no external libraries).

    Improvements over your best (#6):
      - Keeps the good parts (DE + diagonal-CMA sampling) but removes restart overhead.
      - Uses a *single continuous* L-SHADE-style DE engine with:
          * current-to-pbest/1 + archive
          * success-history adaptation of F/CR
          * linear population-size reduction over time
      - Adds *cheap but strong* exploitation:
          * diagonal NES/CMA-like sampling around best (no full covariance)
          * adaptive coordinate local search + occasional quadratic 1D step
      - Better stagnation handling:
          * time-based trigger (robust to expensive objectives)
          * replaces worst fraction using a mix of near-best, midpoint-opposition and global samples
      - Safe bound handling via reflection.
      - Optional small quantized cache for dim<=14 to avoid repeated evals.

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ----------------- helpers -----------------
    def now():
        return time.time()

    def clip(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def reflect(v, lo, hi):
        # reflect once; if still outside due to huge step, snap to boundary
        if v < lo:
            v = lo + (lo - v)
            if v > hi:
                v = lo
        elif v > hi:
            v = hi - (v - hi)
            if v < lo:
                v = hi
        return v

    def gauss01():
        # fast approx N(0,1): sum 12 uniforms - 6
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ----------------- edge cases / precompute -----------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    spans = [0.0] * dim
    mids = [0.0] * dim
    for j in range(dim):
        lo, hi = bounds[j]
        s = hi - lo
        if not (s > 0.0):
            x = [0.5 * (b[0] + b[1]) for b in bounds]
            try:
                return float(func(x))
            except Exception:
                return float("inf")
        spans[j] = s
        mids[j] = 0.5 * (lo + hi)

    # ----------------- small quantized cache -----------------
    cache = {}
    CACHE_MAX = 25000
    Q = 8192  # bins per dimension

    def eval_f(x):
        if dim <= 14:
            key = []
            for j in range(dim):
                lo, hi = bounds[j]
                q = int(Q * (x[j] - lo) / (hi - lo) + 0.5)
                if q < 0: q = 0
                if q > Q: q = Q
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

    # ----------------- initialization -----------------
    pop0 = max(28, min(160, 10 * dim + 40))
    pop_min = max(10, min(60, 4 * dim + 12))
    pop_size = pop0

    pop, fit = [], []
    best = float("inf")
    best_x = None

    # mixed init: stratified-ish + random + a few opposition points
    half = max(1, pop_size // 2)
    for i in range(pop_size):
        if now() >= deadline - eps_time:
            return best
        x = [0.0] * dim
        if i < half:
            u0 = (i + random.random()) / float(half)
            for j in range(dim):
                lo, hi = bounds[j]
                u = u0 + 0.17 * (random.random() - 0.5)
                u -= math.floor(u)
                x[j] = lo + (hi - lo) * u
        else:
            x = rand_vec()

        if i < max(2, pop_size // 10) and random.random() < 0.7:
            for j in range(dim):
                x[j] = clip(bounds[j][0] + bounds[j][1] - x[j], bounds[j][0], bounds[j][1])

        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    # ----------------- L-SHADE-ish DE state -----------------
    archive = []
    archive_max = pop_size

    H = 6
    M_F = [0.55] * H
    M_CR = [0.75] * H
    mem_k = 0
    pmax = 0.22

    gen = 0
    sorted_idx = list(range(pop_size))
    sorted_stamp = -1

    def ensure_sorted():
        nonlocal sorted_idx, sorted_stamp
        if sorted_stamp == gen:
            return
        sorted_idx = list(range(pop_size))
        sorted_idx.sort(key=lambda i: fit[i])
        sorted_stamp = gen

    def sample_pbest_index():
        ensure_sorted()
        pmin = 2.0 / pop_size
        p = pmin + (pmax - pmin) * random.random()
        k = max(2, int(p * pop_size))
        return sorted_idx[random.randrange(k)]

    # ----------------- exploitation: diag-NES + local search -----------------
    diag = [1.0] * dim         # per-dim scale in z-space
    sigma = 0.20               # global scale (fraction of span)
    mean = best_x[:] if best_x is not None else pop[0][:]

    ls_sigma = 0.10
    ls_succ = 0
    ls_tr = 0

    def nes_step(lam):
        nonlocal best, best_x, mean, diag, sigma
        cand = []
        for _ in range(lam):
            if now() >= deadline - eps_time:
                break
            z = [gauss01() for _ in range(dim)]
            x = [0.0] * dim
            for j in range(dim):
                xj = mean[j] + (sigma * spans[j]) * (math.sqrt(diag[j]) * z[j])
                x[j] = reflect(xj, bounds[j][0], bounds[j][1])
            fx = eval_f(x)
            cand.append((fx, x, z))

        if not cand:
            return

        cand.sort(key=lambda t: t[0])
        if cand[0][0] < best:
            best = cand[0][0]
            best_x = cand[0][1][:]
            mean = best_x[:]

        mu = max(2, len(cand) // 2)
        w = [math.log(mu + 0.5) - math.log(i) for i in range(1, mu + 1)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]

        # mean update
        new_mean = mean[:]
        for j in range(dim):
            s = 0.0
            for i in range(mu):
                s += w[i] * cand[i][1][j]
            new_mean[j] = s

        # diag update in z-space
        lr = 0.20
        for j in range(dim):
            v = 0.0
            for i in range(mu):
                zj = cand[i][2][j]
                v += w[i] * (zj * zj)
            if v < 0.25: v = 0.25
            if v > 4.0:  v = 4.0
            diag[j] = max(1e-12, (1.0 - lr) * diag[j] + lr * v)

        # sigma adaptation (gentle)
        if cand[0][0] < best + 1e-12:
            sigma *= 1.03
        else:
            sigma *= 0.92
        if sigma < 1e-12: sigma = 1e-12
        if sigma > 0.70:  sigma = 0.70

        mean = new_mean

    def local_improve(budget):
        nonlocal best, best_x, mean, ls_sigma, ls_succ, ls_tr
        if best_x is None:
            return
        xbest = best_x[:]
        fbest = best

        for _ in range(budget):
            if now() >= deadline - eps_time:
                break
            ls_tr += 1

            x = xbest[:]
            if random.random() < 0.78:
                j = random.randrange(dim)
                step = gauss01() * spans[j] * ls_sigma
                x[j] = reflect(x[j] + step, bounds[j][0], bounds[j][1])
            else:
                for j in range(dim):
                    step = gauss01() * spans[j] * (ls_sigma * 0.25)
                    x[j] = reflect(x[j] + step, bounds[j][0], bounds[j][1])

            fx = eval_f(x)
            if fx < fbest:
                xbest, fbest = x, fx
                ls_succ += 1
                if fx < best:
                    best, best_x = fx, x[:]
                    mean = best_x[:]

            # occasional 1D quadratic interpolation
            if random.random() < 0.12 and now() < deadline - eps_time:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                a = xbest[j]
                delta = spans[j] * max(1e-8, ls_sigma * 0.55)

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
                            mean = best_x[:]

        if ls_tr >= 30:
            rate = ls_succ / float(ls_tr)
            ls_sigma *= (1.15 if rate > 0.22 else 0.85)
            if ls_sigma < 1e-10: ls_sigma = 1e-10
            if ls_sigma > 0.35:  ls_sigma = 0.35
            ls_succ = 0
            ls_tr = 0

        best_x = xbest
        best = fbest

    # ----------------- main loop -----------------
    last_best = best
    last_improve_t = now()
    stagnation_seconds = max(0.20, 0.06 * max_time)

    while now() < deadline - eps_time:
        gen += 1

        # L-SHADE population reduction (linear over time)
        if pop_size > pop_min:
            frac = (now() - t0) / max(1e-12, max_time)
            target = int(round(pop0 - (pop0 - pop_min) * min(1.0, frac)))
            if target < pop_size:
                ensure_sorted()
                remove_n = pop_size - target
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
                sorted_stamp = -1

        ensure_sorted()
        S_F, S_CR, dF = [], [], []

        # -------- DE generation --------
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
            if Fi > 1.0:  Fi = 1.0

            pbest = pop[sample_pbest_index()]

            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            if archive and random.random() < 0.55:
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
                    mean = best_x[:]
                    last_improve_t = now()

                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(max(0.0, fi - fu))

        # -------- SHADE memory update --------
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

        # -------- exploitation schedule --------
        time_frac = (now() - t0) / max(1e-12, max_time)

        # small local search always
        if best_x is not None and now() < deadline - eps_time:
            local_budget = 2 + dim // 6 + (1 if time_frac > 0.55 else 0)
            local_improve(local_budget)

        # diag-NES block more often later
        if best_x is not None and random.random() < (0.16 + 0.70 * min(1.0, time_frac)):
            lam = max(12, min(48, 12 + dim // 2))
            nes_step(lam)

        # -------- stagnation injection --------
        if best < last_best - 1e-12:
            last_best = best
            last_improve_t = now()

        if (now() - last_improve_t) >= stagnation_seconds and now() < deadline - eps_time:
            ensure_sorted()
            k = max(2, pop_size // 4)
            worst_idx = sorted_idx[-k:]

            for t, idx in enumerate(worst_idx):
                if now() >= deadline - eps_time:
                    return best

                if best_x is not None and t < k // 2:
                    # near-best jitter + occasional opposition around midpoint
                    x = best_x[:]
                    rad = 0.22 + 0.10 * random.random()
                    for j in range(dim):
                        if random.random() < 0.12:
                            x[j] = clip(bounds[j][0] + bounds[j][1] - x[j], bounds[j][0], bounds[j][1])
                        else:
                            x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * spans[j] * rad,
                                        bounds[j][0], bounds[j][1])
                else:
                    # global / midpoint-biased reseed
                    if random.random() < 0.40:
                        x = [reflect(mids[j] + gauss01() * spans[j] * 0.33, bounds[j][0], bounds[j][1])
                             for j in range(dim)]
                    else:
                        x = rand_vec()

                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    mean = best_x[:]

            # loosen local scales a bit to help basin hopping
            sigma = min(0.35, sigma * 1.30)
            ls_sigma = min(0.25, ls_sigma * 1.20)
            if len(archive) > archive_max // 2:
                archive = archive[:archive_max // 2]
            last_improve_t = now()
            sorted_stamp = -1

    return best
