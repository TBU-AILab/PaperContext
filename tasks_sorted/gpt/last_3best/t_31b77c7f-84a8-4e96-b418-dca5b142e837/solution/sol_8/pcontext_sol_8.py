import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries).

    Why this version improves vs the last CMA-heavy one:
      - Uses a strong, robust global core: L-SHADE style DE (current-to-pbest/1 + archive)
        with success-history parameter memory + linear population reduction.
      - Adds a *very cheap* local refinement around the incumbent best: randomized
        coordinate pattern search + occasional quadratic interpolation.
      - Uses low-discrepancy seeding (scrambled Halton) + stratified/LHS-like sampling
        + opposition points for better initial coverage.
      - Has a normalized-space cache (coarse) to avoid repeated evaluations (cheap win).
      - Restart/diversification triggered by stagnation + low diversity.
      - Strictly respects max_time and returns best fitness found.

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    # ---------------------------- helpers ----------------------------

    def now():
        return time.time()

    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    scale = [(w if w > 0.0 else 1.0) for w in widths]
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]
    avgw = sum(scale) / float(dim)

    def reflect_coord(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly if out-of-bounds
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            else:
                x = hi - (x - hi)
        return clamp(x, lo, hi)

    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = reflect_coord(y[i], lo, hi)
        return y

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ---- cache: normalized quantization (coarse) ----
    cache = {}
    q = 1e-10  # normalized resolution

    def key_of(x):
        k = []
        for i in range(dim):
            lo, hi = bounds[i]
            if hi == lo:
                k.append(0)
            else:
                u = (x[i] - lo) / (hi - lo)
                k.append(int(round(u / q)))
        return tuple(k)

    def eval_f(x):
        k = key_of(x)
        v = cache.get(k, None)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        return fx

    # diversity: avg normalized L1 distance to best
    def diversity(pop, best_x):
        if not pop:
            return 0.0
        s = 0.0
        for x in pop:
            d = 0.0
            for i in range(dim):
                d += abs(x[i] - best_x[i]) / (scale[i] + 1e-12)
            s += d / dim
        return s / len(pop)

    # ---------------------------- seeding ----------------------------

    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            ok = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(dim)
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def vdc_scrambled(n, base):
        v = 0.0
        denom = 1.0
        perm = digit_perm[base]
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += perm[rem] / denom
        return v

    def halton_point(index):
        x = []
        for i in range(dim):
            u = vdc_scrambled(index, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    def opposition(x):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(lo + hi - x[i])
        return y

    # ---------------------------- local search ----------------------------

    coord_step = [0.18 * s for s in scale]
    coord_min = [1e-14 * (s + 1.0) for s in scale]
    coord_max = [0.75 * s for s in scale]

    def local_best_search(x0, f0, budget_evals):
        x = x0[:]
        f = f0
        used = 0

        while used < budget_evals and now() < deadline:
            idxs = list(range(dim))
            random.shuffle(idxs)
            m = min(dim, max(2, dim // 3))
            improved = False

            for t in range(m):
                if used >= budget_evals or now() >= deadline:
                    break
                j = idxs[t]
                s = coord_step[j]
                if s < coord_min[j]:
                    continue
                lo, hi = bounds[j]

                xp = x[:]
                xm = x[:]
                xp[j] = clamp(xp[j] + s, lo, hi)
                xm[j] = clamp(xm[j] - s, lo, hi)

                fp = eval_f(xp); used += 1
                if used >= budget_evals or now() >= deadline:
                    if fp < f:
                        x, f = xp, fp
                        improved = True
                    break
                fm = eval_f(xm); used += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    coord_step[j] = min(coord_max[j], coord_step[j] * 1.22)
                    improved = True
                else:
                    # occasional quadratic probe in [-1,1] * s
                    if random.random() < 0.20 and used < budget_evals and now() < deadline:
                        denom = (fm - 2.0 * f + fp)
                        if denom != 0.0:
                            tstar = 0.5 * (fm - fp) / denom
                            if -1.0 <= tstar <= 1.0:
                                xv = x[:]
                                xv[j] = clamp(xv[j] + tstar * s, lo, hi)
                                fv = eval_f(xv); used += 1
                                if fv < f:
                                    x, f = xv, fv
                                    coord_step[j] = min(coord_max[j], coord_step[j] * 1.18)
                                    improved = True
                    coord_step[j] = max(coord_min[j], coord_step[j] * 0.70)

            if not improved:
                for j in range(dim):
                    coord_step[j] = max(coord_min[j], coord_step[j] * 0.90)
                break

        return f, x

    # ---------------------------- initial population ----------------------------

    NP_max = max(30, min(160, 12 * dim + 24))
    NP_min = max(12, min(44, 4 * dim + 8))
    NP = NP_max

    seed_n = max(NP, min(900, 18 * dim + 80))
    candidates = [centers[:]]

    # LHS-like stratified
    lhs_n = max(12, seed_n // 3)
    strata = []
    for i in range(dim):
        idx = list(range(lhs_n))
        random.shuffle(idx)
        strata.append(idx)
    for k in range(lhs_n):
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            u = (strata[i][k] + random.random()) / lhs_n
            x.append(lo + u * (hi - lo))
        candidates.append(x)

    # Halton
    halton_n = max(12, seed_n // 3)
    offset = random.randint(1, 60000)
    for k in range(1, halton_n + 1):
        candidates.append(halton_point(offset + k))

    # Random fill
    while len(candidates) < seed_n:
        candidates.append(rand_uniform_vec())

    # Opposition for subset
    for x in candidates[:max(16, len(candidates) // 6)]:
        candidates.append(opposition(x))

    best = float("inf")
    best_x = centers[:]

    scored = []
    for x in candidates:
        if now() >= deadline:
            return best
        x = reflect_into_bounds(x)
        fx = eval_f(x)
        scored.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    pop = [scored[i][1][:] for i in range(min(NP, len(scored)))]
    while len(pop) < NP and now() < deadline:
        x = reflect_into_bounds(rand_uniform_vec())
        fx = eval_f(x)
        pop.append(x)
        if fx < best:
            best, best_x = fx, x[:]

    fit = [eval_f(x) for x in pop]
    for i in range(NP):
        if fit[i] < best:
            best, best_x = fit[i], pop[i][:]

    # ---------------------------- L-SHADE state ----------------------------

    archive = []
    archive_max = 2 * NP_max

    H = 10  # memory size
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    def sample_F(mf):
        # Cauchy around mf, resample if <=0
        for _ in range(25):
            u = random.random()
            F = mf + 0.1 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                return 1.0 if F > 1.0 else F
        return max(0.05, min(1.0, mf))

    def sample_CR(mcr):
        cr = random.gauss(mcr, 0.1)
        if cr < 0.0:
            return 0.0
        if cr > 1.0:
            return 1.0
        return cr

    # ---------------------------- main loop ----------------------------

    gen = 0
    stagn = 0
    last_best = best
    last_improve_time = now()
    T = max(1e-9, float(max_time))

    while now() < deadline:
        gen += 1

        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
            last_improve_time = now()
        else:
            stagn += 1

        # linear population reduction
        progress = (now() - t0) / T
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0

        target_NP = int(round(NP_max - progress * (NP_max - NP_min)))
        if target_NP < NP_min:
            target_NP = NP_min
        if target_NP < NP:
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            keep_idx = order[:target_NP]
            pop = [pop[i] for i in keep_idx]
            fit = [fit[i] for i in keep_idx]
            NP = target_NP
            if len(archive) > 2 * NP:
                archive = archive[-(2 * NP):]

        # pbest fraction: more exploratory early
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        pfrac = 0.25 - 0.12 * progress
        if pfrac < 0.08:
            pfrac = 0.08
        pcount = max(2, int(pfrac * NP))

        S_F, S_CR, W = [], [], []

        idxs = list(range(NP))
        random.shuffle(idxs)

        for i in idxs:
            if now() >= deadline:
                return best

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            pbest = order[random.randrange(pcount)]
            x_i = pop[i]
            x_p = pop[pbest]

            # r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 from pop U archive
            union = NP + len(archive)
            if union <= 2:
                r2u = random.randrange(NP)
            else:
                for _ in range(12):
                    r2u = random.randrange(union)
                    if r2u != i and r2u != r1:
                        break

            x_r1 = pop[r1]
            x_r2 = pop[r2u] if r2u < NP else archive[r2u - NP]

            # current-to-pbest/1 with tiny jitter
            mutant = [0.0] * dim
            for d in range(dim):
                jitter = 0.0005 * scale[d] * (random.random() - 0.5)
                mutant[d] = (x_i[d]
                             + F * (x_p[d] - x_i[d])
                             + F * (x_r1[d] - x_r2[d])
                             + jitter)

            mutant = reflect_into_bounds(mutant)

            # binomial crossover
            trial = x_i[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = mutant[d]
            trial = reflect_into_bounds(trial)

            f_trial = eval_f(trial)

            if f_trial <= fit[i]:
                # archive parent
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                df = abs(fit[i] - f_trial)
                S_F.append(F)
                S_CR.append(CR)
                W.append(df if df > 0.0 else 1e-12)

                pop[i] = trial
                fit[i] = f_trial

                if f_trial < best:
                    best, best_x = f_trial, trial[:]

        # update memories (weighted)
        if S_F:
            wsum = sum(W) + 1e-18
            numF = 0.0
            denF = 0.0
            meanCR = 0.0
            for f, cr, w in zip(S_F, S_CR, W):
                ww = w / wsum
                numF += ww * f * f
                denF += ww * f
                meanCR += ww * cr
            new_MF = numF / (denF + 1e-18)
            new_MCR = meanCR
            if new_MF <= 0.0:
                new_MF = 0.5
            if new_MF > 1.0:
                new_MF = 1.0
            if new_MCR < 0.0:
                new_MCR = 0.0
            if new_MCR > 1.0:
                new_MCR = 1.0
            M_F[k_mem] = new_MF
            M_CR[k_mem] = new_MCR
            k_mem = (k_mem + 1) % H

        # local refinement
        if now() < deadline and (gen % 3 == 0 or stagn > 12):
            local_budget = 6 + min(32, dim)
            if progress > 0.65:
                local_budget += 10
            f_loc, x_loc = local_best_search(best_x, best, local_budget)
            if f_loc < best:
                best, best_x = f_loc, x_loc[:]
                stagn = 0
                last_improve_time = now()

        # restart/diversify if stuck & low diversity
        if now() < deadline:
            div = diversity(pop, best_x)
            stuck_time = now() - last_improve_time
            if (stagn > 45 and div < 0.06) or (stuck_time > 0.40 * T and div < 0.04):
                order = list(range(NP))
                order.sort(key=lambda i: fit[i])
                keep = max(4, NP // 4)
                elites = [pop[order[k]][:] for k in range(keep)]
                elites_fit = [fit[order[k]] for k in range(keep)]

                pop = elites[:]
                fit = elites_fit[:]

                refill = NP - keep
                offset = random.randint(1, 90000)
                for k in range(refill):
                    if now() >= deadline:
                        return best
                    r = random.random()
                    if r < 0.30:
                        x = rand_uniform_vec()
                    elif r < 0.65:
                        x = halton_point(offset + k + 1)
                    else:
                        x = best_x[:]
                        for d in range(dim):
                            sd = (0.18 + 0.40 * random.random()) * scale[d]
                            if random.random() < 0.10:
                                sd *= 2.5
                            x[d] += random.gauss(0.0, sd)
                    x = reflect_into_bounds(x)
                    fx = eval_f(x)
                    pop.append(x)
                    fit.append(fx)
                    if fx < best:
                        best, best_x = fx, x[:]

                if len(archive) > 2 * NP:
                    archive = archive[-(2 * NP):]

                for d in range(dim):
                    coord_step[d] = max(coord_step[d], 0.20 * scale[d])

                stagn = 0
                last_best = best
                last_improve_time = now()

    return best
