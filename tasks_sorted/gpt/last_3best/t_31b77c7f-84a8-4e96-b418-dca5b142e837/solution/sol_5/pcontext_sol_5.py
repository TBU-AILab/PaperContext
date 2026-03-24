import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries).

    Core ideas (anytime, robust):
      - Strong initial design: center + LHS-like + scrambled Halton + opposition.
      - DE "current-to-pbest/1" with JADE-style success-history adaptation + external archive.
      - Explicit evaluation cache (memoization) to avoid wasting calls on duplicates.
      - Better constraint handling: reflection + random re-injection for persistent violators.
      - More effective local search: (a) small trust-region coordinate search on best,
        (b) short Powell-like directional search using cached best directions,
        (c) occasional SPSA for cheap gradient-ish steps.
      - Stagnation handling: adaptive restart that preserves elites and re-seeds diversely.

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        return float(func([]))

    # ---------------------------- helpers ----------------------------

    def now():
        return time.time()

    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    avgw = sum(widths) / float(dim)
    # numeric scale (avoid 0 widths issues)
    scale = [(w if w > 0 else 1.0) for w in widths]
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    def reflect_coord(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly (handles large excursions)
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            elif x > hi:
                x = hi - (x - hi)
        # safety clamp
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        return x

    def reflect_into_bounds(v):
        out = v[:]
        for i in range(dim):
            lo, hi = bounds[i]
            out[i] = reflect_coord(out[i], lo, hi)
        return out

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ---- evaluation cache (memoization) ----
    # Use coarse rounding to merge near-duplicates (common in DE + local).
    # Rounding step proportional to domain scale.
    quant = []
    for i in range(dim):
        q = 1e-12 * (scale[i] + 1.0)
        # make quant a bit coarser to increase cache hits
        q = max(q, 1e-10 * (scale[i] + 1.0))
        quant.append(q)

    cache = {}

    def key_of(x):
        # tuple of ints to avoid float hashing instability
        return tuple(int(round(x[i] / quant[i])) for i in range(dim))

    def eval_f(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        fx = float(func(x))
        cache[k] = fx
        return fx

    def opposition(x):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(lo + hi - x[i])
        return y

    # ---- scrambled Halton for deterministic-ish diversity ----
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

    # ---------------------------- local search ----------------------------

    def coord_trust_search(x0, f0, steps, per_iter_budget):
        """
        Trust-region coordinate search around x0 using adaptive step sizes.
        steps: per-dim step sizes (modified in place outside is ok, but here we copy).
        per_iter_budget: how many coordinates to try per call.
        """
        x = x0[:]
        f = f0
        # pick subset of coords
        idxs = list(range(dim))
        random.shuffle(idxs)
        m = min(dim, max(1, per_iter_budget))
        for t in range(m):
            if now() >= deadline:
                break
            j = idxs[t]
            sj = steps[j]
            if sj <= 0:
                continue
            lo, hi = bounds[j]

            # try + and -
            xp = x[:]
            xm = x[:]
            xp[j] = clamp(xp[j] + sj, lo, hi)
            xm[j] = clamp(xm[j] - sj, lo, hi)
            fp = eval_f(xp)
            fm = eval_f(xm)

            if fp < f or fm < f:
                if fp <= fm:
                    x, f = xp, fp
                else:
                    x, f = xm, fm
                steps[j] = min(0.7 * scale[j], steps[j] * 1.25)
            else:
                steps[j] = max(1e-14 * (scale[j] + 1.0), steps[j] * 0.6)

        return f, x

    def directional_search(x0, f0, dirs, step0):
        """
        Very small Powell-like line search along a handful of directions.
        dirs: list of direction vectors (will be updated outside).
        """
        xbest = x0[:]
        fbest = f0
        if not dirs:
            return fbest, xbest

        # only try a few directions per call
        order = list(range(len(dirs)))
        random.shuffle(order)
        tried = 0
        for idx in order:
            if now() >= deadline:
                break
            d = dirs[idx]
            # normalize direction scale (avoid huge steps)
            norm = 0.0
            for i in range(dim):
                norm += (d[i] / (scale[i] + 1e-12)) ** 2
            norm = math.sqrt(norm) + 1e-12
            # step schedule: try +/- step, then a smaller step
            for mult in (1.0, -1.0, 0.35, -0.35):
                if now() >= deadline:
                    break
                a = (step0 * mult) / norm
                xt = [xbest[i] + a * d[i] for i in range(dim)]
                xt = reflect_into_bounds(xt)
                ft = eval_f(xt)
                if ft < fbest:
                    fbest, xbest = ft, xt
            tried += 1
            if tried >= 3:
                break
        return fbest, xbest

    def spsa_refine(x, f_x, iters, base_step):
        best_loc_x = x[:]
        best_loc_f = f_x
        a = base_step
        c = 0.1 * base_step + 1e-12
        for k in range(1, iters + 1):
            if now() >= deadline:
                break
            ck = c / (k ** 0.101)
            ak = a / (k ** 0.602)

            delta = [1 if random.random() < 0.5 else -1 for _ in range(dim)]
            xp = best_loc_x[:]
            xm = best_loc_x[:]
            for i in range(dim):
                xp[i] += ck * delta[i]
                xm[i] -= ck * delta[i]
            xp = reflect_into_bounds(xp)
            xm = reflect_into_bounds(xm)

            fp = eval_f(xp)
            fm = eval_f(xm)

            diff = (fp - fm) / (2.0 * ck + 1e-12)
            xn = best_loc_x[:]
            for i in range(dim):
                # g_i approx diff * delta_i
                xn[i] = xn[i] - ak * (diff * delta[i])
            xn = reflect_into_bounds(xn)
            fn = eval_f(xn)
            if fn < best_loc_f:
                best_loc_f = fn
                best_loc_x = xn
        return best_loc_f, best_loc_x

    # ---------------------------- initialization ----------------------------

    # population size
    NP = max(24, min(110, 9 * dim + 12))

    # initial candidate pool size
    seed_n = max(NP, 14 * dim + 40)

    candidates = []
    candidates.append(centers[:])

    # LHS-like stratified
    lhs_n = max(10, seed_n // 3)
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
    halton_n = max(10, seed_n // 3)
    offset = random.randint(1, 20000)
    for k in range(1, halton_n + 1):
        candidates.append(halton_point(offset + k))

    # random fill
    while len(candidates) < seed_n:
        candidates.append(rand_uniform_vec())

    # opposition augmentation (subset)
    for x in candidates[:max(12, len(candidates) // 6)]:
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
        x = rand_uniform_vec()
        x = reflect_into_bounds(x)
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        pop.append(x)

    fit = [eval_f(pop[i]) for i in range(NP)]
    for i in range(NP):
        if fit[i] < best:
            best, best_x = fit[i], pop[i][:]

    # ---------------------------- JADE DE state ----------------------------

    archive = []
    archive_max = max(NP, 2 * NP)

    mu_F = 0.55
    mu_CR = 0.60
    c_adapt = 0.12

    def sample_F():
        # Cauchy around mu_F, truncated to (0,1]
        for _ in range(50):
            u = random.random()
            F = mu_F + 0.12 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                return 1.0 if F > 1.0 else F
        return max(0.05, min(1.0, mu_F))

    def sample_CR():
        CR = random.gauss(mu_CR, 0.12)
        if CR < 0.0:
            CR = 0.0
        elif CR > 1.0:
            CR = 1.0
        return CR

    # local search state
    coord_steps = [0.10 * s for s in scale]
    # keep some evolving directions (start with coordinate axes-like random)
    directions = []
    for _ in range(min(6, dim)):
        d = [0.0] * dim
        j = random.randrange(dim)
        d[j] = scale[j]
        directions.append(d)

    stagn = 0
    last_best = best
    gen = 0

    # ---------------------------- main loop ----------------------------

    while now() < deadline:
        gen += 1
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = 0.14
        pcount = max(2, int(p * NP))

        succ_F = []
        succ_CR = []

        idxs = list(range(NP))
        random.shuffle(idxs)

        for i in idxs:
            if now() >= deadline:
                return best

            F = sample_F()
            CR = sample_CR()

            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_p = pop[pbest_idx]

            # r1 from pop (not i)
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 from union(pop, archive)
            union_size = NP + len(archive)
            if union_size <= 2:
                r2u = random.randrange(NP)
            else:
                r2u = random.randrange(union_size)
                # mild loop to avoid trivial duplicates
                tries = 0
                while tries < 6 and (r2u == i or r2u == r1):
                    r2u = random.randrange(union_size)
                    tries += 1

            x_r1 = pop[r1]
            x_r2 = pop[r2u] if r2u < NP else archive[r2u - NP]

            mutant = [0.0] * dim
            for d in range(dim):
                mutant[d] = x_i[d] + F * (x_p[d] - x_i[d]) + F * (x_r1[d] - x_r2[d])

            mutant = reflect_into_bounds(mutant)

            trial = x_i[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = mutant[d]

            trial = reflect_into_bounds(trial)
            f_trial = eval_f(trial)

            if f_trial <= fit[i]:
                # archive replaced parent
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                # update directions using successful displacement sometimes
                if random.random() < 0.35:
                    disp = [trial[d] - x_i[d] for d in range(dim)]
                    directions.append(disp)
                    if len(directions) > 10:
                        directions.pop(0)

                pop[i] = trial
                fit[i] = f_trial
                succ_F.append(F)
                succ_CR.append(CR)

                if f_trial < best:
                    best, best_x = f_trial, trial[:]

        if succ_F:
            num = 0.0
            den = 0.0
            for f in succ_F:
                num += f * f
                den += f
            lehmer_F = num / (den + 1e-12)
            mean_CR = sum(succ_CR) / float(len(succ_CR))
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmer_F
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * mean_CR

        # -------------------- intensified local search --------------------
        # More frequent but very small budgets (anytime-friendly).
        if now() < deadline and (gen % 3 == 0 or stagn > 16):
            # coordinate trust search
            f_loc, x_loc = coord_trust_search(best_x, best, coord_steps, per_iter_budget=min(dim, 12))
            if f_loc < best:
                best, best_x = f_loc, x_loc[:]

            # directional search along learned directions
            step0 = 0.12 * (avgw if avgw > 0 else 1.0)
            f_loc2, x_loc2 = directional_search(best_x, best, directions, step0)
            if f_loc2 < best:
                best, best_x = f_loc2, x_loc2[:]

            # occasional SPSA "kick" when really stuck
            if stagn > 28 and now() < deadline:
                base_step = 0.10 * (avgw if avgw > 0 else 1.0)
                iters = 2 + max(1, dim // 10)
                f_s, x_s = spsa_refine(best_x, best, iters, base_step)
                if f_s < best:
                    best, best_x = f_s, x_s[:]

        # -------------------- restart / diversification --------------------
        if stagn >= 65 and now() < deadline:
            stagn = 0
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            keep = max(5, NP // 4)

            elites = [pop[order[k]][:] for k in range(keep)]
            elites_fit = [fit[order[k]] for k in range(keep)]

            pop = elites[:]
            fit = elites_fit[:]

            # adjust adaptation slightly to explore more
            mu_F = min(0.85, max(0.25, mu_F * 1.08))
            mu_CR = min(0.95, max(0.05, mu_CR * 0.92))

            refill = NP - keep
            offset = random.randint(1, 50000)
            for k in range(refill):
                if now() >= deadline:
                    return best
                r = random.random()
                if r < 0.35:
                    x = rand_uniform_vec()
                elif r < 0.70:
                    x = halton_point(offset + k + 1)
                else:
                    # sample around best (heavy-tailed)
                    x = best_x[:]
                    for d in range(dim):
                        sd = 0.22 * scale[d]
                        # occasional larger jumps
                        if random.random() < 0.15:
                            sd *= 2.5
                        x[d] += random.gauss(0.0, sd)

                x = reflect_into_bounds(x)
                fx = eval_f(x)
                pop.append(x)
                fit.append(fx)
                if fx < best:
                    best, best_x = fx, x[:]

            # shrink archive to keep relevant
            if len(archive) > archive_max // 2:
                archive = archive[-(archive_max // 2):]

            # re-expand coord steps a bit after restart
            for d in range(dim):
                coord_steps[d] = max(coord_steps[d], 0.12 * scale[d])

    return best
