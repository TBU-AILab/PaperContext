import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libraries).

    Main changes vs your last version (aimed at better robustness + consistency):
      - Switch core optimizer to L-SHADE-style DE (p-best, success-history memory of F/CR,
        linear population size reduction). This is typically stronger than plain JADE.
      - Use "current-to-pbest/1" with archive, plus per-dimension jitter (small) to avoid
        repeated points / stagnation.
      - Better cache keying: normalize by bounds and quantize in that normalized space.
      - Add a compact best-centric local phase: randomized coordinate pattern search with
        step decay + occasional 2-point quadratic interpolation (cheap).
      - Restart logic based on improvement rate + diversity estimate.

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

    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    scale = [(w if w > 0.0 else 1.0) for w in widths]
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]
    avgw = sum(scale) / float(dim)

    def reflect_coord(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly
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

    # ---- cache: quantize in normalized [0,1] coordinates ----
    # This avoids huge/small domains breaking the cache resolution.
    # Resolution chosen to be coarse enough to get hits, but not too coarse.
    cache = {}
    q = 1e-10  # normalized quantization

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
        if k in cache:
            return cache[k]
        fx = float(func(x))
        cache[k] = fx
        return fx

    # ---- simple diversity estimate (avg normalized L1 distance to best) ----
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

    # scrambled Halton (cheap)
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

    # initial NP and max NP for reduction
    NP_max = max(30, min(140, 12 * dim + 20))
    NP_min = max(12, min(40, 4 * dim + 8))
    NP = NP_max

    # initial candidate pool
    seed_n = max(NP, 16 * dim + 50)
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
    offset = random.randint(1, 30000)
    for k in range(1, halton_n + 1):
        candidates.append(halton_point(offset + k))

    # random fill
    while len(candidates) < seed_n:
        candidates.append(rand_uniform_vec())

    # opposition for a subset
    for x in candidates[:max(14, len(candidates) // 6)]:
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

    # external archive
    archive = []
    archive_max = 2 * NP_max

    # memory size
    H = 8
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    def sample_F(mf):
        # Cauchy around mf, resample if <= 0, cap at 1
        for _ in range(30):
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

    # ---------------------------- local search ----------------------------

    # best-centric coordinate pattern search with adaptive steps
    coord_step = [0.18 * s for s in scale]
    coord_min = [1e-14 * (s + 1.0) for s in scale]
    coord_max = [0.75 * s for s in scale]

    def local_best_search(x0, f0, budget_evals):
        # A small randomized coordinate search; uses ~2 evals per coordinate attempt.
        x = x0[:]
        f = f0
        evals_used = 0

        while evals_used < budget_evals and now() < deadline:
            # pick a small random subset of coords
            idxs = list(range(dim))
            random.shuffle(idxs)
            m = min(dim, max(2, dim // 3))
            improved = False

            for t in range(m):
                if evals_used >= budget_evals or now() >= deadline:
                    break
                j = idxs[t]
                s = coord_step[j]
                if s < coord_min[j]:
                    continue
                lo, hi = bounds[j]

                # try +/- step
                xp = x[:]
                xm = x[:]
                xp[j] = clamp(xp[j] + s, lo, hi)
                xm[j] = clamp(xm[j] - s, lo, hi)
                fp = eval_f(xp); evals_used += 1
                if evals_used >= budget_evals or now() >= deadline:
                    if fp < f:
                        x, f = xp, fp
                        improved = True
                    break
                fm = eval_f(xm); evals_used += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    coord_step[j] = min(coord_max[j], coord_step[j] * 1.22)
                    improved = True
                else:
                    # cheap quadratic probe occasionally
                    if random.random() < 0.20 and evals_used < budget_evals and now() < deadline:
                        denom = (fm - 2.0 * f + fp)
                        if denom != 0.0:
                            tstar = 0.5 * (fm - fp) / denom  # in [-1,1] ideally
                            if -1.0 <= tstar <= 1.0:
                                xv = x[:]
                                xv[j] = clamp(xv[j] + tstar * s, lo, hi)
                                fv = eval_f(xv); evals_used += 1
                                if fv < f:
                                    x, f = xv, fv
                                    coord_step[j] = min(coord_max[j], coord_step[j] * 1.18)
                                    improved = True
                    coord_step[j] = max(coord_min[j], coord_step[j] * 0.70)

            if not improved:
                # global decay when nothing improved in this batch
                for j in range(dim):
                    coord_step[j] = max(coord_min[j], coord_step[j] * 0.90)
                break

        return f, x

    # ---------------------------- main loop ----------------------------

    gen = 0
    stagn = 0
    last_best = best
    last_improve_time = now()
    T = max(1e-6, float(max_time))

    while now() < deadline:
        gen += 1

        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
            last_improve_time = now()
        else:
            stagn += 1

        # linear population size reduction over time
        progress = (now() - t0) / T
        target_NP = int(round(NP_max - progress * (NP_max - NP_min)))
        if target_NP < NP_min:
            target_NP = NP_min
        if target_NP < NP:
            # reduce population by removing worst
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            keep_idx = order[:target_NP]
            pop = [pop[i] for i in keep_idx]
            fit = [fit[i] for i in keep_idx]
            NP = target_NP
            if len(archive) > 0 and len(archive) > 2 * NP:
                archive = archive[-(2 * NP):]

        # rank for pbest
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = 0.12 + 0.18 * (1.0 - min(1.0, progress))  # larger early, smaller later
        pcount = max(2, int(p * NP))

        S_F = []
        S_CR = []
        weights = []

        idxs = list(range(NP))
        random.shuffle(idxs)

        for i in idxs:
            if now() >= deadline:
                return best

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            # choose pbest
            pbest = order[random.randrange(pcount)]
            x_i = pop[i]
            x_p = pop[pbest]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # choose r2 from pop U archive, avoid i and r1 by retries
            union_size = NP + len(archive)
            if union_size <= 2:
                r2u = random.randrange(NP)
            else:
                for _ in range(10):
                    r2u = random.randrange(union_size)
                    if r2u != i and r2u != r1:
                        break

            x_r1 = pop[r1]
            x_r2 = pop[r2u] if r2u < NP else archive[r2u - NP]

            # mutation: current-to-pbest/1 + tiny per-dim jitter
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

                # success history
                df = abs(fit[i] - f_trial)
                S_F.append(F)
                S_CR.append(CR)
                weights.append(df if df > 0.0 else 1e-12)

                pop[i] = trial
                fit[i] = f_trial

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]

        # update memories
        if S_F:
            wsum = sum(weights) + 1e-18
            # weighted Lehmer mean for F, weighted mean for CR
            numF = 0.0
            denF = 0.0
            meanCR = 0.0
            for f, cr, w in zip(S_F, S_CR, weights):
                ww = w / wsum
                numF += ww * f * f
                denF += ww * f
                meanCR += ww * cr
            new_MF = numF / (denF + 1e-18)
            new_MCR = meanCR
            # clamp memories a bit
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

        # local refinement occasionally (budget is small + time-aware)
        if now() < deadline and (gen % 3 == 0 or stagn > 12):
            # local budget grows slightly late in the run
            local_budget = 6 + min(30, dim)  # evaluations
            if progress > 0.65:
                local_budget += 10
            f_loc, x_loc = local_best_search(best_x, best, local_budget)
            if f_loc < best:
                best, best_x = f_loc, x_loc[:]
                stagn = 0
                last_improve_time = now()

        # restart if stuck and diversity is low
        if now() < deadline:
            div = diversity(pop, best_x)
            stuck_time = now() - last_improve_time
            if (stagn > 45 and div < 0.06) or (stuck_time > 0.40 * T and div < 0.04):
                # keep elites
                order = list(range(NP))
                order.sort(key=lambda i: fit[i])
                keep = max(4, NP // 4)
                elites = [pop[order[k]][:] for k in range(keep)]
                elites_fit = [fit[order[k]] for k in range(keep)]

                pop = elites[:]
                fit = elites_fit[:]

                # refill with mix: random / Halton / around best (heavy-tailed)
                refill = NP - keep
                offset = random.randint(1, 60000)
                for k in range(refill):
                    if now() >= deadline:
                        return best
                    r = random.random()
                    if r < 0.35:
                        x = rand_uniform_vec()
                    elif r < 0.70:
                        x = halton_point(offset + k + 1)
                    else:
                        x = best_x[:]
                        for d in range(dim):
                            sd = (0.18 + 0.35 * random.random()) * scale[d]
                            if random.random() < 0.12:
                                sd *= 2.5
                            x[d] += random.gauss(0.0, sd)
                    x = reflect_into_bounds(x)
                    fx = eval_f(x)
                    pop.append(x)
                    fit.append(fx)
                    if fx < best:
                        best, best_x = fx, x[:]

                # shrink archive to remain relevant
                if len(archive) > 2 * NP:
                    archive = archive[-(2 * NP):]

                # relax local steps a bit after restart
                for d in range(dim):
                    coord_step[d] = max(coord_step[d], 0.20 * scale[d])

                stagn = 0
                last_best = best
                last_improve_time = now()

    return best
