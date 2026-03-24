import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    Key upgrades vs previous best:
      - L-SHADE style *population size reduction* (more exploration early, faster exploitation late)
      - SHADE success-history for F/CR (as before) + p-best fraction annealing
      - Opposition-based / Latin-ish init (stratified per-dimension) + immediate best-centered cloud
      - Stronger but still cheap local search: adaptive coordinate pattern + occasional 2-point quadratic step
      - Diversity injection when convergence detected (variance collapse), not only time-based stagnation
      - Tight time checks + safe evaluation

    Returns
    -------
    best : float
        Best (minimum) fitness found within time budget.
    """

    # ------------------------- helpers -------------------------
    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def clip_inplace(x):
        for i in range(dim):
            lo, hi = bounds[i]
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def bounce(vj, lo, hi):
        # reflection-like boundary handling
        if vj < lo:
            vj = lo + (lo - vj) * 0.5
            if vj > hi:
                vj = lo
        elif vj > hi:
            vj = hi - (vj - hi) * 0.5
            if vj < lo:
                vj = hi
        return vj

    def cauchy(mu, gamma):
        u = random.random() - 0.5
        return mu + gamma * math.tan(math.pi * u)

    def time_frac(now, start, deadline):
        den = max(1e-12, deadline - start)
        t = (now - start) / den
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t

    widths = []
    mids = []
    for i in range(dim):
        lo, hi = bounds[i]
        w = hi - lo
        widths.append(w if w > 0 else 1.0)
        mids.append((lo + hi) * 0.5)

    # ------------------------- time -------------------------
    start = time.perf_counter()
    deadline = start + float(max_time)

    # ------------------------- initialization -------------------------
    # Initial pop size and final pop size for L-SHADE style reduction
    NP0 = max(18, min(90, 22 + 4 * dim))
    NPmin = max(8, min(24, 8 + dim))

    # stratified-ish init per dimension (reduces pure-random clumping)
    pop = []
    # create NP0 candidates, each coordinate uses a random stratum index
    for i in range(NP0):
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            u = (i + random.random()) / float(NP0)  # stratum
            # scramble strata per dimension
            if random.random() < 0.5:
                u = 1.0 - u
            x[d] = lo + u * (hi - lo)
        pop.append(x)

    # add a small best-centered Gaussian cloud around midpoints to catch easy basins
    cloud = max(2, min(10, NP0 // 8))
    for _ in range(cloud):
        x = []
        for d in range(dim):
            lo, hi = bounds[d]
            s = 0.18 * widths[d]
            v = random.gauss(mids[d], s)
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            x.append(v)
        pop.append(x)

    # if cloud added too many, trim; if too few, fill randomly
    if len(pop) > NP0:
        pop = pop[:NP0]
    while len(pop) < NP0:
        pop.append(rand_vec())

    fit = [safe_eval(x) for x in pop]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # JADE archive
    archive = []
    archive_max = NP0

    # SHADE memories
    H = 10
    MF = [0.55] * H
    MCR = [0.75] * H
    mem_idx = 0

    # restart / diversity logic
    last_best = best_val
    no_improve_gens = 0
    restart_after = max(24, 9 * dim)

    # local search trust region
    tr = 0.32

    # ------------------------- local search -------------------------
    def local_search(x0, f0, budget, scale):
        # adaptive coordinate pattern search + occasional 2-point quadratic step
        x = x0[:]
        f = f0
        steps = [max(1e-12, scale * w) for w in widths]

        # store a few best seen points on single coordinates for parabolic guess
        # (x_minus, f_minus), (x0, f0), (x_plus, f_plus) for a chosen j
        for _ in range(budget):
            if time.perf_counter() >= deadline:
                break

            trial = x[:]
            if dim == 1 or random.random() < 0.78:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                s = steps[j]

                # mostly coordinate steps; sometimes try a small "bracket + parabola"
                if random.random() < 0.18:
                    xj0 = x[j]
                    # propose minus/plus
                    xm = max(lo, xj0 - s)
                    xp = min(hi, xj0 + s)

                    if xm == xj0 and xp == xj0:
                        # can't move; fallback
                        trial[j] = bounce(xj0 + random.gauss(0.0, s), lo, hi)
                        clip_inplace(trial)
                        ft = safe_eval(trial)
                    else:
                        # evaluate bracket points (2 evals max)
                        tm = x[:]
                        tp = x[:]
                        tm[j] = xm
                        tp[j] = xp
                        fm = safe_eval(tm) if xm != xj0 else f
                        if time.perf_counter() >= deadline:
                            return x, f
                        fp = safe_eval(tp) if xp != xj0 else f

                        # quadratic minimizer for three points: (xm,fm), (xj0,f), (xp,fp)
                        # compute only if distinct and well-conditioned
                        denom = (xm - xj0) * (xm - xp) * (xj0 - xp)
                        if abs(denom) > 1e-18 and xm != xj0 and xp != xj0 and xm != xp:
                            a = (xp * (f - fm) + xj0 * (fm - fp) + xm * (fp - f)) / denom
                            b = (xp * xp * (fm - f) + xj0 * xj0 * (fp - fm) + xm * xm * (f - fp)) / denom
                            if abs(a) > 1e-18:
                                xq = -b / (2.0 * a)
                                # clip to bracket
                                if xq < lo:
                                    xq = lo
                                elif xq > hi:
                                    xq = hi
                                tq = x[:]
                                tq[j] = xq
                                fq = safe_eval(tq)
                                ft = fq
                                trial = tq
                            else:
                                # fallback pick best of bracket
                                if fm <= fp and fm < f:
                                    trial, ft = tm, fm
                                elif fp < f:
                                    trial, ft = tp, fp
                                else:
                                    trial, ft = x, f
                        else:
                            # fallback: choose best among fm/fp/current
                            if fm <= fp and fm < f:
                                trial, ft = tm, fm
                            elif fp < f:
                                trial, ft = tp, fp
                            else:
                                trial, ft = x, f
                else:
                    # plain coordinate poke
                    ss = s * (1.0 if random.random() < 0.88 else 2.8)
                    trial[j] = trial[j] + (ss if random.random() < 0.5 else -ss)
                    clip_inplace(trial)
                    ft = safe_eval(trial)
            else:
                # small random direction move
                for j in range(dim):
                    trial[j] += random.gauss(0.0, steps[j])
                clip_inplace(trial)
                ft = safe_eval(trial)

            if ft < f:
                x, f = trial[:], ft
                # expand a touch on success
                if random.random() < 0.35:
                    steps = [s * 1.16 for s in steps]
            else:
                # contract on failure
                if random.random() < 0.50:
                    steps = [s * 0.86 for s in steps]

        return x, f

    # ------------------------- main loop -------------------------
    gen = 0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            return best_val

        gen += 1
        t = time_frac(now, start, deadline)

        # L-SHADE population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * t))
        if target_NP < NPmin:
            target_NP = NPmin
        if len(pop) > target_NP:
            # remove worst to reach target
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            # also cap archive with current pop size
            archive_max = max(8, len(pop))

        NP = len(pop)
        order = sorted(range(NP), key=lambda i: fit[i])

        # p-best fraction anneals: larger early, smaller late for exploitation
        p_best_frac = 0.35 * (1.0 - t) + 0.12 * t
        pcount = max(2, int(math.ceil(p_best_frac * NP)))

        succ_F, succ_CR, succ_dF = [], [], []

        # convergence / diversity indicator (cheap): normalized spread on a few dims
        # used for injection when collapsed
        collapsed = False
        if NP >= 10 and dim >= 1 and gen % 6 == 0:
            # sample up to 5 dims
            dd = min(dim, 5)
            dims = random.sample(range(dim), dd) if dim > dd else list(range(dim))
            spread = 0.0
            for d in dims:
                vals = [pop[i][d] for i in range(NP)]
                mn = min(vals); mx = max(vals)
                spread += (mx - mn) / max(1e-12, widths[d])
            spread /= float(dd)
            if spread < 0.015 + 0.02 * (1.0 - t):
                collapsed = True

        # DE generation
        for i in range(NP):
            if time.perf_counter() >= deadline:
                return best_val

            r = random.randrange(H)
            muF, muCR = MF[r], MCR[r]

            CRi = random.gauss(muCR, 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            Fi = cauchy(muF, 0.12)
            tries = 0
            while Fi <= 0.0 and tries < 7:
                Fi = cauchy(muF, 0.12)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.07
            if Fi > 1.0:
                Fi = 1.0

            # anneal Fi slightly late
            Fi = max(0.05, min(1.0, Fi * (1.0 - 0.25 * t)))

            pbest = order[random.randrange(pcount)]

            # pick r1 distinct from i,pbest
            while True:
                r1 = random.randrange(NP)
                if r1 != i and r1 != pbest:
                    break

            # pick r2 from pop U archive
            pool = NP + len(archive)
            if pool <= 1:
                r2v = pop[random.randrange(NP)]
            else:
                while True:
                    k = random.randrange(pool)
                    if k < NP:
                        if k != i and k != pbest and k != r1:
                            r2v = pop[k]
                            break
                    else:
                        r2v = archive[k - NP]
                        break

            xi = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]
            xr2 = r2v

            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                lo, hi = bounds[j]
                v[j] = bounce(vj, lo, hi)

            trial = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]

            ft = safe_eval(trial)

            if ft <= fit[i]:
                # archive update
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                df = fit[i] - ft
                pop[i] = trial
                fit[i] = ft

                succ_F.append(Fi)
                succ_CR.append(CRi)
                succ_dF.append(df if df > 0.0 else 1e-12)

                if ft < best_val:
                    best_val = ft
                    best_x = trial[:]

        # update SHADE memories
        if succ_F:
            wsum = sum(succ_dF)
            if wsum <= 0.0:
                wsum = float(len(succ_dF))

            numF = 0.0
            denF = 0.0
            meanCR = 0.0
            for Fi, CRi, dfi in zip(succ_F, succ_CR, succ_dF):
                w = dfi / wsum
                numF += w * Fi * Fi
                denF += w * Fi
                meanCR += w * CRi

            if denF > 1e-12:
                MF[mem_idx] = max(0.05, min(0.95, numF / denF))
            MCR[mem_idx] = max(0.0, min(1.0, meanCR))
            mem_idx = (mem_idx + 1) % H

        # local search schedule: more late, but still cheap
        if time.perf_counter() < deadline and (gen % (3 if t < 0.60 else 2) == 0):
            base = 0.22 * (1.0 - t) + 0.010 * t
            ls_budget = max(10, min(70, 4 * dim + 12))
            bx, bv = local_search(best_x, best_val, ls_budget, max(1e-10, tr * base))
            if bv < best_val:
                best_x, best_val = bx, bv
                tr = min(2.8, tr * 1.20)
            else:
                tr = max(0.14, tr * 0.90)

        # stagnation bookkeeping
        if best_val < last_best - 1e-12:
            last_best = best_val
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # diversity injection when collapsed OR stagnating
        if (collapsed or no_improve_gens >= restart_after) and time.perf_counter() < deadline:
            no_improve_gens = 0
            NP = len(pop)
            order = sorted(range(NP), key=lambda i: fit[i])
            elite = max(2, NP // 5)

            new_pop = [pop[idx][:] for idx in order[:elite]]
            new_fit = [fit[idx] for idx in order[:elite]]

            # reseed scale: tighter late
            center_scale = 0.60 * (1.0 - t) + 0.045 * t

            while len(new_pop) < NP and time.perf_counter() < deadline:
                r = random.random()
                if r < 0.70:
                    # best-centered gaussian
                    x = []
                    for d in range(dim):
                        lo, hi = bounds[d]
                        s = center_scale * widths[d]
                        v = random.gauss(best_x[d], s)
                        if v < lo:
                            v = lo
                        elif v > hi:
                            v = hi
                        x.append(v)
                elif r < 0.88 and elite >= 2:
                    # interpolate between two elites (useful when basin is on a ridge)
                    a = new_pop[random.randrange(elite)]
                    b = new_pop[random.randrange(elite)]
                    alpha = random.random()
                    x = [a[d] + alpha * (b[d] - a[d]) for d in range(dim)]
                    clip_inplace(x)
                else:
                    # global
                    x = rand_vec()

                fx = safe_eval(x)
                new_pop.append(x)
                new_fit.append(fx)
                if fx < best_val:
                    best_val = fx
                    best_x = x[:]

            pop, fit = new_pop, new_fit
            archive = []
            tr = 0.32
