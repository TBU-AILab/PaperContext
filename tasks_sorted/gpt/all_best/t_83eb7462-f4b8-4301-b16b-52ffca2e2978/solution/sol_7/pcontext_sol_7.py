import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements over your current best (JADE/SHADE + LS + restart):
      - Better initialization: scrambled Latin-ish + opposition + midpoint cloud + a few 1D line probes
      - L-SHADE style linear population size reduction (NP decreases with time)
      - Dual mutation strategies with time-adaptive mixing:
          * current-to-pbest/1 (fast convergence)
          * rand-to-pbest/1 (diversity when stuck)
      - "External archive" as before + occasional archive refresh on collapse
      - Collapse-aware + stagnation-aware injection using BOTH spread and improvement-rate
      - Stronger local search: adaptive coordinate search + small random subspace steps
      - Tight time checks and robust safe_eval

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

    def clip_inplace(x):
        for i in range(dim):
            lo, hi = bounds[i]
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def bounce(v, lo, hi):
        # reflection-like boundary handling (better for DE dynamics than hard clip)
        if v < lo:
            v = lo + (lo - v) * 0.5
            if v > hi:
                v = lo
        elif v > hi:
            v = hi - (v - hi) * 0.5
            if v < lo:
                v = hi
        return v

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def cauchy(mu, gamma):
        u = random.random() - 0.5
        return mu + gamma * math.tan(math.pi * u)

    # ------------------------- precompute scales -------------------------
    widths = [max(1e-12, bounds[i][1] - bounds[i][0]) for i in range(dim)]
    mids = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    # ------------------------- time -------------------------
    start = time.perf_counter()
    deadline = start + float(max_time)

    def time_frac():
        now = time.perf_counter()
        den = max(1e-12, deadline - start)
        t = (now - start) / den
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t

    # ------------------------- initialization -------------------------
    # L-SHADE sizes
    NP0 = max(22, min(110, 26 + 5 * dim))
    NPmin = max(8, min(28, 8 + dim))

    # scrambled stratified init (Latin-ish per dimension via random permutation)
    # build per-dimension permutations of strata indices
    strata = list(range(NP0))
    perms = []
    for _ in range(dim):
        p = strata[:]
        random.shuffle(p)
        perms.append(p)

    pop = []
    for i in range(NP0):
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            u = (perms[d][i] + random.random()) / float(NP0)
            # occasional mirroring to reduce edge bias
            if random.random() < 0.3:
                u = 1.0 - u
            x[d] = lo + u * (hi - lo)
        pop.append(x)

        # opposition candidate sometimes
        if random.random() < 0.35:
            ox = [bounds[d][0] + bounds[d][1] - x[d] for d in range(dim)]
            clip_inplace(ox)
            pop.append(ox)

    # midpoint/best-agnostic cloud
    cloud = max(3, min(14, NP0 // 6))
    for _ in range(cloud):
        x = []
        for d in range(dim):
            lo, hi = bounds[d]
            s = 0.20 * widths[d]
            v = random.gauss(mids[d], s)
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            x.append(v)
        pop.append(x)

    # trim/fill to NP0
    if len(pop) > NP0:
        random.shuffle(pop)
        pop = pop[:NP0]
    while len(pop) < NP0:
        pop.append(rand_vec())

    fit = [safe_eval(x) for x in pop]
    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # quick 1D line probes from best (cheap early win on separable-ish problems)
    # (kept small so it won't hurt on expensive funcs)
    if dim > 0:
        probes = min(2 * dim, 12)
        for _ in range(probes):
            if time.perf_counter() >= deadline:
                return best_val
            j = random.randrange(dim)
            lo, hi = bounds[j]
            x = best_x[:]
            # try a couple of points along that coordinate
            for frac in (0.25, 0.75):
                if time.perf_counter() >= deadline:
                    return best_val
                x[j] = lo + frac * (hi - lo)
                fx = safe_eval(x)
                if fx < best_val:
                    best_val = fx
                    best_x = x[:]

    # ------------------------- adaptive DE state -------------------------
    # external archive
    archive = []
    archive_max = len(pop)

    # SHADE memories
    H = 12
    MF = [0.55] * H
    MCR = [0.75] * H
    mem_idx = 0

    # stagnation / collapse tracking
    last_best = best_val
    no_improve_gens = 0
    restart_after = max(22, 8 * dim)

    # local-search trust region
    tr = 0.30

    # ------------------------- local search -------------------------
    def local_search(x0, f0, budget, rel_scale):
        """
        Adaptive coordinate search + occasional small random subspace step.
        Very cheap; uses bounce boundaries.
        """
        x = x0[:]
        f = f0
        steps = [max(1e-12, rel_scale * w) for w in widths]

        for _ in range(budget):
            if time.perf_counter() >= deadline:
                break

            if dim == 1 or random.random() < 0.82:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                s = steps[j]

                # 3-point bracket + quadratic step sometimes
                if random.random() < 0.22:
                    xj0 = x[j]
                    xm = max(lo, xj0 - s)
                    xp = min(hi, xj0 + s)

                    tm = x[:]; tm[j] = xm
                    tp = x[:]; tp[j] = xp

                    fm = safe_eval(tm) if xm != xj0 else f
                    if time.perf_counter() >= deadline:
                        return x, f
                    fp = safe_eval(tp) if xp != xj0 else f

                    # if both sides worse, shrink
                    if fm >= f and fp >= f:
                        steps[j] = max(1e-12, steps[j] * 0.75)
                        continue

                    denom = (xm - xj0) * (xm - xp) * (xj0 - xp)
                    if abs(denom) > 1e-18 and xm != xj0 and xp != xj0 and xm != xp:
                        a = (xp * (f - fm) + xj0 * (fm - fp) + xm * (fp - f)) / denom
                        b = (xp * xp * (fm - f) + xj0 * xj0 * (fp - fm) + xm * xm * (f - fp)) / denom
                        if abs(a) > 1e-18:
                            xq = -b / (2.0 * a)
                            if xq < lo:
                                xq = lo
                            elif xq > hi:
                                xq = hi
                            tq = x[:]; tq[j] = xq
                            fq = safe_eval(tq)
                            if fq < f:
                                x, f = tq, fq
                                steps[j] = min(widths[j], steps[j] * 1.15)
                            else:
                                steps[j] = max(1e-12, steps[j] * 0.86)
                            continue

                    # fallback: take best neighbor
                    if fm < f and fm <= fp:
                        x, f = tm, fm
                        steps[j] = min(widths[j], steps[j] * 1.10)
                    elif fp < f:
                        x, f = tp, fp
                        steps[j] = min(widths[j], steps[j] * 1.10)
                    else:
                        steps[j] = max(1e-12, steps[j] * 0.86)
                else:
                    # simple +/- coordinate poke
                    trial = x[:]
                    ss = s * (1.0 if random.random() < 0.9 else 2.6)
                    trial[j] = bounce(trial[j] + (ss if random.random() < 0.5 else -ss), lo, hi)
                    ft = safe_eval(trial)
                    if ft < f:
                        x, f = trial, ft
                        steps[j] = min(widths[j], steps[j] * 1.12)
                    else:
                        steps[j] = max(1e-12, steps[j] * 0.90)
            else:
                # random small subspace step
                trial = x[:]
                k = min(dim, 1 + (dim >= 6) * 2 + (dim >= 18) * 2)  # 1/3/5 dims
                idx = random.sample(range(dim), k) if dim > k else list(range(dim))
                for j in idx:
                    lo, hi = bounds[j]
                    trial[j] = bounce(trial[j] + random.gauss(0.0, steps[j]), lo, hi)
                ft = safe_eval(trial)
                if ft < f:
                    x, f = trial, ft
                    if random.random() < 0.25:
                        steps = [min(widths[j], steps[j] * 1.08) for j in range(dim)]
                else:
                    if random.random() < 0.45:
                        steps = [max(1e-12, steps[j] * 0.93) for j in range(dim)]

        return x, f

    # ------------------------- main loop -------------------------
    gen = 0
    while True:
        if time.perf_counter() >= deadline:
            return best_val

        gen += 1
        t = time_frac()

        # L-SHADE population size reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * t))
        if target_NP < NPmin:
            target_NP = NPmin
        if len(pop) > target_NP:
            order_rm = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = order_rm[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            archive_max = max(8, len(pop))
            # also trim archive to new cap
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        order = sorted(range(NP), key=lambda i: fit[i])

        # p-best annealing
        p_best_frac = 0.40 * (1.0 - t) + 0.10 * t
        pcount = max(2, int(math.ceil(p_best_frac * NP)))

        # collapse detection via normalized spread
        collapsed = False
        if NP >= 10 and gen % 5 == 0:
            dd = min(dim, 6)
            dims = random.sample(range(dim), dd) if dim > dd else list(range(dim))
            spread = 0.0
            for d in dims:
                vals = [pop[i][d] for i in range(NP)]
                spread += (max(vals) - min(vals)) / widths[d]
            spread /= float(max(1, dd))
            if spread < (0.012 + 0.025 * (1.0 - t)):
                collapsed = True

        succ_F, succ_CR, succ_dF = [], [], []

        # time-adaptive strategy mix:
        # early: more rand-to-pbest (diversity); late: more current-to-pbest (exploit)
        p_curr = 0.55 + 0.35 * t  # in [0.55..0.90]

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

            # modest anneal for fine tuning
            Fi = max(0.05, min(1.0, Fi * (1.0 - 0.28 * t)))

            pbest = order[random.randrange(pcount)]

            # pick r1 distinct
            while True:
                r1 = random.randrange(NP)
                if r1 != i and r1 != pbest:
                    break

            # pick r2 from pop U archive
            pool = NP + len(archive)
            if pool <= 1:
                xr2 = pop[random.randrange(NP)]
            else:
                while True:
                    k = random.randrange(pool)
                    if k < NP:
                        if k != i and k != pbest and k != r1:
                            xr2 = pop[k]
                            break
                    else:
                        xr2 = archive[k - NP]
                        break

            xi = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]

            # mutation: choose between two strategies
            # current-to-pbest/1 OR rand-to-pbest/1
            if random.random() < p_curr:
                base = xi
            else:
                base = pop[random.randrange(NP)]

            v = [0.0] * dim
            for j in range(dim):
                vj = base[j] + Fi * (xp[j] - base[j]) + Fi * (xr1[j] - xr2[j])
                lo, hi = bounds[j]
                v[j] = bounce(vj, lo, hi)

            # crossover
            trial = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]

            ft = safe_eval(trial)

            if ft <= fit[i]:
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

        # local search (more frequent late)
        if time.perf_counter() < deadline and (gen % (3 if t < 0.55 else 2) == 0):
            base = 0.24 * (1.0 - t) + 0.010 * t
            ls_budget = max(10, min(80, 4 * dim + 14))
            bx, bv = local_search(best_x, best_val, ls_budget, max(1e-10, tr * base))
            if bv < best_val:
                best_x, best_val = bx, bv
                tr = min(3.0, tr * 1.18)
            else:
                tr = max(0.12, tr * 0.90)

        # stagnation bookkeeping
        if best_val < last_best - 1e-12:
            last_best = best_val
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # diversity injection: stagnation OR collapse
        if (collapsed or no_improve_gens >= restart_after) and time.perf_counter() < deadline:
            no_improve_gens = 0

            # keep elites
            NP = len(pop)
            order = sorted(range(NP), key=lambda i: fit[i])
            elite = max(2, NP // 5)
            new_pop = [pop[idx][:] for idx in order[:elite]]
            new_fit = [fit[idx] for idx in order[:elite]]

            # reseed scale (tighter late)
            center_scale = 0.65 * (1.0 - t) + 0.040 * t

            while len(new_pop) < NP and time.perf_counter() < deadline:
                r = random.random()
                if r < 0.68:
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
                elif r < 0.86 and elite >= 2:
                    # interpolate between elites
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

            # refresh archive more aggressively on collapse (prevents re-collapsing)
            archive = []
            archive_max = max(8, len(pop))

            # slightly reset memories to avoid lock-in, but keep some learning
            if collapsed:
                for k in range(H):
                    MF[k] = 0.55 + 0.10 * (random.random() - 0.5)
                    if MF[k] < 0.05:
                        MF[k] = 0.05
                    elif MF[k] > 0.95:
                        MF[k] = 0.95
                    MCR[k] = 0.75 + 0.15 * (random.random() - 0.5)
                    if MCR[k] < 0.0:
                        MCR[k] = 0.0
                    elif MCR[k] > 1.0:
                        MCR[k] = 1.0

            tr = 0.30
