import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements vs your current best (JADE/SHADE + tiny LS + restart):
      - Better initialization: stratified + opposition + best-centered cloud (often big early gain)
      - L-SHADE style linear population size reduction (explore early, exploit late)
      - p-best fraction annealing (larger early, smaller late)
      - Diversity injection based on BOTH stagnation and population spread-collapse
      - Stronger local search: coordinate search + occasional 3-point parabolic 1D step
      - Tight time checks, safe_eval, reflection boundaries

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
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def bounce(v, lo, hi):
        # reflection-like handling (often better than hard clipping for DE dynamics)
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
    # L-SHADE style sizes (kept modest for speed)
    NP0 = max(18, min(90, 22 + 4 * dim))
    NPmin = max(8, min(26, 8 + dim))

    # Stratified + opposition mix initialization (better coverage than pure random)
    pop = []
    for i in range(NP0):
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            u = (i + random.random()) / float(NP0)
            if random.random() < 0.5:
                u = 1.0 - u
            x[d] = lo + u * (hi - lo)
        if random.random() < 0.35:
            ox = [bounds[d][0] + bounds[d][1] - x[d] for d in range(dim)]
            clip_inplace(ox)
            pop.append(ox)
        pop.append(x)

    if len(pop) > NP0:
        random.shuffle(pop)
        pop = pop[:NP0]
    while len(pop) < NP0:
        pop.append(rand_vec())

    # Add a small cloud around midpoints (often catches easy basins)
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

    if len(pop) > NP0:
        pop = pop[:NP0]

    fit = [safe_eval(x) for x in pop]
    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # JADE archive
    archive = []
    archive_max = len(pop)

    # SHADE memories
    H = 10
    MF = [0.55] * H
    MCR = [0.75] * H
    mem_idx = 0

    # stagnation / collapse tracking
    last_best = best_val
    no_improve_gens = 0
    restart_after = max(24, 9 * dim)

    # local search trust
    tr = 0.32

    # ------------------------- local search -------------------------
    def local_search(x0, f0, budget, rel_scale):
        # Coordinate pattern search + occasional 1D quadratic step
        x = x0[:]
        f = f0
        steps = [max(1e-12, rel_scale * w) for w in widths]

        for _ in range(budget):
            if time.perf_counter() >= deadline:
                break

            trial = x[:]

            if dim == 1 or random.random() < 0.80:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                s = steps[j]

                if random.random() < 0.20:
                    # 3-point bracket + parabola
                    xj0 = x[j]
                    xm = max(lo, xj0 - s)
                    xp = min(hi, xj0 + s)

                    tm = x[:]; tm[j] = xm
                    tp = x[:]; tp[j] = xp

                    fm = safe_eval(tm) if xm != xj0 else f
                    if time.perf_counter() >= deadline:
                        return x, f
                    fp = safe_eval(tp) if xp != xj0 else f

                    if fm >= f and fp >= f:
                        steps[j] = max(1e-12, steps[j] * 0.75)
                        continue

                    denom = (xm - xj0) * (xm - xp) * (xj0 - xp)
                    if abs(denom) > 1e-18 and xm != xp and xm != xj0 and xp != xj0:
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
                                steps[j] = max(1e-12, steps[j] * 0.85)
                            continue

                    if fm < f and fm <= fp:
                        x, f = tm, fm
                        steps[j] = min(widths[j], steps[j] * 1.10)
                    elif fp < f:
                        x, f = tp, fp
                        steps[j] = min(widths[j], steps[j] * 1.10)
                    else:
                        steps[j] = max(1e-12, steps[j] * 0.85)
                else:
                    # plain coordinate poke
                    ss = s * (1.0 if random.random() < 0.9 else 2.5)
                    trial[j] = bounce(trial[j] + (ss if random.random() < 0.5 else -ss), lo, hi)
                    ft = safe_eval(trial)
                    if ft < f:
                        x, f = trial, ft
                        steps[j] = min(widths[j], steps[j] * 1.12)
                    else:
                        steps[j] = max(1e-12, steps[j] * 0.88)
            else:
                # small random direction
                for j in range(dim):
                    trial[j] += random.gauss(0.0, steps[j])
                clip_inplace(trial)
                ft = safe_eval(trial)
                if ft < f:
                    x, f = trial, ft
                    if random.random() < 0.3:
                        steps = [s * 1.10 for s in steps]
                else:
                    if random.random() < 0.5:
                        steps = [max(1e-12, s * 0.90) for s in steps]

        return x, f

    # ------------------------- main loop -------------------------
    gen = 0
    while True:
        if time.perf_counter() >= deadline:
            return best_val

        gen += 1
        t = time_frac()

        # population size reduction (linear)
        target_NP = int(round(NP0 - (NP0 - NPmin) * t))
        if target_NP < NPmin:
            target_NP = NPmin
        if len(pop) > target_NP:
            order_rm = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = order_rm[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            archive_max = max(8, len(pop))

        NP = len(pop)
        order = sorted(range(NP), key=lambda i: fit[i])

        # p-best annealing (bigger early, smaller late)
        p_best_frac = 0.34 * (1.0 - t) + 0.12 * t
        pcount = max(2, int(math.ceil(p_best_frac * NP)))

        # collapse detection: normalized spread on a few dims
        collapsed = False
        if NP >= 10 and gen % 6 == 0:
            dd = min(dim, 5)
            dims = random.sample(range(dim), dd) if dim > dd else list(range(dim))
            spread = 0.0
            for d in dims:
                vals = [pop[i][d] for i in range(NP)]
                spread += (max(vals) - min(vals)) / widths[d]
            spread /= float(dd)
            if spread < 0.015 + 0.02 * (1.0 - t):
                collapsed = True

        succ_F, succ_CR, succ_dF = [], [], []

        # DE loop
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
            Fi = max(0.05, min(1.0, Fi * (1.0 - 0.25 * t)))

            pbest = order[random.randrange(pcount)]

            while True:
                r1 = random.randrange(NP)
                if r1 != i and r1 != pbest:
                    break

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

        # local search (cheap but effective late)
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

        # diversity injection: stagnation OR collapse
        if (collapsed or no_improve_gens >= restart_after) and time.perf_counter() < deadline:
            no_improve_gens = 0

            NP = len(pop)
            order = sorted(range(NP), key=lambda i: fit[i])
            elite = max(2, NP // 5)

            new_pop = [pop[idx][:] for idx in order[:elite]]
            new_fit = [fit[idx] for idx in order[:elite]]

            center_scale = 0.60 * (1.0 - t) + 0.045 * t

            while len(new_pop) < NP and time.perf_counter() < deadline:
                r = random.random()
                if r < 0.70:
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
                    a = new_pop[random.randrange(elite)]
                    b = new_pop[random.randrange(elite)]
                    alpha = random.random()
                    x = [a[d] + alpha * (b[d] - a[d]) for d in range(dim)]
                    clip_inplace(x)
                else:
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
