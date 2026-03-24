import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    What is improved vs your current best (multi-strategy L-SHADE/SHADE+LS):
      1) Better step-size control near convergence:
         - Add "best/1" and "best-to-rand/1" strategies to speed exploitation and escaping
         - Use success-weighted strategy adaptation based on *fitness gain*, not just counts
      2) More robust parameter adaptation:
         - Separate memories per-strategy (MF/MCR per arm) to reduce interference
         - Jitter + late-stage shrink to stabilize fine-tuning
      3) Stronger, cheaper local search:
         - Occasional simultaneous ± coordinate test (2 evals) + quadratic fit when beneficial
         - Late-stage "polish burst" if stagnating but not collapsed
      4) Smarter re-injection:
         - Distinguish collapse vs stagnation: collapse -> global spread restore; stagnation -> best-centered
         - Use mirrored points and elite-difference steps to inject structured diversity
      5) Tighter boundary handling:
         - True reflection until within bounds (not just one bounce)

    Returns:
        best fitness (float)
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

    def reflect_into(v, lo, hi):
        # reflect repeatedly until within bounds (robust when large steps happen)
        if lo > hi:
            lo, hi = hi, lo
        if v != v or v == float("inf") or v == -float("inf"):
            return (lo + hi) * 0.5
        w = hi - lo
        if w <= 0:
            return lo
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
        # numerical safety
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def clip_inplace(x):
        for i in range(dim):
            lo, hi = bounds[i]
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def gaussian_around(center, scale):
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            s = scale * widths[d]
            v = random.gauss(center[d], s)
            x[d] = reflect_into(v, lo, hi)
        return x

    def cauchy(mu, gamma):
        u = random.random() - 0.5
        return mu + gamma * math.tan(math.pi * u)

    # ------------------------- scales/time -------------------------
    widths = [max(1e-12, bounds[i][1] - bounds[i][0]) for i in range(dim)]
    mids = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

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
    # slightly larger early pop can help, but still bounded
    NP0 = max(26, min(140, 30 + 6 * dim))
    NPmin = max(10, min(34, 10 + dim))

    # scrambled stratified init (Latin-ish)
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
            # random mirroring
            if random.random() < 0.35:
                u = 1.0 - u
            x[d] = lo + u * (hi - lo)
        pop.append(x)

        # opposition
        if random.random() < 0.40:
            ox = [bounds[d][0] + bounds[d][1] - x[d] for d in range(dim)]
            clip_inplace(ox)
            pop.append(ox)

    # midpoint cloud
    cloud = max(4, min(20, NP0 // 6))
    for _ in range(cloud):
        x = []
        for d in range(dim):
            lo, hi = bounds[d]
            v = random.gauss(mids[d], 0.20 * widths[d])
            x.append(reflect_into(v, lo, hi))
        pop.append(x)

    # trim/fill
    if len(pop) > NP0:
        random.shuffle(pop)
        pop = pop[:NP0]
    while len(pop) < NP0:
        pop.append(rand_vec())

    fit = [safe_eval(x) for x in pop]
    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # early coordinate scan from current best (cheap)
    probes = min(2 * dim, 16)
    for _ in range(probes):
        if time.perf_counter() >= deadline:
            return best_val
        j = random.randrange(dim)
        lo, hi = bounds[j]
        x = best_x[:]
        for frac in (0.15, 0.50, 0.85):
            if time.perf_counter() >= deadline:
                return best_val
            x[j] = lo + frac * (hi - lo)
            fx = safe_eval(x)
            if fx < best_val:
                best_val = fx
                best_x = x[:]

    # ------------------------- DE state -------------------------
    archive = []
    archive_max = len(pop)

    # Strategies:
    # 0: current-to-pbest/1
    # 1: rand-to-pbest/1
    # 2: rand/1
    # 3: best/1
    # 4: best-to-rand/1  (helps escape local basins while exploiting best)
    K = 5

    # separate SHADE memories per strategy (reduces negative transfer)
    H = 10
    MF = [[0.55] * H for _ in range(K)]
    MCR = [[0.75] * H for _ in range(K)]
    mem_idx = [0] * K

    # strategy probabilities, adapted by *fitness-gain mass*
    strat_p = [0.42, 0.18, 0.12, 0.18, 0.10]
    gain_ema = [1.0] * K

    # stagnation/collapse
    last_best = best_val
    no_improve_gens = 0
    restart_after = max(22, 8 * dim)

    # local search trust
    tr = 0.28

    # ------------------------- local search -------------------------
    def local_search(x0, f0, budget, rel_scale, polish=False):
        x = x0[:]
        f = f0
        steps = [max(1e-12, rel_scale * w) for w in widths]

        for _ in range(budget):
            if time.perf_counter() >= deadline:
                break

            # coordinate-focused late; occasional subspace move early
            if dim == 1 or polish or random.random() < 0.85:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                s = steps[j]

                xj0 = x[j]
                xm = reflect_into(xj0 - s, lo, hi)
                xp = reflect_into(xj0 + s, lo, hi)

                # evaluate both sides (up to 2 evals)
                tm = x[:]; tm[j] = xm
                tp = x[:]; tp[j] = xp

                fm = safe_eval(tm) if xm != xj0 else f
                if time.perf_counter() >= deadline:
                    return x, f
                fp = safe_eval(tp) if xp != xj0 else f

                # take best neighbor if improved
                if fm < f or fp < f:
                    if fm <= fp:
                        x, f = tm, fm
                    else:
                        x, f = tp, fp
                    steps[j] = min(widths[j], steps[j] * 1.12)
                    continue

                # quadratic fit occasionally (1 extra eval)
                if random.random() < (0.28 if polish else 0.18):
                    denom = (xm - xj0) * (xm - xp) * (xj0 - xp)
                    if abs(denom) > 1e-18 and xm != xj0 and xp != xj0 and xm != xp:
                        a = (xp * (f - fm) + xj0 * (fm - fp) + xm * (fp - f)) / denom
                        b = (xp * xp * (fm - f) + xj0 * xj0 * (fp - fm) + xm * xm * (f - fp)) / denom
                        if abs(a) > 1e-18:
                            xq = -b / (2.0 * a)
                            xq = reflect_into(xq, lo, hi)
                            tq = x[:]; tq[j] = xq
                            fq = safe_eval(tq)
                            if fq < f:
                                x, f = tq, fq
                                steps[j] = min(widths[j], steps[j] * 1.10)
                                continue

                # no improvement -> shrink
                steps[j] = max(1e-12, steps[j] * (0.86 if polish else 0.90))
            else:
                # small subspace gaussian
                trial = x[:]
                k = min(dim, 3 if dim >= 6 else 1)
                idx = random.sample(range(dim), k) if dim > k else list(range(dim))
                for j in idx:
                    lo, hi = bounds[j]
                    trial[j] = reflect_into(trial[j] + random.gauss(0.0, steps[j]), lo, hi)
                ft = safe_eval(trial)
                if ft < f:
                    x, f = trial, ft
                    if random.random() < 0.25:
                        steps = [min(widths[j], steps[j] * 1.05) for j in range(dim)]
                else:
                    if random.random() < 0.40:
                        steps = [max(1e-12, steps[j] * 0.94) for j in range(dim)]

        return x, f

    # ------------------------- main loop -------------------------
    gen = 0
    while True:
        if time.perf_counter() >= deadline:
            return best_val

        gen += 1
        t = time_frac()

        # L-SHADE population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * t))
        if target_NP < NPmin:
            target_NP = NPmin
        if len(pop) > target_NP:
            order_rm = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = order_rm[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            archive_max = max(10, len(pop))
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        order = sorted(range(NP), key=lambda i: fit[i])

        # p-best annealing
        p_best_frac = 0.42 * (1.0 - t) + 0.08 * t
        pcount = max(2, int(math.ceil(p_best_frac * NP)))

        # collapse detection
        collapsed = False
        if NP >= 10 and gen % 6 == 0 and dim > 0:
            dd = min(dim, 6)
            dims = random.sample(range(dim), dd) if dim > dd else list(range(dim))
            spread = 0.0
            for d in dims:
                vals = [pop[i][d] for i in range(NP)]
                spread += (max(vals) - min(vals)) / widths[d]
            spread /= float(max(1, dd))
            if spread < (0.010 + 0.025 * (1.0 - t)):
                collapsed = True

        # pick strategy by probabilities
        def pick_strategy():
            r = random.random()
            s = 0.0
            for k in range(K):
                s += strat_p[k]
                if r <= s:
                    return k
            return K - 1

        # success pools per strategy for memory updates
        succ_F = [[] for _ in range(K)]
        succ_CR = [[] for _ in range(K)]
        succ_dF = [[] for _ in range(K)]
        gain_mass = [0.0] * K

        for i in range(NP):
            if time.perf_counter() >= deadline:
                return best_val

            strat = pick_strategy()

            # sample memory slot for this strategy
            rmem = random.randrange(H)
            muF, muCR = MF[strat][rmem], MCR[strat][rmem]

            CRi = random.gauss(muCR, 0.10)
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            Fi = cauchy(muF, 0.12)
            tries = 0
            while Fi <= 0.0 and tries < 7:
                Fi = cauchy(muF, 0.12)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.07
            if Fi > 1.0:
                Fi = 1.0

            # late shrink for fine tuning
            Fi = max(0.04, min(1.0, Fi * (1.0 - 0.30 * t)))

            # choose pbest
            pbest = order[random.randrange(pcount)]
            xi = pop[i]
            xp = pop[pbest]

            # r1, r2 selection
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

            xr1 = pop[r1]

            # mutation
            v = [0.0] * dim
            if strat == 0:
                # current-to-pbest/1
                for j in range(dim):
                    vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                    lo, hi = bounds[j]
                    v[j] = reflect_into(vj, lo, hi)
            elif strat == 1:
                # rand-to-pbest/1
                base = pop[random.randrange(NP)]
                for j in range(dim):
                    vj = base[j] + Fi * (xp[j] - base[j]) + Fi * (xr1[j] - xr2[j])
                    lo, hi = bounds[j]
                    v[j] = reflect_into(vj, lo, hi)
            elif strat == 2:
                # rand/1
                while True:
                    r0 = random.randrange(NP)
                    if r0 != i and r0 != r1 and r0 != pbest:
                        break
                x0 = pop[r0]
                for j in range(dim):
                    vj = x0[j] + Fi * (xr1[j] - xr2[j])
                    lo, hi = bounds[j]
                    v[j] = reflect_into(vj, lo, hi)
            elif strat == 3:
                # best/1
                for j in range(dim):
                    vj = best_x[j] + Fi * (xr1[j] - xr2[j])
                    lo, hi = bounds[j]
                    v[j] = reflect_into(vj, lo, hi)
            else:
                # best-to-rand/1
                base = best_x
                xrand = pop[random.randrange(NP)]
                for j in range(dim):
                    vj = base[j] + Fi * (xrand[j] - base[j]) + Fi * (xr1[j] - xr2[j])
                    lo, hi = bounds[j]
                    v[j] = reflect_into(vj, lo, hi)

            # crossover
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

                if df <= 0.0:
                    df = 1e-12

                succ_F[strat].append(Fi)
                succ_CR[strat].append(CRi)
                succ_dF[strat].append(df)
                gain_mass[strat] += df

                if ft < best_val:
                    best_val = ft
                    best_x = trial[:]

        # update per-strategy SHADE memories
        for k in range(K):
            if succ_F[k]:
                wsum = sum(succ_dF[k])
                if wsum <= 0.0:
                    wsum = float(len(succ_dF[k]))

                numF = 0.0
                denF = 0.0
                meanCR = 0.0
                for Fi, CRi, dfi in zip(succ_F[k], succ_CR[k], succ_dF[k]):
                    w = dfi / wsum
                    numF += w * Fi * Fi
                    denF += w * Fi
                    meanCR += w * CRi

                if denF > 1e-12:
                    MF[k][mem_idx[k]] = max(0.04, min(0.95, numF / denF))
                MCR[k][mem_idx[k]] = max(0.0, min(1.0, meanCR))
                mem_idx[k] = (mem_idx[k] + 1) % H

        # update strategy probabilities using gain mass (EMA + floor)
        for k in range(K):
            gain_ema[k] = 0.85 * gain_ema[k] + 0.15 * (gain_mass[k] + 1e-12)
        ssum = sum(gain_ema)
        if ssum > 0:
            raw = [g / ssum for g in gain_ema]
            floor = 0.06
            strat_p = [max(floor, r) for r in raw]
            ps = sum(strat_p)
            strat_p = [p / ps for p in strat_p]

        # local search: scheduled + "polish burst" on late stagnation
        if time.perf_counter() < deadline:
            do_ls = (gen % (3 if t < 0.55 else 2) == 0)
            if do_ls:
                base = 0.22 * (1.0 - t) + 0.009 * t
                ls_budget = max(14, min(95, 4 * dim + 18))
                bx, bv = local_search(best_x, best_val, ls_budget, max(1e-10, tr * base), polish=False)
                if bv < best_val:
                    best_x, best_val = bx, bv
                    tr = min(3.2, tr * 1.16)
                else:
                    tr = max(0.10, tr * 0.90)

            # late polish burst if stagnating (but not collapsed)
            if (t > 0.60) and (not collapsed) and (no_improve_gens > max(6, dim // 2)):
                base = 0.08 * (1.0 - t) + 0.004 * t
                bx, bv = local_search(best_x, best_val, max(10, min(40, 2 * dim + 10)),
                                      max(1e-12, tr * base), polish=True)
                if bv < best_val:
                    best_x, best_val = bx, bv

        # stagnation bookkeeping
        if best_val < last_best - 1e-12:
            last_best = best_val
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # diversity injection (collapse vs stagnation treated differently)
        if (collapsed or no_improve_gens >= restart_after) and time.perf_counter() < deadline:
            no_improve_gens = 0

            NP = len(pop)
            order = sorted(range(NP), key=lambda i: fit[i])
            elite = max(2, NP // 5)

            new_pop = [pop[idx][:] for idx in order[:elite]]
            new_fit = [fit[idx] for idx in order[:elite]]

            if collapsed:
                # stronger global restore on collapse
                center_scale = 0.85 * (1.0 - t) + 0.10 * t
                mix_best = 0.45
                mix_global = 0.35
                mix_struct = 0.20
            else:
                # stagnation: more best-centered
                center_scale = 0.60 * (1.0 - t) + 0.05 * t
                mix_best = 0.65
                mix_global = 0.20
                mix_struct = 0.15

            while len(new_pop) < NP and time.perf_counter() < deadline:
                r = random.random()
                if r < mix_best:
                    x = gaussian_around(best_x, center_scale)
                elif r < mix_best + mix_struct and elite >= 2:
                    # structured step: best + diff of two elites (often good on ridges)
                    a = new_pop[random.randrange(elite)]
                    b = new_pop[random.randrange(elite)]
                    s = (0.9 - 0.6 * t) * random.random()
                    x = [0.0] * dim
                    for d in range(dim):
                        lo, hi = bounds[d]
                        v = best_x[d] + s * (a[d] - b[d])
                        # occasional mirror to increase spread
                        if random.random() < 0.15:
                            v = lo + hi - v
                        x[d] = reflect_into(v, lo, hi)
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
            archive_max = max(10, len(pop))

            # reset probabilities a bit on collapse
            if collapsed:
                strat_p = [0.40, 0.18, 0.14, 0.18, 0.10]
                gain_ema = [1.0] * K

            tr = 0.28

    # unreachable
