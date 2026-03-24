import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Changes vs previous:
      - Uses L-SHADE-style linear population size reduction (often faster convergence).
      - Adds "two-strategy" mutation mix: current-to-pbest/1 (exploit) + rand/1 (explore) with adaptive mixing.
      - Uses jittered/limited F and CR sampling; success-history memory updates kept.
      - Adds a proper intensification stage near the end: bounded, step-controlled coordinate search around best.
      - More robust sampling of distinct indices without relying on object identity.
      - Better stagnation handling and partial reinitialization.

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    # ---------------- utils ----------------
    def time_left():
        return time.time() < deadline

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def reflect_into_bounds(x):
        # fold into [lo, hi] by reflection; stable for overshoots
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            span = hi - lo
            if span <= 0.0:
                x[i] = lo
                continue
            v = x[i]
            if v < lo or v > hi:
                u = (v - lo) % (2.0 * span)
                if u < 0.0:
                    u += 2.0 * span
                if u > span:
                    u = 2.0 * span - u
                v = lo + u
            x[i] = v
        return x

    # Box-Muller normal
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare = z1
        _has_spare = True
        return z0

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def choose_distinct(k, n, banned=set()):
        # returns list of k distinct indices from [0,n), excluding banned
        # simple rejection with fallback shuffle (n typically small/moderate)
        res = []
        tries = 0
        banned_local = set(banned)
        while len(res) < k and tries < 50 * k:
            r = random.randrange(n)
            tries += 1
            if r in banned_local:
                continue
            banned_local.add(r)
            res.append(r)
        if len(res) < k:
            pool = [i for i in range(n) if i not in banned]
            random.shuffle(pool)
            need = k - len(res)
            res.extend(pool[:need])
        return res

    # ---------------- setup ----------------
    # Initial and minimum population sizes for L-SHADE style reduction
    NP_init = max(24, min(120, 12 * dim + 12))
    NP_min = max(8, min(24, 4 * dim + 4))
    NP = NP_init

    # success-history memory
    H = max(8, min(30, NP_init // 2))
    MF = [0.6] * H
    MCR = [0.5] * H
    mem_ptr = 0

    # archive
    archive = []
    archive_max = NP_init

    # pbest range
    p_min, p_max = 0.05, 0.20

    # mutation strategy mixing: probability of using exploit strategy
    p_exploit = 0.70

    # init population
    pop = [rand_vec() for _ in range(NP)]
    fit = [float("inf")] * NP

    best = float("inf")
    best_x = None
    for i in range(NP):
        if not time_left():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # -------------- intensification (late-stage coordinate search) --------------
    def coord_refine(x0, f0, eval_budget, start_scale=0.08):
        if x0 is None:
            return x0, f0
        x = x0[:]
        f = f0
        # per-dimension step sizes
        step = [max(1e-12, start_scale * spans[j]) for j in range(dim)]
        for _ in range(eval_budget):
            if not time_left():
                break
            j = random.randrange(dim)
            if step[j] <= 1e-14 * (spans[j] + 1.0):
                continue

            base = x[j]
            # try + step
            cand = x[:]
            cand[j] = base + step[j]
            reflect_into_bounds(cand)
            fc = safe_eval(cand)
            if fc < f:
                x, f = cand, fc
                step[j] *= 1.15
                continue

            # try - step
            cand = x[:]
            cand[j] = base - step[j]
            reflect_into_bounds(cand)
            fc = safe_eval(cand)
            if fc < f:
                x, f = cand, fc
                step[j] *= 1.15
                continue

            # no improvement: shrink
            step[j] *= 0.6
        return x, f

    # -------------- main loop --------------
    gen = 0
    stagn = 0
    last_best = best

    # For L-SHADE reduction: we reduce with "generation count proxy".
    # We'll compute a target NP from elapsed time fraction.
    def target_np():
        frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        # linear reduction NP_init -> NP_min
        return int(round(NP_init - frac * (NP_init - NP_min)))

    while time_left():
        gen += 1

        # rank
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        if fit[idx_sorted[0]] < best:
            best = fit[idx_sorted[0]]
            best_x = pop[idx_sorted[0]][:]
        if best < last_best - 1e-12:
            stagn = 0
            last_best = best
        else:
            stagn += 1

        # late-stage refinement increases near end
        time_frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        if best_x is not None:
            if time_frac > 0.70 and (gen % 3 == 0):
                bx, bf = coord_refine(best_x, best, eval_budget=max(8, 2 * dim + 10), start_scale=0.06)
                if bf < best:
                    best, best_x = bf, bx[:]
                    stagn = 0
                    last_best = best
            elif time_frac > 0.45 and (gen % 6 == 0):
                bx, bf = coord_refine(best_x, best, eval_budget=max(6, dim + 6), start_scale=0.10)
                if bf < best:
                    best, best_x = bf, bx[:]
                    stagn = 0
                    last_best = best

        # stagnation handling: re-seed some worst
        if stagn >= max(20, 6 + dim):
            k = max(2, NP // 5)
            worst = idx_sorted[-k:]
            for wi in worst:
                if not time_left():
                    return best
                if best_x is not None and random.random() < 0.7:
                    # sample around best
                    x = best_x[:]
                    rad = 0.18
                    for d in range(dim):
                        x[d] = x[d] + rad * spans[d] * randn()
                    reflect_into_bounds(x)
                else:
                    x = rand_vec()
                pop[wi] = x
                fit[wi] = safe_eval(x)
            archive.clear()
            stagn = 0

        # adapt exploit probability: more exploit later
        if time_frac < 0.25:
            p_exploit = 0.60
        elif time_frac < 0.60:
            p_exploit = 0.72
        else:
            p_exploit = 0.82

        # compute p for pbest selection
        p = p_max - (p_max - p_min) * min(1.0, max(0.0, time_frac))

        # accumulate for memory update
        SCR, SF, dF = [], [], []

        # union for r2 (pop + archive)
        union = pop + archive
        union_n = len(union)

        for i in range(NP):
            if not time_left():
                return best

            xi = pop[i]
            fi = fit[i]

            # memory slot
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # sample CR ~ N(mu_cr, 0.1) clipped
            CR = clamp01(mu_cr + 0.10 * randn())

            # sample F ~ Cauchy(mu_f, 0.1), retry; then clamp to (0,1]
            F = cauchy(mu_f, 0.10)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 16:
                F = cauchy(mu_f, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.3 + 0.2 * random.random()
            if F > 1.0:
                F = 1.0
            # mild jitter to avoid collapse
            F = min(1.0, max(1e-6, F * (0.95 + 0.10 * random.random())))

            # choose pbest from top p%
            pcount = max(2, int(math.ceil(p * NP)))
            pbest_idx = idx_sorted[random.randrange(pcount)]
            x_pbest = pop[pbest_idx]

            # choose indices for mutation
            # We'll need r1, r2 (and maybe r3) depending on strategy
            # ensure distinct among {i, pbest_idx}
            if random.random() < p_exploit:
                # current-to-pbest/1 with archive
                # r1 from pop, r2 from union
                r1 = choose_distinct(1, NP, banned={i, pbest_idx})[0]
                # pick r2 from union, disallow using the same vector index if it is within pop and banned
                # easiest: pick index in union and reject if maps to banned pop indices
                r2u = None
                for _ in range(40):
                    cand = random.randrange(union_n)
                    if cand < NP and cand in (i, pbest_idx, r1):
                        continue
                    r2u = cand
                    break
                if r2u is None:
                    r2u = random.randrange(union_n)

                x_r1 = pop[r1]
                x_r2 = union[r2u]

                donor = [0.0] * dim
                for j in range(dim):
                    donor[j] = xi[j] + F * (x_pbest[j] - xi[j]) + F * (x_r1[j] - x_r2[j])
            else:
                # rand/1 (pure exploration): v = xr1 + F*(xr2 - xr3)
                r1, r2, r3 = choose_distinct(3, NP, banned={i})
                x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                donor = [0.0] * dim
                for j in range(dim):
                    donor[j] = x1[j] + F * (x2[j] - x3[j])

            # binomial crossover
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]
            reflect_into_bounds(trial)

            ftrial = safe_eval(trial)
            if ftrial <= fi:
                # archive: store replaced
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = trial
                fit[i] = ftrial

                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

                df = fi - ftrial
                if df < 0.0:
                    df = 0.0
                SCR.append(CR)
                SF.append(F)
                dF.append(df)

        # update memory
        if dF:
            wsum = sum(dF)
            if wsum <= 1e-18:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / wsum for di in dF]

            # weighted mean CR
            mcr = 0.0
            for wi, cri in zip(w, SCR):
                mcr += wi * cri

            # weighted Lehmer mean F
            num = 0.0
            den = 0.0
            for wi, fi_ in zip(w, SF):
                num += wi * fi_ * fi_
                den += wi * fi_
            mf = (num / den) if den > 1e-12 else 0.5

            MCR[mem_ptr] = clamp01(mcr)
            MF[mem_ptr] = min(1.0, max(1e-6, mf))
            mem_ptr = (mem_ptr + 1) % H

        # L-SHADE population size reduction
        targ = target_np()
        if targ < NP:
            # remove worst individuals
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            remove = NP - targ
            # keep best targ indices
            keep = idx_sorted[:targ]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = targ
            # also shrink archive max
            archive_max = max(NP, NP_min)
            if len(archive) > archive_max:
                # random prune
                random.shuffle(archive)
                archive = archive[:archive_max]

    return best
