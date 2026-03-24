import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements over prior DE/ES versions:
      1) Evaluation cache with quantization + LRU-ish eviction (saves calls in local phases).
      2) Stronger endgame: Powell-like coordinate pattern search (deterministic accept) + random-order axes.
      3) Triangular DE: L-SHADE (current-to-pbest/1 + archive) as core, with occasional "rand/1" bursts
         when diversity collapses.
      4) Budget-aware: dynamically shifts effort from global search -> local refinement as time runs out.
      5) Better stagnation recovery: diversified partial restart using (a) around-best Gaussian, (b) opposition,
         (c) full random.

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

    # ---------------- helpers ----------------
    def time_left():
        return time.time() < deadline

    def frac_time():
        den = (deadline - t0)
        if den <= 0:
            return 1.0
        x = (time.time() - t0) / den
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

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

    def reflect_into_bounds(x):
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

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def choose_distinct(k, n, banned):
        banned = set(banned)
        out = []
        tries = 0
        while len(out) < k and tries < 80 * k:
            r = random.randrange(n)
            tries += 1
            if r in banned:
                continue
            banned.add(r)
            out.append(r)
        if len(out) < k:
            pool = [i for i in range(n) if i not in banned]
            random.shuffle(pool)
            out.extend(pool[:(k - len(out))])
        return out

    # -------------- cache (quantized) --------------
    # Quantization balances re-use in local search with not over-collapsing distinct points.
    # Scale quantization by dimension (slightly coarser in high-d).
    q = 20000 if dim <= 10 else (12000 if dim <= 30 else 8000)

    cache = {}
    cache_keys = []
    cache_max = 30000

    def key_of(x):
        # integer bins in [0,q]
        k = []
        for i in range(dim):
            if spans[i] <= 0.0:
                k.append(0)
            else:
                u = (x[i] - lows[i]) / spans[i]
                if u < 0.0: u = 0.0
                if u > 1.0: u = 1.0
                k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = safe_eval(x)
        if len(cache) >= cache_max:
            # evict ~1% oldest keys
            m = max(1, cache_max // 100)
            for _ in range(m):
                if not cache_keys:
                    break
                kk = cache_keys.pop(0)
                cache.pop(kk, None)
        cache[k] = fx
        cache_keys.append(k)
        return fx

    # ---------------- initialization ----------------
    NP_init = max(26, min(140, 10 * dim + 40))
    NP_min = max(8, min(28, 3 * dim + 6))
    NP = NP_init

    H = max(8, min(30, NP_init // 2))
    MF = [0.6] * H
    MCR = [0.5] * H
    mem_ptr = 0

    archive = []
    archive_max = NP_init

    p_min, p_max = 0.05, 0.20

    pop = [rand_vec() for _ in range(NP)]
    fit = [float("inf")] * NP

    best = float("inf")
    best_x = None

    for i in range(NP):
        if not time_left():
            return best
        fi = eval_cached(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # population reduction target
    def target_np():
        ft = frac_time()
        return int(round(NP_init - ft * (NP_init - NP_min)))

    # --------- diversity metric (cheap) ----------
    def diversity_estimate():
        # estimate relative spread using 6 random pairs
        if NP < 2:
            return 0.0
        m = 6 if NP >= 10 else 3
        s = 0.0
        for _ in range(m):
            a = random.randrange(NP)
            b = random.randrange(NP)
            if a == b:
                b = (b + 1) % NP
            xa, xb = pop[a], pop[b]
            d2 = 0.0
            for j in range(dim):
                t = (xa[j] - xb[j]) / (spans[j] + 1e-300)
                d2 += t * t
            s += math.sqrt(d2 / max(1, dim))
        return s / m

    # ---------------- local refinement: coordinate pattern search ----------------
    def pattern_refine(x0, f0, eval_budget, init_rel_step):
        """
        Powell-like coordinate search with adaptive step per dimension.
        - random order of axes each cycle
        - accept improving move immediately
        - shrink step when no improvement
        """
        if x0 is None:
            return x0, f0
        x = x0[:]
        f = f0
        step = [max(1e-16, init_rel_step * spans[j]) for j in range(dim)]
        noimp_cycles = 0

        while eval_budget > 0 and time_left():
            improved_any = False
            axes = list(range(dim))
            random.shuffle(axes)

            for j in axes:
                if eval_budget <= 0 or not time_left():
                    break
                if step[j] <= 1e-16 * (spans[j] + 1.0):
                    continue

                base = x[j]

                # try +step
                cand = x[:]
                cand[j] = base + step[j]
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    improved_any = True
                    step[j] *= 1.25
                    continue

                if eval_budget <= 0 or not time_left():
                    break

                # try -step
                cand = x[:]
                cand[j] = base - step[j]
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    improved_any = True
                    step[j] *= 1.25
                else:
                    step[j] *= 0.60

            if improved_any:
                noimp_cycles = 0
            else:
                noimp_cycles += 1
                # global shrink to settle
                if noimp_cycles >= 2:
                    for j in range(dim):
                        step[j] *= 0.75
                    noimp_cycles = 0

            # early stop if steps tiny
            tiny = True
            for j in range(dim):
                if step[j] > 1e-14 * (spans[j] + 1.0):
                    tiny = False
                    break
            if tiny:
                break

        return x, f

    # ---------------- main loop ----------------
    stagn = 0
    last_best = best

    while time_left():
        ft = frac_time()

        # sort
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        if fit[idx_sorted[0]] < best:
            best = fit[idx_sorted[0]]
            best_x = pop[idx_sorted[0]][:]
        if best < last_best - 1e-12:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # allocate more time to refinement near end
        if best_x is not None and ft > 0.70:
            # bigger budget at the very end
            budget = max(10, 6 * dim + 30)
            x2, f2 = pattern_refine(best_x, best, budget, init_rel_step=0.06)
            if f2 < best:
                best, best_x = f2, x2[:]
                stagn = 0
                last_best = best
        elif best_x is not None and ft > 0.45 and (stagn % 3 == 0):
            budget = max(8, 3 * dim + 16)
            x2, f2 = pattern_refine(best_x, best, budget, init_rel_step=0.10)
            if f2 < best:
                best, best_x = f2, x2[:]
                stagn = 0
                last_best = best

        # stagnation recovery: partial restart
        if stagn >= max(18, 6 + dim):
            k = max(2, NP // 4)
            worst = idx_sorted[-k:]
            for wi in worst:
                if not time_left():
                    return best
                r = random.random()
                if best_x is not None and r < 0.55:
                    # gaussian around best
                    x = best_x[:]
                    rad = 0.22
                    for d in range(dim):
                        x[d] += rad * spans[d] * randn()
                    reflect_into_bounds(x)
                elif best_x is not None and r < 0.80:
                    # opposition wrt best
                    x = [best_x[d] + (best_x[d] - pop[wi][d]) for d in range(dim)]
                    # small noise
                    for d in range(dim):
                        x[d] += 0.02 * spans[d] * randn()
                    reflect_into_bounds(x)
                else:
                    x = rand_vec()
                pop[wi] = x
                fit[wi] = eval_cached(x)
            archive.clear()
            stagn = 0

        # diversity-triggered exploration bursts
        div = diversity_estimate()
        # lower diversity => more rand/1
        p_explore = 0.10
        if div < 0.08:
            p_explore = 0.45
        elif div < 0.15:
            p_explore = 0.25
        # more exploitation later
        if ft > 0.60:
            p_explore *= 0.7

        # choose p for pbest
        p = p_max - (p_max - p_min) * ft
        pcount = max(2, int(math.ceil(p * NP)))

        SCR, SF, dF = [], [], []

        union = pop + archive
        union_n = len(union)

        for i in range(NP):
            if not time_left():
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            CR = clamp01(mu_cr + 0.10 * randn())

            F = cauchy(mu_f, 0.10)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 16:
                F = cauchy(mu_f, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.3 + 0.2 * random.random()
            if F > 1.0:
                F = 1.0
            F = min(1.0, max(1e-6, F * (0.95 + 0.10 * random.random())))

            use_explore = (random.random() < p_explore)

            if not use_explore:
                # current-to-pbest/1 with archive
                pbest_idx = idx_sorted[random.randrange(pcount)]
                x_pbest = pop[pbest_idx]
                r1 = choose_distinct(1, NP, banned={i, pbest_idx})[0]

                r2u = None
                for _ in range(50):
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
                # rand/1
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

            ftrial = eval_cached(trial)
            if ftrial <= fi:
                # archive store replaced
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = trial
                fit[i] = ftrial

                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]
                    stagn = 0
                    last_best = best

                df = fi - ftrial
                if df < 0.0:
                    df = 0.0
                SCR.append(CR)
                SF.append(F)
                dF.append(df)

        # update memories
        if dF:
            wsum = sum(dF)
            if wsum <= 1e-18:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / wsum for di in dF]

            mcr = 0.0
            for wi, cri in zip(w, SCR):
                mcr += wi * cri

            num = 0.0
            den = 0.0
            for wi, fi_ in zip(w, SF):
                num += wi * fi_ * fi_
                den += wi * fi_
            mf = (num / den) if den > 1e-12 else 0.5

            MCR[mem_ptr] = clamp01(mcr)
            MF[mem_ptr] = min(1.0, max(1e-6, mf))
            mem_ptr = (mem_ptr + 1) % H

        # reduce population
        targ = target_np()
        if targ < NP:
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            keep = idx_sorted[:targ]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = targ
            archive_max = max(NP, NP_min)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

    return best
