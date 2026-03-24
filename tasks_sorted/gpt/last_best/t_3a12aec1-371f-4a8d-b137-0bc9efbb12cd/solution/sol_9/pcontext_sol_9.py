import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvement focus vs your best L-SHADE variant:
      1) Adds a real "memetic" layer: DE (global) + Powell-like pattern/coordinate local search (strong final squeeze).
      2) Better handling of boundary constraints with reflection + optional random re-injection when repeatedly clipped.
      3) Uses a lightweight quantized cache to reduce redundant evaluations during local search and late DE.
      4) Stronger stagnation response: multi-start around best with mixed radii + partial restart of subpopulation.
      5) More stable parameter adaptation: success-history for F/CR + per-individual "pbest pressure" schedule.

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

    # ---------------- timing ----------------
    def time_left():
        return time.time() < deadline

    def frac_time():
        den = max(1e-12, deadline - t0)
        x = (time.time() - t0) / den
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    # ---------------- RNG helpers ----------------
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

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # ---------------- bounds ----------------
    def reflect_into_bounds(x):
        # reflection fold; returns x in-place
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            span = hi - lo
            v = x[i]
            if v < lo or v > hi:
                u = (v - lo) % (2.0 * span)
                if u > span:
                    u = 2.0 * span - u
                v = lo + u
            x[i] = v
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------------- evaluation + cache ----------------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # quantized cache (helps late-stage + local search)
    # keep modest to avoid overhead; resolution depends on dim
    q = 24000 if dim <= 12 else (14000 if dim <= 40 else 8000)
    cache = {}
    cache_fifo = []
    cache_max = 45000

    def key_of(x):
        k = []
        for i in range(dim):
            s = spans[i]
            if s <= 0.0:
                k.append(0)
            else:
                u = (x[i] - lows[i]) / s
                if u < 0.0: u = 0.0
                elif u > 1.0: u = 1.0
                k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        kk = key_of(x)
        v = cache.get(kk)
        if v is not None:
            return v
        fx = safe_eval(x)
        if len(cache) >= cache_max:
            # evict small FIFO chunk
            m = max(1, cache_max // 60)
            for _ in range(m):
                if not cache_fifo:
                    break
                k2 = cache_fifo.pop(0)
                cache.pop(k2, None)
        cache[kk] = fx
        cache_fifo.append(kk)
        return fx

    # ---------------- selection helpers ----------------
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

    # ---------------- local search (memetic) ----------------
    def local_memetic(best_x, best_f, eval_budget):
        """
        Powell-ish pattern + coordinate search with adaptive step sizes.
        Designed to be safe for black-box; uses reflection bounds; small cache hits.
        """
        if best_x is None or eval_budget <= 0:
            return best_x, best_f

        x = best_x[:]
        f = best_f

        # initial steps: larger earlier, smaller later
        tf = frac_time()
        base = 0.10 if tf < 0.6 else (0.06 if tf < 0.85 else 0.035)
        step = [max(1e-15, base * spans[i]) for i in range(dim)]

        # pattern direction accumulator (very lightweight)
        pat = [0.0] * dim
        noimp = 0

        while eval_budget > 0 and time_left():
            improved = False

            # coordinate tries (random order subset)
            # do a few coordinates per iteration
            trials = 1 if dim <= 8 else 2
            for _ in range(trials):
                if eval_budget <= 0 or not time_left():
                    break
                j = random.randrange(dim)
                sj = step[j]
                if sj <= 1e-14 * (spans[j] + 1.0):
                    continue

                basev = x[j]
                # try +/- with slight jitter
                for sgn in (1.0, -1.0):
                    if eval_budget <= 0 or not time_left():
                        break
                    cand = x[:]
                    cand[j] = basev + sgn * sj * (0.8 + 0.4 * random.random())
                    reflect_into_bounds(cand)
                    fc = eval_cached(cand)
                    eval_budget -= 1
                    if fc < f:
                        # update pattern direction
                        pat[j] = (cand[j] - x[j])
                        x, f = cand, fc
                        step[j] *= 1.18
                        improved = True
                        break
                if not improved:
                    step[j] *= 0.72

            # pattern move (if we made improvements recently)
            if improved and eval_budget > 0 and time_left():
                cand = x[:]
                # small pattern step; shrink late
                alpha = 0.9 if frac_time() < 0.8 else 0.6
                for d in range(dim):
                    cand[d] = cand[d] + alpha * pat[d]
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    for d in range(dim):
                        step[d] *= 1.05
                else:
                    for d in range(dim):
                        pat[d] *= 0.5

            if improved:
                noimp = 0
            else:
                noimp += 1

            # escape small basin occasionally (heavy-tail micro-kick)
            if noimp >= max(10, dim // 2) and eval_budget > 0 and time_left():
                noimp = 0
                cand = x[:]
                kick = (0.10 if frac_time() < 0.85 else 0.06) * (0.6 + 0.8 * random.random())
                for d in range(dim):
                    cand[d] += kick * spans[d] * math.tan(math.pi * (random.random() - 0.5)) * 0.08
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    for d in range(dim):
                        step[d] *= 1.08
                else:
                    for d in range(dim):
                        step[d] *= 0.95

            # stop if steps tiny
            if random.random() < 0.06:
                tiny = True
                for d in range(dim):
                    if step[d] > 1e-14 * (spans[d] + 1.0):
                        tiny = False
                        break
                if tiny:
                    break

        return x, f

    # ---------------- DE: L-SHADE-like with stronger restarts ----------------
    NP_init = max(26, min(130, 12 * dim + 18))
    NP_min = max(8, min(26, 4 * dim + 6))
    NP = NP_init

    H = max(8, min(32, NP_init // 2))
    MF = [0.62] * H
    MCR = [0.50] * H
    mem_ptr = 0

    archive = []
    archive_max = NP_init

    # pbest schedule
    p_min, p_max = 0.04, 0.22

    # initialize
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
            best, best_x = fi, pop[i][:]

    last_best = best
    stagn = 0
    gen = 0

    def target_np():
        ft = frac_time()
        return int(round(NP_init - ft * (NP_init - NP_min)))

    while time_left():
        gen += 1

        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        if fit[idx_sorted[0]] < best:
            best = fit[idx_sorted[0]]
            best_x = pop[idx_sorted[0]][:]

        if best < last_best - 1e-12:
            stagn = 0
            last_best = best
        else:
            stagn += 1

        ft = frac_time()

        # periodic memetic refinement (increasing strength near end)
        if best_x is not None:
            if ft > 0.78 and (gen % 2 == 0):
                bx, bf = local_memetic(best_x, best, eval_budget=max(20, 6 * dim))
                if bf < best:
                    best, best_x = bf, bx[:]
                    last_best = best
                    stagn = 0
            elif ft > 0.50 and (gen % 5 == 0):
                bx, bf = local_memetic(best_x, best, eval_budget=max(12, 3 * dim))
                if bf < best:
                    best, best_x = bf, bx[:]
                    last_best = best
                    stagn = 0

        # stronger stagnation response: partial restart around best + some global
        if stagn >= max(18, 6 + dim):
            k = max(3, NP // 4)
            worst = idx_sorted[-k:]
            for wi in worst:
                if not time_left():
                    return best
                x = None
                if best_x is not None and random.random() < 0.78:
                    # mixture of radii
                    rad = 0.30 if ft < 0.5 else (0.18 if ft < 0.8 else 0.10)
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += rad * spans[d] * randn()
                    reflect_into_bounds(x)
                else:
                    x = rand_vec()
                pop[wi] = x
                fit[wi] = eval_cached(x)
            archive.clear()
            stagn = 0

        # pbest percent (more exploit later)
        p = p_max - (p_max - p_min) * ft
        pcount = max(2, int(math.ceil(p * NP)))

        # success pools for memory updates
        SCR, SF, dF = [], [], []

        union = pop + archive
        union_n = len(union)

        # mix strategies: exploit more as time passes
        p_exploit = 0.62 if ft < 0.25 else (0.74 if ft < 0.65 else 0.84)

        for i in range(NP):
            if not time_left():
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            CR = clamp01(MCR[r] + 0.10 * randn())

            F = cauchy(MF[r], 0.10)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 16:
                F = cauchy(MF[r], 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.3 + 0.25 * random.random()
            if F > 1.0:
                F = 1.0
            # small jitter + avoid tiny F
            F = min(1.0, max(1e-6, F * (0.94 + 0.12 * random.random())))

            pbest_idx = idx_sorted[random.randrange(pcount)]
            x_pbest = pop[pbest_idx]

            if random.random() < p_exploit:
                # current-to-pbest/1 with archive
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
                # rand/1 (explore)
                r1, r2, r3 = choose_distinct(3, NP, banned={i})
                x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                donor = [0.0] * dim
                for j in range(dim):
                    donor[j] = x1[j] + F * (x2[j] - x3[j])

            # crossover
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]

            # if too many coords got clipped often, inject some random dims (helps with tight bounds)
            reflect_into_bounds(trial)
            if dim >= 6 and random.random() < 0.03:
                # replace a small random subset with uniform samples
                m = 1 if dim < 14 else 2
                for _ in range(m):
                    j = random.randrange(dim)
                    trial[j] = lows[j] + random.random() * spans[j]

            ftrial = eval_cached(trial)

            if ftrial <= fi:
                # archive management
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

        # memory update
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
            mf = (num / den) if den > 1e-12 else 0.6

            MCR[mem_ptr] = clamp01(mcr)
            MF[mem_ptr] = min(1.0, max(1e-6, mf))
            mem_ptr = (mem_ptr + 1) % H

        # reduce population (L-SHADE style)
        targ = target_np()
        if targ < NP:
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            keep = idx_sorted[:targ]
            pop = [pop[ii] for ii in keep]
            fit = [fit[ii] for ii in keep]
            NP = targ

            archive_max = max(NP, NP_min)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

    return best
