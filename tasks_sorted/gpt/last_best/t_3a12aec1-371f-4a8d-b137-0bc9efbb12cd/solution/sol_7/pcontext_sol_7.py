import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs your best (~12.26):
      1) Keeps a strong global engine (L-SHADE current-to-pbest/1 + archive + LS population reduction).
      2) Adds *proper* restart logic (IPOP-style) with increasing population and re-seeding around best.
      3) Adds inexpensive surrogate-like “selection pressure” via opposition / mirrored sampling on restarts.
      4) Replaces the weak coordinate refinement with a stronger derivative-free local solver:
         - stochastic hillclimb + adaptive step per-dimension
         - occasional 2D subspace rotate moves (helps on non-separable functions)
      5) Adds evaluation cache (quantized) to reduce repeated evaluations during local search/restarts.
      6) Better stagnation detection and “shake” operator (heavy-tail + directed).

    Returns:
        best (float): best objective found within max_time seconds
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
        den = max(1e-12, (deadline - t0))
        x = (time.time() - t0) / den
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    # ---------------- safe eval + cache ----------------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # quantized cache (helps local search + restarts)
    q = 25000 if dim <= 12 else (14000 if dim <= 40 else 9000)
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
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = safe_eval(x)
        if len(cache) >= cache_max:
            # evict ~1% FIFO
            m = max(1, cache_max // 100)
            for _ in range(m):
                if not cache_fifo:
                    break
                kk = cache_fifo.pop(0)
                cache.pop(kk, None)
        cache[k] = fx
        cache_fifo.append(k)
        return fx

    # ---------------- sampling / bounds handling ----------------
    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def reflect_into_bounds(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            span = hi - lo
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

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

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

    # ---------------- local search (stronger than coord refine) ----------------
    def local_refine(best_x, best_f, eval_budget):
        """
        Adaptive stochastic pattern search with:
          - per-dimension step sizes
          - occasional 2D rotation moves
          - heavy-tail “kick” when stuck
        """
        if best_x is None or eval_budget <= 0:
            return best_x, best_f

        x = best_x[:]
        f = best_f

        # initialize steps relative to spans
        base_scale = 0.12 if frac_time() < 0.6 else 0.06
        step = [max(1e-14, base_scale * spans[i]) for i in range(dim)]

        # counters
        noimp = 0
        it = 0

        while eval_budget > 0 and time_left():
            it += 1
            improved = False

            # 70% axis moves, 30% 2D subspace moves
            if dim >= 2 and random.random() < 0.30:
                i, j = random.sample(range(dim), 2)
                # small random rotation in (i,j)
                ang = (random.random() * 2.0 - 1.0) * 0.9  # radians-ish
                ci = math.cos(ang)
                si = math.sin(ang)
                di = step[i] * (0.6 + 0.8 * random.random())
                dj = step[j] * (0.6 + 0.8 * random.random())
                # propose a few variants
                for sgn in (1.0, -1.0):
                    if eval_budget <= 0 or not time_left():
                        break
                    cand = x[:]
                    # rotate a vector (di, dj)
                    vi = sgn * di
                    vj = sgn * dj
                    cand[i] = x[i] + ci * vi - si * vj
                    cand[j] = x[j] + si * vi + ci * vj
                    reflect_into_bounds(cand)
                    fc = eval_cached(cand)
                    eval_budget -= 1
                    if fc < f:
                        x, f = cand, fc
                        step[i] *= 1.10
                        step[j] *= 1.10
                        improved = True
                        break
            else:
                # axis move
                k = random.randrange(dim)
                s = step[k]
                if s > 1e-16 * (spans[k] + 1.0):
                    base = x[k]
                    # try + and -
                    for direction in (1.0, -1.0):
                        if eval_budget <= 0 or not time_left():
                            break
                        cand = x[:]
                        cand[k] = base + direction * s
                        reflect_into_bounds(cand)
                        fc = eval_cached(cand)
                        eval_budget -= 1
                        if fc < f:
                            x, f = cand, fc
                            step[k] *= 1.18
                            improved = True
                            break
                    if not improved:
                        step[k] *= 0.62

            if improved:
                noimp = 0
            else:
                noimp += 1

            # occasional heavy-tail kick if stuck
            if noimp >= max(8, dim // 2) and eval_budget > 0:
                noimp = 0
                cand = x[:]
                # kick scale shrinks with time
                kick = (0.20 if frac_time() < 0.7 else 0.10) * (0.5 + random.random())
                for d in range(dim):
                    # cauchy-like heavy tail using tan
                    u = random.random()
                    kickn = math.tan(math.pi * (u - 0.5))
                    cand[d] = cand[d] + kick * spans[d] * 0.25 * kickn
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    for d in range(dim):
                        step[d] *= 1.05
                else:
                    for d in range(dim):
                        step[d] *= 0.92

            # termination if steps tiny
            if it % 25 == 0:
                tiny = True
                for d in range(dim):
                    if step[d] > 1e-13 * (spans[d] + 1.0):
                        tiny = False
                        break
                if tiny:
                    break

        return x, f

    # ---------------- DE core (L-SHADE) ----------------
    def de_phase(NP_init, phase_time_frac, seed_best_x, seed_best_f, restart_id):
        """
        One DE phase until time ends or phase budget consumed (soft budget based on time fraction).
        Returns best_x, best_f, and last-pop state for possible restart heuristics.
        """
        # phase soft end time
        phase_start = time.time()
        # allocate a portion of remaining time; later restarts get less
        remaining = max(0.0, deadline - phase_start)
        phase_budget = remaining * phase_time_frac
        phase_deadline = min(deadline, phase_start + phase_budget)

        def phase_left():
            return time.time() < phase_deadline

        NP_min = max(8, min(26, 4 * dim + 6))
        NP = max(NP_min, NP_init)

        H = max(8, min(30, NP // 2))
        MF = [0.65] * H
        MCR = [0.5] * H
        mem_ptr = 0

        archive = []
        archive_max = NP

        p_min, p_max = 0.05, 0.20

        # --- init population with mixture (random + around best + opposition) ---
        pop = []
        fit = []

        # seed around best if available
        if seed_best_x is not None and seed_best_f < float("inf"):
            # include best itself
            pop.append(seed_best_x[:])
        # fill
        while len(pop) < NP:
            r = random.random()
            if seed_best_x is not None and r < (0.55 if restart_id == 0 else 0.70):
                # gaussian around best (radius decreases with restarts)
                rad = (0.25 / (1.0 + 0.5 * restart_id))
                x = seed_best_x[:]
                for d in range(dim):
                    x[d] += rad * spans[d] * randn()
                reflect_into_bounds(x)
                pop.append(x)
            elif seed_best_x is not None and r < 0.82:
                # opposition/mirror sample around best: x = lo+hi - (best + noise)
                x = seed_best_x[:]
                for d in range(dim):
                    x[d] += 0.15 * spans[d] * randn()
                    x[d] = (lows[d] + highs[d]) - x[d]
                reflect_into_bounds(x)
                pop.append(x)
            else:
                pop.append(rand_vec())

        for x in pop:
            if not time_left():
                break
            fx = eval_cached(x)
            fit.append(fx)

        best_f = seed_best_f
        best_x = seed_best_x[:] if seed_best_x is not None else None
        for i in range(len(pop)):
            if fit[i] < best_f:
                best_f = fit[i]
                best_x = pop[i][:]

        # linear pop size reduction based on *phase* progress
        def target_np():
            if NP <= NP_min:
                return NP
            denom = max(1e-12, (phase_deadline - phase_start))
            ft = (time.time() - phase_start) / denom
            if ft < 0.0: ft = 0.0
            if ft > 1.0: ft = 1.0
            return int(round(NP_init - ft * (NP_init - NP_min)))

        # strategy mixing
        stagn = 0
        last_best = best_f

        while time_left() and phase_left():
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            if fit[idx_sorted[0]] < best_f:
                best_f = fit[idx_sorted[0]]
                best_x = pop[idx_sorted[0]][:]
            if best_f < last_best - 1e-12:
                last_best = best_f
                stagn = 0
            else:
                stagn += 1

            # adapt explore probability from diversity proxy + time
            # quick diversity estimate from a few pairs
            div = 0.0
            if NP >= 2:
                m = 5 if NP >= 10 else 3
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
                div = s / m

            ft_global = frac_time()
            p_explore = 0.10
            if div < 0.08:
                p_explore = 0.40
            elif div < 0.15:
                p_explore = 0.22
            if ft_global > 0.7:
                p_explore *= 0.65

            p = p_max - (p_max - p_min) * min(1.0, max(0.0, ft_global))
            pcount = max(2, int(math.ceil(p * NP)))

            SCR, SF, dF = [], [], []
            union = pop + archive
            union_n = len(union)

            # stagnation: partial refresh of worst
            if stagn >= max(14, 5 + dim // 2):
                k = max(2, NP // 6)
                worst = idx_sorted[-k:]
                for wi in worst:
                    if not time_left() or not phase_left():
                        break
                    rr = random.random()
                    if best_x is not None and rr < 0.75:
                        x = best_x[:]
                        rad = 0.22
                        for d in range(dim):
                            x[d] += rad * spans[d] * randn()
                        reflect_into_bounds(x)
                    else:
                        x = rand_vec()
                    pop[wi] = x
                    fit[wi] = eval_cached(x)
                archive.clear()
                stagn = 0

            for i in range(NP):
                if not time_left() or not phase_left():
                    break

                xi = pop[i]
                fi = fit[i]

                rmem = random.randrange(H)
                mu_cr = MCR[rmem]
                mu_f = MF[rmem]

                CR = clamp01(mu_cr + 0.10 * randn())

                F = cauchy(mu_f, 0.10)
                tries = 0
                while (F <= 0.0 or F > 1.0) and tries < 16:
                    F = cauchy(mu_f, 0.10)
                    tries += 1
                if F <= 0.0:
                    F = 0.35 + 0.25 * random.random()
                if F > 1.0:
                    F = 1.0
                F = min(1.0, max(1e-6, F * (0.95 + 0.10 * random.random())))

                use_explore = (random.random() < p_explore)

                if not use_explore:
                    pbest_idx = idx_sorted[random.randrange(pcount)]
                    r1 = choose_distinct(1, NP, banned={i, pbest_idx})[0]
                    # r2 from union (pop+archive)
                    r2u = None
                    for _ in range(50):
                        cand = random.randrange(union_n)
                        if cand < NP and cand in (i, pbest_idx, r1):
                            continue
                        r2u = cand
                        break
                    if r2u is None:
                        r2u = random.randrange(union_n)

                    x_pbest = pop[pbest_idx]
                    x_r1 = pop[r1]
                    x_r2 = union[r2u]

                    donor = [0.0] * dim
                    for j in range(dim):
                        donor[j] = xi[j] + F * (x_pbest[j] - xi[j]) + F * (x_r1[j] - x_r2[j])
                else:
                    r1, r2, r3 = choose_distinct(3, NP, banned={i})
                    x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                    donor = [0.0] * dim
                    for j in range(dim):
                        donor[j] = x1[j] + F * (x2[j] - x3[j])

                jrand = random.randrange(dim)
                trial = xi[:]
                for j in range(dim):
                    if j == jrand or random.random() < CR:
                        trial[j] = donor[j]
                reflect_into_bounds(trial)

                ftrial = eval_cached(trial)
                if ftrial <= fi:
                    # archive update
                    if len(archive) < archive_max:
                        archive.append(xi[:])
                    else:
                        archive[random.randrange(archive_max)] = xi[:]

                    pop[i] = trial
                    fit[i] = ftrial

                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial[:]
                        stagn = 0
                        last_best = best_f

                    df = fi - ftrial
                    if df < 0.0: df = 0.0
                    SCR.append(CR)
                    SF.append(F)
                    dF.append(df)

            # memory update
            if dF:
                wsum = sum(dF)
                if wsum <= 1e-18:
                    wts = [1.0 / len(dF)] * len(dF)
                else:
                    wts = [di / wsum for di in dF]

                mcr = 0.0
                for wi, cri in zip(wts, SCR):
                    mcr += wi * cri

                num = 0.0
                den = 0.0
                for wi, fi_ in zip(wts, SF):
                    num += wi * fi_ * fi_
                    den += wi * fi_
                mf = (num / den) if den > 1e-12 else 0.6

                MCR[mem_ptr] = clamp01(mcr)
                MF[mem_ptr] = min(1.0, max(1e-6, mf))
                mem_ptr = (mem_ptr + 1) % H

            # population reduction
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

        return best_x, best_f

    # ---------------- main: multi-restart schedule ----------------
    best = float("inf")
    best_x = None

    # initial quick probe (helps if func is expensive, improves seed for DE)
    if time_left():
        for _ in range(min(6 + dim, 40)):
            if not time_left():
                break
            x = rand_vec()
            fx = eval_cached(x)
            if fx < best:
                best, best_x = fx, x[:]

    restart = 0
    base_NP = max(26, min(110, 12 * dim + 30))

    while time_left():
        ft = frac_time()

        # allocate more time to early restarts; keep at least one later intensification
        # phase_time_frac is fraction of *remaining* time
        if restart == 0:
            phase_frac = 0.55
        elif restart == 1:
            phase_frac = 0.45
        else:
            phase_frac = 0.35

        # IPOP-style: increase pop size on restarts (but cap it)
        NP0 = int(min(220, base_NP * (2 ** min(restart, 2))))
        bx, bf = de_phase(NP0, phase_frac, best_x, best, restart)

        if bf < best:
            best, best_x = bf, bx[:]

        # stronger local refinement after each phase (budget depends on time left)
        if best_x is not None and time_left():
            remaining = max(0.0, deadline - time.time())
            # conservative eval budget since eval cost unknown
            lb = int(min(2500, max(60, 14 * dim + 60)))
            if remaining < 0.20 * max_time:
                lb = int(min(lb, max(40, 8 * dim + 40)))
            bx2, bf2 = local_refine(best_x, best, lb)
            if bf2 < best:
                best, best_x = bf2, bx2[:]

        # if very late, just keep refining
        if ft > 0.93:
            while time_left():
                bx2, bf2 = local_refine(best_x, best, eval_budget=max(20, 6 * dim))
                if bf2 < best:
                    best, best_x = bf2, bx2[:]
                else:
                    break
            break

        restart += 1

    return best
