import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs prior L-SHADE attempt:
      - Adds a CMA-ES-like evolution strategy with diagonal covariance adaptation (very strong on many continuous problems).
      - Uses mirrored/antithetic sampling and elite recombination for efficiency.
      - Keeps a DE (current-to-pbest/1 + archive) phase as a robust global fallback.
      - Uses a best-centered trust-region local search with adaptive radius + occasional random subspace steps.
      - Uses lightweight evaluation cache (quantized) to avoid wasting time in late search.
      - Multi-stage schedule by time fraction: probe -> CMA-like -> DE -> local intensification.

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

    # ---------- timing ----------
    def time_left():
        return time.time() < deadline

    def time_frac():
        den = max(1e-12, (deadline - t0))
        x = (time.time() - t0) / den
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    # ---------- RNG helpers ----------
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

    # ---------- bounds handling ----------
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

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # ---------- evaluation + cache ----------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # quantized cache; helps during intensification / repeated tries
    q = 30000 if dim <= 12 else (16000 if dim <= 40 else 9000)
    cache = {}
    cache_fifo = []
    cache_max = 60000

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
            # evict ~2% FIFO
            m = max(1, cache_max // 50)
            for _ in range(m):
                if not cache_fifo:
                    break
                kk = cache_fifo.pop(0)
                cache.pop(kk, None)
        cache[k] = fx
        cache_fifo.append(k)
        return fx

    # ---------- choose distinct indices ----------
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

    # ---------- Local trust-region refinement ----------
    def trust_refine(best_x, best_f, eval_budget):
        if best_x is None or eval_budget <= 0:
            return best_x, best_f
        x = best_x[:]
        f = best_f

        # trust radius per dimension (start moderate, shrink with time)
        base = 0.18 if time_frac() < 0.6 else 0.10
        rad = [max(1e-15, base * spans[i]) for i in range(dim)]

        noimp = 0
        while eval_budget > 0 and time_left():
            improved = False

            # mostly coordinate/diagonal steps; sometimes random subspace
            if dim >= 2 and random.random() < 0.25:
                # random 2D move
                i, j = random.sample(range(dim), 2)
                cand = x[:]
                cand[i] += rad[i] * randn()
                cand[j] += rad[j] * randn()
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    rad[i] *= 1.12
                    rad[j] *= 1.12
                    improved = True
            else:
                k = random.randrange(dim)
                # try +/- along k
                step = rad[k] * (0.6 + 0.8 * random.random())
                basev = x[k]
                for sgn in (1.0, -1.0):
                    if eval_budget <= 0 or not time_left():
                        break
                    cand = x[:]
                    cand[k] = basev + sgn * step
                    reflect_into_bounds(cand)
                    fc = eval_cached(cand)
                    eval_budget -= 1
                    if fc < f:
                        x, f = cand, fc
                        rad[k] *= 1.20
                        improved = True
                        break
                if not improved:
                    rad[k] *= 0.65

            if improved:
                noimp = 0
            else:
                noimp += 1

            # occasional heavy-tail kick if stuck
            if noimp >= max(10, dim // 2) and eval_budget > 0:
                noimp = 0
                cand = x[:]
                kick = (0.22 if time_frac() < 0.7 else 0.12) * (0.4 + 0.8 * random.random())
                for d in range(dim):
                    cand[d] += kick * spans[d] * 0.15 * math.tan(math.pi * (random.random() - 0.5))
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    for d in range(dim):
                        rad[d] *= 1.05
                else:
                    for d in range(dim):
                        rad[d] *= 0.92

            # stop if radii tiny
            if random.random() < 0.05:
                tiny = True
                for d in range(dim):
                    if rad[d] > 1e-14 * (spans[d] + 1.0):
                        tiny = False
                        break
                if tiny:
                    break

        return x, f

    # ---------- CMA-ES-like diagonal strategy ----------
    def cma_diag_phase(seed_x, seed_f, phase_deadline):
        """
        Diagonal CMA-ES-like:
          - maintains mean m and diagonal std sigma_i
          - samples lambda candidates (mirrored pairs) ~ N(m, diag(sigma^2))
          - updates mean by weighted recombination of best mu
          - adapts sigma_i by tracking successful steps (simple evolution path / variance update)
        """
        # set initial mean
        if seed_x is None:
            m = rand_vec()
            fm = eval_cached(m)
        else:
            m = seed_x[:]
            fm = seed_f

        best_x = m[:]
        best_f = fm

        # initial sigmas
        # start broad early, narrower later
        tf = time_frac()
        s0 = 0.35 if tf < 0.35 else (0.22 if tf < 0.70 else 0.12)
        sigma = [max(1e-12, s0 * spans[i]) for i in range(dim)]

        # population size
        lam = max(8, min(40, 4 + int(3.0 * math.sqrt(dim)) + dim // 6))
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2

        # recombination weights (log)
        ws = []
        for i in range(mu):
            ws.append(math.log(mu + 0.5) - math.log(i + 1.0))
        wsum = sum(ws)
        ws = [w / wsum for w in ws]

        # learning rates (simple, diagonal)
        c_mean = 1.0
        c_sig = 0.20 if dim <= 20 else 0.12
        floor_sigma = 1e-15

        # track a smoothed successful step (diagonal)
        p = [0.0] * dim

        def phase_left():
            return time.time() < phase_deadline

        noimp = 0
        while time_left() and phase_left():
            # sample lam candidates, use mirrored sampling
            cand = []
            for k in range(lam // 2):
                z = [randn() for _ in range(dim)]
                x1 = [m[i] + sigma[i] * z[i] for i in range(dim)]
                x2 = [m[i] - sigma[i] * z[i] for i in range(dim)]
                reflect_into_bounds(x1)
                reflect_into_bounds(x2)
                f1 = eval_cached(x1)
                if f1 < best_f:
                    best_f, best_x = f1, x1[:]
                    noimp = 0
                f2 = eval_cached(x2)
                if f2 < best_f:
                    best_f, best_x = f2, x2[:]
                    noimp = 0
                cand.append((f1, x1, z))
                cand.append((f2, x2, [-zz for zz in z]))
                if not time_left() or not phase_left():
                    break
            if not cand:
                break

            cand.sort(key=lambda t: t[0])
            if cand[0][0] >= best_f - 1e-14:
                noimp += 1
            else:
                noimp = 0

            # recombine mean
            m_old = m[:]
            m = [0.0] * dim
            for i in range(mu):
                _, xi, _ = cand[i]
                wi = ws[i]
                for d in range(dim):
                    m[d] += wi * xi[d]
            reflect_into_bounds(m)
            fm = eval_cached(m)
            if fm < best_f:
                best_f, best_x = fm, m[:]

            # update "path" and sigma diagonals based on successful steps (using z of elites)
            # approximate step in normalized coordinates: dz = sum(w_i * z_i)
            dz = [0.0] * dim
            for i in range(mu):
                _, _, zi = cand[i]
                wi = ws[i]
                for d in range(dim):
                    dz[d] += wi * zi[d]

            # path update (EMA)
            for d in range(dim):
                p[d] = (1.0 - c_sig) * p[d] + math.sqrt(c_sig * (2.0 - c_sig)) * dz[d]

            # sigma update: if |p| big -> increase, else decrease; plus per-dim scaling
            # keep very stable and bounded to avoid blow-ups
            for d in range(dim):
                # target ~ N(0,1): E|N| ~ 0.798. Compare |p[d]|
                a = abs(p[d])
                # smooth multiplicative factor
                if a > 0.95:
                    fac = 1.08
                elif a < 0.55:
                    fac = 0.93
                else:
                    fac = 1.00
                sigma[d] = max(floor_sigma * (spans[d] + 1.0), min(0.8 * spans[d], sigma[d] * fac))

            # if stuck, broaden a bit and re-center at best
            if noimp >= max(10, 4 + dim // 6):
                noimp = 0
                if best_x is not None:
                    m = best_x[:]
                # broaden modestly
                for d in range(dim):
                    sigma[d] = min(0.8 * spans[d], sigma[d] * 1.35)

        return best_x, best_f

    # ---------- DE phase (robust global fallback) ----------
    def de_phase(seed_x, seed_f, phase_deadline):
        NP_init = max(24, min(120, 10 * dim + 30))
        NP_min = max(8, min(26, 4 * dim + 6))
        NP = NP_init

        H = max(8, min(30, NP // 2))
        MF = [0.65] * H
        MCR = [0.5] * H
        mem_ptr = 0

        archive = []
        archive_max = NP

        p_min, p_max = 0.05, 0.20

        def phase_left():
            return time.time() < phase_deadline

        pop = []
        if seed_x is not None:
            pop.append(seed_x[:])
        while len(pop) < NP:
            if seed_x is not None and random.random() < 0.65:
                rad = 0.25
                x = seed_x[:]
                for d in range(dim):
                    x[d] += rad * spans[d] * randn()
                reflect_into_bounds(x)
                pop.append(x)
            else:
                pop.append(rand_vec())

        fit = [eval_cached(x) for x in pop]

        best_x = seed_x[:] if seed_x is not None else None
        best_f = seed_f
        for i in range(NP):
            if fit[i] < best_f:
                best_f = fit[i]
                best_x = pop[i][:]

        def target_np():
            # time-based linear reduction within this phase
            frac = (time.time() - t0) / max(1e-12, (deadline - t0))
            if frac < 0.0: frac = 0.0
            if frac > 1.0: frac = 1.0
            return int(round(NP_init - frac * (NP_init - NP_min)))

        while time_left() and phase_left():
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            if fit[idx_sorted[0]] < best_f:
                best_f = fit[idx_sorted[0]]
                best_x = pop[idx_sorted[0]][:]

            tf = time_frac()
            p = p_max - (p_max - p_min) * tf
            pcount = max(2, int(math.ceil(p * NP)))

            SCR, SF, dF = [], [], []
            union = pop + archive
            union_n = len(union)

            for i in range(NP):
                if not time_left() or not phase_left():
                    break
                xi = pop[i]
                fi = fit[i]

                rmem = random.randrange(H)
                CR = clamp01(MCR[rmem] + 0.10 * randn())

                F = cauchy(MF[rmem], 0.10)
                tries = 0
                while (F <= 0.0 or F > 1.0) and tries < 12:
                    F = cauchy(MF[rmem], 0.10)
                    tries += 1
                if F <= 0.0:
                    F = 0.35 + 0.25 * random.random()
                if F > 1.0:
                    F = 1.0
                F = min(1.0, max(1e-6, F * (0.95 + 0.10 * random.random())))

                pbest_idx = idx_sorted[random.randrange(pcount)]
                r1 = choose_distinct(1, NP, banned={i, pbest_idx})[0]

                r2u = None
                for _ in range(40):
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

                jrand = random.randrange(dim)
                trial = xi[:]
                for j in range(dim):
                    if j == jrand or random.random() < CR:
                        trial[j] = donor[j]
                reflect_into_bounds(trial)

                ftrial = eval_cached(trial)
                if ftrial <= fi:
                    if len(archive) < archive_max:
                        archive.append(xi[:])
                    else:
                        archive[random.randrange(archive_max)] = xi[:]
                    pop[i] = trial
                    fit[i] = ftrial

                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial[:]

                    df = fi - ftrial
                    if df < 0.0:
                        df = 0.0
                    SCR.append(CR)
                    SF.append(F)
                    dF.append(df)

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

    # ---------- main schedule ----------
    best = float("inf")
    best_x = None

    # quick probe
    probe = min(60, 10 + 5 * dim)
    for _ in range(probe):
        if not time_left():
            return best
        x = rand_vec()
        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

    # Stage A: CMA-like diagonal (usually very strong)
    if time_left():
        phase_deadline = min(deadline, time.time() + 0.55 * max(0.0, deadline - time.time()))
        bx, bf = cma_diag_phase(best_x, best, phase_deadline)
        if bf < best:
            best, best_x = bf, bx[:]

    # Stage B: DE refinement / escape
    if time_left():
        phase_deadline = min(deadline, time.time() + 0.35 * max(0.0, deadline - time.time()))
        bx, bf = de_phase(best_x, best, phase_deadline)
        if bf < best:
            best, best_x = bf, bx[:]

    # Stage C: final trust-region intensification
    while time_left():
        # small, repeated budgets; adapts naturally to unknown eval cost
        budget = max(30, 8 * dim)
        if time_frac() > 0.85:
            budget = max(20, 6 * dim)
        bx, bf = trust_refine(best_x, best, budget)
        if bf < best:
            best, best_x = bf, bx[:]
        else:
            # if not improving late, stop early
            if time_frac() > 0.92:
                break

    return best
