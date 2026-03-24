import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only, time-bounded black-box minimizer.

    Improved vs your current best code mainly by:
      - Much faster bound handling (mirror-modulo; no while loops).
      - Evaluation cache with adaptive rounding + small random jitter to reduce
        pathological cache misses.
      - Stronger DE core: L-SHADE style with linear pop reduction + archive,
        AND a 3-strategy ensemble with online reward adaptation:
            0) current-to-pbest/1 (fast convergence)
            1) rand/1 (exploration)
            2) current-to-best/1 (aggressive exploitation late)
      - Low-cost “polish” local search:
            * opportunistic quadratic step along a coordinate (3 evals) sometimes
            * coordinate step-halving
            * small random subspace pattern search
      - Better stagnation recovery: refresh worst via mixture of
        opposition-to-best, around-best heavy-tail, and random.

    Returns: best (minimum) fitness found within max_time seconds.
    """

    t0 = time.time()
    end_time = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ---------------- timing ----------------
    def timed_out():
        return time.time() >= end_time

    def frac_time():
        if max_time <= 0:
            return 1.0
        return min(1.0, max(0.0, (time.time() - t0) / (max_time + 1e-12)))

    # ---------------- bounds (fast) ----------------
    def reflect_fast_inplace(x):
        # Mirror with modulo into [lo, hi] with period 2*span; O(dim), no loops.
        for i in range(dim):
            lo = lows[i]
            s = spans[i]
            if s <= 0.0:
                x[i] = lo
                continue
            hi = highs[i]
            period = 2.0 * s
            y = (x[i] - lo) % period  # [0, 2s)
            if y > s:
                y = period - y
            xi = lo + y
            if xi < lo: xi = lo
            if xi > hi: xi = hi
            x[i] = xi
        return x

    # ---------------- random helpers ----------------
    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def cauchy(scale=1.0):
        # Cauchy(0, scale)
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # ---------------- eval + cache ----------------
    eval_cache = {}
    cache_fifo = []
    cache_cap = 22000

    def safe_eval(x):
        # Adaptive rounding key; add tiny jitter to reduce systematic collisions.
        ft = frac_time()
        digits = 4 + int(6 * ft)  # 4..10
        # jitter decreases over time
        jit = (1e-12 + 1e-7 * (1.0 - ft))
        key = tuple(round(float(xi + (random.random() - 0.5) * jit), digits) for xi in x)
        v = eval_cache.get(key)
        if v is not None:
            return v
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if math.isnan(v) or math.isinf(v):
            v = float("inf")
        eval_cache[key] = v
        cache_fifo.append(key)
        if len(cache_fifo) > cache_cap:
            # evict chunk
            for _ in range(450):
                if not cache_fifo:
                    break
                k = cache_fifo.pop(0)
                eval_cache.pop(k, None)
        return v

    # ---------------- SHADE sampling ----------------
    def sample_F(mu):
        for _ in range(16):
            f = mu + 0.10 * cauchy(1.0)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return 0.6 if mu <= 0.0 else (1.0 if mu > 1.0 else mu)

    def sample_CR(mu):
        cr = mu + 0.10 * random.gauss(0.0, 1.0)
        if cr < 0.0: return 0.0
        if cr > 1.0: return 1.0
        return cr

    # ---------------- init ----------------
    if dim <= 0:
        # Edge case: no parameters.
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    NP_init = max(40, 18 * dim)
    NP_init = min(NP_init, 220)
    NP_min = max(12, 4 * dim)
    if NP_min > NP_init:
        NP_min = max(12, NP_init // 2)

    # SHADE memory
    H = 18
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    p_best_rate = 0.12
    p_best_rate = max(0.06, min(0.30, p_best_rate))

    archive = []
    archive_cap = NP_init

    # Seeding: corners + random + opposition + center sprays
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    pop = []

    # a few axis corners (good for boundary optima)
    for _ in range(min(8, max(2, NP_init // 12))):
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        pop.append(x)

    while len(pop) < NP_init:
        x = rand_vec()
        pop.append(x)
        if len(pop) < NP_init:
            xo = opposition(x)
            reflect_fast_inplace(xo)
            pop.append(xo)
        if len(pop) < NP_init and random.random() < 0.50:
            xs = center[:]
            rad = 0.25
            for d in range(dim):
                xs[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
            reflect_fast_inplace(xs)
            pop.append(xs)

    pop = pop[:NP_init]
    fit = [float("inf")] * len(pop)

    best = float("inf")
    best_x = pop[0][:]

    for i in range(len(pop)):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # ---------------- DE primitives ----------------
    def pick_idx(NP, excl):
        while True:
            j = random.randrange(NP)
            if j not in excl:
                return j

    def binom_cross(x, v, cr):
        jrand = random.randrange(dim)
        u = [0.0] * dim
        for d in range(dim):
            u[d] = v[d] if (d == jrand or random.random() < cr) else x[d]
        return u

    # ---------------- local search around best ----------------
    ls_sigma = 0.13
    ls_sigma_min = 1e-12
    ls_sigma_max = 0.55

    def local_refine(xb, fb, ft):
        nonlocal ls_sigma
        x = xb[:]
        f = fb

        # budgets: modest but increase with time
        coord_budget = min(dim + 2, 6 + int(16 * ft))
        pattern_budget = 1 + int(4 * ft)

        # occasionally do a 1D quadratic fit along a coord (3 evals)
        if random.random() < (0.10 + 0.20 * ft) and not timed_out():
            d = random.randrange(dim)
            s = spans[d]
            if s > 0.0:
                step = max(1e-18, 0.60 * ls_sigma * s)
                x0 = x[:]
                f0 = f

                x1 = x[:]; x1[d] += step
                x2 = x[:]; x2[d] -= step
                reflect_fast_inplace(x1)
                reflect_fast_inplace(x2)
                f1 = safe_eval(x1)
                if timed_out():
                    return x, f
                f2 = safe_eval(x2)

                # parabola vertex for points at -step,0,+step
                denom = (f1 + f2 - 2.0 * f0)
                if denom != 0.0 and math.isfinite(denom):
                    t = 0.5 * step * (f2 - f1) / denom  # offset from x0
                    # clamp to reasonable region
                    if t > 2.0 * step: t = 2.0 * step
                    if t < -2.0 * step: t = -2.0 * step
                    xt = x[:]
                    xt[d] += t
                    reflect_fast_inplace(xt)
                    ftv = safe_eval(xt)
                    if ftv < f:
                        x, f = xt, ftv
                        ls_sigma = max(ls_sigma_min, ls_sigma * 0.90)
                    else:
                        ls_sigma = min(ls_sigma_max, ls_sigma * 1.02)

        # coordinate step-halving
        idxs = list(range(dim))
        random.shuffle(idxs)
        for t in range(coord_budget):
            if timed_out():
                break
            d = idxs[t % dim]
            s = spans[d]
            if s <= 0.0:
                continue
            step = max(1e-18, ls_sigma * s)
            improved = False
            for _ in range(3):
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[d] += sgn * step
                    reflect_fast_inplace(xt)
                    ftv = safe_eval(xt)
                    if ftv < f:
                        x, f = xt, ftv
                        improved = True
                        break
                if improved:
                    break
                step *= 0.5
                if step < 1e-18:
                    break
            if improved:
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.92)
            else:
                ls_sigma = min(ls_sigma_max, ls_sigma * 1.015)

        # small random pattern (subspace) steps
        for _ in range(pattern_budget):
            if timed_out():
                break
            xt = x[:]
            prob = 0.06 + 0.18 * (1.0 - ft)
            scale = (0.35 * (1.0 - ft) + 0.10) * ls_sigma
            for d in range(dim):
                if random.random() < prob:
                    xt[d] += (random.random() * 2.0 - 1.0) * scale * spans[d]
            reflect_fast_inplace(xt)
            ftv = safe_eval(xt)
            if ftv < f:
                x, f = xt, ftv
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.94)

        return x, f

    # ---------------- strategy ensemble ----------------
    # 0: current-to-pbest/1
    # 1: rand/1
    # 2: current-to-best/1 (late exploit)
    strat_w = [1.0, 1.0, 0.25]
    strat_decay = 0.90
    minw = 1e-8

    no_improve_gens = 0
    stagnation_limit = max(12, 3 * dim)

    # ---------------- main loop ----------------
    while not timed_out():
        ft = frac_time()

        # linear population reduction
        NP_target = int(round(NP_init - ft * (NP_init - NP_min)))
        if NP_target < NP_min:
            NP_target = NP_min

        NP = len(pop)
        if NP_target < NP:
            keep = sorted(range(NP), key=lambda i: fit[i])[:NP_target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = NP_target
            archive_cap = max(NP, NP_init // 2)
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        p_count = max(2, int(math.ceil(p_best_rate * NP)))

        # increase exploitation over time
        strat_w[2] = max(strat_w[2], 0.08 + 1.00 * ft)

        S_F, S_CR, S_df = [], [], []
        strat_gain = [0.0, 0.0, 0.0]
        improved_gen = False

        for i in range(NP):
            if timed_out():
                return best

            xi, fi = pop[i], fit[i]

            rmem = random.randrange(H)
            F = sample_F(M_F[rmem])
            CR = sample_CR(M_CR[rmem])

            sw = strat_w[0] + strat_w[1] + strat_w[2]
            z = random.random() * sw
            if z < strat_w[0]:
                strat = 0
            elif z < strat_w[0] + strat_w[1]:
                strat = 1
            else:
                strat = 2

            if strat == 0:
                pbest = pop[idx_sorted[random.randrange(p_count)]]
                r1 = pick_idx(NP, {i})
                use_arch = (archive and random.random() < 0.55)
                if use_arch:
                    combined_n = NP + len(archive)
                    while True:
                        j = random.randrange(combined_n)
                        if j == i or j == r1:
                            continue
                        xr2 = pop[j] if j < NP else archive[j - NP]
                        break
                else:
                    r2 = pick_idx(NP, {i, r1})
                    xr2 = pop[r2]
                xr1 = pop[r1]
                v = [xi[d] + F * (pbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            elif strat == 1:
                r1 = pick_idx(NP, {i})
                r2 = pick_idx(NP, {i, r1})
                x1 = pop[r1]
                x2 = pop[r2]
                use_arch = (archive and random.random() < 0.35)
                if use_arch:
                    x3 = archive[random.randrange(len(archive))]
                else:
                    r3 = pick_idx(NP, {i, r1, r2})
                    x3 = pop[r3]
                v = [x1[d] + F * (x2[d] - x3[d]) for d in range(dim)]

            else:
                # current-to-best/1
                base = best_x
                r1 = pick_idx(NP, {i})
                use_arch = (archive and random.random() < 0.45)
                if use_arch:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_idx(NP, {i, r1})
                    x2 = pop[r2]
                x1 = pop[r1]
                v = [xi[d] + F * (base[d] - xi[d]) + F * (x1[d] - x2[d]) for d in range(dim)]

            reflect_fast_inplace(v)
            u = binom_cross(xi, v, CR)
            reflect_fast_inplace(u)

            # mild best injection late
            if random.random() < (0.006 + 0.080 * ft):
                d = random.randrange(dim)
                u[d] = 0.78 * u[d] + 0.22 * best_x[d]
                reflect_fast_inplace(u)

            fu = safe_eval(u)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > archive_cap:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = u
                fit[i] = fu

                df = fi - fu
                if df > 0.0:
                    S_F.append(F)
                    S_CR.append(CR)
                    S_df.append(df)
                    strat_gain[strat] += df

                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_gen = True

        # SHADE memory update
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                weights = [1.0 / float(len(S_df))] * len(S_df)
            else:
                inv = 1.0 / wsum
                weights = [df * inv for df in S_df]

            meanCR = 0.0
            for w, cr in zip(weights, S_CR):
                meanCR += w * cr

            num = 0.0
            den = 0.0
            for w, fval in zip(weights, S_F):
                num += w * fval * fval
                den += w * fval
            meanF = (num / den) if den != 0.0 else M_F[mem_idx]

            M_CR[mem_idx] = min(1.0, max(0.0, meanCR))
            M_F[mem_idx] = min(1.0, max(0.05, meanF))
            mem_idx = (mem_idx + 1) % H

        # bandit update
        for s in range(3):
            strat_w[s] = strat_w[s] * strat_decay + (strat_gain[s] + 1e-12)
            if strat_w[s] < minw:
                strat_w[s] = minw

        # local polish
        if not timed_out():
            if improved_gen or random.random() < (0.05 + 0.65 * ft):
                bx, bf = local_refine(best_x, best, ft)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

        # stagnation rescue
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= stagnation_limit and not timed_out():
            no_improve_gens = 0
            NP = len(pop)
            k = max(2, NP // 4)  # refresh 25%
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]

            for wi in worst:
                if timed_out():
                    return best
                rmode = random.random()
                if rmode < 0.35:
                    x = opposition(best_x)
                    rad = 0.20 * (1.0 - ft) + 0.02
                    for d in range(dim):
                        if random.random() < 0.35:
                            x[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
                    reflect_fast_inplace(x)
                elif rmode < 0.80:
                    x = best_x[:]
                    kick = 0.26 * (1.0 - ft) + 0.03
                    m = 1 + int(0.18 * dim)
                    for _ in range(m):
                        d = random.randrange(dim)
                        x[d] += cauchy(1.0) * kick * spans[d]
                    reflect_fast_inplace(x)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, x[:]

            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

    return best
