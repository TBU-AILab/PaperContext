import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only, time-bounded black-box minimizer.

    Improvements vs your current best (DE + local search):
      - Much faster bound handling (mirror-modulo reflection; no while loops).
      - Time-adaptive evaluation cache (rounded keys) to avoid redundant evals.
      - L-SHADE style DE with:
          * current-to-pbest/1 + rand/1 + best/2 (late) ensemble
          * success-history adaptation for F and CR
          * linear population size reduction
          * external archive
      - Stronger local refinement around best:
          * SPSA-like 2-eval step (handles coupling/noise)
          * coordinate step-halving with occasional pairwise (2D) refinement
      - Restart/rescue when stagnating: mix of opposition-to-best, heavy-tail kicks,
        and random immigrants; intensity scheduled by time progress.

    Returns: best (minimum) fitness found within max_time seconds.
    """

    t0 = time.time()
    end_time = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def timed_out():
        return time.time() >= end_time

    def frac_time():
        if max_time <= 0:
            return 1.0
        return min(1.0, max(0.0, (time.time() - t0) / (max_time + 1e-12)))

    # ---- fast mirror reflection using modulo over period 2*span ----
    def reflect_fast_inplace(x):
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

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def cauchy(scale=1.0):
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # ---- evaluation cache (time-adaptive rounding) ----
    eval_cache = {}
    cache_fifo = []
    cache_cap = 18000

    def safe_eval(x):
        ft = frac_time()
        # coarser early (diversity), finer late (avoid duplicates)
        digits = 4 + int(6 * ft)  # 4..10
        key = tuple(round(float(xi), digits) for xi in x)
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
            for _ in range(350):
                if not cache_fifo:
                    break
                k = cache_fifo.pop(0)
                eval_cache.pop(k, None)
        return v

    # ---- SHADE sampling ----
    def sample_F(mu):
        # Cauchy around mu; ensure (0,1]
        for _ in range(16):
            f = mu + 0.10 * cauchy(1.0)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        if mu <= 0.0:
            return 0.6
        return 1.0 if mu > 1.0 else mu

    def sample_CR(mu):
        cr = mu + 0.10 * random.gauss(0.0, 1.0)
        if cr < 0.0: return 0.0
        if cr > 1.0: return 1.0
        return cr

    # ---- initialization ----
    # Slightly larger start; then reduce linearly.
    NP_init = max(36, 18 * dim)
    NP_init = min(NP_init, 200)
    NP_min = max(12, 4 * dim)
    if NP_min > NP_init:
        NP_min = max(12, NP_init // 2)

    H = 16
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    p_best_rate = 0.12
    p_best_rate = max(0.05, min(0.30, p_best_rate))

    archive = []
    archive_cap = NP_init

    # Seeding: random + opposition + center sprays + a couple axis-corners
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    pop = []

    # axis corners (few) to help when optimum lies on boundaries
    if dim > 0:
        for _ in range(min(6, NP_init // 10 + 1)):
            x = []
            for i in range(dim):
                x.append(lows[i] if random.random() < 0.5 else highs[i])
            pop.append(x)

    while len(pop) < NP_init:
        x = rand_vec()
        pop.append(x)
        if len(pop) < NP_init:
            xo = opposition(x)
            reflect_fast_inplace(xo)
            pop.append(xo)
        if len(pop) < NP_init and random.random() < 0.45:
            xs = center[:]
            rad = 0.22
            for d in range(dim):
                xs[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
            reflect_fast_inplace(xs)
            pop.append(xs)
    pop = pop[:NP_init]

    fit = [float("inf")] * len(pop)
    best = float("inf")
    best_x = None

    for i in range(len(pop)):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    def binomial_crossover(x, v, cr):
        jrand = random.randrange(dim) if dim > 0 else 0
        u = [0.0] * dim
        for d in range(dim):
            u[d] = v[d] if (d == jrand or random.random() < cr) else x[d]
        return u

    # ---- local search around best ----
    ls_sigma = 0.13
    ls_sigma_min = 1e-12
    ls_sigma_max = 0.55

    def local_refine(xb, fb, ft):
        nonlocal ls_sigma
        if xb is None or dim == 0:
            return xb, fb

        x = xb[:]
        f = fb

        # budgets: small but increase later
        coord_budget = min(dim + 2, 6 + int(14 * ft))
        pair_budget = 1 + int(3 * ft)
        use_spsa = (dim >= 2 and random.random() < (0.18 + 0.22 * ft))

        # SPSA-like 2-eval attempt
        if use_spsa and not timed_out():
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            c = max(1e-12, 0.50 * ls_sigma)
            a = max(1e-12, 0.42 * ls_sigma)

            x_plus = x[:]
            x_minus = x[:]
            for d in range(dim):
                step = c * spans[d]
                x_plus[d] += delta[d] * step
                x_minus[d] -= delta[d] * step
            reflect_fast_inplace(x_plus)
            reflect_fast_inplace(x_minus)

            f_plus = safe_eval(x_plus)
            if not timed_out():
                f_minus = safe_eval(x_minus)
                g = f_plus - f_minus
                sgn = 1.0 if g > 0.0 else -1.0
                xt = x[:]
                for d in range(dim):
                    xt[d] -= a * spans[d] * sgn * delta[d]
                reflect_fast_inplace(xt)
                ftv = safe_eval(xt)
                if ftv < f:
                    x, f = xt, ftv
                    ls_sigma = max(ls_sigma_min, ls_sigma * 0.87)
                else:
                    ls_sigma = min(ls_sigma_max, ls_sigma * 1.03)

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
            for _ in range(3):  # try, halve, halve
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

        # occasional 2D pair refinement (helps coupled variables)
        for _ in range(pair_budget):
            if timed_out():
                break
            if dim < 2:
                break
            a = random.randrange(dim)
            b = random.randrange(dim - 1)
            if b >= a:
                b += 1
            sa = max(1e-18, ls_sigma * spans[a])
            sb = max(1e-18, ls_sigma * spans[b])

            xt = x[:]
            xt[a] += (random.random() * 2.0 - 1.0) * sa
            xt[b] += (random.random() * 2.0 - 1.0) * sb
            reflect_fast_inplace(xt)
            ftv = safe_eval(xt)
            if ftv < f:
                x, f = xt, ftv
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.94)

        return x, f

    # ---- strategy ensemble bandit ----
    # 0: current-to-pbest/1
    # 1: rand/1
    # 2: best/2 (activated more late)
    strat_w = [1.0, 1.0, 0.25]
    strat_decay = 0.90
    minw = 1e-8

    no_improve_gens = 0
    stagnation_limit = max(12, 3 * dim)

    def pick_idx(NP, excl):
        while True:
            j = random.randrange(NP)
            if j not in excl:
                return j

    while not timed_out():
        ft = frac_time()

        # linear pop reduction
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

        S_F, S_CR, S_df = [], [], []
        strat_gain = [0.0, 0.0, 0.0]
        improved_gen = False

        # increase best/2 usage later
        strat_w[2] = max(strat_w[2], 0.10 + 0.80 * ft)

        for i in range(NP):
            if timed_out():
                return best

            xi, fi = pop[i], fit[i]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            # pick strategy
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
                base = best_x if best_x is not None else pop[idx_sorted[0]]
                r1 = pick_idx(NP, {i})
                r2 = pick_idx(NP, {i, r1})
                r3 = pick_idx(NP, {i, r1, r2})
                x1 = pop[r1]
                x2 = pop[r2]
                x3 = pop[r3]
                # best/2: base + F*(x1-x2) + F*(x3-xi) gives strong pull + diversity
                v = [base[d] + F * (x1[d] - x2[d]) + F * (x3[d] - xi[d]) for d in range(dim)]

            reflect_fast_inplace(v)
            u = binomial_crossover(xi, v, CR)
            reflect_fast_inplace(u)

            # mild best injection late
            if best_x is not None and random.random() < (0.010 + 0.070 * ft):
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

        # update SHADE memories
        if S_F:
            wsum = sum(S_df)
            inv = 1.0 / wsum if wsum > 0.0 else 0.0
            weights = [(df * inv if wsum > 0.0 else 1.0 / len(S_df)) for df in S_df]

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

        # local refine schedule
        if best_x is not None and not timed_out():
            if improved_gen or random.random() < (0.05 + 0.60 * ft):
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
                if best_x is not None and rmode < 0.38:
                    x = opposition(best_x)
                    rad = 0.18 * (1.0 - ft) + 0.02
                    for d in range(dim):
                        if random.random() < 0.30:
                            x[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
                    reflect_fast_inplace(x)
                elif best_x is not None and rmode < 0.82:
                    x = best_x[:]
                    kick = 0.24 * (1.0 - ft) + 0.03
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
