import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only bounded minimizer (anytime).

    Improvements vs previous hybrid:
      - Adds (1+1)-ES + 1/5 success rule on the current best (very strong local driver).
      - Keeps L-SHADE-like DE with archive for global search.
      - Uses "as-you-go" best-driven normalized sampling (CMA-diag flavor) but cheaper.
      - Uses budget-aware scheduling: more exploration early, more exploitation late.
      - Robust bound handling + NaN/inf safety + cheap dedup cache.

    Returns:
        best (float): best fitness found within max_time seconds
    """

    # ------------------ helpers ------------------
    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    def safe_float(y):
        try:
            y = float(y)
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == -float("inf"):
            return float("inf")
        return y

    # allow func(x) or func(*x)
    def eval_f(x):
        # quantized cache key to avoid repeated evals on identical/near-identical points
        key = tuple(int((x[i] - lo[i]) * inv_quant[i]) if inv_quant[i] > 0.0 else 0 for i in range(dim))
        v = cache.get(key)
        if v is not None:
            return v
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        y = safe_float(y)
        cache[key] = y
        return y

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                x[i] = random.uniform(lo[i], hi[i])
            else:
                x[i] = lo[i]
        return x

    def opposite_vec(x):
        ox = x[:]
        for i in range(dim):
            if span_ok[i]:
                ox[i] = (lo[i] + hi[i]) - x[i]
                if ox[i] < lo[i]: ox[i] = lo[i]
                elif ox[i] > hi[i]: ox[i] = hi[i]
            else:
                ox[i] = lo[i]
        return ox

    # quasi latin hypercube
    def qlhs(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    x[d] = lo[d]
                else:
                    u = (perms[d][i] + random.random()) / n
                    x[d] = lo[d] + u * span[d]
            pop.append(x)
        return pop

    def cauchy(scale):
        # clipped cauchy (heavy tail) for DE F sampling / kicks
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        if t > 30.0: t = 30.0
        if t < -30.0: t = -30.0
        return scale * t

    def reflect_into_bounds(x):
        # reflection + clamp (better than clamp-only for evolutionary steps)
        for d in range(dim):
            if not span_ok[d]:
                x[d] = lo[d]
                continue
            a = lo[d]; b = hi[d]
            v = x[d]
            if v < a:
                v = a + (a - v)
                if v > b:
                    v = a + random.random() * (b - a)
            elif v > b:
                v = b - (v - b)
                if v < a:
                    v = a + random.random() * (b - a)
            if v < a: v = a
            elif v > b: v = b
            x[d] = v
        return x

    def to_norm01(x):
        z = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                z[i] = (x[i] - lo[i]) / span[i]
            else:
                z[i] = 0.0
        return z

    def from_norm01(z):
        x = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                x[i] = lo[i] + z[i] * span[i]
            else:
                x[i] = lo[i]
        return x

    # ------------------ edge cases / setup ------------------
    if dim <= 0:
        try:
            return safe_float(func([]))
        except TypeError:
            return safe_float(func())

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_ok = [span[i] > 0.0 for i in range(dim)]

    inv_quant = []
    for i in range(dim):
        if not span_ok[i]:
            inv_quant.append(0.0)
        else:
            q = span[i] * 1e-7  # finer than before; still bounded keys
            inv_quant.append(0.0 if q <= 0.0 else (1.0 / q))

    cache = {}

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ------------------ initialization ------------------
    NP = max(20, min(110, 10 * dim + 10))
    init = qlhs(NP)
    init += [opposite_vec(x) for x in init]

    pop, fit = [], []
    for x in init:
        if time.time() >= deadline:
            break
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)

    if not fit:
        return float("inf")

    while len(pop) < NP and time.time() < deadline:
        x = rand_vec()
        pop.append(x)
        fit.append(eval_f(x))

    if len(pop) > NP:
        idx = sorted(range(len(pop)), key=lambda i: fit[i])[:NP]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    best_i = min(range(NP), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    # ------------------ L-SHADE-ish DE state ------------------
    H = 6
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_pos = 0
    archive = []
    arch_max = NP

    # ------------------ cheap "CMA-diag-ish" sampling state ------------------
    # variance per dimension in normalized space, adapted by successes
    vdiag = [0.20] * dim  # std in [0,1] space
    vmin = 1e-4
    vmax = 0.50

    # ------------------ (1+1)-ES local search state ------------------
    # sigma_es lives in normalized space
    sigma_es = 0.18
    es_succ = 0
    es_trials = 0

    # ------------------ stagnation / restart ------------------
    stall = 0
    last_best = best
    max_stall = max(12, 4 + dim)

    gen = 0
    while time.time() < deadline:
        gen += 1
        tl = deadline - time.time()
        frac_left = tl / max(1e-12, float(max_time))
        near_end = frac_left <= 0.25
        very_end = frac_left <= 0.08

        # =========================================================
        # 1) One DE generation (global / mid-range search)
        # =========================================================
        p = 0.20 if not near_end else 0.10
        pnum = max(2, int(p * NP))
        order = sorted(range(NP), key=lambda i: fit[i])

        SF, SCR, dF = [], [], []
        new_pop = [None] * NP
        new_fit = [None] * NP

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() - 1.5)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = -1.0
            for _ in range(6):
                F = mu_f + cauchy(0.10)
                if 0.0 < F <= 1.0:
                    break
            if not (0.0 < F <= 1.0):
                F = 0.5

            pbest = order[random.randrange(pnum)]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            total_pool = NP + len(archive)
            if total_pool <= 2:
                r2 = (i + 1) % NP
                x_r2 = pop[r2]
            else:
                while True:
                    r2 = random.randrange(total_pool)
                    if r2 < NP:
                        if r2 != i and r2 != r1:
                            x_r2 = pop[r2]
                            break
                    else:
                        x_r2 = archive[r2 - NP]
                        break

            x_r1 = pop[r1]
            x_pb = pop[pbest]

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    v[d] = lo[d]
                else:
                    v[d] = xi[d] + F * (x_pb[d] - xi[d]) + F * (x_r1[d] - x_r2[d])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if (random.random() < CR) or (d == jrand):
                    u[d] = v[d]

            reflect_into_bounds(u)
            fu = eval_f(u)

            if fu <= fi:
                new_pop[i] = u
                new_fit[i] = fu

                if fu < fi:
                    # archive parent for diversity
                    if arch_max > 0:
                        if len(archive) < arch_max:
                            archive.append(xi[:])
                        else:
                            archive[random.randrange(len(archive))] = xi[:]
                    SF.append(F)
                    SCR.append(CR)
                    dF.append(fi - fu)

                if fu < best:
                    best = fu
                    bestx = u[:]
            else:
                new_pop[i] = xi
                new_fit[i] = fi

        if new_fit[0] is not None:
            pop, fit = new_pop, new_fit

        if dF:
            wsum = sum(dF)
            ww = ([1.0 / len(dF)] * len(dF)) if wsum <= 0.0 else [di / wsum for di in dF]

            mcr = 0.0
            for wi, cri in zip(ww, SCR):
                mcr += wi * cri

            num = 0.0
            den = 0.0
            for wi, Fi in zip(ww, SF):
                num += wi * Fi * Fi
                den += wi * Fi
            mf = (num / den) if den > 1e-12 else MF[mem_pos]

            MCR[mem_pos] = mcr
            MF[mem_pos] = mf
            mem_pos = (mem_pos + 1) % H

        # =========================================================
        # 2) Best-centered diagonal sampling (cheap CMA-diag flavor)
        # =========================================================
        if time.time() < deadline:
            bz = to_norm01(bestx)
            # small batch size; larger early, smaller late
            batch = 10 if not near_end else (6 if not very_end else 4)

            succ = 0
            trials = 0
            for _ in range(batch):
                if time.time() >= deadline:
                    break
                trials += 1

                # sample each dim independently (gaussian via Box-Muller)
                z = [0.0] * dim
                for d in range(dim):
                    if not span_ok[d]:
                        z[d] = 0.0
                        continue
                    u1 = max(1e-12, random.random())
                    u2 = random.random()
                    g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    # step in [0,1] space
                    step = g * vdiag[d]
                    z[d] = bz[d] + step

                # clamp in [0,1], map back
                for d in range(dim):
                    if z[d] < 0.0: z[d] = 0.0
                    elif z[d] > 1.0: z[d] = 1.0

                x = from_norm01(z)
                fx = eval_f(x)
                if fx < best:
                    best, bestx = fx, x[:]
                    bz = to_norm01(bestx)
                    succ += 1

            # adapt vdiag (slightly shrink if no success, expand a bit if some)
            if trials > 0:
                rate = succ / float(trials)
                # target small success probability
                if rate > 0.25:
                    for d in range(dim):
                        vdiag[d] = min(vmax, vdiag[d] * 1.15)
                elif rate < 0.10:
                    for d in range(dim):
                        vdiag[d] = max(vmin, vdiag[d] * 0.85)

        # =========================================================
        # 3) (1+1)-ES on the current best with 1/5 success rule
        #    Very effective for squeezing improvements under time limits.
        # =========================================================
        if time.time() < deadline:
            bz = to_norm01(bestx)

            # ES attempts per outer iteration: more near the end
            es_iters = 8 if near_end else 4
            if very_end:
                es_iters = 14

            for _ in range(es_iters):
                if time.time() >= deadline:
                    break

                # mutate best in normalized space with scalar sigma_es
                candz = bz[:]
                for d in range(dim):
                    if not span_ok[d]:
                        candz[d] = 0.0
                        continue
                    u1 = max(1e-12, random.random())
                    u2 = random.random()
                    g = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                    candz[d] = candz[d] + sigma_es * g
                    if candz[d] < 0.0: candz[d] = 0.0
                    elif candz[d] > 1.0: candz[d] = 1.0

                candx = from_norm01(candz)
                fc = eval_f(candx)

                es_trials += 1
                if fc < best:
                    best = fc
                    bestx = candx[:]
                    bz = candz[:]  # accept
                    es_succ += 1

                # update sigma periodically (1/5 success rule)
                if es_trials >= 10:
                    p_succ = es_succ / float(es_trials)
                    # if success rate > 1/5 -> increase step, else decrease
                    if p_succ > 0.20:
                        sigma_es *= 1.25
                    else:
                        sigma_es *= 0.82
                    if sigma_es < 1e-6: sigma_es = 1e-6
                    if sigma_es > 0.60: sigma_es = 0.60
                    es_succ = 0
                    es_trials = 0

        # =========================================================
        # 4) Stagnation handling: partial refresh + kick best
        # =========================================================
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0

            # refresh weakest part of population
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(5, NP // 6)

            for idx in order[keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.65:
                    # sample around best (scaled by current vdiag / sigma_es)
                    bz = to_norm01(bestx)
                    z = bz[:]
                    for d in range(dim):
                        if not span_ok[d]:
                            z[d] = 0.0
                            continue
                        # uniform-ish around best; wider earlier
                        wid = (0.35 if not near_end else 0.22)
                        z[d] = bz[d] + (random.random() * 2.0 - 1.0) * wid
                        if z[d] < 0.0: z[d] = 0.0
                        elif z[d] > 1.0: z[d] = 1.0
                    x = from_norm01(z)
                else:
                    x = rand_vec()
                pop[idx] = x
                fit[idx] = eval_f(x)

            # also "kick" ES step a bit to escape tight local minima
            sigma_es = min(0.35, max(0.05, sigma_es * 1.35))
            for d in range(dim):
                vdiag[d] = min(vmax, max(vmin, vdiag[d] * 1.10))

            # trim archive a bit
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif archive and random.random() < 0.3:
                for _ in range(max(1, len(archive) // 4)):
                    if archive:
                        archive.pop(random.randrange(len(archive)))

            bi = min(range(NP), key=lambda i: fit[i])
            if fit[bi] < best:
                best = fit[bi]
                bestx = pop[bi][:]

    return best
