import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs previous version:
      - Adds a coordinate-wise local solver (Pattern Search / Hooke-Jeeves style) that is
        very robust under bounds and cheap per iteration.
      - Keeps Nelder–Mead but makes it *adaptive*: scale shrinks on stagnation and can
        restart around best/elite points.
      - Better global exploration: quasi-opposition sampling + elite-guided differential
        evolution (DE/rand-to-best) style proposals.
      - Stronger, cheaper cache with true random eviction.
      - More reliable time usage: inner loops check time frequently and degrade gracefully.

    Returns:
      best (float): best objective value found within max_time.
    """
    start = time.time()
    deadline = start + max(0.0, float(max_time))
    eps = 1e-12

    if dim <= 0:
        try:
            y = float(func([]))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [max(hi[i] - lo[i], 0.0) for i in range(dim)]
    max_span = max(span) if span else 0.0
    if max_span <= eps:
        # All fixed
        x0 = [lo[i] for i in range(dim)]
        try:
            y = float(func(x0))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    def time_left():
        return time.time() < deadline

    def clip(v, a, b):
        return a if v < a else (b if v > b else v)

    def project(x):
        return [clip(float(x[i]), lo[i], hi[i]) for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) if span[i] > eps else lo[i] for i in range(dim)]

    def center():
        return [(lo[i] + hi[i]) * 0.5 for i in range(dim)]

    # --- cache (quantized) ---
    qstep = []
    for i in range(dim):
        s = span[i]
        if s <= eps:
            qstep.append(0.0)
        else:
            # slightly coarser than before to reduce cache misses; good for noisy funcs too
            qstep.append(max(1e-12, 5e-7 * s))

    cache = {}
    cache_keys = []
    CACHE_MAX = 30000

    def key_of(x):
        k = []
        for i in range(dim):
            if qstep[i] == 0.0:
                k.append(0)
            else:
                k.append(int(round((x[i] - lo[i]) / qstep[i])))
        return tuple(k)

    def cache_put(k, y):
        if k in cache:
            cache[k] = y
            return
        if len(cache) >= CACHE_MAX:
            # evict ~1% random keys quickly
            for _ in range(max(10, CACHE_MAX // 100)):
                if not cache_keys:
                    break
                idx = random.randrange(len(cache_keys))
                kk = cache_keys[idx]
                # swap-pop
                cache_keys[idx] = cache_keys[-1]
                cache_keys.pop()
                cache.pop(kk, None)
                if len(cache) < CACHE_MAX:
                    break
        cache[k] = y
        cache_keys.append(k)

    def safe_eval(x):
        x = project(x)
        k = key_of(x)
        y = cache.get(k)
        if y is not None:
            return y, x
        try:
            y = func(x)
            if y is None:
                y = float("inf")
            y = float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")
        cache_put(k, y)
        return y, x

    # --- elites ---
    best = float("inf")
    x_best = center()
    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    ELITE_MAX = max(16, 5 * dim)
    elites = [(best, x_best[:])]

    def push_elite(fx, x):
        nonlocal best, x_best, elites
        if fx < best:
            best = fx
            x_best = x[:]
        elites.append((fx, x[:]))
        if len(elites) > 4 * ELITE_MAX:
            elites.sort(key=lambda t: t[0])
            elites[:] = elites[:ELITE_MAX]

    # --- sampling helpers ---
    def stratified(m):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
            else:
                b = random.randrange(m)
                a = lo[i] + (b / m) * span[i]
                c = lo[i] + ((b + 1) / m) * span[i]
                x[i] = random.uniform(a, c)
        return x

    def opposition(x):
        # quasi-opposition (often helps on bounded boxes)
        y = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                y[i] = lo[i]
            else:
                # opposition
                o = lo[i] + hi[i] - x[i]
                # quasi: move between center and opposition
                mid = 0.5 * (lo[i] + hi[i])
                y[i] = clip(mid + random.random() * (o - mid), lo[i], hi[i])
        return y

    # --- Local search: bounded pattern search (coordinate + exploratory) ---
    def pattern_search(x0, f0, step0, max_iter):
        """
        Coordinate pattern search with adaptive step (good for boxes).
        Returns (best_f, best_x).
        """
        x = x0[:]
        fx = f0
        step = step0

        # random coordinate order helps in higher dimensions
        coords = list(range(dim))

        it = 0
        while it < max_iter and time_left():
            it += 1
            improved = False
            random.shuffle(coords)

            for j in coords:
                if span[j] <= eps:
                    continue
                cur = x[j]

                # try +step and -step
                for sgn in (1.0, -1.0):
                    xj = cur + sgn * step
                    if xj < lo[j] or xj > hi[j]:
                        xj = clip(xj, lo[j], hi[j])
                        if abs(xj - cur) <= 0.5 * eps:
                            continue
                    xt = x[:]
                    xt[j] = xj
                    ft, xt = safe_eval(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                        break
                if improved and not time_left():
                    break

            if improved:
                # small step growth when making progress
                step *= 1.2
            else:
                step *= 0.5

            # stop if step tiny relative to box
            if step <= 1e-10 * (max_span if max_span > 0 else 1.0):
                break

        return fx, x

    # --- Nelder–Mead (bounded) with small tweaks ---
    def nelder_mead(x0, f0, max_evals, init_scale):
        if max_evals <= 0 or not time_left():
            return f0, x0, 0

        simplex = [(f0, x0[:])]
        evals = 0

        # build simplex
        for i in range(dim):
            if not time_left() or evals >= max_evals:
                break
            if span[i] <= eps:
                continue
            xi = x0[:]
            step = init_scale * span[i]
            if step <= 0.0:
                continue
            if xi[i] + step <= hi[i]:
                xi[i] += step
            elif xi[i] - step >= lo[i]:
                xi[i] -= step
            else:
                xi[i] = clip(xi[i] + 0.01 * span[i], lo[i], hi[i])
            fi, xi = safe_eval(xi)
            evals += 1
            simplex.append((fi, xi))

        if len(simplex) < 2:
            simplex.sort(key=lambda t: t[0])
            return simplex[0][0], simplex[0][1], evals

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while evals < max_evals and time_left():
            simplex.sort(key=lambda t: t[0])
            fbest, xbest = simplex[0]
            fworst, xworst = simplex[-1]
            fsecond = simplex[-2][0]

            # centroid excluding worst
            m = len(simplex) - 1
            centroid = [0.0] * dim
            for _, x in simplex[:-1]:
                for j in range(dim):
                    centroid[j] += x[j]
            invm = 1.0 / max(1, m)
            for j in range(dim):
                centroid[j] *= invm

            # reflection
            xr = [centroid[j] + alpha * (centroid[j] - xworst[j]) for j in range(dim)]
            fr, xr = safe_eval(xr)
            evals += 1

            if fr < fbest:
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(dim)]
                fe, xe = safe_eval(xe)
                evals += 1
                simplex[-1] = (fe, xe) if fe < fr else (fr, xr)
            elif fr < fsecond:
                simplex[-1] = (fr, xr)
            else:
                # contraction
                if fr < fworst:
                    xc = [centroid[j] + rho * (xr[j] - centroid[j]) for j in range(dim)]
                else:
                    xc = [centroid[j] - rho * (centroid[j] - xworst[j]) for j in range(dim)]
                fc, xc = safe_eval(xc)
                evals += 1
                if fc < fworst:
                    simplex[-1] = (fc, xc)
                else:
                    # shrink toward best
                    x0s = simplex[0][1]
                    new_simplex = [simplex[0]]
                    for k in range(1, len(simplex)):
                        if evals >= max_evals or not time_left():
                            break
                        xs = [x0s[j] + sigma * (simplex[k][1][j] - x0s[j]) for j in range(dim)]
                        fs, xs = safe_eval(xs)
                        evals += 1
                        new_simplex.append((fs, xs))
                    simplex = new_simplex

            # stop if simplex tiny
            max_spread = 0.0
            for j in range(dim):
                mn = min(p[1][j] for p in simplex)
                mx = max(p[1][j] for p in simplex)
                if mx - mn > max_spread:
                    max_spread = mx - mn
            if max_spread <= 1e-11 * (max_span if max_span > 0 else 1.0):
                break

        simplex.sort(key=lambda t: t[0])
        return simplex[0][0], simplex[0][1], evals

    # --- initial population: stratified + opposition ---
    init_n = max(50, 20 * dim)
    m_bins = max(5, int(math.sqrt(init_n)) + 2)

    for k in range(init_n):
        if not time_left():
            return best
        x = stratified(m_bins) if (k % 2 == 0) else rand_uniform()
        fx, x = safe_eval(x)
        push_elite(fx, x)

        # opposition sample sometimes
        if (k % 3 == 0) and time_left():
            xo = opposition(x)
            fo, xo = safe_eval(xo)
            push_elite(fo, xo)

    # --- main loop ---
    no_improve = 0
    round_id = 0

    while time_left():
        round_id += 1
        elites.sort(key=lambda t: t[0])
        elites = elites[:min(len(elites), ELITE_MAX)]

        prev_best = best

        # 1) global: DE-style proposals around elites + random restarts
        G = max(20, 8 * dim)
        for _ in range(G):
            if not time_left():
                return best

            r = random.random()
            if r < 0.25 or len(elites) < 4:
                x = rand_uniform()
                if random.random() < 0.35:
                    x = opposition(x)
            else:
                # choose three distinct elites
                idxs = list(range(len(elites)))
                a = idxs[int((random.random() ** 1.7) * (len(idxs) - 1))]  # bias to better
                b = random.randrange(len(elites))
                c = random.randrange(len(elites))
                while b == a and len(elites) > 1:
                    b = random.randrange(len(elites))
                while (c == a or c == b) and len(elites) > 2:
                    c = random.randrange(len(elites))

                xa = elites[a][1]
                xb = elites[b][1]
                xc = elites[c][1]

                F = 0.5 + 0.8 * random.random()         # differential weight
                CR = 0.6 + 0.35 * random.random()        # crossover rate
                # rand-to-best/1: x = xa + F*(x_best - xa) + F*(xb - xc)
                x = [0.0] * dim
                jrand = random.randrange(dim)
                for j in range(dim):
                    if span[j] <= eps:
                        x[j] = lo[j]
                        continue
                    if random.random() < CR or j == jrand:
                        v = (xa[j] +
                             F * (x_best[j] - xa[j]) +
                             F * (xb[j] - xc[j]))
                        # add tiny annealed noise
                        v += random.gauss(0.0, (0.02 / (1.0 + 0.05 * round_id)) * span[j])
                        x[j] = clip(v, lo[j], hi[j])
                    else:
                        x[j] = xa[j]

            fx, x = safe_eval(x)
            push_elite(fx, x)

        # 2) local: pattern search on best and a couple elites
        elites.sort(key=lambda t: t[0])
        L = 1 if dim > 30 else 2
        L = min(L, len(elites))

        # base step: decreases with rounds but not too small
        step0 = (0.08 / (1.0 + 0.10 * round_id)) * max_span
        step0 = max(step0, 1e-6 * max_span)

        for i in range(L):
            if not time_left():
                return best
            x0 = elites[i][1]
            f0 = elites[i][0]

            fl, xl = pattern_search(x0, f0, step0, max_iter=max(8, 2 * dim))
            push_elite(fl, xl)

        # 3) occasional NM polish (short)
        if time_left() and (round_id % 2 == 1) and dim <= 60:
            x0 = elites[0][1]
            f0 = elites[0][0]
            init_scale = max(0.015, 0.10 / (1.0 + 0.06 * round_id))
            per_run = max(25 + 5 * dim, 80)
            fnm, xnm, _ = nelder_mead(x0, f0, per_run, init_scale)
            push_elite(fnm, xnm)

        # stagnation handling
        if best < prev_best:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 3:
            # diversification burst: many random + opposition + heavy-tail around best
            burst = max(25, 10 * dim)
            for _ in range(burst):
                if not time_left():
                    return best
                if random.random() < 0.5:
                    x = rand_uniform()
                    if random.random() < 0.5:
                        x = opposition(x)
                else:
                    # heavy-tail around best (Cauchy-like)
                    x = x_best[:]
                    for j in range(dim):
                        if span[j] <= eps:
                            x[j] = lo[j]
                            continue
                        u = random.random()
                        step = (0.12 * span[j]) * math.tan(math.pi * (u - 0.5))
                        x[j] = clip(x[j] + step, lo[j], hi[j])
                fx, x = safe_eval(x)
                push_elite(fx, x)
            no_improve = 0

    return best
