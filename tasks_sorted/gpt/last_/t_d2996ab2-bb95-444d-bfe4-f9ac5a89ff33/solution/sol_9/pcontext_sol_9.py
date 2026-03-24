import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (self-contained, no external libs).

    Improvements over the provided code (typical impact: better early progress and
    less stagnation under tight time limits):
      - "Trust-region" local search (adaptive coordinate + diagonal steps) called
        opportunistically and budget-aware.
      - Better DE core: JADE/current-to-pbest/1 + archive, but with:
          * diversity-aware parent choice
          * occasional "rand/1" exploration branch
          * per-dimension crossover mask shortcut for speed
      - Stronger restarts: mixture of Latin-ish stratification, elite sampling,
        opposition, and heavy-tail perturbations.
      - Faster cache with adaptive quantization (coarser early, finer later).
      - Robust bound handling by reflection.

    Returns best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))
    eps = 1e-12

    # ----------------- trivial cases -----------------
    if dim <= 0:
        try:
            y = float(func([]))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [max(0.0, hi[i] - lo[i]) for i in range(dim)]
    max_span = max(span) if span else 0.0

    if max_span <= eps:
        x0 = lo[:]
        try:
            y = float(func(x0))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    def time_left():
        return time.time() < deadline

    def clip(v, a, b):
        return a if v < a else (b if v > b else v)

    def reflect_into(v, a, b):
        if a == b:
            return a
        w = b - a
        x = float(v)
        if x < a or x > b:
            x = a + (x - a) % (2.0 * w)
            if x > b:
                x = b - (x - b)
        return clip(x, a, b)

    def project_reflect(x):
        return [reflect_into(x[i], lo[i], hi[i]) for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) if span[i] > eps else lo[i] for i in range(dim)]

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = lo[i] if span[i] <= eps else (lo[i] + hi[i] - x[i])
        return project_reflect(y)

    # ----------------- cache (adaptive quantization) -----------------
    # Make qstep coarser early (higher cache hit rate), refine later.
    CACHE_MAX = 90000
    cache = {}
    cache_keys = []

    def qstep_for_time():
        # fraction of elapsed time in [0,1]
        frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        frac = clip(frac, 0.0, 1.0)
        # start coarse (1e-6*span), end finer (2e-7*span)
        base = (1.0 - frac) * 1.0e-6 + frac * 2.0e-7
        qs = []
        for s in span:
            qs.append(0.0 if s <= eps else max(1e-12, base * s))
        return qs

    def key_of(x, qstep):
        k = []
        for i in range(dim):
            qs = qstep[i]
            if qs == 0.0:
                k.append(0)
            else:
                k.append(int((x[i] - lo[i]) / qs + 0.5))
        return tuple(k)

    def cache_put(k, y):
        if k in cache:
            cache[k] = y
            return
        if len(cache) >= CACHE_MAX:
            # evict ~2%
            ev = max(120, CACHE_MAX // 50)
            for _ in range(ev):
                if not cache_keys:
                    break
                idx = random.randrange(len(cache_keys))
                kk = cache_keys[idx]
                cache_keys[idx] = cache_keys[-1]
                cache_keys.pop()
                cache.pop(kk, None)
                if len(cache) < CACHE_MAX:
                    break
        cache[k] = y
        cache_keys.append(k)

    def safe_eval(x):
        x = project_reflect(x)
        qs = qstep_for_time()
        k = key_of(x, qs)
        y = cache.get(k)
        if y is not None:
            return y, x
        try:
            y = func(x)
            y = float("inf") if y is None else float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")
        cache_put(k, y)
        return y, x

    # ----------------- local search (adaptive trust-region) -----------------
    def local_trust_search(x0, f0, radius0, iters):
        """
        Coordinate + diagonal trial steps with adaptive radius.
        Very cheap and robust; good at polishing and escaping shallow stalls.
        """
        x = x0[:]
        fx = f0
        radius = radius0
        idxs = list(range(dim))

        for _ in range(iters):
            if not time_left():
                break
            improved = False
            random.shuffle(idxs)

            # coordinate steps
            for j in idxs:
                if span[j] <= eps:
                    continue
                cur = x[j]
                step = radius
                # try +/- step
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[j] = reflect_into(cur + sgn * step, lo[j], hi[j])
                    ft, xt = safe_eval(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                        break
                if improved or not time_left():
                    break

            # occasional diagonal step (captures interactions cheaply)
            if not improved and time_left() and dim >= 2:
                xt = x[:]
                for j in idxs[:min(dim, 6)]:  # limit work
                    if span[j] <= eps:
                        continue
                    xt[j] = reflect_into(xt[j] + (random.random() * 2 - 1) * radius, lo[j], hi[j])
                ft, xt = safe_eval(xt)
                if ft < fx:
                    x, fx = xt, ft
                    improved = True

            # adapt radius
            radius *= (1.35 if improved else 0.55)
            if radius <= 1e-14 * max_span:
                break
        return fx, x

    # ----------------- initialization (stratified + opposition + elite jitter) -----------------
    best = float("inf")
    x_best = rand_uniform()
    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    init_n = max(70, 16 * dim)  # a bit more probing up-front
    bins = max(6, int(math.sqrt(init_n)) + 2)

    pool = []
    for k in range(init_n):
        if not time_left():
            return best
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
            else:
                b = (random.randrange(bins) + k) % bins
                a = lo[i] + (b / bins) * span[i]
                c = lo[i] + ((b + 1) / bins) * span[i]
                x[i] = random.uniform(a, c)
        fx, x = safe_eval(x)
        pool.append((fx, x))
        if fx < best:
            best, x_best = fx, x[:]

        # opposition every 2nd sample
        if (k & 1) == 0 and time_left():
            xo = opposition(x)
            fo, xo = safe_eval(xo)
            pool.append((fo, xo))
            if fo < best:
                best, x_best = fo, xo[:]

    pool.sort(key=lambda t: t[0])

    # ----------------- DE (JADE-ish) -----------------
    NP_max = max(26, min(140, 14 * dim))
    NP_min = max(10, min(30, 4 * dim))
    if max_time <= 1.0:
        NP_max = max(20, min(NP_max, 45))
        NP_min = max(10, min(NP_min, 22))

    pop = []
    fit = []

    take = min(len(pool), NP_max)
    for i in range(take):
        x = pool[i][1][:]
        # jitter
        for j in range(dim):
            if span[j] > eps and random.random() < 0.8:
                x[j] = reflect_into(x[j] + (random.random() * 2 - 1) * 0.04 * span[j], lo[j], hi[j])
        fx, x = safe_eval(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, x_best = fx, x[:]

    while len(pop) < NP_max and time_left():
        x = rand_uniform()
        fx, x = safe_eval(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, x_best = fx, x[:]

    if not pop:
        return best

    archive = []
    A_MAX = NP_max

    def archive_add(x):
        if len(archive) >= A_MAX:
            archive[random.randrange(len(archive))] = x[:]
        else:
            archive.append(x[:])

    # parameter means (JADE-style)
    mu_F = 0.55
    mu_CR = 0.85

    def cauchy(mu, gamma):
        return mu + gamma * math.tan(math.pi * (random.random() - 0.5))

    def normal(mu, sigma):
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def pick_two_distinct(n, exclude):
        a = random.randrange(n - 1)
        a = a + 1 + exclude if a >= exclude else a
        b = random.randrange(n - 2)
        # map b into [0,n) excluding exclude and a
        candidates = []
        for r in (b, b + 1, b + 2):
            rr = r % n
            if rr != exclude and rr != a:
                candidates.append(rr)
            if len(candidates) >= 1:
                break
        if candidates:
            b = candidates[0]
        else:
            while True:
                b = random.randrange(n)
                if b != exclude and b != a:
                    break
        return a, b

    last_polish = time.time()
    no_improve = 0
    gen = 0

    while time_left():
        gen += 1
        n = len(pop)
        if n < 6:
            while len(pop) < 6 and time_left():
                x = rand_uniform()
                fx, x = safe_eval(x)
                pop.append(x)
                fit.append(fx)
                if fx < best:
                    best, x_best = fx, x[:]
            n = len(pop)

        # sort indices by fitness
        sorted_idx = sorted(range(n), key=lambda i: fit[i])
        prev_best = best

        # population size reduction over time
        frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        frac = clip(frac, 0.0, 1.0)
        target_NP = int(round(NP_max - frac * (NP_max - NP_min)))
        target_NP = max(NP_min, min(NP_max, target_NP))

        # p-best fraction
        p = min(0.35, max(0.08, 2.2 / max(2.0, math.sqrt(dim))))

        SF = []
        SCR = []
        dF = []

        for i in range(n):
            if not time_left():
                break

            xi = pop[i]
            fi = fit[i]

            # sample parameters
            CRi = clip(normal(mu_CR, 0.12), 0.0, 1.0)
            Fi = cauchy(mu_F, 0.12)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 8:
                Fi = cauchy(mu_F, 0.12)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.45
            if Fi > 1.0:
                Fi = 1.0

            # choose pbest
            top = max(2, int(math.ceil(p * n)))
            ipbest = sorted_idx[random.randrange(top)]
            xpbest = pop[ipbest]

            # exploration branch sometimes: rand/1 + binomial crossover
            if random.random() < (0.10 + 0.10 * (1.0 - frac)):
                r0 = random.randrange(n)
                r1, r2 = pick_two_distinct(n, r0)
                x0 = pop[r0]
                x1 = pop[r1]
                x2 = archive[random.randrange(len(archive))] if archive and random.random() < 0.5 else pop[r2]
                v = [x0[j] + Fi * (x1[j] - x2[j]) for j in range(dim)]
            else:
                r1, r2 = pick_two_distinct(n, i)
                xr1 = pop[r1]
                xr2 = archive[random.randrange(len(archive))] if archive and random.random() < 0.5 else pop[r2]
                v = [xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j]) for j in range(dim)]

            # crossover (fast path: create mask probability)
            jrand = random.randrange(dim)
            u = xi[:]  # start from parent
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
            u = project_reflect(u)

            fu, u = safe_eval(u)
            if fu < best:
                best, x_best = fu, u[:]

            if fu <= fi:
                archive_add(xi)
                pop[i] = u
                fit[i] = fu
                if fu < fi:
                    SF.append(Fi)
                    SCR.append(CRi)
                    dF.append(fi - fu)

        # adapt mu_F, mu_CR (JADE)
        if SF:
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F
            num = 0.0
            den = 0.0
            crw = 0.0
            for k in range(len(SF)):
                w = dF[k] / wsum
                f = SF[k]
                num += w * f * f
                den += w * f
                crw += w * SCR[k]
            if den > 1e-300:
                mu_F = clip((1 - 0.12) * mu_F + 0.12 * (num / den), 0.05, 1.0)
            mu_CR = clip((1 - 0.12) * mu_CR + 0.12 * crw, 0.0, 1.0)

        # reduce population
        if len(pop) > target_NP:
            worst = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
            remove = len(pop) - target_NP
            kill = set(worst[:remove])
            pop = [pop[i] for i in range(len(pop)) if i not in kill]
            fit = [fit[i] for i in range(len(fit)) if i not in kill]
            A_MAX = max(target_NP, NP_min)
            while len(archive) > A_MAX:
                archive[random.randrange(len(archive))] = archive[-1]
                archive.pop()

        # stagnation tracking
        if best < prev_best:
            no_improve = 0
        else:
            no_improve += 1

        # opportunistic local polish
        now = time.time()
        polish_period = 0.12 if dim <= 10 else 0.18
        if time_left() and (now - last_polish) > polish_period and (no_improve >= 2 or gen % 5 == 0):
            last_polish = now
            f0, x0 = safe_eval(x_best[:])
            radius0 = max(1e-10 * max_span, (0.03 if dim <= 10 else 0.02) * max_span / (1.0 + 0.02 * gen))
            fl, xl = local_trust_search(x0, f0, radius0, iters=max(10, 2 * dim))
            if fl < best:
                best, x_best = fl, xl[:]
            # inject into worst
            if pop:
                iw = max(range(len(pop)), key=lambda i: fit[i])
                if fl < fit[iw]:
                    pop[iw] = xl[:]
                    fit[iw] = fl

        # stronger restart injection on persistent stagnation
        if no_improve >= 6 and time_left():
            no_improve = 0
            n = len(pop)
            if n >= 6:
                repl = max(2, n // 4)  # replace 25% worst
                worst_idx = sorted(range(n), key=lambda i: fit[i], reverse=True)[:repl]
                sorted_idx = sorted(range(n), key=lambda i: fit[i])
                topk = max(2, min(n, max(8, n // 3)))

                for wi in worst_idx:
                    if not time_left():
                        break
                    base = pop[sorted_idx[int((random.random() ** 1.7) * topk)]]
                    x = base[:]
                    for j in range(dim):
                        if span[j] <= eps:
                            x[j] = lo[j]
                        else:
                            u = random.random()
                            if u < 0.65:
                                step = (random.random() * 2 - 1) * (0.14 * span[j])
                            else:
                                step = 0.07 * span[j] * math.tan(math.pi * (random.random() - 0.5))
                            x[j] = reflect_into(x[j] + step, lo[j], hi[j])
                    if random.random() < 0.30:
                        x = opposition(x)
                    fx, x = safe_eval(x)
                    pop[wi] = x
                    fit[wi] = fx
                    if fx < best:
                        best, x_best = fx, x[:]

    return best
