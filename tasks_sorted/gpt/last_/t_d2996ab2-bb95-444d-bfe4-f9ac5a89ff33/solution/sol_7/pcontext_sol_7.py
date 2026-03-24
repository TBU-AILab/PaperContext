import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (self-contained, no external libs).

    Improvements vs the provided code:
      - Uses a robust global-to-local portfolio centered on Differential Evolution (DE)
        with adaptive parameters (jDE-style), which tends to work very well on wide
        classes of continuous bounded problems.
      - Keeps an elite archive + occasional local coordinate/pattern refinement.
      - Stronger bound handling (reflection) to reduce boundary bias.
      - Quantized cache to avoid wasting evaluations on near-duplicates.
      - Automatically scales population size to time/dim and remains anytime.

    Returns:
      best (float): best objective found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))
    eps = 1e-12

    # ---------- trivial/degenerate cases ----------
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
        # reflection keeps distribution less biased than hard clipping
        if a == b:
            return a
        w = b - a
        x = v
        # bring to [a, b] by reflection
        if x < a or x > b:
            x = a + (x - a) % (2.0 * w)
            if x > b:
                x = b - (x - b)
        return clip(x, a, b)

    def project_reflect(x):
        return [reflect_into(float(x[i]), lo[i], hi[i]) for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) if span[i] > eps else lo[i] for i in range(dim)]

    # --------- quantized cache ----------
    qstep = []
    for i in range(dim):
        s = span[i]
        qstep.append(0.0 if s <= eps else max(1e-12, 2e-7 * s))
    cache = {}
    cache_keys = []
    CACHE_MAX = 60000

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
            # evict random ~3%
            ev = max(50, CACHE_MAX // 33)
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
        k = key_of(x)
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

    # ---------- elites ----------
    best = float("inf")
    x_best = rand_uniform()
    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    ELITE_MAX = max(20, 5 * dim)
    elites = [(best, x_best[:])]

    def push_elite(fx, x):
        nonlocal best, x_best, elites
        if fx < best:
            best = fx
            x_best = x[:]
        elites.append((fx, x[:]))
        if len(elites) > 6 * ELITE_MAX:
            elites.sort(key=lambda t: t[0])
            elites[:] = elites[:ELITE_MAX]

    # ---------- quick local pattern search ----------
    def pattern_search(x0, f0, step0, max_iter):
        x = x0[:]
        fx = f0
        step = step0
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
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[j] = reflect_into(cur + sgn * step, lo[j], hi[j])
                    ft, xt = safe_eval(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                        break
                if improved and not time_left():
                    break
            step *= (1.25 if improved else 0.55)
            if step <= 1e-14 * max_span:
                break
        return fx, x

    # ---------- initialization (stratified-ish + opposition) ----------
    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                y[i] = lo[i]
            else:
                y[i] = lo[i] + hi[i] - x[i]
        return project_reflect(y)

    init_n = max(40, 12 * dim)
    bins = max(5, int(math.sqrt(init_n)) + 1)

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
        push_elite(fx, x)
        if (k % 2) == 0 and time_left():
            xo = opposition(x)
            fo, xo = safe_eval(xo)
            push_elite(fo, xo)

    # ---------- Differential Evolution (DE) core ----------
    # Population size: tradeoff between diversity and eval rate.
    # Keep it moderate for high dim.
    NP = max(18, min(90, 10 * dim))
    # If time is tiny, reduce
    if max_time <= 1.0:
        NP = max(12, min(NP, 24))

    # Build initial population biased toward elites (good starts) + random
    pop = []
    fit = []

    elites.sort(key=lambda t: t[0])
    elite_seeds = min(len(elites), max(2, NP // 4))
    for i in range(elite_seeds):
        x = elites[i][1][:]
        # small jitter to create diversity
        for j in range(dim):
            if span[j] > eps and random.random() < 0.7:
                x[j] = reflect_into(x[j] + (random.random() * 2 - 1) * 0.05 * span[j], lo[j], hi[j])
        fx, x = safe_eval(x)
        pop.append(x)
        fit.append(fx)
        push_elite(fx, x)

    while len(pop) < NP and time_left():
        x = rand_uniform()
        fx, x = safe_eval(x)
        pop.append(x)
        fit.append(fx)
        push_elite(fx, x)

    if not pop:
        return best

    # jDE-style per-individual parameters (self-adaptation)
    F = [0.5 for _ in range(len(pop))]
    CR = [0.9 for _ in range(len(pop))]
    tau1, tau2 = 0.1, 0.1

    def rand_idx_excluding(n, exc):
        r = random.randrange(n - 1)
        return r + 1 + exc if r >= exc else r

    def pick3(n, exclude):
        a = rand_idx_excluding(n, exclude)
        b = rand_idx_excluding(n, exclude)
        while b == a:
            b = rand_idx_excluding(n, exclude)
        c = rand_idx_excluding(n, exclude)
        while c == a or c == b:
            c = rand_idx_excluding(n, exclude)
        return a, b, c

    # Main loop: DE + occasional local polish + restart injections on stagnation
    gen = 0
    no_improve = 0

    while time_left():
        gen += 1
        prev_best = best

        n = len(pop)
        # If population shrank (shouldn't), fix
        if n < 6:
            while n < 6 and time_left():
                x = rand_uniform()
                fx, x = safe_eval(x)
                pop.append(x)
                fit.append(fx)
                F.append(0.5)
                CR.append(0.9)
                push_elite(fx, x)
                n += 1

        # Find current best index
        ibest = min(range(n), key=lambda i: fit[i])
        xbest = pop[ibest]

        # One DE generation
        for i in range(n):
            if not time_left():
                break

            # parameter self-adaptation
            Fi = F[i]
            CRi = CR[i]
            if random.random() < tau1:
                Fi = 0.1 + 0.9 * random.random()
            if random.random() < tau2:
                CRi = random.random()

            # Mutation: "current-to-best/1" with occasional "rand/1" to keep exploration
            r = random.random()
            a, b, c = pick3(n, i)

            xi = pop[i]
            if r < 0.75:
                # current-to-best/1
                v = [0.0] * dim
                for j in range(dim):
                    v[j] = xi[j] + Fi * (xbest[j] - xi[j]) + Fi * (pop[a][j] - pop[b][j])
            else:
                # rand/1
                v = [0.0] * dim
                for j in range(dim):
                    v[j] = pop[a][j] + Fi * (pop[b][j] - pop[c][j])

            # Binomial crossover (ensure at least one dimension from mutant)
            jrand = random.randrange(dim)
            u = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
            u = project_reflect(u)

            fu, u = safe_eval(u)
            push_elite(fu, u)

            # Selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                F[i] = Fi
                CR[i] = CRi

        # Local polish occasionally
        if time_left() and (gen % 7 == 0):
            f0, x0 = safe_eval(x_best[:])
            step0 = max(1e-10 * max_span, 0.03 * max_span / (1.0 + 0.02 * gen))
            fl, xl = pattern_search(x0, f0, step0, max_iter=max(10, 2 * dim))
            push_elite(fl, xl)
            # Inject polished solution into worst individual
            iworst = max(range(len(pop)), key=lambda i: fit[i])
            if fl < fit[iworst]:
                pop[iworst] = xl
                fit[iworst] = fl

        # Stagnation handling: inject new points around elites + random restarts
        if best < prev_best:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 4 and time_left():
            elites.sort(key=lambda t: t[0])
            # replace a fraction of worst individuals
            repl = max(2, len(pop) // 6)
            worst_idx = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)[:repl]
            for wi in worst_idx:
                if not time_left():
                    break
                if random.random() < 0.65 and elites:
                    # sample around a good elite with heavy-tail-ish steps
                    base = elites[int((random.random() ** 1.7) * min(len(elites), ELITE_MAX))][1]
                    x = base[:]
                    for j in range(dim):
                        if span[j] <= eps:
                            x[j] = lo[j]
                        else:
                            # mixture of gaussian-ish and cauchy-ish
                            u = random.random()
                            if u < 0.75:
                                step = (random.random() * 2 - 1) * 0.12 * span[j]
                            else:
                                t = math.tan(math.pi * (random.random() - 0.5))
                                step = 0.06 * span[j] * t
                            x[j] = reflect_into(x[j] + step, lo[j], hi[j])
                else:
                    x = rand_uniform()
                    if random.random() < 0.35:
                        x = opposition(x)
                fx, x = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                F[wi] = 0.3 + 0.7 * random.random()
                CR[wi] = random.random()
                push_elite(fx, x)
            no_improve = 0

    return best
