import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (self-contained).

    Hybrid strategy:
      1) Space-filling init: scrambled Halton + corners + midpoint jitter + opposition
      2) Global search: (1+lambda) evolution strategy w/ 1/5 success rule step adaptation
      3) Local refinement: bounded Nelder–Mead on elites + cheap coordinate pattern search
      4) Restarts: stagnation-triggered injection + sigma expansion

    Returns:
        best (float): best function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if bounds is None or len(bounds) != dim:
        raise ValueError("bounds must be a list of (low, high) pairs, one per dimension")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if not (s > 0.0):
            raise ValueError("Each bound must satisfy high > low")

    # -------------------- utilities --------------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def midpoint():
        return [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # approx N(0,1) without external libs
    def gauss01():
        return (sum(random.random() for _ in range(12)) - 6.0)

    # -------------------- Halton (scrambled/shifted) --------------------
    def first_primes(n):
        primes = []
        c = 2
        while len(primes) < n:
            ok = True
            r = int(c ** 0.5)
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(c)
            c += 1
        return primes

    primes = first_primes(dim)
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = vdc(idx, primes[i]) + halton_shift[i]
            u -= int(u)  # wrap to [0,1)
            x[i] = lows[i] + u * spans[i]
        return x

    # -------------------- elite pool --------------------
    K = 14 if dim <= 16 else 10
    elites = []  # list of (f, x, age)

    best = float("inf")
    best_x = None

    def push_elite(x, fx):
        nonlocal best, best_x, elites
        if not math.isfinite(fx):
            return
        if fx < best:
            best, best_x = fx, x[:]

        # age everyone
        for i in range(len(elites)):
            f_i, x_i, a_i = elites[i]
            elites[i] = (f_i, x_i, a_i + 1)

        if len(elites) < K:
            elites.append((fx, x[:], 0))
            elites.sort(key=lambda t: t[0])
            return

        elites.sort(key=lambda t: t[0])
        # replace worst if better
        if fx < elites[-1][0]:
            elites[-1] = (fx, x[:], 0)
        else:
            # sometimes replace the oldest if near-competitive (diversity/anti-stale)
            if random.random() < 0.10:
                oldest_idx = max(range(len(elites)), key=lambda i: elites[i][2])
                if fx <= elites[oldest_idx][0] * 1.03:
                    elites[oldest_idx] = (fx, x[:], 0)
        elites.sort(key=lambda t: t[0])

    # -------------------- bounded Nelder–Mead (local refine) --------------------
    def nelder_mead(x_start, f_start, time_limit, max_evals):
        n = dim
        evals = 0

        x0 = x_start[:]
        f0 = f_start

        # initial simplex step sizes
        scale = 0.06
        steps = [max(1e-14 * spans[i], scale * spans[i]) for i in range(n)]

        simplex = [(f0, x0)]
        for i in range(n):
            xi = x0[:]
            xi[i] += steps[i] * (1.0 if random.random() < 0.5 else -1.0)
            clip_inplace(xi)
            fi = safe_eval(xi); evals += 1
            simplex.append((fi, xi))

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        def centroid(points):
            c = [0.0] * n
            m = len(points)
            for _, x in points:
                for j in range(n):
                    c[j] += x[j]
            inv = 1.0 / m
            for j in range(n):
                c[j] *= inv
            return c

        best_local_x = simplex[0][1][:]
        best_local_f = simplex[0][0]

        while evals < max_evals and time.time() < time_limit:
            simplex.sort(key=lambda t: t[0])
            if simplex[0][0] < best_local_f:
                best_local_f = simplex[0][0]
                best_local_x = simplex[0][1][:]

            f_best, x_best = simplex[0]
            f_worst, x_worst = simplex[-1]
            f_second = simplex[-2][0]

            c = centroid(simplex[:-1])

            # reflect
            xr = [c[j] + alpha * (c[j] - x_worst[j]) for j in range(n)]
            clip_inplace(xr)
            fr = safe_eval(xr); evals += 1

            if fr < f_best:
                # expand
                xe = [c[j] + gamma * (xr[j] - c[j]) for j in range(n)]
                clip_inplace(xe)
                fe = safe_eval(xe); evals += 1
                simplex[-1] = (fe, xe) if fe < fr else (fr, xr)
            elif fr < f_second:
                simplex[-1] = (fr, xr)
            else:
                # contract
                if fr < f_worst:
                    xc = [c[j] + rho * (xr[j] - c[j]) for j in range(n)]  # outside
                else:
                    xc = [c[j] - rho * (c[j] - x_worst[j]) for j in range(n)]  # inside
                clip_inplace(xc)
                fc = safe_eval(xc); evals += 1
                if fc < f_worst:
                    simplex[-1] = (fc, xc)
                else:
                    # shrink
                    xb = simplex[0][1]
                    new_simplex = [simplex[0]]
                    for k in range(1, len(simplex)):
                        xk = simplex[k][1]
                        xs = [xb[j] + sigma * (xk[j] - xb[j]) for j in range(n)]
                        clip_inplace(xs)
                        fs = safe_eval(xs); evals += 1
                        new_simplex.append((fs, xs))
                        if evals >= max_evals or time.time() >= time_limit:
                            break
                    simplex = new_simplex

            # stop if simplex is tiny
            simplex.sort(key=lambda t: t[0])
            xb = simplex[0][1]
            size = 0.0
            for _, xk in simplex[1:]:
                for j in range(n):
                    d = (xk[j] - xb[j]) / (spans[j] + 1e-300)
                    if abs(d) > size:
                        size = abs(d)
            if size < 1e-9:
                break

        return best_local_x, best_local_f

    # -------------------- cheap coordinate pattern search --------------------
    def pattern_search(x0, f0, time_limit, eval_limit):
        x = x0[:]
        fx = f0
        evals = 0

        step = [0.05 * spans[i] for i in range(dim)]
        min_step = [1e-14 * spans[i] for i in range(dim)]

        while evals < eval_limit and time.time() < time_limit:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if evals >= eval_limit or time.time() >= time_limit:
                    break
                for sgn in (1.0, -1.0):
                    xn = x[:]
                    xn[i] += sgn * step[i]
                    clip_inplace(xn)
                    fn = safe_eval(xn); evals += 1
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        break
                if improved:
                    break

            if improved:
                for i in range(dim):
                    step[i] = min(0.25 * spans[i], step[i] * 1.25)
            else:
                tiny = True
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.6)
                    if step[i] > 1e-7 * spans[i]:
                        tiny = False
                if tiny:
                    break

        return x, fx

    # -------------------- initialization --------------------
    # corners + jitter
    probes = min(56, 4 * dim + 16)
    for _ in range(probes):
        if time.time() >= deadline:
            return best
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        for _j in range(max(1, dim // 3)):
            i = random.randrange(dim)
            x[i] = lows[i] + random.random() * spans[i]
        fx = safe_eval(x)
        push_elite(x, fx)

    # midpoint + gaussian jitter cloud
    if time.time() < deadline:
        xm = midpoint()
        fm = safe_eval(xm)
        push_elite(xm, fm)
        cloud = min(16, 4 + dim)
        for _ in range(cloud):
            if time.time() >= deadline:
                return best
            x = [xm[i] + gauss01() * (0.18 * spans[i]) for i in range(dim)]
            clip_inplace(x)
            fx = safe_eval(x)
            push_elite(x, fx)

    # space-filling burst (Halton + opposition)
    init_n = max(120, min(650, 120 + 18 * dim))
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_point() if random.random() < 0.88 else rand_point()
        fx = safe_eval(x)
        push_elite(x, fx)

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = safe_eval(xo)
        push_elite(xo, fo)

    if best_x is None:
        return best

    # -------------------- main loop: ES + local refinements --------------------
    sigma = 0.22
    sigma_min, sigma_max = 1e-8, 0.75

    succ = 0
    trials = 0
    last_improve_t = time.time()
    last_best = best

    lam = max(12, min(44, 12 + int(3.0 * math.log(dim + 2.0))))

    def mutate(center, sig):
        x = center[:]
        if random.random() < 0.22:
            z = gauss01() * sig
            for i in range(dim):
                x[i] += z * spans[i]
        else:
            for i in range(dim):
                x[i] += gauss01() * (sig * spans[i])
        return clip_inplace(x)

    nm_cooldown = 0

    while time.time() < deadline:
        elites.sort(key=lambda t: t[0])
        if elites:
            # bias towards best, but sometimes pick other elites for diversity
            if random.random() < 0.70:
                parent = elites[0][1]
                parent_f = elites[0][0]
            else:
                idx = int((random.random() ** 2.4) * len(elites))
                idx = min(idx, len(elites) - 1)
                parent = elites[idx][1]
                parent_f = elites[idx][0]
        else:
            parent = best_x
            parent_f = best

        # offspring batch
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            r = random.random()
            if r < 0.74:
                x = mutate(parent, sigma)
            elif r < 0.88:
                # coordinate kick from current best
                x = (best_x[:] if best_x is not None else parent[:])
                k = 1 if dim == 1 else random.randint(1, max(1, dim // 2))
                for _k in range(k):
                    i = random.randrange(dim)
                    x[i] += gauss01() * (0.35 * sigma * spans[i] + 1e-18)
                clip_inplace(x)
            elif r < 0.95:
                # mix two elites (box crossover)
                if len(elites) >= 2:
                    a = elites[int(random.random() * min(len(elites), 6))][1]
                    b = elites[int(random.random() * min(len(elites), 6))][1]
                    x = [a[i] + random.random() * (b[i] - a[i]) for i in range(dim)]
                    clip_inplace(x)
                else:
                    x = mutate(parent, sigma * 1.2)
            else:
                # global sample
                x = halton_point() if random.random() < 0.7 else rand_point()

            fx = safe_eval(x)
            push_elite(x, fx)

            trials += 1
            if fx < best:
                succ += 1
                last_improve_t = time.time()

        # 1/5 success-ish rule
        if trials >= 50:
            rate = succ / float(trials)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.25)
            elif rate < 0.15:
                sigma = max(sigma_min, sigma * 0.78)
            succ = 0
            trials = 0

        # frequent cheap pattern search around incumbent
        if best_x is not None and random.random() < 0.28 and time.time() < deadline:
            tl = min(deadline, time.time() + 0.03 * max_time)
            x1, f1 = pattern_search(best_x, best, tl, eval_limit=14 + 6 * dim)
            push_elite(x1, f1)

        # occasional Nelder–Mead on top elites when stuck / after improvements
        if nm_cooldown > 0:
            nm_cooldown -= 1
        else:
            improved_since = (best < last_best - 1e-12)
            stuck = (time.time() - last_improve_t > max(0.18 * max_time, 0.8))
            if (stuck or (improved_since and random.random() < 0.45)) and time.time() < deadline:
                elites.sort(key=lambda t: t[0])
                topm = min(len(elites), 5)
                pick = 0 if random.random() < 0.55 else random.randrange(topm)
                x0 = elites[pick][1][:]
                f0 = elites[pick][0]
                tl = min(deadline, time.time() + 0.12 * max_time)
                x2, f2 = nelder_mead(x0, f0, tl, max_evals=55 + 14 * dim)
                push_elite(x2, f2)

                sigma = max(sigma_min, sigma * 0.92)
                nm_cooldown = 3

        last_best = best

        # restart injection if stagnant
        if time.time() - last_improve_t > max(0.32 * max_time, 1.4):
            last_improve_t = time.time()
            sigma = min(sigma_max, sigma * 1.55)

            inject = max(20, 4 * dim)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                x = rand_point() if random.random() < 0.55 else halton_point()
                fx = safe_eval(x)
                push_elite(x, fx)

                if time.time() >= deadline:
                    return best
                xo = opposite(x)
                fo = safe_eval(xo)
                push_elite(xo, fo)

    return best
