import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no deps).

    Changes vs previous versions:
      - Better global exploration: scrambled Halton + opposition points + corners
      - Strong local exploitation: multi-start Nelder–Mead (bounded) on elites
      - Robust sampling: occasional coordinate/pattern steps + adaptive scales
      - Restarts + elite memory to avoid getting stuck

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
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

    # ---------- utilities ----------
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
        # opposition-based point (reflect around midpoint)
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Approx N(0,1) without libraries
    def gauss01():
        return (sum(random.random() for _ in range(12)) - 6.0)

    # ---------- scrambled Halton ----------
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
            u -= int(u)  # wrap
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite pool ----------
    # keep a few best points (diversity by age and random replacement)
    K = 12 if dim <= 16 else 8
    elites = []  # list of (f, x, age)

    best = float("inf")
    best_x = None

    def push_elite(x, fx):
        nonlocal best, best_x, elites
        if not math.isfinite(fx):
            return
        if fx < best:
            best, best_x = fx, x[:]
        # age
        for i in range(len(elites)):
            f_i, x_i, a_i = elites[i]
            elites[i] = (f_i, x_i, a_i + 1)

        if len(elites) < K:
            elites.append((fx, x[:], 0))
            elites.sort(key=lambda t: t[0])
            return

        # replace worst, but occasionally replace an old one to keep diversity
        if fx < elites[-1][0]:
            elites[-1] = (fx, x[:], 0)
        else:
            # chance to replace an old elite with a decent new point
            # (prevents stale pool if objective is noisy / deceptive)
            if random.random() < 0.08:
                oldest_idx = max(range(len(elites)), key=lambda i: elites[i][2])
                if fx <= elites[oldest_idx][0] * 1.05:  # near as good
                    elites[oldest_idx] = (fx, x[:], 0)
        elites.sort(key=lambda t: t[0])

    # ---------- bounded Nelder–Mead ----------
    def nelder_mead(x_start, f_start, time_limit, max_evals):
        # Simple NM with bound clipping; good local refinement for black-box
        n = dim
        evals = 0

        # build initial simplex: x0 plus coordinate steps
        x0 = x_start[:]
        f0 = f_start

        # local step size relative to bounds
        scale = 0.08
        steps = [max(1e-12 * spans[i], scale * spans[i]) for i in range(n)]

        simplex = [(f0, x0)]
        for i in range(n):
            xi = x0[:]
            xi[i] += steps[i] * (1.0 if random.random() < 0.5 else -1.0)
            clip_inplace(xi)
            fi = safe_eval(xi); evals += 1
            simplex.append((fi, xi))

        simplex.sort(key=lambda t: t[0])

        # NM coefficients
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

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

        best_local_f, best_local_x = simplex[0][0], simplex[0][1][:]

        while evals < max_evals and time.time() < time_limit:
            simplex.sort(key=lambda t: t[0])
            if simplex[0][0] < best_local_f:
                best_local_f, best_local_x = simplex[0][0], simplex[0][1][:]

            f_best, x_best = simplex[0]
            f_worst, x_worst = simplex[-1]
            f_second, x_second = simplex[-2]

            c = centroid(simplex[:-1])  # centroid excluding worst

            # reflect
            xr = [c[j] + alpha * (c[j] - x_worst[j]) for j in range(n)]
            clip_inplace(xr)
            fr = safe_eval(xr); evals += 1
            if fr < f_best:
                # expand
                xe = [c[j] + gamma * (xr[j] - c[j]) for j in range(n)]
                clip_inplace(xe)
                fe = safe_eval(xe); evals += 1
                if fe < fr:
                    simplex[-1] = (fe, xe)
                else:
                    simplex[-1] = (fr, xr)
            elif fr < f_second:
                simplex[-1] = (fr, xr)
            else:
                # contract
                if fr < f_worst:
                    # outside contraction
                    xc = [c[j] + rho * (xr[j] - c[j]) for j in range(n)]
                else:
                    # inside contraction
                    xc = [c[j] - rho * (c[j] - x_worst[j]) for j in range(n)]
                clip_inplace(xc)
                fc = safe_eval(xc); evals += 1
                if fc < f_worst:
                    simplex[-1] = (fc, xc)
                else:
                    # shrink towards best
                    new_simplex = [simplex[0]]
                    xb = simplex[0][1]
                    for k in range(1, len(simplex)):
                        xk = simplex[k][1]
                        xs = [xb[j] + sigma * (xk[j] - xb[j]) for j in range(n)]
                        clip_inplace(xs)
                        fs = safe_eval(xs); evals += 1
                        new_simplex.append((fs, xs))
                        if evals >= max_evals or time.time() >= time_limit:
                            break
                    simplex = new_simplex

            # small termination: if simplex is tiny, stop early
            if evals >= max_evals or time.time() >= time_limit:
                break
            size = 0.0
            xb = simplex[0][1]
            for _, xk in simplex[1:]:
                for j in range(n):
                    d = (xk[j] - xb[j]) / (spans[j] + 1e-300)
                    size = max(size, abs(d))
            if size < 1e-8:
                break

        return best_local_x, best_local_f, evals

    # ---------- quick pattern search around a point ----------
    def pattern_search(x0, f0, time_limit, eval_limit):
        x = x0[:]
        fx = f0
        evals = 0
        step = [0.06 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]

        while evals < eval_limit and time.time() < time_limit:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if evals >= eval_limit or time.time() >= time_limit:
                    break
                # try + and -
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
                for i in range(dim):
                    step[i] = min(0.25 * spans[i], step[i] * 1.2)
            else:
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.55)
                # if really tiny, stop
                tiny = True
                for i in range(dim):
                    if step[i] > 1e-8 * spans[i]:
                        tiny = False
                        break
                if tiny:
                    break
        return x, fx, evals

    # ---------- initialization: mix of boundary/corners + halton + opposition ----------
    # corners/bounds (helps if optimum on boundary)
    probes = min(48, 4 * dim + 12)
    for _ in range(probes):
        if time.time() >= deadline:
            return best
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        # jitter a few dims
        for _j in range(max(1, dim // 4)):
            i = random.randrange(dim)
            x[i] = lows[i] + random.random() * spans[i]
        fx = safe_eval(x)
        push_elite(x, fx)

    # mid + random around mid
    if time.time() < deadline:
        xm = midpoint()
        fm = safe_eval(xm)
        push_elite(xm, fm)
        for _ in range(min(10, 2 + dim // 2)):
            if time.time() >= deadline:
                return best
            x = [xm[i] + gauss01() * (0.15 * spans[i]) for i in range(dim)]
            clip_inplace(x)
            fx = safe_eval(x)
            push_elite(x, fx)

    # space-filling burst
    init_n = max(80, min(420, 80 + 14 * dim))
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_point() if random.random() < 0.85 else rand_point()
        fx = safe_eval(x)
        push_elite(x, fx)

        # opposition point often gives a big jump "for free"
        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = safe_eval(xo)
        push_elite(xo, fo)

    if best_x is None:
        return best

    # ---------- main loop: evolution around elites + periodic Nelder–Mead refinements ----------
    sigma = 0.20
    sigma_min, sigma_max = 1e-6, 0.7

    succ = 0
    trials = 0
    last_improve = time.time()
    best_at_last = best

    lam = max(10, min(34, 10 + int(2.5 * math.log(dim + 1.0))))

    def mutate(center, sig):
        x = center[:]
        # correlated-ish: sometimes same noise scale, sometimes coord noise
        if random.random() < 0.25:
            z = gauss01() * (sig)
            for i in range(dim):
                x[i] += z * spans[i]
        else:
            for i in range(dim):
                x[i] += gauss01() * (sig * spans[i])
        return clip_inplace(x)

    nm_cooldown = 0  # avoid calling NM too frequently

    while time.time() < deadline:
        elites.sort(key=lambda t: t[0])
        parent = elites[0][1] if elites else best_x
        parent_f = elites[0][0] if elites else best

        # occasionally pick another elite for diversity
        if elites and random.random() < 0.35:
            idx = int((random.random() ** 2.2) * len(elites))
            idx = min(idx, len(elites) - 1)
            parent = elites[idx][1]
            parent_f = elites[idx][0]

        # generate offspring
        local_inject = (time.time() - last_improve < 0.25 * max_time)
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            r = random.random()
            if r < (0.86 if local_inject else 0.70):
                x = mutate(parent, sigma)
            elif r < 0.92:
                # coordinate kick from best_x
                x = (best_x[:] if best_x is not None else parent[:])
                k = 1 if dim == 1 else random.randint(1, max(1, dim // 3))
                for _k in range(k):
                    i = random.randrange(dim)
                    x[i] += gauss01() * (0.25 * sigma * spans[i] + 1e-15)
                clip_inplace(x)
            else:
                # global
                x = halton_point() if random.random() < 0.7 else rand_point()

            fx = safe_eval(x)
            push_elite(x, fx)

            trials += 1
            if fx < best:
                succ += 1
                last_improve = time.time()

        # 1/5 success rule-ish adaptation
        if trials >= 40:
            rate = succ / float(trials)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.22)
            elif rate < 0.16:
                sigma = max(sigma_min, sigma * 0.80)
            succ = 0
            trials = 0

        # small local pattern search frequently (cheap)
        if time.time() < deadline and best_x is not None and random.random() < 0.20:
            tl = min(deadline, time.time() + 0.03 * max_time)
            x1, f1, _ = pattern_search(best_x, best, tl, eval_limit=18 + 5 * dim)
            push_elite(x1, f1)

        # Nelder–Mead refinement occasionally on the best elites
        if nm_cooldown > 0:
            nm_cooldown -= 1
        else:
            improved_recently = (best < best_at_last - 1e-12)
            stuck = (time.time() - last_improve > max(0.20 * max_time, 0.7))
            if time.time() < deadline and (stuck or (improved_recently and random.random() < 0.35)):
                # pick one of top few elites, not always the best
                elites.sort(key=lambda t: t[0])
                topm = min(len(elites), 4)
                pick = 0 if random.random() < 0.6 else random.randrange(topm)
                x0 = elites[pick][1][:]
                f0 = elites[pick][0]

                tl = min(deadline, time.time() + 0.10 * max_time)
                x2, f2, _ = nelder_mead(x0, f0, tl, max_evals=40 + 12 * dim)
                push_elite(x2, f2)

                # after NM, slightly reduce sigma to exploit around new basin
                sigma = max(sigma_min, sigma * 0.92)
                nm_cooldown = 3  # wait a few loops

        best_at_last = best

        # restart if very stuck: enlarge sigma + fresh samples + opposition
        if time.time() - last_improve > max(0.33 * max_time, 1.2):
            last_improve = time.time()
            sigma = min(sigma_max, sigma * 1.45)
            inject = max(16, 3 * dim)
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
