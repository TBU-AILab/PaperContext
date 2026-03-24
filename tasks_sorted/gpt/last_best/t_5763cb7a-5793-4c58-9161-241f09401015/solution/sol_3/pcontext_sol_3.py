import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (self-contained, no deps).

    Core ideas (robust across many black-box functions):
      - Better global coverage: scrambled Halton + occasional boundary/corner probes
      - Maintain an elite pool (top-K points) with aging
      - Main engine: (1+λ)-ES with decaying sigma + success-based adaptation
      - Local refinement: opportunistic coordinate/pattern search around best elites
      - Restarts when stagnating: increase sigma + inject fresh samples

    Returns:
        best (float): minimum function value found within max_time seconds.
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

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = func(x)
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # "Gaussian-ish" generator (no numpy): sum of uniforms -> approx N(0,1)
    def gauss01():
        return (sum(random.random() for _ in range(12)) - 6.0)

    # ---------------- Halton (scrambled) ----------------
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
            u -= int(u)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- elite pool ----------------
    # store: (f, x, age)
    K = 10 if dim <= 12 else 6
    elites = []

    def push_elite(x, fx):
        nonlocal elites
        if not math.isfinite(fx):
            return
        # age all
        for k in range(len(elites)):
            f_k, x_k, age_k = elites[k]
            elites[k] = (f_k, x_k, age_k + 1)
        if len(elites) < K:
            elites.append((fx, x[:], 0))
            elites.sort(key=lambda t: t[0])
            return
        if fx < elites[-1][0]:
            elites[-1] = (fx, x[:], 0)
            elites.sort(key=lambda t: t[0])

    # ---------------- local: robust pattern search ----------------
    def pattern_search(x0, f0, time_limit, eval_limit):
        x = x0[:]
        fx = f0
        evals = 0

        step = [0.12 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]
        max_step = [0.35 * spans[i] for i in range(dim)]

        last_dx = [0.0] * dim
        noimp = 0

        while evals < eval_limit and time.time() < time_limit:
            improved = False
            base_x = x[:]
            base_f = fx

            coords = list(range(dim))
            random.shuffle(coords)

            for i in coords:
                if evals >= eval_limit or time.time() >= time_limit:
                    break

                # + step
                xp = x[:]
                xp[i] += step[i]
                clip_inplace(xp)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    last_dx = [xp[j] - x[j] for j in range(dim)]
                    x, fx = xp, fp
                    improved = True
                    continue

                if evals >= eval_limit or time.time() >= time_limit:
                    break

                # - step
                xm = x[:]
                xm[i] -= step[i]
                clip_inplace(xm)
                fm = safe_eval(xm); evals += 1
                if fm < fx:
                    last_dx = [xm[j] - x[j] for j in range(dim)]
                    x, fx = xm, fm
                    improved = True

            # pattern move
            if improved and evals < eval_limit and time.time() < time_limit:
                xt = [x[i] + 0.9 * last_dx[i] for i in range(dim)]
                clip_inplace(xt)
                ft = safe_eval(xt); evals += 1
                if ft < fx:
                    x, fx = xt, ft

            if fx < base_f:
                noimp = 0
                for i in range(dim):
                    step[i] = min(max_step[i], step[i] * 1.25)
            else:
                noimp += 1
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.55)

                # small escape
                if noimp >= 3 and evals < eval_limit and time.time() < time_limit:
                    noimp = 0
                    xr = x[:]
                    for i in range(dim):
                        xr[i] += gauss01() * (0.05 * spans[i])
                    clip_inplace(xr)
                    fr = safe_eval(xr); evals += 1
                    if fr < fx:
                        x, fx = xr, fr

        return x, fx, evals

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # Corner/boundary probes (often helps a lot)
    if time.time() < deadline:
        probes = min(3 * dim + 8, 40)
        for _ in range(probes):
            if time.time() >= deadline:
                return best
            x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
            # jitter a few dims to avoid strict corners only
            for _j in range(max(1, dim // 5)):
                i = random.randrange(dim)
                u = random.random()
                x[i] = lows[i] + u * spans[i]
            fx = safe_eval(x)
            if fx < best:
                best, best_x = fx, x[:]
            push_elite(x, fx)

    # Space-filling burst: Halton + random
    init_n = max(60, min(320, 60 + 12 * dim))
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_point() if (random.random() < 0.80) else rand_point()
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(x, fx)

    if best_x is None:
        return best

    # ---------------- main: (1+λ)-ES around elites ----------------
    # sigma in normalized [0,1] coordinates; map to spans
    # start relatively broad; adapt with 1/5-ish success logic
    sigma = 0.22
    sigma_min = 1e-6
    sigma_max = 0.6

    # evaluation accounting for adaptation
    succ = 0
    trials = 0
    last_improve_time = time.time()
    stagnation = 0

    # population size
    lam = 8 + int(2.0 * math.log(dim + 1.0))
    lam = max(8, min(28, lam))

    # helper to sample around center in normalized space
    def mutate(center, sig):
        x = center[:]
        for i in range(dim):
            x[i] += gauss01() * (sig * spans[i])
        return clip_inplace(x)

    while time.time() < deadline:
        # choose a parent: mostly best, sometimes other elites (diversity)
        elites.sort(key=lambda t: t[0])
        if elites and random.random() < 0.75:
            parent = elites[0][1]
            parent_f = elites[0][0]
        else:
            idx = int((random.random() ** 2.0) * max(1, len(elites)))  # bias to better
            idx = min(idx, len(elites) - 1) if elites else 0
            if elites:
                parent = elites[idx][1]
                parent_f = elites[idx][0]
            else:
                parent = best_x
                parent_f = best

        # generate offspring
        improved_any = False
        best_child_f = float("inf")
        best_child_x = None

        for _ in range(lam):
            if time.time() >= deadline:
                return best
            # mix of local gaussian and occasional global sample
            if random.random() < (0.88 if stagnation < 10 else 0.72):
                x = mutate(parent, sigma)
            else:
                # global injection (Halton preferred)
                x = halton_point() if random.random() < 0.7 else rand_point()

            fx = safe_eval(x)
            push_elite(x, fx)

            trials += 1
            if fx < best:
                best, best_x = fx, x[:]
                improved_any = True
                best_child_f, best_child_x = fx, x[:]
                succ += 1
                last_improve_time = time.time()
                stagnation = 0
            else:
                stagnation += 1
                if fx < best_child_f:
                    best_child_f, best_child_x = fx, x[:]

        # success-based sigma adaptation (smoothed)
        if trials >= 30:
            rate = succ / float(trials)
            # target success ~ 0.20
            if rate > 0.23:
                sigma = min(sigma_max, sigma * 1.18)
            elif rate < 0.17:
                sigma = max(sigma_min, sigma * 0.82)
            # reset window
            succ = 0
            trials = 0

        # opportunistic local refinement if we found a promising child / or occasionally on best
        now = time.time()
        if now < deadline and (improved_any or random.random() < 0.12):
            x0 = best_x if best_x is not None else parent
            f0 = best if best_x is not None else parent_f
            # small slice
            tl = min(deadline, now + max(0.01, 0.06 * (deadline - now)))
            x1, f1, _ = pattern_search(x0, f0, time_limit=tl, eval_limit=25 + 6 * dim)
            push_elite(x1, f1)
            if f1 < best:
                best, best_x = f1, x1[:]
                last_improve_time = time.time()
                stagnation = 0

        # restart logic when stuck for time
        if time.time() - last_improve_time > max(0.18 * max_time, 0.6):
            last_improve_time = time.time()
            # broaden search and inject fresh points
            sigma = min(sigma_max, sigma * 1.35)
            inject = max(12, 2 * dim)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                x = rand_point() if random.random() < 0.5 else halton_point()
                fx = safe_eval(x)
                push_elite(x, fx)
                if fx < best:
                    best, best_x = fx, x[:]
                    stagnation = 0

    return best
