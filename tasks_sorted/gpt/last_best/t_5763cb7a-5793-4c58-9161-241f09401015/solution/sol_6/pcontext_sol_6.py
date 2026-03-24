import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no numpy).

    Key improvements vs prior versions:
      - Better global exploration via low-discrepancy sampling (Halton) + opposition
      - Stronger exploitation via:
          * (mu, lambda)-style Evolution Strategy with diagonal sigma adaptation
          * Occasional finite-difference gradient step (SPSA-style, 2 evals)
          * Local bounded Nelder–Mead on the current best (short bursts)
      - Robust stagnation handling with adaptive restarts and sigma resets
      - Safety: handles exceptions/NaN/inf from func

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

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # approx N(0,1) (CLT)
    def gauss01():
        return sum(random.random() for _ in range(12)) - 6.0

    # ---------------- Halton sequence ----------------
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

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---------------- local Nelder–Mead (short budget) ----------------
    def nelder_mead(x_start, f_start, time_limit, max_evals):
        n = dim
        evals = 0
        x0 = x_start[:]
        f0 = f_start

        # initial simplex steps
        scale = 0.08
        steps = [max(1e-12 * spans[i], scale * spans[i]) for i in range(n)]

        simplex = [(f0, x0)]
        for i in range(n):
            xi = x0[:]
            xi[i] += steps[i] * (1.0 if random.random() < 0.5 else -1.0)
            clip_inplace(xi)
            fi = safe_eval(xi)
            evals += 1
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

        best_x = simplex[0][1][:]
        best_f = simplex[0][0]

        while evals < max_evals and time.time() < time_limit:
            simplex.sort(key=lambda t: t[0])
            if simplex[0][0] < best_f:
                best_f = simplex[0][0]
                best_x = simplex[0][1][:]

            f_best, x_best = simplex[0]
            f_worst, x_worst = simplex[-1]
            f_second = simplex[-2][0]

            c = centroid(simplex[:-1])

            # reflect
            xr = [c[j] + alpha * (c[j] - x_worst[j]) for j in range(n)]
            clip_inplace(xr)
            fr = safe_eval(xr)
            evals += 1

            if fr < f_best:
                # expand
                xe = [c[j] + gamma * (xr[j] - c[j]) for j in range(n)]
                clip_inplace(xe)
                fe = safe_eval(xe)
                evals += 1
                simplex[-1] = (fe, xe) if fe < fr else (fr, xr)
            elif fr < f_second:
                simplex[-1] = (fr, xr)
            else:
                # contract
                if fr < f_worst:
                    xc = [c[j] + rho * (xr[j] - c[j]) for j in range(n)]
                else:
                    xc = [c[j] - rho * (c[j] - x_worst[j]) for j in range(n)]
                clip_inplace(xc)
                fc = safe_eval(xc)
                evals += 1
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
                        fs = safe_eval(xs)
                        evals += 1
                        new_simplex.append((fs, xs))
                        if evals >= max_evals or time.time() >= time_limit:
                            break
                    simplex = new_simplex

            # tiny simplex stop
            simplex.sort(key=lambda t: t[0])
            xb = simplex[0][1]
            size = 0.0
            for _, xk in simplex[1:]:
                for j in range(n):
                    d = abs((xk[j] - xb[j]) / (spans[j] + 1e-300))
                    if d > size:
                        size = d
            if size < 1e-10:
                break

        return best_x, best_f

    # ---------------- SPSA-like gradient step (2 evals) ----------------
    def spsa_step(x, fx, alpha, c):
        # delta in {-1,+1}^dim
        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
        x_plus = [x[i] + c * delta[i] * spans[i] for i in range(dim)]
        x_minus = [x[i] - c * delta[i] * spans[i] for i in range(dim)]
        clip_inplace(x_plus)
        clip_inplace(x_minus)
        f_plus = safe_eval(x_plus)
        f_minus = safe_eval(x_minus)

        # grad estimate: (f+ - f-) / (2c*delta_i)
        g = [0.0] * dim
        denom = 2.0 * c + 1e-300
        df = (f_plus - f_minus) / denom
        for i in range(dim):
            g[i] = df / (delta[i] * (spans[i] + 1e-300))

        # step: x - alpha * g (normalize by dim to avoid huge steps)
        scale = alpha / (1.0 + 0.2 * dim)
        xn = [x[i] - scale * g[i] * spans[i] for i in range(dim)]
        clip_inplace(xn)
        fn = safe_eval(xn)
        return xn, fn, f_plus, f_minus

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # corners/jitter probes
    probes = min(40 + 2 * dim, 90)
    for _ in range(probes):
        if time.time() >= deadline:
            return best
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        # jitter a few coordinates
        for _j in range(max(1, dim // 3)):
            i = random.randrange(dim)
            x[i] = lows[i] + random.random() * spans[i]
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x[:]

    # halton + opposition burst
    init_n = max(120, min(900, 160 + 25 * dim))
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_point() if random.random() < 0.85 else rand_point()
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x[:]
        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = safe_eval(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        return best

    # ---------------- main loop: ES + SPSA + NM ----------------
    # Diagonal sigmas (per-dimension) improve conditioning vs single sigma
    sig = [0.18] * dim
    sig_min = 1e-10
    sig_max = 0.75

    # ES parameters
    lam = max(16, min(64, 16 + int(6.0 * math.log(dim + 2.0))))
    mu = max(4, lam // 4)

    # track stagnation
    last_improve_t = time.time()
    best_at_last_check = best

    # reuse small pool of good points
    pool = [(best, best_x[:])]

    def mutate(x):
        # correlated kick sometimes, else diagonal
        y = x[:]
        if random.random() < 0.20:
            z = gauss01()
            for i in range(dim):
                y[i] += z * sig[i] * spans[i]
        else:
            for i in range(dim):
                y[i] += gauss01() * sig[i] * spans[i]
        return clip_inplace(y)

    cooldown_nm = 0
    spsa_alpha = 0.35
    spsa_c = 0.04

    while time.time() < deadline:
        # --- offspring generation ---
        off = []
        # parent choice: mostly best, sometimes other pool entry
        if len(pool) > 1 and random.random() < 0.35:
            parent = pool[int((random.random() ** 2.0) * len(pool))][1]
        else:
            parent = best_x

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            r = random.random()
            if r < 0.78:
                x = mutate(parent)
            elif r < 0.92:
                # crossover between two pool points
                a = pool[int(random.random() * len(pool))][1]
                b = pool[int(random.random() * len(pool))][1]
                x = [a[i] + random.random() * (b[i] - a[i]) for i in range(dim)]
                clip_inplace(x)
            else:
                # global injection
                x = halton_point() if random.random() < 0.65 else rand_point()

            fx = safe_eval(x)
            off.append((fx, x))
            if fx < best:
                best, best_x = fx, x[:]
                last_improve_t = time.time()

        off.sort(key=lambda t: t[0])
        # update pool with best offspring + incumbent
        pool = off[:mu] + [(best, best_x[:])]
        pool.sort(key=lambda t: t[0])
        pool = pool[:max(mu, 8)]

        # --- sigma adaptation (1/5-ish using offspring quality) ---
        # success = fraction beating incumbent-at-start-of-gen proxy
        success = 0
        thresh = pool[-1][0]  # weak proxy
        for fx, _ in off[:mu]:
            if fx <= thresh:
                success += 1
        rate = success / float(mu)

        if rate > 0.25:
            for i in range(dim):
                sig[i] = min(sig_max, sig[i] * 1.18)
        elif rate < 0.12:
            for i in range(dim):
                sig[i] = max(sig_min, sig[i] * 0.82)

        # --- occasional SPSA step around best (cheap directional improvement) ---
        if time.time() < deadline and random.random() < 0.35:
            xn, fn, _, _ = spsa_step(best_x, best, spsa_alpha, spsa_c)
            if fn < best:
                best, best_x = fn, xn[:]
                last_improve_t = time.time()
                # mild tighten after good progress
                for i in range(dim):
                    sig[i] = max(sig_min, sig[i] * 0.93)
            else:
                # slowly anneal SPSA params
                spsa_alpha *= 0.995
                spsa_c *= 0.997
                if spsa_alpha < 0.05:
                    spsa_alpha = 0.05
                if spsa_c < 0.005:
                    spsa_c = 0.005

        # --- occasional Nelder–Mead burst (exploitation) ---
        if cooldown_nm > 0:
            cooldown_nm -= 1
        else:
            stuck = (time.time() - last_improve_t) > max(0.18 * max_time, 0.8)
            if stuck or (random.random() < 0.12):
                tl = min(deadline, time.time() + 0.08 * max_time)
                x2, f2 = nelder_mead(best_x, best, tl, max_evals=40 + 10 * dim)
                if f2 < best:
                    best, best_x = f2, x2[:]
                    last_improve_t = time.time()
                cooldown_nm = 3

        # --- stagnation restart ---
        if best < best_at_last_check - 1e-12:
            best_at_last_check = best
        else:
            if (time.time() - last_improve_t) > max(0.33 * max_time, 1.2):
                last_improve_t = time.time()
                # reset sigmas and inject diversified samples
                for i in range(dim):
                    sig[i] = min(sig_max, max(0.12, sig[i] * 1.6))
                inject = max(10, 2 * dim)
                for _ in range(inject):
                    if time.time() >= deadline:
                        return best
                    x = rand_point() if random.random() < 0.55 else halton_point()
                    fx = safe_eval(x)
                    if fx < best:
                        best, best_x = fx, x[:]
                        best_at_last_check = best
                    if time.time() >= deadline:
                        return best
                    xo = opposite(x)
                    fo = safe_eval(xo)
                    if fo < best:
                        best, best_x = fo, xo[:]
                        best_at_last_check = best

    return best
