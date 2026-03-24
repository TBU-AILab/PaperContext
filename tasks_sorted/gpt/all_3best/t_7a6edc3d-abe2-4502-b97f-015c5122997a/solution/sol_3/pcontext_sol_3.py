import random
import time
import math

def run(func, dim, bounds, max_time):
    start = time.time()
    deadline = start + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def eval_point(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Halton (for deterministic-ish global coverage)
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(n, base):
        vdc = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(index, bases):
        u = [van_der_corput(index, b) for b in bases]
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    # Normal(0,1) via Box-Muller
    def randn():
        u1 = max(1e-15, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---------------- core state ----------------
    best = float("inf")
    best_x = None

    # ensure we have a valid incumbent
    x0 = rand_point()
    f0 = eval_point(x0)
    best, best_x = f0, list(x0)

    # ---------------- Phase 1: global seeding (Halton + opposition + random) ----------------
    bases = first_primes(dim)

    # conservative but strong seeding; time-limited anyway
    init_n = max(120, 50 * dim)

    elite = []  # list of (f, x)
    def elite_add(fx, x, cap):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > cap:
            del elite[cap:]

    elite_cap = 16

    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best

        x = halton_point(k, bases)
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
        elite_add(fx, x, elite_cap)

        if time.time() >= deadline:
            return best

        # opposition
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        fo = eval_point(xo)
        if fo < best:
            best, best_x = fo, list(xo)
        elite_add(fo, xo, elite_cap)

    # extra random diversification (helps for deceptive functions)
    extra = max(40, 15 * dim)
    for _ in range(extra):
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
        # only store if competitive
        if not elite or fx < elite[-1][0] or len(elite) < elite_cap:
            elite_add(fx, x, elite_cap)

    # ---------------- Phase 2: CMA-ES-like (diagonal) + occasional DE mixing ----------------
    # We use a lightweight separable (diagonal covariance) evolution strategy with
    # rank-1 update-ish behavior and step-size adaptation (very cheap, robust).
    #
    # Representation: sample in normalized z-space, map to x = mean + sigma * diag_s * z
    # Bound handling: reflection to keep samples feasible.

    def reflect_to_bounds(x):
        # reflect repeatedly if far out (rare)
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect into [lo,hi]
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
            y[i] = v
        return y

    # initialize mean from best elite and dispersion from bounds
    mean = list(best_x)
    # per-dim scale (acts like diagonal covariance factor)
    diag_s = [0.3 * spans[i] for i in range(dim)]
    # global step-size multiplier
    sigma = 1.0

    # population size
    lam = max(10, min(30, 4 + int(3.0 * math.log(dim + 1.0)) * 4))
    mu = max(2, lam // 2)

    # recombination weights (log)
    weights = []
    for i in range(1, mu + 1):
        weights.append(math.log(mu + 0.5) - math.log(i))
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    # evolution paths (diagonal-ish)
    ps = [0.0] * dim

    # parameters
    cs = 0.3  # cumulation for sigma
    ds = 1.0 + 0.5 * math.log(dim + 1.0)  # damping
    # diag_s learning: slow to avoid instability
    cdiag = 0.2

    # expected length of N(0,I) in dim (approx)
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)) if dim > 1 else 0.7978845608

    # seed mean toward best few elites (faster start)
    if elite:
        top = elite[:min(4, len(elite))]
        # weighted by rank
        a = [0.55, 0.25, 0.13, 0.07]
        m = [0.0] * dim
        asum = 0.0
        for r, (fx, x) in enumerate(top):
            ar = a[r] if r < len(a) else 0.0
            asum += ar
            for j in range(dim):
                m[j] += ar * x[j]
        if asum > 0:
            mean = [m[j] / asum for j in range(dim)]

    # a tiny DE-mix helper to kick diversity when stuck
    def de_mix(xa, xb, xc, F):
        y = [0.0] * dim
        for j in range(dim):
            y[j] = xa[j] + F * (xb[j] - xc[j])
        return reflect_to_bounds(y)

    no_improve_gens = 0
    gen = 0

    while time.time() < deadline:
        gen += 1

        # sample population
        pop = []
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            z = [randn() for _ in range(dim)]
            x = [0.0] * dim
            for j in range(dim):
                x[j] = mean[j] + sigma * diag_s[j] * z[j]
            x = reflect_to_bounds(x)

            fx = eval_point(x)
            pop.append((fx, x, z))

            if fx < best:
                best, best_x = fx, list(x)

        pop.sort(key=lambda t: t[0])

        # update improvement tracker
        if pop[0][0] < best:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # recombine mean in x-space and also track z-mean
        new_mean = [0.0] * dim
        zmean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            fx, x, z = pop[i]
            for j in range(dim):
                new_mean[j] += w * x[j]
                zmean[j] += w * z[j]
        mean = new_mean

        # update sigma via path ps (separable CMA-ES style)
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + math.sqrt(cs * (2.0 - cs) * mu) * zmean[j]

        ps_norm = math.sqrt(sum(v * v for v in ps))
        sigma *= math.exp((cs / ds) * (ps_norm / (chi_n if chi_n > 1e-12 else 1.0) - 1.0))

        # keep sigma sane
        if sigma < 1e-12:
            sigma = 1e-12
        if sigma > 5.0:
            sigma = 5.0

        # adapt diagonal scales towards successful directions
        # use weighted spread of top solutions around mean
        # (encourages exploring dimensions that matter)
        spread = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            fx, x, _ = pop[i]
            for j in range(dim):
                d = (x[j] - mean[j]) / (spans[j] if spans[j] > 0 else 1.0)
                spread[j] += w * (d * d)

        for j in range(dim):
            # target scale proportional to sqrt(spread) but never collapse too much
            tgt = max(1e-12, math.sqrt(spread[j]) * spans[j] * 1.2)
            diag_s[j] = (1.0 - cdiag) * diag_s[j] + cdiag * tgt
            # cap within bounds scale
            cap = 0.5 * spans[j]
            if diag_s[j] > cap:
                diag_s[j] = cap
            if diag_s[j] < 1e-15 * (spans[j] if spans[j] > 0 else 1.0):
                diag_s[j] = 1e-15 * (spans[j] if spans[j] > 0 else 1.0)

        # occasionally inject best into mean (stabilize)
        if gen % 7 == 0 and best_x is not None:
            bblend = 0.25
            for j in range(dim):
                mean[j] = (1.0 - bblend) * mean[j] + bblend * best_x[j]

        # stagnation escape: DE mixing using elites from current pop
        if no_improve_gens >= 10 and len(pop) >= 6 and time.time() < deadline:
            no_improve_gens = 0
            # take some of the best and mix
            pool = [p[1] for p in pop[:min(8, len(pop))]]
            for _ in range(3):
                if time.time() >= deadline:
                    return best
                a, b, c = random.sample(pool, 3)
                y = de_mix(a, b, c, F=0.8 + 0.3 * random.random())
                fy = eval_point(y)
                if fy < best:
                    best, best_x = fy, list(y)
                    # pull mean toward it and slightly expand sigma
                    for j in range(dim):
                        mean[j] = 0.7 * mean[j] + 0.3 * y[j]
                    sigma *= 1.2
                    if sigma > 5.0:
                        sigma = 5.0

    return best
