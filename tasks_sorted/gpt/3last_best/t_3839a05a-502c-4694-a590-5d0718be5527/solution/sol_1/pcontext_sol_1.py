import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer:
      - low-discrepancy (Halton) initialization + opposition points
      - (1+1)-Evolution Strategy style local search with 1/5 success rule
      - occasional Cauchy / Gaussian heavy-tail perturbations
      - adaptive per-dimension step sizes, restarts with shrinking/increasing radius
      - simple evaluation cache to avoid duplicate calls

    Returns:
        best (float): best fitness found
    """
    t0 = time.time()
    deadline = t0 + max_time

    def time_left():
        return time.time() < deadline

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]

    # Handle degenerate spans
    for i in range(dim):
        if span[i] < 0:
            lo[i], hi[i] = hi[i], lo[i]
            span[i] = -span[i]

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    # ---- small LRU-ish cache (dict with pruning) ----
    cache = {}
    cache_max = 5000

    def key_of(x):
        # quantize to reduce near-duplicates; keeps cache effective without numpy
        # scale by span to be roughly relative; fall back for zero span.
        k = []
        for i in range(dim):
            s = span[i] if span[i] > 0 else 1.0
            k.append(int(round((x[i] - lo[i]) / s * 1e6)))
        return tuple(k)

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        if len(cache) > cache_max:
            # prune random ~20% to keep overhead low
            for kk in random.sample(list(cache.keys()), k=min(len(cache)//5, 500)):
                cache.pop(kk, None)
        return fx

    # ---- Halton sequence for diversified initial sampling (no external libs) ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
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

    primes = first_primes(max(1, dim))

    def van_der_corput(index, base):
        # index >= 1
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            i, rem = divmod(i, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(index):
        # index starts at 1
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(index, primes[i])
            x[i] = lo[i] + u * span[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # ---- initialization ----
    best = float("inf")
    best_x = [lo[i] + 0.5 * span[i] for i in range(dim)]  # mid-point seed

    # budget a chunk of time for initialization
    # (works even if max_time is tiny)
    init_target = max(16, min(200, 12 * dim))
    idx = 1

    # include midpoint
    if time_left():
        best = evaluate(best_x)

    while time_left() and idx <= init_target:
        x = halton_point(idx)
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition-based sample (often helps in bounded spaces)
        xo = [lo[i] + hi[i] - x[i] for i in range(dim)]
        clamp_inplace(xo)
        fxo = evaluate(xo)
        if fxo < best:
            best, best_x = fxo, xo[:]

        idx += 1

    # ---- local search: (1+1)-ES with adaptive step sizes ----
    # initial sigma: moderate fraction of range; keep a floor
    sigma = [0.25 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
    sigma_min = [1e-12 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
    sigma_max = [2.0 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]

    # success rule bookkeeping
    window = 18
    succ = 0
    trials = 0

    # restart control
    no_improve = 0
    restart_after = 60 + 8 * dim

    # heavy-tail mixing probabilities
    p_cauchy = 0.18
    p_global = 0.03

    def randn_approx():
        # ~N(0,1) via sum of uniforms
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0)

    def cauchy_approx():
        # Cauchy(0,1) via tan(pi*(u-0.5)); clamp extremes to avoid inf
        u = random.random()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # main loop
    while time_left():
        # occasional global try (keeps exploration alive)
        if random.random() < p_global:
            xg = rand_point()
            fg = evaluate(xg)
            if fg < best:
                best, best_x = fg, xg[:]
                no_improve = 0
                succ += 1
            else:
                no_improve += 1
            trials += 1
        else:
            # propose mutation around incumbent
            x = best_x[:]
            # choose noise type
            use_cauchy = (random.random() < p_cauchy)

            # mutate in random subset to reduce wasted evaluations in high dim
            # (but still allows all dims over time)
            # expected mutated dims ~ sqrt(dim) but at least 1
            m = max(1, int(math.sqrt(dim)))
            # random unique indices
            if m >= dim:
                idxs = range(dim)
            else:
                idxs = random.sample(range(dim), m)

            for i in idxs:
                if span[i] <= 0:
                    continue
                z = cauchy_approx() if use_cauchy else randn_approx()
                x[i] += z * sigma[i]

            clamp_inplace(x)
            fx = evaluate(x)

            trials += 1
            if fx < best:
                best, best_x = fx, x[:]
                succ += 1
                no_improve = 0
            else:
                no_improve += 1

        # adapt step sizes every 'window' trials using 1/5 success rule
        if trials >= window:
            rate = succ / float(trials)
            # if too successful, increase; else decrease
            if rate > 0.2:
                factor = 1.22
            else:
                factor = 0.82

            # mild anisotropic jitter to avoid lockstep sigmas
            for i in range(dim):
                s = sigma[i] * factor
                # small random drift
                s *= (0.98 + 0.04 * random.random())
                if s < sigma_min[i]:
                    s = sigma_min[i]
                elif s > sigma_max[i]:
                    s = sigma_max[i]
                sigma[i] = s

            succ = 0
            trials = 0

        # restart if stagnating: sample near-best at varying radii
        if no_improve >= restart_after and time_left():
            no_improve = 0

            # choose restart radius based on current sigmas and dimension
            # sometimes broaden, sometimes tighten
            if random.random() < 0.5:
                rad = 0.35
            else:
                rad = 0.12

            xr = best_x[:]
            for i in range(dim):
                if span[i] <= 0:
                    continue
                z = randn_approx()
                xr[i] += z * rad * span[i]
            clamp_inplace(xr)
            fr = evaluate(xr)
            if fr < best:
                best, best_x = fr, xr[:]

            # reset sigmas moderately (not too large to waste time)
            for i in range(dim):
                base = span[i] if span[i] > 0 else 1.0
                sigma[i] = max(sigma_min[i], min(0.22 * base, sigma_max[i]))

    return best
