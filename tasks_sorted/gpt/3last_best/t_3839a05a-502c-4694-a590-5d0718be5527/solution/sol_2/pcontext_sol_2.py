import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs prior version:
      - keeps a *small population* of elites (not just best-so-far) to reduce premature convergence
      - does *DE-style current-to-best/1* proposals + isotropic/anisotropic Gaussian noise
      - retains Halton + opposition initialization, plus quick "winner refinement" stage
      - uses *two-scale* step control (global sigma + per-dim sigma) with success adaptation
      - restarts draw from elites + occasional global samples
      - lightweight evaluation cache

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return time.time() < deadline

    # ---- bounds / helpers ----
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if span[i] < 0.0:
            lo[i], hi[i] = hi[i], lo[i]
            span[i] = -span[i]

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # ---- cache (quantized) ----
    cache = {}
    cache_max = 8000

    def key_of(x):
        k = []
        for i in range(dim):
            s = span[i] if span[i] > 0 else 1.0
            # quantize to 1e-7 of range
            k.append(int((x[i] - lo[i]) / s * 1e7 + 0.5))
        return tuple(k)

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        if len(cache) > cache_max:
            # prune ~15% randomly
            kill = min(len(cache) // 7, 1200)
            for kk in random.sample(list(cache.keys()), k=kill):
                cache.pop(kk, None)
        return fx

    # ---- low discrepancy (Halton) ----
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
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            i, rem = divmod(i, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(index, primes[i])
            x[i] = lo[i] + u * span[i]
        return x

    # ---- random variates (no external libs) ----
    def randn():
        # ~N(0,1) via 12 uniforms
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy():
        u = random.random()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---- Elite archive utilities ----
    # We store tuples (f, x_list)
    elite_k = max(5, min(18, 2 + dim // 2))  # small, robust
    elites = []

    def try_add_elite(x, fx):
        nonlocal elites
        # insert sorted by fitness, keep unique-ish
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        # remove near-duplicates beyond top few
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # ---- initialization ----
    best = float("inf")
    best_x = [lo[i] + 0.5 * span[i] for i in range(dim)]
    if time_left():
        fb = evaluate(best_x)
        best = fb
        try_add_elite(best_x, fb)

    init_n = max(24, min(260, 16 * dim))
    idx = 1
    while time_left() and idx <= init_n:
        x = halton_point(idx)
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]
        try_add_elite(x, fx)

        # opposition point
        xo = [lo[i] + hi[i] - x[i] for i in range(dim)]
        clamp_inplace(xo)
        fxo = evaluate(xo)
        if fxo < best:
            best, best_x = fxo, xo[:]
        try_add_elite(xo, fxo)

        # occasional pure random to de-correlate from Halton patterns
        if (idx & 7) == 0:
            xr = rand_point()
            fr = evaluate(xr)
            if fr < best:
                best, best_x = fr, xr[:]
            try_add_elite(xr, fr)

        idx += 1

    # quick refinement on current elites (cheap coordinate poke)
    # helps a lot on separable-ish landscapes, low overhead
    if time_left() and elites:
        poke = max(1, min(dim, 6))
        for (f0, x0) in elites[:min(4, len(elites))]:
            if not time_left():
                break
            x = x0[:]
            base_step = [0.06 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
            for _ in range(poke):
                if not time_left():
                    break
                i = random.randrange(dim)
                if span[i] <= 0:
                    continue
                for sgn in (+1.0, -1.0):
                    xt = x[:]
                    xt[i] += sgn * base_step[i]
                    clamp_inplace(xt)
                    ft = evaluate(xt)
                    if ft < best:
                        best, best_x = ft, xt[:]
                    if ft < f0:
                        x, f0 = xt, ft
            try_add_elite(x, f0)

    # ---- Main search (DE/current-to-best + ES noise) ----
    base = [(span[i] if span[i] > 0 else 1.0) for i in range(dim)]
    # per-dim sigma + global multiplier
    sigma_d = [0.18 * base[i] for i in range(dim)]
    sigma_min = [1e-14 * base[i] for i in range(dim)]
    sigma_max = [1.2 * base[i] for i in range(dim)]
    sigma_g = 1.0

    # parameters
    p_global = 0.02      # full random sample
    p_cauchy = 0.10      # heavy-tail in noise
    p_restart = 0.005    # rare archive-based restart
    CR = 0.6             # crossover rate
    F0 = 0.7             # DE scale baseline

    # success adaptation (1/5-ish)
    win = 24
    succ = 0
    trials = 0
    no_improve = 0
    stagnate_after = 80 + 10 * dim

    while time_left():
        # occasional restarts from elites / random
        if elites and (random.random() < p_restart or no_improve > stagnate_after):
            no_improve = 0
            if random.random() < 0.7 and elites:
                # sample around a random elite, radius tied to current sigmas
                _, xe = random.choice(elites)
                x = xe[:]
                rad = 0.35 if random.random() < 0.5 else 0.12
                for i in range(dim):
                    if span[i] > 0:
                        x[i] += randn() * rad * base[i]
                clamp_inplace(x)
            else:
                x = rand_point()
            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x[:]
                succ += 1
            try_add_elite(x, fx)
            trials += 1
            continue

        # global random sample (keeps exploration alive)
        if random.random() < p_global:
            x = rand_point()
            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x[:]
                succ += 1
                no_improve = 0
            else:
                no_improve += 1
            try_add_elite(x, fx)
            trials += 1
        else:
            # choose base vector from elites (or best)
            if elites and random.random() < 0.85:
                _, xb = random.choice(elites)
                x_base = xb
            else:
                x_base = best_x

            # pick two other distinct elite points for difference vector if possible
            if len(elites) >= 3:
                a = random.randrange(len(elites))
                b = random.randrange(len(elites))
                while b == a:
                    b = random.randrange(len(elites))
                _, xa = elites[a]
                _, xb2 = elites[b]
            else:
                xa = rand_point()
                xb2 = rand_point()

            # current-to-best/1 style
            # v = x_base + F*(best - x_base) + F*(xa - xb2)
            F = F0 * (0.85 + 0.3 * random.random())
            v = x_base[:]
            for i in range(dim):
                if span[i] <= 0:
                    continue
                v[i] += F * (best_x[i] - x_base[i]) + F * (xa[i] - xb2[i])

            # binomial crossover + added noise (ES-style)
            j_rand = random.randrange(dim) if dim > 0 else 0
            u = x_base[:]  # trial
            use_c = (random.random() < p_cauchy)
            m = max(1, int(math.sqrt(dim)))  # additional mutated dims for noise
            noise_idxs = None
            if m >= dim:
                noise_idxs = range(dim)
            else:
                noise_idxs = random.sample(range(dim), m)

            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    u[j] = v[j]

            # add noise on subset (keeps local search alive even if DE collapses)
            for j in noise_idxs:
                if span[j] <= 0:
                    continue
                z = cauchy() if use_c else randn()
                u[j] += z * (sigma_g * sigma_d[j])

            clamp_inplace(u)
            fu = evaluate(u)

            trials += 1
            if fu < best:
                best, best_x = fu, u[:]
                succ += 1
                no_improve = 0
            else:
                no_improve += 1

            try_add_elite(u, fu)

        # adapt sigmas
        if trials >= win:
            rate = succ / float(trials)
            # move towards ~0.2 success
            if rate > 0.22:
                sigma_g *= 1.18
            elif rate < 0.18:
                sigma_g *= 0.84
            sigma_g = max(0.05, min(5.0, sigma_g))

            # slight per-dim drift to avoid lockstep
            for i in range(dim):
                s = sigma_d[i] * (0.985 + 0.03 * random.random())
                # if we're stagnating, slowly increase some dims
                if no_improve > stagnate_after // 2 and random.random() < 0.15:
                    s *= 1.10
                # clamp
                if s < sigma_min[i]:
                    s = sigma_min[i]
                elif s > sigma_max[i]:
                    s = sigma_max[i]
                sigma_d[i] = s

            succ = 0
            trials = 0

    return best
