import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free optimizer (no external libs).

    Key upgrades vs prior best (Halton + (1+1)-ES):
      - Uses CMA-ES style adaptation (diagonal/separable CMA): much better on ill-scaled problems
      - Keeps a small "archive" and does occasional heavy-tail injections (escape local minima)
      - Starts from Halton coverage, then runs repeated short CMA epochs with restarts
      - Time-aware (checks deadline frequently) and exception-safe objective calls

    Returns:
      best (float): best objective value found
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ----- bounds / scaling -----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    def clip01(u):
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def u_to_x(u):
        return [lows[i] + u[i] * span_safe[i] for i in range(dim)]

    def safe_eval_u(u):
        try:
            return float(func(u_to_x(u)))
        except Exception:
            return float("inf")

    # ----- RNG helpers -----
    _bm_has = False
    _bm_next = 0.0
    def randn():
        nonlocal _bm_has, _bm_next
        if _bm_has:
            _bm_has = False
            return _bm_next
        u1 = random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(max(1e-300, u1)))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_next = z1
        _bm_has = True
        return z0

    def rand_u():
        return [random.random() for _ in range(dim)]

    # approximate heavy tail: z / |z2|
    def rand_heavy():
        return randn() / max(0.25, abs(randn()))

    # ----- Halton init (low discrepancy) -----
    def _first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def _halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = _first_primes(max(1, dim))
    def halton_point(k):  # k>=1
        return [_halton_value(k, primes[d]) for d in range(dim)]

    # ----- quick timing probe for eval cost (very conservative) -----
    if max_time <= 0:
        return float("inf")

    probe = rand_u()
    t_probe = time.time()
    _ = safe_eval_u(probe)
    dt1 = time.time() - t_probe
    t_probe = time.time()
    _ = safe_eval_u(rand_u())
    dt2 = time.time() - t_probe
    eval_dt = max(1e-6, 0.5 * (dt1 + dt2))
    # keep slack for Python overhead / variance in func time
    max_evals = int(max(30, 0.88 * (max(0.0, deadline - time.time()) / eval_dt)))

    evals = 0

    # ----- best tracking + small archive -----
    best_val = float("inf")
    best_u = rand_u()

    arch_cap = max(12, min(60, 4 * dim + 20))
    archive = []  # (val, u)

    def arch_add(v, u):
        nonlocal best_val, best_u, archive
        if v < best_val:
            best_val, best_u = v, u[:]
        archive.append((v, u[:]))
        if len(archive) > arch_cap:
            # keep best half + a few random survivors for diversity
            archive.sort(key=lambda t: t[0])
            keep = max(10, arch_cap // 2)
            survivors = archive[:keep]
            rest = archive[keep:]
            random.shuffle(rest)
            survivors.extend(rest[:max(0, arch_cap - len(survivors))])
            archive = survivors

    # ----- initialization: Halton + random -----
    init_n = max(16, min(120, 8 * dim + 40))
    init_n = min(init_n, max(16, max_evals // 4))

    k = 1
    while k <= init_n and time.time() < deadline and evals < max_evals:
        u = halton_point(k)
        v = safe_eval_u(u)
        evals += 1
        arch_add(v, u)
        k += 1

    extra = min(max(0, init_n // 3), max(0, max_evals // 12))
    for _ in range(extra):
        if time.time() >= deadline or evals >= max_evals:
            return best_val
        u = rand_u()
        v = safe_eval_u(u)
        evals += 1
        arch_add(v, u)

    # If everything failed somehow
    if not archive:
        v = safe_eval_u(best_u)
        evals += 1
        arch_add(v, best_u)

    # ----- separable CMA-ES epoch -----
    # This is a diagonal-CMA variant: covariance is diag(D^2).
    # Much cheaper than full CMA and typically far stronger than (1+1)-ES in moderate dims.
    def cma_epoch(start_u, start_sigma, epoch_evals_cap):
        nonlocal evals

        n = dim
        # strategy parameters (standard-ish defaults)
        lam = max(8, 4 + int(3 * math.log(n + 1.0)))  # population
        mu = lam // 2

        # log weights
        ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(ws)
        w = [wi / wsum for wi in ws]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # adaptation params
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        # expected length of N(0,I)
        chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # state
        m = start_u[:]
        sigma = start_sigma
        D = [1.0] * n  # diagonal std multipliers
        pc = [0.0] * n
        ps = [0.0] * n

        # scratch
        def norm(v):
            return math.sqrt(sum(x * x for x in v))

        # iterate
        used = 0
        while (time.time() < deadline and evals < max_evals and used < epoch_evals_cap):
            # sample population
            pop = []
            for _ in range(lam):
                if time.time() >= deadline or evals >= max_evals or used >= epoch_evals_cap:
                    break
                z = [randn() for _ in range(n)]
                y = [D[i] * z[i] for i in range(n)]
                x = [m[i] + sigma * y[i] for i in range(n)]
                clip01(x)
                fx = safe_eval_u(x)
                evals += 1
                used += 1
                pop.append((fx, x, z, y))
                arch_add(fx, x)

            if len(pop) < mu:
                return

            pop.sort(key=lambda t: t[0])
            # recombination
            old_m = m[:]
            m = [0.0] * n
            zmean = [0.0] * n
            ymean = [0.0] * n
            for i in range(mu):
                _, x, z, y = pop[i]
                wi = w[i]
                for j in range(n):
                    m[j] += wi * x[j]
                    zmean[j] += wi * z[j]
                    ymean[j] += wi * y[j]

            # update evolution path ps (approx invsqrt(C) * (m-old_m)/sigma using diagonal)
            invD_zmean = [zmean[j] for j in range(n)]  # since y = D*z, invsqrt(C) uses 1/D, cancels
            for j in range(n):
                ps[j] = (1.0 - cs) * ps[j] + math.sqrt(cs * (2.0 - cs) * mueff) * invD_zmean[j]

            # step-size control
            sigma *= math.exp((cs / damps) * (norm(ps) / chiN - 1.0))
            if sigma < 1e-8:
                sigma = 1e-8
            if sigma > 0.8:
                sigma = 0.8

            # hsig
            hsig = 1.0 if (norm(ps) / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * (used / float(lam) + 1.0))) < (1.4 + 2.0 / (n + 1.0))) else 0.0

            # update pc
            for j in range(n):
                pc[j] = (1.0 - cc) * pc[j] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (m[j] - old_m[j]) / max(1e-12, sigma)

            # covariance diagonal update
            # compute weighted sum of y_i^2 for selected individuals
            artmp = [0.0] * n
            for i in range(mu):
                _, _, _, y = pop[i]
                wi = w[i]
                for j in range(n):
                    artmp[j] += wi * (y[j] * y[j])

            for j in range(n):
                # D^2 update (diagonal of C)
                Dj2 = D[j] * D[j]
                Dj2 = (1.0 - c1 - cmu) * Dj2 + c1 * (pc[j] * pc[j]) + cmu * artmp[j]
                Dj2 = max(1e-18, Dj2)
                D[j] = math.sqrt(Dj2)

            # occasional tiny local coordinate poke around current best (cheap but helps on ridges)
            if random.random() < 0.12 and time.time() < deadline and evals < max_evals and used < epoch_evals_cap:
                u = best_u[:]
                step = max(0.002, 0.35 * sigma)
                j = random.randrange(n)
                up = u[:]; up[j] += step
                um = u[:]; um[j] -= step
                clip01(up); clip01(um)
                vp = safe_eval_u(up); evals += 1; used += 1; arch_add(vp, up)
                if time.time() >= deadline or evals >= max_evals or used >= epoch_evals_cap:
                    continue
                vm = safe_eval_u(um); evals += 1; used += 1; arch_add(vm, um)

    # ----- main restart loop -----
    # Choose starting points mostly from archive, sometimes global/heavy jumps.
    base_sigma = 0.30 / math.sqrt(max(1, dim))  # in [0,1] space
    restarts = 0

    while time.time() < deadline and evals < max_evals:
        restarts += 1

        # pick start
        if archive and random.random() < 0.82:
            archive.sort(key=lambda t: t[0])
            # bias to best but not always the best
            idx = int((random.random() ** 2.0) * min(len(archive), 30))
            start = archive[idx][1][:]
        else:
            start = rand_u()

        # jitter start
        jit = base_sigma * (1.8 if restarts <= 3 else 1.0)
        for i in range(dim):
            start[i] += (randn() * jit)
        clip01(start)

        # sometimes do a heavy-tail "teleport" in a subset of dims
        if random.random() < 0.25:
            for i in range(dim):
                if random.random() < 0.35:
                    start[i] += 0.25 * base_sigma * rand_heavy()
            clip01(start)

        # epoch budget: short-and-many tends to work well under strict time limits
        epoch_cap = max(40, min(400, 30 * dim + 120))
        # reduce cap if budget is small
        epoch_cap = min(epoch_cap, max(20, (max_evals - evals)))

        # sigma schedule: larger early, smaller later; also restart-dependent
        sigma0 = base_sigma * (2.2 / (1.0 + 0.25 * (restarts - 1)))
        sigma0 = max(0.004, min(0.45, sigma0))

        cma_epoch(start, sigma0, epoch_cap)

        # if lots of time, also inject a fresh Halton point periodically
        if time.time() < deadline and evals < max_evals and random.random() < 0.30:
            u = halton_point(1 + (restarts * 11) % 10007)
            v = safe_eval_u(u)
            evals += 1
            arch_add(v, u)

    return best_val
