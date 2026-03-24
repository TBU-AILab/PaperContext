import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free optimizer (no external libs).

    Main improvements vs the provided diagonal-CMA:
      1) Hybrid global->local: Sobol-ish (van der Corput per-dim with Owen-like scrambling)
         + archive-based restarts + tri-level local search.
      2) Strong local finisher: adaptive coordinate pattern search (very fast per-eval),
         periodically invoked around the current best to harvest easy gains.
      3) Better restart logic: mix of (a) best-archive starts, (b) diverse-archive starts,
         (c) fresh quasi-random starts; plus directed "difference" jumps between elites.
      4) Time-aware evaluation budgeting and frequent deadline checks.

    Notes:
      - Works in scaled space u in [0,1]^dim, then maps to x in given bounds.
      - Returns best (float): best objective value found.
    """
    if max_time is None or max_time <= 0:
        return float("inf")

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- bounds and scaling ----
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

    # ---- RNG: normal + heavy tail ----
    _bm_has = False
    _bm_next = 0.0

    def randn():
        nonlocal _bm_has, _bm_next
        if _bm_has:
            _bm_has = False
            return _bm_next
        u1 = max(1e-300, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_next = z1
        _bm_has = True
        return z0

    def rand_u():
        return [random.random() for _ in range(dim)]

    def rand_heavy():
        # bounded heavy-tail
        return randn() / max(0.20, abs(randn()))

    # ---- quasi-random sampler (fast, no big-int Sobol; still better than pure random) ----
    # We use per-dimension van der Corput with distinct odd bases + light scrambling.
    def first_primes(n):
        ps = []
        k = 2
        while len(ps) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in ps:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(k)
            k += 1
        return ps

    primes = first_primes(max(1, dim))

    def vdc(k, base):
        # van der Corput radical inverse
        inv = 1.0 / base
        x = 0.0
        f = inv
        while k:
            x += (k % base) * f
            k //= base
            f *= inv
        return x

    # Light "scramble": xor-like digit perturbation by adding a fixed random shift in [0,1)
    scr_shifts = [random.random() for _ in range(dim)]

    def qrand_point(idx):
        # idx starts from 1
        u = [0.0] * dim
        for d in range(dim):
            base = primes[d]
            u[d] = vdc(idx, base) + scr_shifts[d]
            u[d] -= int(u[d])  # mod 1
        return u

    # ---- time/budget probe ----
    # Do 2 probes for better estimate, but keep it cheap.
    u_probe = rand_u()
    t_probe = time.time()
    _ = safe_eval_u(u_probe)
    dt1 = time.time() - t_probe

    t_probe = time.time()
    _ = safe_eval_u(rand_u())
    dt2 = time.time() - t_probe

    eval_dt = max(1e-6, 0.5 * (dt1 + dt2))
    time_left = max(0.0, deadline - time.time())
    # Keep slack for overhead and occasional extra checks
    max_evals = int(max(25, 0.92 * (time_left / eval_dt)))
    evals = 0

    # ---- archive / best ----
    best_val = float("inf")
    best_u = rand_u()

    arch_cap = max(18, min(90, 6 * dim + 30))
    archive = []  # list of (val, u)

    def arch_add(v, u):
        nonlocal best_val, best_u, archive
        if v < best_val:
            best_val, best_u = v, u[:]
        archive.append((v, u[:]))
        if len(archive) > arch_cap:
            # keep top ~60% + a few diverse randoms
            archive.sort(key=lambda t: t[0])
            keep = max(12, int(0.6 * arch_cap))
            kept = archive[:keep]
            rest = archive[keep:]
            random.shuffle(rest)
            kept.extend(rest[:max(0, arch_cap - len(kept))])
            archive = kept

    def eval_and_add(u):
        nonlocal evals
        v = safe_eval_u(u)
        evals += 1
        arch_add(v, u)
        return v

    # ---- initialization: quasi-random + some random ----
    if time.time() >= deadline:
        return best_val

    init_n = max(20, min(160, 10 * dim + 60))
    init_n = min(init_n, max(20, max_evals // 3))

    idx = 1
    while idx <= init_n and time.time() < deadline and evals < max_evals:
        u = qrand_point(idx)
        eval_and_add(u)
        idx += 1

    # extra random insurance
    extra = min(max(0, init_n // 4), max(0, max_evals // 12))
    for _ in range(extra):
        if time.time() >= deadline or evals >= max_evals:
            return best_val
        eval_and_add(rand_u())

    if not archive:
        return best_val

    # ---- local search 1: small-step coordinate pattern search (fast finisher) ----
    def coord_search(start_u, step0, min_step, max_passes, eval_cap):
        nonlocal evals
        u = start_u[:]
        clip01(u)
        fu = safe_eval_u(u)
        evals += 1
        arch_add(fu, u)

        step = step0
        used = 1
        passes = 0
        while passes < max_passes and used < eval_cap and time.time() < deadline and evals < max_evals:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if used + 2 > eval_cap or time.time() >= deadline or evals >= max_evals:
                    return
                up = u[:]
                um = u[:]
                up[j] += step
                um[j] -= step
                clip01(up)
                clip01(um)
                vp = safe_eval_u(up); evals += 1; used += 1
                vm = safe_eval_u(um); evals += 1; used += 1
                arch_add(vp, up)
                arch_add(vm, um)
                if vp < fu or vm < fu:
                    if vp <= vm:
                        u, fu = up, vp
                    else:
                        u, fu = um, vm
                    improved = True

            if not improved:
                step *= 0.5
                if step < min_step:
                    return
            passes += 1

    # ---- local search 2: compact diagonal-CMA epoch (short, restart-friendly) ----
    def diag_cma_epoch(start_u, start_sigma, eval_cap):
        nonlocal evals

        n = dim
        lam = max(10, 4 + int(4 * math.log(n + 1.0)))
        mu = lam // 2

        ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(ws)
        w = [wi / wsum for wi in ws]
        mueff = 1.0 / sum(wi * wi for wi in w)

        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        m = start_u[:]
        sigma = start_sigma
        D = [1.0] * n
        pc = [0.0] * n
        ps = [0.0] * n

        def norm(v):
            return math.sqrt(sum(x * x for x in v))

        used = 0
        while used < eval_cap and time.time() < deadline and evals < max_evals:
            pop = []
            k = 0
            while k < lam and used < eval_cap and time.time() < deadline and evals < max_evals:
                z = [randn() for _ in range(n)]
                y = [D[i] * z[i] for i in range(n)]
                x = [m[i] + sigma * y[i] for i in range(n)]
                clip01(x)
                fx = safe_eval_u(x)
                evals += 1
                used += 1
                pop.append((fx, x, z, y))
                arch_add(fx, x)
                k += 1

            if len(pop) < mu:
                return

            pop.sort(key=lambda t: t[0])
            old_m = m[:]
            m = [0.0] * n
            zmean = [0.0] * n

            for i in range(mu):
                _, x, z, _y = pop[i]
                wi = w[i]
                for j in range(n):
                    m[j] += wi * x[j]
                    zmean[j] += wi * z[j]

            for j in range(n):
                ps[j] = (1.0 - cs) * ps[j] + math.sqrt(cs * (2.0 - cs) * mueff) * zmean[j]

            sigma *= math.exp((cs / damps) * (norm(ps) / max(1e-12, chiN) - 1.0))
            sigma = max(1e-8, min(0.9, sigma))

            hsig = 1.0 if norm(ps) < (1.4 + 2.0 / (n + 1.0)) * chiN else 0.0

            for j in range(n):
                pc[j] = (1.0 - cc) * pc[j] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (m[j] - old_m[j]) / max(1e-12, sigma)

            # diag covariance update via selected steps
            artmp = [0.0] * n
            for i in range(mu):
                _fx, _x, _z, y = pop[i]
                wi = w[i]
                for j in range(n):
                    artmp[j] += wi * (y[j] * y[j])

            for j in range(n):
                Dj2 = D[j] * D[j]
                Dj2 = (1.0 - c1 - cmu) * Dj2 + c1 * (pc[j] * pc[j]) + cmu * artmp[j]
                D[j] = math.sqrt(max(1e-18, Dj2))

    # ---- main loop: restart portfolio ----
    base_sigma = 0.32 / math.sqrt(max(1, dim))
    restart = 0

    while time.time() < deadline and evals < max_evals:
        restart += 1
        archive.sort(key=lambda t: t[0])

        # choose a start:
        r = random.random()
        if r < 0.65 and archive:
            # biased-to-best pick
            top = min(len(archive), 35)
            idx = int((random.random() ** 2.2) * top)
            start = archive[idx][1][:]
        elif r < 0.85 and len(archive) >= 2:
            # "difference" jump between two elites (DE-style)
            top = min(len(archive), 40)
            a = archive[int((random.random() ** 1.8) * top)][1]
            b = archive[int((random.random() ** 1.8) * top)][1]
            start = best_u[:]
            F = 0.7 + 0.6 * random.random()
            for i in range(dim):
                if random.random() < 0.8:
                    start[i] = start[i] + F * (a[i] - b[i]) + 0.05 * base_sigma * randn()
            clip01(start)
        else:
            # fresh quasi-random
            start = qrand_point(1 + (restart * 97) % 1000003)

        # jitter/teleport
        sig0 = base_sigma * (2.4 / (1.0 + 0.22 * (restart - 1)))
        sig0 = max(0.004, min(0.55, sig0))

        if random.random() < 0.30:
            for i in range(dim):
                if random.random() < 0.35:
                    start[i] += 0.20 * sig0 * rand_heavy()
                else:
                    start[i] += 0.35 * sig0 * randn()
            clip01(start)
        else:
            for i in range(dim):
                start[i] += 0.25 * sig0 * randn()
            clip01(start)

        # allocate evals per restart (short epochs)
        remaining = max_evals - evals
        if remaining <= 5:
            break

        # CMA budget then coordinate-search budget (finisher)
        cma_cap = int(max(30, min(360, 22 * dim + 120, 0.65 * remaining)))
        diag_cma_epoch(start, sig0, cma_cap)

        # fast finisher around current best
        remaining = max_evals - evals
        if remaining <= 6 or time.time() >= deadline:
            break
        step0 = max(0.0025, 0.30 * base_sigma)
        min_step = 1e-6
        coord_cap = int(max(10, min(220, 10 * dim + 60, 0.35 * remaining)))
        coord_search(best_u, step0=step0, min_step=min_step, max_passes=3, eval_cap=coord_cap)

        # occasional single extra global probe
        if time.time() < deadline and evals < max_evals and random.random() < 0.20:
            eval_and_add(qrand_point(1 + (restart * 193) % 1000003))

    return best_val
