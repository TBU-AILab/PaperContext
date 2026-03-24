import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Portfolio optimizer (no external libs) tuned to improve over the previous best.

    Key improvements vs your current best (#2, value ~31.38):
      - Uses *active-set bound handling* (reflect + clip) everywhere (more stable near bounds).
      - Keeps an elite + diverse archive, but restart logic is upgraded with:
          * "center-of-mass of top-k" starts (often better than picking a single elite)
          * DE/current-to-best + jitter
      - Local search upgraded:
          * short diagonal-CMA epochs (as before)
          * + a fast trust-region finite-difference "quasi-Newton diagonal" step with backtracking
            (more consistent exploitation than coordinate search on smooth problems)
          * + small coordinate polish only as a cheap finisher
      - Time-aware evaluation budgeting with frequent checks; exception-safe objective.

    Returns:
      best (float): best objective value found within time
    """
    if not max_time or max_time <= 0:
        return float("inf")

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- bounds / scaling ----------------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    def u_to_x(u):
        return [lows[i] + u[i] * span_safe[i] for i in range(dim)]

    def clip01_inplace(u):
        for i in range(dim):
            x = u[i]
            if x < 0.0:
                u[i] = 0.0
            elif x > 1.0:
                u[i] = 1.0
        return u

    def reflect01_inplace(u):
        # reflect at bounds; handles moderate overshoots nicely
        for i in range(dim):
            x = u[i]
            if x < 0.0 or x > 1.0:
                x = abs(x)
                if x > 2.0:
                    x = x - 2.0 * int(x / 2.0)
                if x > 1.0:
                    x = 2.0 - x
                u[i] = x
        return clip01_inplace(u)

    def safe_eval_u(u):
        try:
            return float(func(u_to_x(u)))
        except Exception:
            return float("inf")

    # ---------------- RNG helpers ----------------
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
        # bounded heavy tail (Cauchy-ish)
        return randn() / max(0.20, abs(randn()))

    # ---------------- scrambled Halton (global coverage) ----------------
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
    scr = [random.random() for _ in range(dim)]

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def shalton_point(k):
        u = [0.0] * dim
        for d in range(dim):
            u[d] = halton_value(k, primes[d]) + scr[d]
            u[d] -= int(u[d])
        return u

    # ---------------- timing probe -> eval budget ----------------
    # Two probes; keep slack for python overhead.
    if time.time() >= deadline:
        return float("inf")

    t = time.time()
    _ = safe_eval_u(rand_u())
    dt1 = time.time() - t
    t = time.time()
    _ = safe_eval_u(rand_u())
    dt2 = time.time() - t
    eval_dt = max(1e-6, 0.5 * (dt1 + dt2))
    time_left = max(0.0, deadline - time.time())
    max_evals = int(max(30, 0.90 * (time_left / eval_dt)))
    evals = 0

    # ---------------- archives ----------------
    best_val = float("inf")
    best_u = rand_u()

    elite_cap = max(12, min(48, 2 * dim + 24))
    div_cap = max(12, min(48, 2 * dim + 24))
    elite = []   # sorted by value
    diverse = [] # reservoir-ish

    def dist2(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    def add_point(v, u):
        nonlocal best_val, best_u, elite, diverse
        if v < best_val:
            best_val, best_u = v, u[:]

        elite.append((v, u[:]))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            elite = elite[:elite_cap]

        if not diverse:
            diverse.append((v, u[:]))
        else:
            thr = 0.03 * dim
            mind = min(dist2(u, p[1]) for p in diverse)
            # keep if far OR good relative to best
            if mind > thr or v <= elite[0][0] * 1.06:
                diverse.append((v, u[:]))
                if len(diverse) > div_cap:
                    diverse.sort(key=lambda t: t[0])
                    keep = max(8, int(0.65 * div_cap))
                    kept = diverse[:keep]
                    rest = diverse[keep:]
                    random.shuffle(rest)
                    kept.extend(rest[:div_cap - len(kept)])
                    diverse = kept

    def eval_and_add(u):
        nonlocal evals
        v = safe_eval_u(u)
        evals += 1
        add_point(v, u)
        return v

    # ---------------- initialization ----------------
    init_n = max(30, min(220, 12 * dim + 90))
    init_n = min(init_n, max(24, max_evals // 3))

    k = 1
    while k <= init_n and evals < max_evals and time.time() < deadline:
        u = shalton_point(k)
        eval_and_add(u)
        if (k % 3 == 0) and evals < max_evals and time.time() < deadline:
            eval_and_add([1.0 - ui for ui in u])  # opposition
        if (k % 9 == 0) and evals < max_evals and time.time() < deadline:
            eval_and_add(rand_u())
        k += 1

    if not elite:
        return best_val

    # ---------------- diagonal CMA-ES epoch (short) ----------------
    def cma_epoch(start_u, start_sigma, eval_cap):
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

        def norm(v):
            return math.sqrt(sum(x * x for x in v))

        m = start_u[:]
        sigma = start_sigma
        D = [1.0] * n
        pc = [0.0] * n
        ps = [0.0] * n

        used = 0
        while used < eval_cap and evals < max_evals and time.time() < deadline:
            pop = []
            for _ in range(lam):
                if used >= eval_cap or evals >= max_evals or time.time() >= deadline:
                    break
                z = [randn() for _ in range(n)]
                y = [D[i] * z[i] for i in range(n)]
                x = [m[i] + sigma * y[i] for i in range(n)]
                if random.random() < 0.10:
                    j = random.randrange(n)
                    x[j] = m[j] + sigma * D[j] * (2.0 * rand_heavy())
                reflect01_inplace(x)
                fx = safe_eval_u(x)
                evals += 1
                used += 1
                pop.append((fx, x, z, y))
                add_point(fx, x)

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

    # ---------------- TR finite-diff "quasi-Newton diagonal" step ----------------
    def tr_diag_newton(center_u, r0, eval_cap):
        nonlocal evals
        n = dim
        c = center_u[:]
        reflect01_inplace(c)

        fc = safe_eval_u(c)
        evals += 1
        add_point(fc, c)
        used = 1
        if used >= eval_cap or evals >= max_evals or time.time() >= deadline:
            return

        # radius in [0,1] space; keep modest
        r = max(1e-6, min(0.25, r0))

        # symmetric stencil -> gradient + diagonal curvature
        g = [0.0] * n
        h = [0.0] * n
        for i in range(n):
            if used + 2 > eval_cap or evals + 2 > max_evals or time.time() >= deadline:
                return
            up = c[:]
            um = c[:]
            up[i] += r
            um[i] -= r
            reflect01_inplace(up)
            reflect01_inplace(um)
            fp = safe_eval_u(up)
            fm = safe_eval_u(um)
            evals += 2
            used += 2
            add_point(fp, up)
            add_point(fm, um)

            g[i] = (fp - fm) / (2.0 * r)
            h[i] = (fp - 2.0 * fc + fm) / (r * r)

        # proposed diagonal-Newton-like step
        damp = 1e-6
        s = [0.0] * n
        for i in range(n):
            denom = abs(h[i]) + damp
            s[i] = -g[i] / denom

        # normalize to trust radius
        ns = math.sqrt(sum(si * si for si in s))
        if ns < 1e-18:
            return
        scale = min(1.0, r / ns)
        for i in range(n):
            s[i] *= scale

        # backtracking line-search on model step
        alphas = (1.0, 0.6, 0.35, 0.2, 0.1)
        for a in alphas:
            if used + 1 > eval_cap or evals + 1 > max_evals or time.time() >= deadline:
                return
            x = [c[i] + a * s[i] for i in range(n)]
            reflect01_inplace(x)
            fx = safe_eval_u(x)
            evals += 1
            used += 1
            add_point(fx, x)
            if fx <= fc:
                # optional second (smaller) TR step if we have budget
                if used + (2 * n + 4) <= eval_cap and evals + (2 * n + 4) <= max_evals and time.time() < deadline:
                    tr_diag_newton(x, max(1e-6, 0.55 * r), eval_cap - used)
                return

    # ---------------- coordinate micro-polish (cheap) ----------------
    def coord_polish(start_u, step0, min_step, eval_cap):
        nonlocal evals
        u = start_u[:]
        reflect01_inplace(u)
        fu = safe_eval_u(u)
        evals += 1
        add_point(fu, u)
        used = 1
        step = step0

        while used + 2 * dim <= eval_cap and evals < max_evals and time.time() < deadline:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if used + 2 > eval_cap or evals + 2 > max_evals or time.time() >= deadline:
                    return
                up = u[:]
                um = u[:]
                up[j] += step
                um[j] -= step
                reflect01_inplace(up)
                reflect01_inplace(um)
                vp = safe_eval_u(up)
                vm = safe_eval_u(um)
                evals += 2
                used += 2
                add_point(vp, up)
                add_point(vm, um)
                if vp <= fu or vm <= fu:
                    if vp <= vm and vp <= fu:
                        u, fu = up, vp
                        improved = True
                    elif vm <= fu:
                        u, fu = um, vm
                        improved = True
            if not improved:
                step *= 0.5
                if step < min_step:
                    return

    # ---------------- helper: top-k center start ----------------
    def topk_center(kmax):
        k = min(kmax, len(elite))
        if k <= 0:
            return rand_u()
        # rank-weighted average in u-space
        wsum = 0.0
        c = [0.0] * dim
        for i in range(k):
            wi = 1.0 / (1.0 + i)  # 1, 1/2, 1/3...
            u = elite[i][1]
            wsum += wi
            for j in range(dim):
                c[j] += wi * u[j]
        inv = 1.0 / wsum
        for j in range(dim):
            c[j] *= inv
        reflect01_inplace(c)
        return c

    # ---------------- main loop (portfolio restarts) ----------------
    base_sigma = 0.26 / math.sqrt(max(1, dim))
    restarts = 0
    last_best = best_val
    stale = 0

    while time.time() < deadline and evals < max_evals:
        restarts += 1
        if best_val < last_best - 1e-12:
            last_best = best_val
            stale = 0
        else:
            stale += 1

        r = random.random()
        if r < 0.30 and elite:
            start = topk_center(min(len(elite), 8))
        elif r < 0.58 and elite:
            idx = int((random.random() ** 2.2) * min(len(elite), 30))
            start = elite[idx][1][:]
        elif r < 0.75 and diverse:
            start = diverse[random.randrange(len(diverse))][1][:]
        elif r < 0.92 and elite and diverse:
            a = elite[int((random.random() ** 1.8) * min(len(elite), 30))][1]
            b = diverse[random.randrange(len(diverse))][1]
            start = best_u[:]
            F = 0.6 + 0.8 * random.random()
            for i in range(dim):
                if random.random() < 0.85:
                    start[i] = start[i] + F * (a[i] - b[i]) + 0.015 * randn()
            reflect01_inplace(start)
        else:
            start = shalton_point(1 + (restarts * 101) % 1000003)

        if random.random() < 0.18:
            start = [1.0 - ui for ui in start]

        sig0 = base_sigma * (2.6 if stale >= 4 else 1.6)
        sig0 *= (2.1 / (1.0 + 0.20 * (restarts - 1)))
        sig0 = max(0.003, min(0.60, sig0))

        for i in range(dim):
            start[i] += (0.22 * sig0 * rand_heavy()) if (random.random() < 0.18) else (0.30 * sig0 * randn())
        reflect01_inplace(start)

        remaining = max_evals - evals
        if remaining < 12:
            break

        # budget split: CMA to explore/shape, TR to exploit, coord to finish
        cma_cap = int(max(30, min(560, 26 * dim + 160, 0.62 * remaining)))
        cma_epoch(start, sig0, cma_cap)

        remaining = max_evals - evals
        if remaining < 10 or time.time() >= deadline:
            break

        tr_cap = int(max(12, min(520, 20 * dim + 180, 0.45 * remaining)))
        tr_r = max(0.0015, 0.85 * sig0)
        tr_diag_newton(best_u, tr_r, tr_cap)

        remaining = max_evals - evals
        if remaining < 8 or time.time() >= deadline:
            break

        coord_cap = int(max(10, min(240, 10 * dim + 80, 0.25 * remaining)))
        coord_polish(best_u, step0=max(0.0015, 0.25 * base_sigma), min_step=1e-7, eval_cap=coord_cap)

        # occasional re-evaluation (noise robustness)
        if evals < max_evals and time.time() < deadline and random.random() < 0.08:
            v = safe_eval_u(best_u)
            evals += 1
            add_point(v, best_u)

    return best_val
