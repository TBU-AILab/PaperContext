import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Upgrades vs your best (~24.265):
      1) Keeps your strong global engine (scrambled Halton + archives + diag-CMA restarts),
         but adds a *very evaluation-efficient* local optimizer:
           - SPSA (2 evaluations) to estimate gradient in full dim
           - a few quasi-Newton (L-BFGS-lite) steps with backtracking line search
         This can dramatically improve exploitation when the objective is even mildly smooth.
      2) More time-aware scheduling:
           - early: coverage + CMA shaping
           - late: heavier focus on SPSA/L-BFGS refinement near best
      3) Slightly improved restart seeding using elite center + differential mixing.

    Returns:
      best (float): best objective value found within max_time.
    """
    if max_time is None or max_time <= 0:
        return float("inf")

    t0 = time.time()
    deadline = t0 + float(max_time)

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
        # Reflect into [0,1] with modulo-2 folding, then clip for safety.
        for i in range(dim):
            x = u[i]
            if x < 0.0 or x > 1.0:
                x = abs(x)
                if x > 2.0:
                    x -= 2.0 * int(x / 2.0)
                if x > 1.0:
                    x = 2.0 - x
                u[i] = x
        return clip01_inplace(u)

    def safe_eval_u(u):
        try:
            return float(func(u_to_x(u)))
        except Exception:
            return float("inf")

    # ---------- RNG helpers ----------
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

    # ---------- scrambled Halton ----------
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
            u[d] -= int(u[d])  # mod 1
        return u

    # ---------- eval budget estimate ----------
    if time.time() >= deadline:
        return float("inf")

    # probe a couple evaluations to estimate rate
    t = time.time()
    _ = safe_eval_u(rand_u())
    dt1 = time.time() - t
    t = time.time()
    _ = safe_eval_u(rand_u())
    dt2 = time.time() - t
    eval_dt = max(1e-6, 0.5 * (dt1 + dt2))

    time_left = max(0.0, deadline - time.time())
    max_evals = int(max(30, 0.92 * (time_left / eval_dt)))
    evals = 0

    # ---------- archives ----------
    best_val = float("inf")
    best_u = rand_u()

    elite_cap = max(12, min(56, 2 * dim + 28))
    div_cap = max(12, min(56, 2 * dim + 28))
    elite = []    # sorted by value
    diverse = []  # diversity reservoir-ish

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

    # ---------- initialization ----------
    init_n = max(30, min(260, 12 * dim + 110))
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

    # ---------- helper: top-k center ----------
    def topk_center(kmax):
        k = min(kmax, len(elite))
        if k <= 0:
            return rand_u()
        c = [0.0] * dim
        wsum = 0.0
        for i in range(k):
            wi = 1.0 / (1.0 + i)
            u = elite[i][1]
            wsum += wi
            for j in range(dim):
                c[j] += wi * u[j]
        inv = 1.0 / wsum
        for j in range(dim):
            c[j] *= inv
        reflect01_inplace(c)
        return c

    # ---------- diagonal CMA epoch ----------
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
            sigma = max(1e-8, min(0.85, sigma))

            hsig = 1.0 if norm(ps) < (1.4 + 2.0 / (n + 1.0)) * chiN else 0.0
            invsig = 1.0 / max(1e-12, sigma)
            for j in range(n):
                pc[j] = (1.0 - cc) * pc[j] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (m[j] - old_m[j]) * invsig

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

    # ---------- SPSA + L-BFGS-lite refinement ----------
    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def norm2(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def lbfgs_refine(start_u, radius, eval_cap):
        """
        Very small-memory quasi-Newton with SPSA gradients.
        Uses only O(m*dim) operations; objective calls dominate anyway.
        """
        nonlocal evals
        n = dim
        m_hist = 6  # memory

        u = start_u[:]
        reflect01_inplace(u)
        fu = safe_eval_u(u)
        evals += 1
        add_point(fu, u)
        used = 1

        if used >= eval_cap or evals >= max_evals or time.time() >= deadline:
            return

        # trust radius in u-space
        tr = max(1e-6, min(0.25, radius))

        s_list = []
        y_list = []
        rho_list = []

        # SPSA step sizes (mildly decreasing)
        a0 = 0.12 * tr
        c0 = 0.08 * tr

        it = 0
        while used + 4 <= eval_cap and evals + 4 <= max_evals and time.time() < deadline:
            it += 1
            # SPSA gradient: 2 evals
            ck = max(1e-6, c0 / (it ** 0.101))
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(n)]
            up = [u[i] + ck * delta[i] for i in range(n)]
            um = [u[i] - ck * delta[i] for i in range(n)]
            reflect01_inplace(up)
            reflect01_inplace(um)

            fp = safe_eval_u(up)
            fm = safe_eval_u(um)
            evals += 2
            used += 2
            add_point(fp, up)
            add_point(fm, um)

            # gradient estimate
            g = [(fp - fm) / (2.0 * ck * delta[i]) for i in range(n)]

            # two-loop recursion to get search direction
            q = g[:]
            alpha = [0.0] * len(s_list)
            for k in range(len(s_list) - 1, -1, -1):
                alpha[k] = rho_list[k] * dot(s_list[k], q)
                yk = y_list[k]
                for i in range(n):
                    q[i] -= alpha[k] * yk[i]

            # initial H0 scaling
            if y_list:
                ys = dot(y_list[-1], s_list[-1])
                yy = dot(y_list[-1], y_list[-1])
                gamma = ys / max(1e-18, yy)
                gamma = max(1e-6, min(1e2, gamma))
            else:
                gamma = 1.0

            r = [gamma * qi for qi in q]

            for k in range(len(s_list)):
                beta = rho_list[k] * dot(y_list[k], r)
                sk = s_list[k]
                for i in range(n):
                    r[i] += sk[i] * (alpha[k] - beta)

            p = [-ri for ri in r]
            np = norm2(p)
            if np < 1e-18:
                return

            # clip to trust radius
            scale = min(1.0, tr / np)
            p = [pi * scale for pi in p]

            # backtracking line search (few tries)
            # also include a "small" step candidate (often helps on noisy/rough surfaces)
            ak = max(1e-4, a0 / (it ** 0.2))
            candidates = [1.0, 0.5, 0.25, 0.12]
            accepted = False
            for cfac in candidates:
                if used + 1 > eval_cap or evals + 1 > max_evals or time.time() >= deadline:
                    return
                step = cfac * ak
                x = [u[i] + step * p[i] for i in range(n)]
                reflect01_inplace(x)
                fx = safe_eval_u(x)
                evals += 1
                used += 1
                add_point(fx, x)
                if fx <= fu:
                    u_new = x
                    fu_new = fx
                    accepted = True
                    break

            if not accepted:
                # shrink trust radius a bit and continue; stop if too small
                tr *= 0.7
                if tr < 1e-5:
                    return
                continue

            # update history (s,y) using new gradient (SPSA again if budget allows, else finite diff not possible)
            if used + 2 > eval_cap or evals + 2 > max_evals or time.time() >= deadline:
                u, fu = u_new, fu_new
                add_point(fu, u)
                return

            # reuse same delta for cheap-ish gradient update around new point
            up2 = [u_new[i] + ck * delta[i] for i in range(n)]
            um2 = [u_new[i] - ck * delta[i] for i in range(n)]
            reflect01_inplace(up2)
            reflect01_inplace(um2)
            fp2 = safe_eval_u(up2)
            fm2 = safe_eval_u(um2)
            evals += 2
            used += 2
            add_point(fp2, up2)
            add_point(fm2, um2)
            g2 = [(fp2 - fm2) / (2.0 * ck * delta[i]) for i in range(n)]

            s = [u_new[i] - u[i] for i in range(n)]
            y = [g2[i] - g[i] for i in range(n)]
            ys = dot(y, s)
            if ys > 1e-12:
                rho = 1.0 / ys
                s_list.append(s)
                y_list.append(y)
                rho_list.append(rho)
                if len(s_list) > m_hist:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)

            u, fu = u_new, fu_new

            # occasionally expand trust region if improving
            if it % 4 == 0 and fu <= best_val + 1e-12:
                tr = min(0.25, tr * 1.15)

    # ---------- main loop ----------
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
            start = topk_center(min(10, len(elite)))
        elif r < 0.58 and elite:
            idx = int((random.random() ** 2.2) * min(len(elite), 30))
            start = elite[idx][1][:]
        elif r < 0.76 and diverse:
            start = diverse[random.randrange(len(diverse))][1][:]
        elif r < 0.94 and elite and diverse:
            a = elite[int((random.random() ** 1.8) * min(len(elite), 30))][1]
            b = diverse[random.randrange(len(diverse))][1]
            start = best_u[:]
            F = 0.55 + 0.85 * random.random()
            for i in range(dim):
                if random.random() < 0.85:
                    start[i] = start[i] + F * (a[i] - b[i]) + 0.012 * randn()
            reflect01_inplace(start)
        else:
            start = shalton_point(1 + (restarts * 101) % 1000003)

        if random.random() < 0.18:
            start = [1.0 - ui for ui in start]

        sig0 = base_sigma * (2.8 if stale >= 4 else 1.6)
        sig0 *= (2.05 / (1.0 + 0.20 * (restarts - 1)))
        sig0 = max(0.003, min(0.60, sig0))

        for i in range(dim):
            start[i] += (0.22 * sig0 * rand_heavy()) if (random.random() < 0.18) else (0.30 * sig0 * randn())
        reflect01_inplace(start)

        remaining = max_evals - evals
        if remaining < 14:
            break

        # Phase A: diagonal-CMA (shape / exploration)
        # give it less budget late in the run
        frac_left = max(0.0, min(1.0, (deadline - time.time()) / max(1e-9, (deadline - t0))))
        cma_weight = 0.62 if frac_left > 0.35 else 0.42
        cma_cap = int(max(30, min(560, 26 * dim + 160, cma_weight * remaining)))
        cma_epoch(start, sig0, cma_cap)

        remaining = max_evals - evals
        if remaining < 10 or time.time() >= deadline:
            break

        # Phase B: SPSA + L-BFGS-lite exploitation near best
        lb_cap = int(max(10, min(520, 18 * dim + 220, 0.70 * remaining)))
        lb_r = max(0.0010, 0.85 * sig0)
        lbfgs_refine(best_u, lb_r, lb_cap)

        # occasional re-evaluation (noise robustness)
        if evals < max_evals and time.time() < deadline and random.random() < 0.06:
            v = safe_eval_u(best_u)
            evals += 1
            add_point(v, best_u)

    return best_val
