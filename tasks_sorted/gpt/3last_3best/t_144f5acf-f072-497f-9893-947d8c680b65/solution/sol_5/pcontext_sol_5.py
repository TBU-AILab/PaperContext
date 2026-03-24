import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free optimizer (no external libs).

    What changed vs your best (diag-CMA + Halton restarts):
      1) Better global coverage: scrambled Halton + opposition points + mix-in random.
      2) More informative archive: keeps a small elite set and a diverse set (not only best).
      3) Stronger restarts: (a) elite-biased, (b) DE/current-to-best style, (c) opposite,
         (d) fresh Halton; restart sigma depends on progress.
      4) Two-phase local refinement: short separable CMA epochs + opportunistic coordinate search.
      5) Noise/plateau robustness: reevaluate-best occasionally (very cheap) and allow "equal" moves.
      6) Strict time-awareness: frequent deadline checks, conservative eval budget.

    Returns:
      best (float): best objective value found within time.
    """
    if max_time is None or max_time <= 0:
        return float("inf")

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- bounds / scaling ----------------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    def clip01(u):
        for i in range(dim):
            x = u[i]
            if x < 0.0:
                u[i] = 0.0
            elif x > 1.0:
                u[i] = 1.0
        return u

    def u_to_x(u):
        return [lows[i] + u[i] * span_safe[i] for i in range(dim)]

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
        # bounded heavy tail: Cauchy-ish
        return randn() / max(0.20, abs(randn()))

    # ---------------- scrambled Halton ----------------
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
    scr = [random.random() for _ in range(dim)]  # additive scramble (mod 1)

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def shalton_point(k):  # k>=1
        u = [0.0] * dim
        for d in range(dim):
            u[d] = halton_value(k, primes[d]) + scr[d]
            u[d] -= int(u[d])  # mod 1
        return u

    # ---------------- timing probe -> eval budget ----------------
    # Use 2 probes; keep slack.
    probe = rand_u()
    t = time.time()
    _ = safe_eval_u(probe)
    dt1 = time.time() - t
    t = time.time()
    _ = safe_eval_u(rand_u())
    dt2 = time.time() - t
    eval_dt = max(1e-6, 0.5 * (dt1 + dt2))
    time_left = max(0.0, deadline - time.time())
    max_evals = int(max(25, 0.90 * (time_left / eval_dt)))
    evals = 0

    # ---------------- archive (elite + diverse) ----------------
    best_val = float("inf")
    best_u = rand_u()

    elite_cap = max(10, min(40, 2 * dim + 20))
    div_cap = max(10, min(40, 2 * dim + 20))
    elite = []   # sorted by val: [(val,u),...]
    diverse = [] # reservoir-ish: [(val,u),...]

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

        # elite insert
        elite.append((v, u[:]))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            elite = elite[:elite_cap]

        # diverse: keep points that are far from existing diverse set OR very good
        if not diverse:
            diverse.append((v, u[:]))
        else:
            # distance threshold adapts with dim
            # (in [0,1]^dim, typical squared distance ~ dim/6)
            thr = 0.03 * dim
            mind = min(dist2(u, p[1]) for p in diverse)
            if mind > thr or v <= elite[0][0] * 1.05:
                diverse.append((v, u[:]))
                if len(diverse) > div_cap:
                    # drop one: prefer dropping worst and redundant (close to others)
                    # cheap heuristic: sort by val, keep a mix
                    diverse.sort(key=lambda t: t[0])
                    keep = max(6, int(0.65 * div_cap))
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
    if time.time() >= deadline:
        return best_val

    init_n = max(24, min(180, 10 * dim + 80))
    init_n = min(init_n, max(20, max_evals // 3))

    k = 1
    while k <= init_n and evals < max_evals and time.time() < deadline:
        u = shalton_point(k)
        eval_and_add(u)

        # opposition point (often helps on bounded problems)
        if evals < max_evals and time.time() < deadline and (k % 3 == 0):
            uo = [1.0 - ui for ui in u]
            eval_and_add(uo)

        # occasional pure random
        if evals < max_evals and time.time() < deadline and (k % 7 == 0):
            eval_and_add(rand_u())
        k += 1

    if not elite:
        return best_val

    # ---------------- local search: coordinate pattern ----------------
    def coord_search(start_u, step0, min_step, eval_cap):
        nonlocal evals
        u = start_u[:]
        clip01(u)
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
                clip01(up)
                clip01(um)
                vp = safe_eval_u(up)
                vm = safe_eval_u(um)
                evals += 2
                used += 2
                add_point(vp, up)
                add_point(vm, um)

                # allow equal moves (plateaus)
                if vp <= fu or vm <= fu:
                    if vp <= vm:
                        if vp <= fu:
                            u, fu = up, vp
                            improved = True
                    else:
                        if vm <= fu:
                            u, fu = um, vm
                            improved = True

            if not improved:
                step *= 0.5
                if step < min_step:
                    return

    # ---------------- local search: separable (diagonal) CMA epoch ----------------
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
            # generate population
            for _ in range(lam):
                if used >= eval_cap or evals >= max_evals or time.time() >= deadline:
                    break
                z = [randn() for _ in range(n)]
                y = [D[i] * z[i] for i in range(n)]
                x = [m[i] + sigma * y[i] for i in range(n)]
                # occasional heavy-tail injection on some restarts
                if random.random() < 0.10:
                    j = random.randrange(n)
                    x[j] = m[j] + sigma * D[j] * (2.0 * rand_heavy())
                clip01(x)
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

            # quick "polish" poke sometimes
            if used + 2 <= eval_cap and evals + 2 <= max_evals and time.time() < deadline and random.random() < 0.08:
                u = best_u[:]
                step = max(0.0015, 0.25 * sigma)
                j = random.randrange(n)
                up = u[:]; up[j] += step
                um = u[:]; um[j] -= step
                clip01(up); clip01(um)
                add_point(safe_eval_u(up), up); evals += 1; used += 1
                if used >= eval_cap or evals >= max_evals or time.time() >= deadline:
                    continue
                add_point(safe_eval_u(um), um); evals += 1; used += 1

    # ---------------- main loop: restart portfolio ----------------
    base_sigma = 0.28 / math.sqrt(max(1, dim))
    restarts = 0
    last_best = best_val
    stale = 0

    while time.time() < deadline and evals < max_evals:
        restarts += 1

        # progress tracking to adapt exploration
        if best_val < last_best - 1e-12:
            last_best = best_val
            stale = 0
        else:
            stale += 1

        # choose start type
        r = random.random()
        if r < 0.55 and elite:
            # elite-biased (not always top-1)
            idx = int((random.random() ** 2.2) * min(len(elite), 25))
            start = elite[idx][1][:]
        elif r < 0.75 and diverse:
            # diverse point
            idx = random.randrange(len(diverse))
            start = diverse[idx][1][:]
        elif r < 0.90 and elite and diverse:
            # current-to-best / DE-like
            a = elite[int((random.random() ** 1.8) * min(len(elite), 30))][1]
            b = diverse[random.randrange(len(diverse))][1]
            start = best_u[:]
            F = 0.6 + 0.7 * random.random()
            for i in range(dim):
                if random.random() < 0.85:
                    start[i] = start[i] + F * (a[i] - b[i]) + 0.02 * randn()
            clip01(start)
        else:
            # fresh global
            start = shalton_point(1 + (restarts * 101) % 1000003)

        # opposition restart sometimes
        if random.random() < 0.18:
            start = [1.0 - ui for ui in start]

        # restart sigma: bigger if stale
        sig0 = base_sigma * (2.2 if stale >= 4 else 1.4)
        sig0 *= (2.0 / (1.0 + 0.22 * (restarts - 1)))
        sig0 = max(0.0035, min(0.55, sig0))

        # jitter the start
        for i in range(dim):
            if random.random() < 0.20:
                start[i] += 0.25 * sig0 * rand_heavy()
            else:
                start[i] += 0.35 * sig0 * randn()
        clip01(start)

        remaining = max_evals - evals
        if remaining < 8:
            break

        # allocate: CMA then coordinate polish
        cma_cap = int(max(30, min(520, 26 * dim + 140, 0.70 * remaining)))
        cma_epoch(start, sig0, cma_cap)

        remaining = max_evals - evals
        if remaining < 6 or time.time() >= deadline:
            break

        # coordinate polish around best
        coord_cap = int(max(12, min(260, 12 * dim + 70, 0.30 * remaining)))
        step0 = max(0.0020, 0.30 * base_sigma)
        coord_search(best_u, step0=step0, min_step=1e-7, eval_cap=coord_cap)

        # occasional re-evaluation of best to fight noisy objectives (cheap: 1 eval)
        if evals < max_evals and time.time() < deadline and random.random() < 0.10:
            v = safe_eval_u(best_u)
            evals += 1
            add_point(v, best_u)

    return best_val
