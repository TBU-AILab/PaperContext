import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained, no external libs).

    Main improvements vs your best (SHADE-like + TR):
      1) Better use of time: measure evaluation time and adapt population / local-search intensity.
      2) Stronger DE core: L-SHADE-ish with linear population reduction + archive + pbest.
      3) More robust boundary handling: reflection ("bounce") to reduce boundary sticking.
      4) Two local optimizers:
           - (1+1)-ES w/ 1/5 success rule (cheap, robust)
           - deterministic coordinate/pattern polish (good for separable-ish problems)
      5) Stagnation response: heavy-tail “kicks” + partial refresh + temporary TR expansion.
      6) "Re-evaluate best occasionally" guard (optional) is omitted to avoid wasting budget.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------- RNG helpers ----------
    _bm_has = False
    _bm_val = 0.0

    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_val = z1
        _bm_has = True
        return z0

    def cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------- helpers ----------
    def bounce_repair(x):
        # reflect at bounds (period 2w folding)
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                xi = a + y
            if xi < a:
                xi = a
            elif xi > b:
                xi = b
            x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def eval_point(x):
        # robust evaluation + inf/NaN guard
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def opposition_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------- low discrepancy seeding (scrambled Halton) ----------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def scrambled_halton_points(n):
        bases = first_primes(max(1, dim))
        shifts = [random.random() for _ in range(dim)]  # CP rotation
        pts = []
        for k in range(1, n + 1):
            x = []
            for d in range(dim):
                u = (halton_value(k, bases[d]) + shifts[d]) % 1.0
                x.append(lo[d] + u * span_safe[d])
            pts.append(x)
        return pts

    # ---------- Local search: (1+1)-ES with 1/5 success ----------
    def es_local(best_x, best_f, sigma, steps):
        if best_x is None:
            return best_x, best_f, sigma

        x = best_x[:]
        fx = best_f
        succ = 0
        trials = 0

        for _ in range(steps):
            if time.time() >= deadline:
                break
            trials += 1
            y = x[:]
            for i in range(dim):
                y[i] += randn() * (sigma[i] * span_safe[i])
            bounce_repair(y)
            fy = eval_point(y)
            if fy < fx:
                x, fx = y, fy
                succ += 1

        if trials > 0:
            rate = succ / trials
            factor = 1.20 if rate > 0.2 else 0.82
            for i in range(dim):
                sigma[i] *= factor
                if sigma[i] < 1e-12:
                    sigma[i] = 1e-12
                elif sigma[i] > 0.6:
                    sigma[i] = 0.6

        return x, fx, sigma

    def coord_polish(best_x, best_f, step_frac, rounds=1):
        if best_x is None:
            return best_x, best_f
        x = best_x[:]
        fx = best_f
        step = [step_frac * span_safe[i] for i in range(dim)]
        for _ in range(rounds):
            if time.time() >= deadline:
                break
            idx = list(range(dim))
            random.shuffle(idx)
            improved = False
            for i in idx:
                if time.time() >= deadline:
                    break
                si = step[i]
                if si <= 0.0:
                    continue
                y1 = x[:]
                y1[i] += si
                bounce_repair(y1)
                f1 = eval_point(y1)
                if f1 < fx:
                    x, fx = y1, f1
                    improved = True
                    continue
                y2 = x[:]
                y2[i] -= si
                bounce_repair(y2)
                f2 = eval_point(y2)
                if f2 < fx:
                    x, fx = y2, f2
                    improved = True
            if not improved:
                for i in range(dim):
                    step[i] *= 0.5
        return x, fx

    # ---------- quick eval-time estimation ----------
    # Helps scale how many points we can afford.
    # If func is very slow, reduce population automatically.
    def estimate_eval_time():
        # do a few evaluations (but keep it tiny)
        k = 3
        times = []
        for _ in range(k):
            if time.time() >= deadline:
                break
            x = rand_point()
            t1 = time.time()
            _ = eval_point(x)
            t2 = time.time()
            times.append(max(1e-6, t2 - t1))
        if not times:
            return 1e-3
        times.sort()
        return times[len(times) // 2]

    eval_dt = estimate_eval_time()
    remaining = max(0.0, deadline - time.time())
    # rough eval budget
    eval_budget = max(30, int(0.80 * remaining / max(eval_dt, 1e-6)))

    # ---------- initialization ----------
    # L-SHADE reduction endpoints
    # scale with budget: if small budget, small NP; if large budget, bigger NP
    NP0 = int(18 + 4.5 * dim)
    NP0 = max(20, min(90, NP0))
    if eval_budget < 250:
        NP0 = max(12, min(NP0, 28))
    elif eval_budget < 600:
        NP0 = max(16, min(NP0, 45))

    NPmin = max(8, min(24, 6 + 2 * dim))

    n_seed = min(max(NP0, 3 * NP0), max(60, min(240, eval_budget // 3)))
    n_halton = max(2, int(0.70 * n_seed))
    n_rand = n_seed - n_halton

    seeds = scrambled_halton_points(n_halton)
    for _ in range(n_rand):
        seeds.append(rand_point())

    # add opposition + boundary biased
    seeds2 = []
    for x in seeds:
        seeds2.append(x)
        seeds2.append(opposition_point(x))

    boundary_k = max(6, min(40, 2 * dim + 8))
    for _ in range(boundary_k):
        x = []
        for d in range(dim):
            r = random.random()
            if r < 0.34:
                u = (random.random() ** 2) * 0.02
                x.append(lo[d] + u * span_safe[d])
            elif r < 0.68:
                u = (random.random() ** 2) * 0.02
                x.append(hi[d] - u * span_safe[d])
            else:
                x.append(lo[d] + random.random() * span_safe[d])
        seeds2.append(x)

    best = float("inf")
    best_x = None
    scored = []

    for x in seeds2:
        if time.time() >= deadline:
            return best
        bounce_repair(x)
        fx = eval_point(x)
        scored.append((fx, x[:]))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    scored = scored[:NP0]
    pop = [x for (fx, x) in scored]
    fit = [fx for (fx, x) in scored]

    # ---------- SHADE memory ----------
    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    archive = []
    archive_max = NP0

    pmin = 2.0 / max(2, NP0)
    pmax = 0.30

    # local params
    sigma = [0.10 for _ in range(dim)]
    polish_step = 0.04

    last_improve = time.time()
    # slightly more aggressive stall detection than before (often helps in time-bounded settings)
    stall_seconds = max(0.20, 0.16 * max_time)

    gen = 0
    while time.time() < deadline:
        gen += 1

        # ---- linear population reduction ----
        elapsed = time.time() - t0
        frac = min(1.0, max(0.0, elapsed / max(1e-9, max_time)))
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac))
        if target_NP < NPmin:
            target_NP = NPmin

        if len(pop) > target_NP:
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            archive_max = max(target_NP, 8)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        if NP < 4:
            return best

        # ---- periodic local improvements ----
        # adapt intensity to dimension + time left
        if best_x is not None and (gen <= 3 or gen % 8 == 0):
            # ES steps: small but a bit more in early time
            es_steps = 4 + (1 if frac < 0.35 else 0) + (1 if dim <= 6 else 0)
            bx, bf, sigma = es_local(best_x, best, sigma, steps=es_steps)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve = time.time()
            if gen % 16 == 0:
                bx, bf = coord_polish(best_x, best, step_frac=polish_step, rounds=1)
                if bf < best:
                    best, best_x = bf, bx[:]
                    last_improve = time.time()
                polish_step = max(1e-7, polish_step * 0.92)

        # ---- pbest ordering ----
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])

        p = pmin + (pmax - pmin) * random.random()
        pcount = max(2, int(math.ceil(p * NP)))

        S_CR, S_F, S_df = [], [], []

        # ---- evolve ----
        for i in range(NP):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)

            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            Fi = cauchy(MF[r], 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            pool_size = NP + len(archive)
            if pool_size <= 2:
                r2 = random.randrange(NP)
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_size)

            x_r1 = pop[r1]
            x_r2 = archive[r2 - NP] if r2 >= NP else pop[r2]

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # binomial crossover
            u = x_i[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
            bounce_repair(u)

            fu = eval_point(u)
            if fu <= fit[i]:
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                df = fit[i] - fu
                if df > 0.0:
                    S_CR.append(CRi)
                    S_F.append(Fi)
                    S_df.append(df)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = time.time()

        # ---- update memories ----
        if S_df:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                wsum = 1.0

            cr_new = 0.0
            for k in range(len(S_df)):
                cr_new += (S_df[k] / wsum) * S_CR[k]

            num = 0.0
            den = 0.0
            for k in range(len(S_df)):
                wk = S_df[k] / wsum
                fk = S_F[k]
                num += wk * fk * fk
                den += wk * fk
            f_new = (num / den) if den > 1e-18 else MF[mem_idx]

            MCR[mem_idx] = cr_new
            MF[mem_idx] = f_new
            mem_idx = (mem_idx + 1) % H

        # ---- stall handling / diversification ----
        if time.time() - last_improve > stall_seconds and time.time() < deadline:
            # heavy-tail kick around best for a portion of worst individuals
            order_desc = sorted(range(NP), key=lambda i: fit[i], reverse=True)
            m = max(2, int(0.40 * NP))

            for t in range(m):
                if time.time() >= deadline:
                    return best
                k = order_desc[t]
                if best_x is not None and random.random() < 0.80:
                    y = best_x[:]
                    # Cauchy-like big jumps sometimes
                    rad = 0.12 + 0.22 * abs(cauchy(0.0, 1.0))
                    if rad > 0.85:
                        rad = 0.85
                    for d in range(dim):
                        y[d] += (random.random() * 2.0 - 1.0) * rad * span_safe[d]
                    bounce_repair(y)
                else:
                    y = rand_point()
                fy = eval_point(y)
                pop[k] = y
                fit[k] = fy
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve = time.time()

            # ES kick (short)
            if best_x is not None:
                bx, bf, sigma = es_local(best_x, best, sigma, steps=8)
                if bf < best:
                    best, best_x = bf, bx[:]
                    last_improve = time.time()

            # temporarily increase exploration
            for i in range(dim):
                sigma[i] = min(0.30, sigma[i] * 1.25)
            polish_step = min(0.08, polish_step * 1.10)
            last_improve = time.time()

    return best
