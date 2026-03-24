import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Key upgrades vs the provided SHADE+TR version:
      1) Better early coverage: scrambled Halton seeding + opposition + a few boundary-biased points.
      2) Stronger DE core: L-SHADE-ish with:
           - current-to-pbest/1
           - external archive
           - success-history adaptation (CR/F)
           - *linear population size reduction* (more exploration early, faster exploitation late)
      3) More reliable local exploitation near the best:
           - lightweight (1+1)-ES with the 1/5 success rule (diagonal sigma)
           - occasional coordinate polishing with shrinking step
      4) Safer constraint handling: bounce-back repair (less boundary sticking than pure clamp).

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---- bounds / spans ----
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---- RNG helpers ----
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

    # ---- vector helpers ----
    def bounce_repair(x):
        # Reflect at bounds; then clamp to be safe.
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                # reflect in [a,b] by folding
                # translate to [0,w], fold with period 2w
                y = (xi - a) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                xi = a + y
            # final clamp
            if xi < a:
                xi = a
            elif xi > b:
                xi = b
            x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def opposition_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def eval_point(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # ---- low-discrepancy seeding: scrambled Halton ----
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
        # radical inverse
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def scrambled_halton_points(n):
        # Simple digit-scramble by random shift per dimension (Cranley-Patterson rotation)
        bases = first_primes(max(1, dim))
        shifts = [random.random() for _ in range(dim)]
        pts = []
        # start index at 1 to avoid all zeros
        for k in range(1, n + 1):
            x = []
            for d in range(dim):
                u = halton_value(k, bases[d])
                u = (u + shifts[d]) % 1.0
                x.append(lo[d] + u * span_safe[d])
            pts.append(x)
        return pts

    # ---- local search: (1+1)-ES with 1/5 rule (diagonal sigma) ----
    def es_local(best_x, best_f, sigma, steps, eval_budget_hint=999999):
        # sigma is per-dim fraction of span_safe (actual step = sigma[i]*span_safe[i])
        # 1/5 success rule: if success rate > 0.2 -> increase sigma else decrease.
        # Here applied in small batches to keep it cheap.
        if best_x is None:
            return best_x, best_f, sigma

        success = 0
        trials = 0
        x = best_x[:]
        fx = best_f

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
                success += 1

        if trials > 0:
            rate = success / trials
            # multiplicative update (gentle)
            if rate > 0.2:
                factor = 1.15
            else:
                factor = 0.85
            for i in range(dim):
                sigma[i] *= factor
                # keep within reasonable range
                if sigma[i] < 1e-12:
                    sigma[i] = 1e-12
                elif sigma[i] > 0.5:
                    sigma[i] = 0.5

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
                # try +/- along coordinate
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
            # shrink if no improvement
            if not improved:
                for i in range(dim):
                    step[i] *= 0.5
        return x, fx

    # ---- initialization ----
    # initial and minimum pop sizes for L-SHADE reduction
    NP0 = max(24, min(90, 14 + 6 * dim))
    NPmin = max(8, min(24, 6 + 2 * dim))

    # seeding budget
    n_seed = max(NP0, min(5 * NP0, 220))
    n_halton = max(2, int(0.70 * n_seed))
    n_rand = n_seed - n_halton

    seeds = scrambled_halton_points(n_halton)
    for _ in range(n_rand):
        seeds.append(rand_point())

    # add opposition and a few boundary-biased points
    seeds2 = []
    for x in seeds:
        seeds2.append(x)
        seeds2.append(opposition_point(x))

    # boundary-biased: put some coordinates near bounds to catch optima on edges
    boundary_k = max(6, min(40, 2 * dim + 8))
    for _ in range(boundary_k):
        x = []
        for d in range(dim):
            r = random.random()
            if r < 0.33:
                u = (random.random() ** 2) * 0.02
                x.append(lo[d] + u * span_safe[d])
            elif r < 0.66:
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

    # ---- SHADE memory ----
    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    archive = []
    archive_max = NP0

    # p-best fraction
    pmin = 2.0 / max(2, NP0)
    pmax = 0.30

    last_improve = time.time()
    stall_seconds = max(0.30, 0.22 * max_time)

    # ES local sigma (per-dim fraction of span)
    sigma = [0.08 for _ in range(dim)]
    # coord polish step fraction
    polish_step = 0.03

    gen = 0
    while time.time() < deadline:
        gen += 1

        # ---- periodic local improvements ----
        # Early + periodic: ES is very robust on many black-boxes.
        if best_x is not None and (gen <= 3 or gen % 9 == 0):
            # small ES batch
            bx, bf, sigma = es_local(best_x, best, sigma, steps=6)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve = time.time()
            # occasional coordinate polish
            if gen % 18 == 0:
                bx, bf = coord_polish(best_x, best, step_frac=polish_step, rounds=1)
                if bf < best:
                    best, best_x = bf, bx[:]
                    last_improve = time.time()
                polish_step = max(1e-6, polish_step * 0.9)

        # ---- L-SHADE population reduction ----
        elapsed = time.time() - t0
        frac = min(1.0, max(0.0, elapsed / max(1e-9, max_time)))
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac))
        if target_NP < NPmin:
            target_NP = NPmin

        if len(pop) > target_NP:
            # remove worst to match target size
            order = sorted(range(len(pop)), key=lambda i: fit[i])  # best to worst
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            # shrink archive cap accordingly
            archive_max = max(target_NP, 8)
            if len(archive) > archive_max:
                # random truncate
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        if NP < 4:
            return best

        # sort indices by fitness for pbest
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])

        # choose p for this generation
        p = pmin + (pmax - pmin) * random.random()
        pcount = max(2, int(math.ceil(p * NP)))

        S_CR, S_F, S_df = [], [], []

        # ---- evolve ----
        for i in range(NP):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)

            # CR
            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # F
            Fi = cauchy(MF[r], 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            # pbest
            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            # r1 in pop != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 in pop U archive != i,r1
            pool_size = NP + len(archive)
            if pool_size <= 2:
                r2 = random.randrange(NP)
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_size)

            x_r1 = pop[r1]
            x_r2 = archive[r2 - NP] if r2 >= NP else pop[r2]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # crossover
            u = x_i[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
            bounce_repair(u)

            fu = eval_point(u)
            if fu <= fit[i]:
                # add replaced parent to archive
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    # random removal
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
                    best = fu
                    best_x = u[:]
                    last_improve = time.time()

        # ---- update memories ----
        if S_df:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                wsum = 1.0

            # weighted mean for CR
            cr_new = 0.0
            for k in range(len(S_df)):
                cr_new += (S_df[k] / wsum) * S_CR[k]

            # weighted Lehmer mean for F
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

        # ---- stall handling ----
        if time.time() - last_improve > stall_seconds and time.time() < deadline:
            # diversify: replace some worst individuals + a short ES burst
            order_desc = sorted(range(NP), key=lambda i: fit[i], reverse=True)
            m = max(2, int(0.35 * NP))
            for t in range(m):
                if time.time() >= deadline:
                    return best
                k = order_desc[t]
                if best_x is not None and random.random() < 0.7:
                    # jitter around best
                    y = best_x[:]
                    rad = 0.20 * (0.3 + 0.7 * random.random())
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

            # ES kick
            if best_x is not None:
                bx, bf, sigma = es_local(best_x, best, sigma, steps=10)
                if bf < best:
                    best, best_x = bf, bx[:]
                    last_improve = time.time()

            # reset stall timer and slightly increase exploration
            last_improve = time.time()
            for i in range(dim):
                sigma[i] = min(0.25, sigma[i] * 1.2)

    return best
