import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (no external libs):

    Core changes vs your DE+SPSA version:
      - Uses SHADE-style adaptive DE (success-history for F/CR) for better stability.
      - Uses "current-to-pbest/1" mutation (better exploitation while staying global).
      - Uses an external archive (JADE/SHADE idea) to preserve diversity.
      - Uses periodic *trust-region* local search around the best (coordinate + 2-point
        directional probes) with shrinking radius.
      - Uses stagnation-triggered partial refresh.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- helpers ----------------
    def clamp_vec(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def eval_point(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def lhs_points(n):
        # fast LHS-like
        perms = []
        for _ in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        invn = 1.0 / max(1, n)
        for k in range(n):
            x = []
            for i in range(dim):
                a = (perms[i][k] + random.random()) * invn
                x.append(lo[i] + a * span_safe[i])
            pts.append(x)
        return pts

    def opposition_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # Normal generator (Box-Muller)
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
        # inverse CDF for Cauchy
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def directional_local_search(x, fx, rad_frac, probes=6):
        """
        Very cheap trust-region search around x:
          - coordinate +/- moves
          - a few random directions (2-point probes)
        """
        rad = [rad_frac * span_safe[i] for i in range(dim)]
        bestx = x[:]
        bestf = fx

        # coordinate probes (one sweep, randomized)
        idx = list(range(dim))
        random.shuffle(idx)
        for i in idx:
            if time.time() >= deadline:
                break
            si = rad[i]
            if si <= 0.0:
                continue
            for sgn in (-1.0, 1.0):
                y = bestx[:]
                y[i] += sgn * si
                clamp_vec(y)
                fy = eval_point(y)
                if fy < bestf:
                    bestf, bestx = fy, y

        # random direction probes
        for _ in range(probes):
            if time.time() >= deadline:
                break
            # random direction
            d = [randn() for _ in range(dim)]
            # normalize-ish (avoid heavy cost)
            norm = 0.0
            for i in range(dim):
                norm += d[i] * d[i]
            if norm <= 1e-18:
                continue
            inv = 1.0 / math.sqrt(norm)
            step = (0.5 + random.random())  # in [0.5,1.5)
            y1 = bestx[:]
            y2 = bestx[:]
            for i in range(dim):
                di = d[i] * inv
                si = rad[i] * step * di
                y1[i] += si
                y2[i] -= si
            clamp_vec(y1)
            clamp_vec(y2)
            f1 = eval_point(y1)
            if f1 < bestf:
                bestf, bestx = f1, y1
            if time.time() >= deadline:
                break
            f2 = eval_point(y2)
            if f2 < bestf:
                bestf, bestx = f2, y2

        return bestx, bestf

    # ---------------- initialization ----------------
    # Population sizing tuned for time-bounded runs
    pop_size = max(18, min(70, 12 + 5 * dim))
    n_seed = max(pop_size, min(4 * pop_size, 160))

    # Seed with LHS + opposition, keep best pop_size
    seeds = lhs_points(max(2, n_seed // 2))
    seeds2 = []
    for s in seeds:
        seeds2.append(s)
        seeds2.append(opposition_point(s))
    while len(seeds2) < n_seed:
        seeds2.append(rand_point())

    scored = []
    best = float("inf")
    best_x = None

    for x in seeds2:
        if time.time() >= deadline:
            return best
        fx = eval_point(x)
        scored.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    scored = scored[:pop_size]

    pop = [x[:] for (fx, x) in scored]
    fit = [fx for (fx, x) in scored]

    # ---------------- SHADE-like adaptation state ----------------
    # memory size
    H = 8
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    # archive for diversity (stores replaced parents)
    archive = []
    archive_max = pop_size

    # p-best fraction for current-to-pbest
    pmin = 2.0 / pop_size
    pmax = 0.25

    # stagnation handling
    last_improve = time.time()
    stall_seconds = max(0.25, 0.20 * max_time)

    # local search trust radius schedule
    tr = 0.12  # fraction of span
    tr_min = 1e-6
    tr_shrink = 0.7

    gen = 0
    while time.time() < deadline:
        gen += 1

        # periodic local improvement near best
        if best_x is not None and (gen <= 2 or gen % 10 == 0):
            bx, bf = directional_local_search(best_x, best, rad_frac=tr, probes=4)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve = time.time()
                tr = min(0.25, tr * 1.05)  # expand a touch if improving
            else:
                tr = max(tr_min, tr * tr_shrink)

        # sort indices by fitness to select pbest quickly
        order = list(range(pop_size))
        order.sort(key=lambda i: fit[i])

        p = pmin + (pmax - pmin) * random.random()
        pcount = max(2, int(math.ceil(p * pop_size)))

        # per-generation success stats for memory update
        S_CR = []
        S_F = []
        S_df = []

        # evolution
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose memory slot
            r = random.randrange(H)

            # sample CR ~ N(MCR[r], 0.1), clipped to [0,1]
            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # sample F from Cauchy(MF[r], 0.1) until in (0,1]
            Fi = cauchy(MF[r], 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 8:
                Fi = cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            # pick pbest
            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            # choose r1 from population, r2 from (population U archive)
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # pool for r2
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pool_size)

            x_r1 = pop[r1]
            x_r2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]

            # current-to-pbest/1 mutation:
            # v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # binomial crossover
            jrand = random.randrange(dim) if dim > 0 else 0
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
            clamp_vec(u)

            fu = eval_point(u)
            if fu <= fit[i]:
                # success: add parent to archive
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    # random removal to keep diversity
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                df = fit[i] - fu
                if df < 0.0:
                    df = 0.0
                if df > 0.0:
                    S_CR.append(CRi)
                    S_F.append(Fi)
                    S_df.append(df)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = time.time()

        # update memories (weighted Lehmer mean for F; weighted mean for CR)
        if S_df:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = 1.0
            # CR mean
            cr_new = 0.0
            for k in range(len(S_df)):
                cr_new += (S_df[k] / wsum) * S_CR[k]
            # F Lehmer mean: sum(w*F^2)/sum(w*F)
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

        # partial refresh if stalled
        if time.time() - last_improve > stall_seconds and time.time() < deadline:
            # replace worst fraction with jittered best and randoms
            order_desc = sorted(range(pop_size), key=lambda i: fit[i], reverse=True)
            m = max(2, int(0.30 * pop_size))
            for t in range(m):
                k = order_desc[t]
                if time.time() >= deadline:
                    return best
                if best_x is not None and random.random() < 0.75:
                    rad = 0.25 * (0.3 + 0.7 * random.random())
                    y = best_x[:]
                    for d in range(dim):
                        y[d] += (random.random() * 2.0 - 1.0) * rad * span_safe[d]
                    clamp_vec(y)
                else:
                    y = rand_point()
                fy = eval_point(y)
                pop[k] = y
                fit[k] = fy
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve = time.time()

            # reset timers/radius a bit
            last_improve = time.time()
            tr = min(0.18, max(0.06, tr))

    return best
