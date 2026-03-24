import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained; no external libs).

    What is improved vs your best (9.2007...) DE/SHADE hybrid:
      1) Adds an explicit *CMA-ES style* local optimizer (diagonal sep-CMA) that is very strong
         for continuous box problems and complements DE well.
      2) Uses a time-scheduled hybrid:
            - Early/mid: L-SHADE-ish DE (current-to-pbest/1 + archive, pop-size reduction).
            - Late: sep-CMA intensification from the best-so-far (with mirrored/reflect repair).
      3) More robust restarts: when DE stalls, inject samples near best + opposition + random.
      4) More careful time budgeting: CMA steps are only attempted when near the end or stalling.

    Returns:
      best (float): best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------------- helpers ----------------------
    def isfinite(x):
        return (x == x) and (x != float("inf")) and (x != float("-inf"))

    def safe_eval(x):
        try:
            v = func(x)
            if isinstance(v, (int, float)):
                v = float(v)
                return v if isfinite(v) else float("inf")
            return float("inf")
        except Exception:
            return float("inf")

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly (handles big steps)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def ensure_reflect(x):
        y = x[:]
        for j in range(dim):
            lo, hi = bounds[j]
            y[j] = reflect(y[j], lo, hi)
        return y

    def rand_vec():
        return [random.uniform(bounds[j][0], bounds[j][1]) for j in range(dim)]

    def opposite(x):
        y = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            y[j] = lo + hi - x[j]
        return ensure_reflect(y)

    def gauss01():
        # approx N(0,1) using CLT
        return (sum(random.random() for _ in range(12)) - 6.0)

    def lhs_init(n):
        # cheap LHS-like init
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for j in range(dim):
                lo, hi = bounds[j]
                u = (perms[j][i] + random.random()) / float(n)
                x[j] = lo + u * (hi - lo)
            pop.append(x)
        return pop

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[j] - lo[j] for j in range(dim)]
    for j in range(dim):
        if span[j] <= 0.0:
            span[j] = 1.0
    avg_span = sum(span) / float(max(1, dim))

    # ---------------------- DE (L-SHADE-ish) setup ----------------------
    NP_init = max(24, min(90, 14 + 5 * dim))
    NP_min = max(8,  min(26, 6 + 2 * dim))
    NP = NP_init

    # SHADE memory
    H = 10
    MF = [0.6] * H
    MCR = [0.6] * H
    mem_ptr = 0

    # pbest schedule
    pbest_max = 0.25
    pbest_min = 0.08

    archive = []

    pop = lhs_init(NP // 2) + [rand_vec() for _ in range(NP - NP // 2)]
    pop = [ensure_reflect(x) for x in pop]
    fit = [float("inf")] * NP

    best = float("inf")
    best_x = None

    for i in range(NP):
        if time.time() >= deadline:
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best, best_x = fi, pop[i][:]

    # opposition on small elite
    ranked = sorted(range(NP), key=lambda i: fit[i])
    for i in ranked[:max(2, NP // 7)]:
        if time.time() >= deadline:
            return best
        xo = opposite(pop[i])
        fo = safe_eval(xo)
        if fo < fit[i]:
            pop[i], fit[i] = xo, fo
        if fo < best:
            best, best_x = fo, xo[:]

    # DE stagnation
    last_best = best
    stall_gens = 0
    last_improve_time = time.time()

    # ---------------------- sep-CMA (diagonal CMA-ES) state ----------------------
    # We run it as an intensifier around best_x when time is late or DE stalls.
    cma_m = best_x[:] if best_x is not None else rand_vec()
    cma_best_f = best
    cma_best_x = best_x[:] if best_x is not None else cma_m[:]

    # diagonal variances
    cma_sigma = 0.20 * avg_span / math.sqrt(max(1, dim))
    cma_sigma = max(1e-12, cma_sigma)
    cma_d = [1.0] * dim  # diagonal scaling ~ std multipliers

    # learning rates (sep-CMA typical heuristics)
    mu = 0
    c1 = 0.0
    cmu = 0.0
    cc = 0.0
    cs = 0.0
    damps = 0.0
    mueff = 0.0

    def cma_recompute_params(lam):
        nonlocal mu, c1, cmu, cc, cs, damps, mueff
        mu = max(2, lam // 2)
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        w2sum = sum(wi * wi for wi in w)
        mueff = 1.0 / max(1e-18, w2sum)

        # sep-CMA coefficients (simple, stable)
        n = float(dim)
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        return w

    # evolution paths
    ps = [0.0] * dim
    pc = [0.0] * dim

    # Expected norm of N(0,I) ~ sqrt(n)*(1 - 1/(4n) + 1/(21n^2))
    n = float(dim)
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)) if dim > 1 else 1.0

    def cma_ask(lam):
        # sample x_k = m + sigma * d * z  (elementwise)
        xs = []
        zs = []
        for _ in range(lam):
            z = [gauss01() for _ in range(dim)]
            x = [0.0] * dim
            for j in range(dim):
                x[j] = cma_m[j] + cma_sigma * cma_d[j] * z[j]
            x = ensure_reflect(x)
            xs.append(x)
            zs.append(z)
        return xs, zs

    def cma_tell(xs, zs, fs, weights):
        nonlocal cma_m, cma_sigma, cma_d, ps, pc

        # sort by fitness
        order = sorted(range(len(fs)), key=lambda i: fs[i])
        # recombination
        old_m = cma_m[:]
        y_w = [0.0] * dim  # weighted z in scaled coordinates
        for idx_w in range(len(weights)):
            k = order[idx_w]
            wk = weights[idx_w]
            zk = zs[k]
            for j in range(dim):
                y_w[j] += wk * zk[j]

        # update mean
        for j in range(dim):
            cma_m[j] = old_m[j] + cma_sigma * cma_d[j] * y_w[j]
        cma_m = ensure_reflect(cma_m)

        # update ps (conjugate evolution path in z-space)
        cs_loc = cs
        for j in range(dim):
            ps[j] = (1.0 - cs_loc) * ps[j] + math.sqrt(cs_loc * (2.0 - cs_loc) * mueff) * y_w[j]

        # update sigma (CSA)
        ps_norm = math.sqrt(sum(v * v for v in ps))
        cma_sigma *= math.exp((cs_loc / damps) * (ps_norm / max(1e-18, chi_n) - 1.0))
        # clamp sigma to sane range relative to box
        cma_sigma = max(1e-15, min(cma_sigma, 0.8 * avg_span))

        # update pc (in x-space normalized by d)
        cc_loc = cc
        for j in range(dim):
            pc[j] = (1.0 - cc_loc) * pc[j] + math.sqrt(cc_loc * (2.0 - cc_loc) * mueff) * (cma_d[j] * y_w[j])

        # update diagonal covariance (d^2) using sep-CMA rule:
        # d_j^2 <- (1-c1-cmu) d_j^2 + c1 pc_j^2 + cmu * sum_k w_k * (d_j*z_kj)^2
        for j in range(dim):
            dj2 = cma_d[j] * cma_d[j]
            rank_mu = 0.0
            for idx_w in range(len(weights)):
                k = order[idx_w]
                wk = weights[idx_w]
                zj = zs[k][j]
                rank_mu += wk * (cma_d[j] * zj) * (cma_d[j] * zj)
            dj2 = (1.0 - c1 - cmu) * dj2 + c1 * (pc[j] * pc[j]) + cmu * rank_mu
            if dj2 < 1e-24:
                dj2 = 1e-24
            if dj2 > 1e24:
                dj2 = 1e24
            cma_d[j] = math.sqrt(dj2)

    # ---------------------- main loop ----------------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # --- DE population size reduction ---
        target_NP = int(round(NP_init - (NP_init - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min
        if target_NP < NP:
            ranked = sorted(range(NP), key=lambda i: fit[i])
            keep = ranked[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            if len(archive) > NP:
                random.shuffle(archive)
                archive = archive[:NP]

        arch_max = NP

        # pbest schedule
        pbest_rate = pbest_max - (pbest_max - pbest_min) * (frac ** 1.15)
        pnum = max(2, int(math.ceil(pbest_rate * NP)))
        ranked = sorted(range(NP), key=lambda i: fit[i])
        pbest_set = ranked[:pnum]

        # DE success memories
        S_F, S_CR, S_w = [], [], []

        # mutation index picker
        def pick_excluding(excl):
            k = random.randrange(NP)
            while k in excl:
                k = random.randrange(NP)
            return k

        improved_gen = False

        # --- DE generation ---
        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mF = MF[r]
            mCR = MCR[r]

            CR = mCR + 0.10 * gauss01()
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # cauchy-like F
            F = -1.0
            for _ in range(6):
                g1 = gauss01()
                g2 = gauss01()
                if abs(g2) < 1e-12:
                    continue
                F = mF + 0.09 * (g1 / g2)
                if F > 0.0:
                    break
            if F <= 0.0: F = mF
            if F > 1.0: F = 1.0
            if F < 0.05: F = 0.05

            # mostly current-to-pbest/1; rare rand/1 early; rare best/1 late
            use_rand1 = (frac < 0.30 and random.random() < 0.18)
            use_best1 = (frac > 0.55 and random.random() < 0.10)

            use_archive = (archive and random.random() < 0.45)

            if use_rand1:
                r0 = pick_excluding({i})
                r1 = pick_excluding({i, r0})
                if use_archive:
                    xr2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding({i, r0, r1})
                    xr2 = pop[r2]
                v = [pop[r0][j] + F * (pop[r1][j] - xr2[j]) for j in range(dim)]
            elif use_best1 and best_x is not None:
                r1 = pick_excluding({i})
                if use_archive:
                    xr2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding({i, r1})
                    xr2 = pop[r2]
                v = [best_x[j] + F * (pop[r1][j] - xr2[j]) for j in range(dim)]
            else:
                pbest = pop[random.choice(pbest_set)]
                r1 = pick_excluding({i})
                if use_archive:
                    xr2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding({i, r1})
                    xr2 = pop[r2]
                v = [xi[j] + F * (pbest[j] - xi[j]) + F * (pop[r1][j] - xr2[j]) for j in range(dim)]

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]
            u = ensure_reflect(u)
            fu = safe_eval(u)

            if fu <= fi:
                # archive parent
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = u
                fit[i] = fu

                df = fi - fu
                w = df if isfinite(df) and df > 0.0 else 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

                improved_gen = True
                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_time = time.time()

        # update SHADE memories
        if S_F:
            wsum = sum(S_w)
            if wsum <= 1e-30:
                wsum = float(len(S_w))
                weights = [1.0 / wsum] * len(S_w)
            else:
                weights = [w / wsum for w in S_w]

            newCR = 0.0
            for k in range(len(S_CR)):
                newCR += weights[k] * S_CR[k]

            num = 0.0
            den = 0.0
            for k in range(len(S_F)):
                fk = S_F[k]
                wk = weights[k]
                num += wk * fk * fk
                den += wk * fk
            newF = (num / den) if den > 1e-18 else 0.6

            MCR[mem_ptr] = newCR
            MF[mem_ptr] = newF
            mem_ptr = (mem_ptr + 1) % H

        # stagnation
        if best < last_best - 1e-12:
            last_best = best
            stall_gens = 0
        else:
            stall_gens += 1

        # restart DE portion if stalling (keeps best)
        if stall_gens >= 14:
            stall_gens = 0
            worst = sorted(range(NP), key=lambda k: fit[k], reverse=True)
            krep = max(1, int(0.30 * NP))
            for idx in worst[:krep]:
                if time.time() >= deadline:
                    return best
                if best_x is not None and pop[idx] == best_x:
                    continue
                r = random.random()
                if best_x is not None and r < 0.65:
                    x = best_x[:]
                    rad = 0.25 + 0.30 * random.random()
                    for j in range(dim):
                        x[j] += random.uniform(-rad, rad) * span[j]
                    x = ensure_reflect(x)
                    if random.random() < 0.25:
                        x = opposite(x)
                elif best_x is not None and r < 0.80:
                    x = opposite(best_x)
                else:
                    x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()
            if archive and random.random() < 0.6:
                random.shuffle(archive)
                archive = archive[:max(0, len(archive)//2)]

        # ---------------------- CMA intensification (late or after stall) ----------------------
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        do_cma = (best_x is not None) and (frac > 0.55 or (now - last_improve_time) > 0.35)
        # Don't run CMA too frequently early; also ensure some time remains
        if do_cma and (deadline - now) > 0.02:
            # (re)anchor CMA mean to current best sometimes
            if best_x is not None and (cma_best_x is None or best < cma_best_f - 1e-12 or random.random() < 0.15):
                cma_m = best_x[:]
                cma_best_x = best_x[:]
                cma_best_f = best
                # reset paths mildly
                ps = [0.0] * dim
                pc = [0.0] * dim
                # slightly shrink sigma when late
                if frac > 0.75:
                    cma_sigma = max(1e-12, min(cma_sigma, 0.12 * avg_span / math.sqrt(max(1, dim))))

            # choose lambda by dimension but keep small
            lam = int(4 + 3 * math.log(max(2.0, float(dim))))
            lam = max(6, min(18, lam))
            weights = cma_recompute_params(lam)

            # do 1-2 CMA iterations depending on remaining time
            iters = 2 if (deadline - time.time()) > 0.10 else 1
            for _ in range(iters):
                if time.time() >= deadline:
                    break
                xs, zs = cma_ask(lam)
                fs = []
                for k in range(lam):
                    if time.time() >= deadline:
                        break
                    fk = safe_eval(xs[k])
                    fs.append(fk)
                    if fk < best:
                        best, best_x = fk, xs[k][:]
                        last_best = best
                        last_improve_time = time.time()
                if len(fs) < lam:
                    break

                cma_tell(xs, zs, fs, weights)

                # update CMA-best bookkeeping
                kbest = min(range(lam), key=lambda i: fs[i])
                if fs[kbest] < cma_best_f:
                    cma_best_f = fs[kbest]
                    cma_best_x = xs[kbest][:]

    return best
