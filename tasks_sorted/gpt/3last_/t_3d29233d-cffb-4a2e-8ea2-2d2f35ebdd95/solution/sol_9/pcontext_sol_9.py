import random, math, time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improved vs prior CMA/DE attempts:
      - Uses a *hybrid* that is very strong under tight time:
          (A) L-SHADE (current-to-pbest + archive + parameter memories + pop reduction)
          (B) Occasional *small-batch* CMA-ES "burst" centered at current best
              (diagonal covariance only; cheap, but very effective for local structure)
          (C) Very cheap coordinate/pattern refinement on best
      - Adaptive scheduling by remaining time: more global early, more local late.
      - Robust bound handling via reflection.
      - Stagnation detection triggers diversifying immigrants.

    Returns:
        best (float): best objective value found within max_time seconds.
    """

    # -------------------------- helpers --------------------------
    def f(x):
        return float(func(x))

    def reflect(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if (t <= w) else (hi - (t - w))

    def clip01(a):
        return 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

    widths = [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]
    fixed = [bounds[i][1] <= bounds[i][0] for i in range(dim)]
    avg_w = sum(widths) / max(1, dim)

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def center_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else 0.5 * (lo + hi)
        return x

    def lhs(n):
        # simple LHS-like (no numpy)
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for j in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    x[d] = lo
                else:
                    u = (perms[d][j] + random.random()) / n
                    x[d] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = lo if hi <= lo else reflect(lo + hi - x[i], lo, hi)
        return y

    def cheap_refine(x0, f0, sweeps=1):
        # very cheap coordinate + one pattern move; step shrinks
        x = x0[:]
        fx = f0
        step = [max(1e-12, 0.02 * widths[i]) for i in range(dim)]
        for _ in range(sweeps):
            base = x[:]
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if fixed[j]:
                    x[j] = bounds[j][0]
                    continue
                lo, hi = bounds[j]
                s = step[j]
                if s <= 0.0 or hi <= lo:
                    continue

                xj = x[j]
                xp = x[:]
                xp[j] = reflect(xj + s, lo, hi)
                fp = f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[j] = reflect(xj - s, lo, hi)
                fm = f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if improved:
                y = x[:]
                for j in range(dim):
                    if fixed[j]:
                        y[j] = bounds[j][0]
                    else:
                        lo, hi = bounds[j]
                        y[j] = reflect(x[j] + (x[j] - base[j]), lo, hi)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
            else:
                for j in range(dim):
                    step[j] *= 0.5
        return x, fx

    # -------------------------- time --------------------------
    start = time.time()
    deadline = start + float(max_time)
    if max_time <= 0 or dim <= 0:
        return f(rand_vec())

    # -------------------------- init probes --------------------------
    best = float("inf")
    best_x = None

    probes = []
    probes.append(center_vec())
    probes += lhs(max(10, min(40, 6 * dim)))
    probes += [opposite(p) for p in probes[:max(5, min(20, len(probes)//2))]]
    for k in range(min(dim, 12)):
        lo, hi = bounds[k]
        if hi > lo:
            c = center_vec()
            c[k] = lo
            probes.append(c)
            c = center_vec()
            c[k] = hi
            probes.append(c)
    for _ in range(10):
        probes.append(rand_vec())

    # evaluate a limited number of probes quickly
    for x in probes:
        if time.time() >= deadline:
            return best
        fx = f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_vec()
        best = f(best_x)

    if avg_w <= 0.0:
        return best

    # -------------------------- L-SHADE state --------------------------
    NP0 = max(28, min(140, 10 * dim if dim > 0 else 28))
    NP_min = max(10, 4 + int(2.5 * math.log(dim + 2.0)))

    # build initial pop around probes (take best-ish + random fill)
    # keep diversity: mix best-centered jitter + random
    pop = []
    fit = []

    # seed with some good probes
    random.shuffle(probes)
    seeds = probes[:min(len(probes), NP0//2)]
    for x in seeds:
        if time.time() >= deadline:
            return best
        fx = f(x)
        pop.append(x[:])
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    while len(pop) < NP0:
        if time.time() >= deadline:
            return best
        if random.random() < 0.45:
            x = best_x[:]
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    x[d] = lo
                else:
                    sd = max(1e-12, 0.18 * widths[d])
                    x[d] = reflect(x[d] + random.gauss(0.0, sd), lo, hi)
        else:
            x = rand_vec()
        fx = f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    NP = len(pop)

    H = 12
    M_F = [0.5] * H
    M_CR = [0.5] * H
    k_mem = 0

    archive = []
    arch_max = NP

    no_global = 0
    refine_every = 18
    cma_every = 28
    restart_after = max(90, 28 * (1 + dim // 5))

    # -------------------------- CMA-ES diagonal "burst" --------------------------
    # keep a diagonal covariance estimate around best that we update from successful steps
    cma_diag = [1.0] * dim
    cma_sigma = max(1e-12, 0.15 * avg_w)

    def cma_burst(center, base_sigma, diag, lam):
        nonlocal best, best_x
        # sample lam points: x = center + sigma * diag * N(0,1)
        # return updated diag based on top points
        cand = []
        for _ in range(lam):
            if time.time() >= deadline:
                break
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            x = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                if hi <= lo:
                    x[i] = lo
                else:
                    x[i] = reflect(center[i] + base_sigma * diag[i] * z[i], lo, hi)
            fx = f(x)
            if fx < best:
                best, best_x = fx, x[:]
            cand.append((fx, z))
        if len(cand) < 4:
            return diag
        cand.sort(key=lambda t: t[0])
        mu = max(2, len(cand) // 3)
        # update diag towards std of best z's (robust-ish)
        newdiag = diag[:]
        for d in range(dim):
            s2 = 0.0
            for i in range(mu):
                s2 += cand[i][1][d] * cand[i][1][d]
            s2 /= mu
            # blend; keep >0
            target = math.sqrt(max(1e-12, s2))
            newdiag[d] = max(1e-6, 0.85 * diag[d] + 0.15 * target)
        return newdiag

    # -------------------------- main loop --------------------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, (deadline - start))

        # sort for pbest selection
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # pbest fraction schedule: larger early, smaller late
        pmax = 0.28
        pmin = 2.0 / max(2, NP)
        p = max(pmin, pmax * (1.0 - 0.70 * tfrac))
        pcount = max(2, int(math.ceil(p * NP)))

        SF, SCR, dF = [], [], []

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # CR ~ N(mu, 0.1)
            CRi = clip01(muCR + 0.1 * random.gauss(0.0, 1.0))

            # F ~ Cauchy(mu,0.1)
            Fi = None
            for _ in range(12):
                u = random.random()
                val = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < val <= 1.0:
                    Fi = val
                    break
            if Fi is None:
                Fi = max(1e-3, min(1.0, muF))

            # choose pbest
            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            # choose r2 from union(pop, archive)
            unionN = NP + len(archive)
            if unionN <= 2:
                r2_is_arch = False
                r2 = random.randrange(NP)
            else:
                pick = random.randrange(unionN)
                r2_is_arch = pick >= NP
                r2 = pick - NP if r2_is_arch else pick
            xr2 = archive[r2] if (r2_is_arch and len(archive) > 0) else pop[r2]

            # subspace mutation sometimes (helps high-dim speed)
            if dim >= 12 and random.random() < 0.55:
                k = max(2, int(0.22 * dim))
                sub = set()
                while len(sub) < k:
                    sub.add(random.randrange(dim))
            else:
                sub = None

            # mutation current-to-pbest/1
            v = xi[:]
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    v[d] = lo
                    continue
                if sub is not None and d not in sub:
                    v[d] = xi[d]
                    continue
                vraw = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                v[d] = reflect(vraw, lo, hi)

            # crossover
            u = xi[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    u[d] = lo
                else:
                    if random.random() < CRi or d == jrand:
                        u[d] = v[d]

            fu = f(u)

            if fu <= fi:
                # archive defeated parent
                if arch_max > 0:
                    if len(archive) < arch_max:
                        archive.append(xi[:])
                    else:
                        archive[random.randrange(arch_max)] = xi[:]

                pop[i] = u
                fit[i] = fu

                SF.append(Fi)
                SCR.append(CRi)
                dF.append(max(0.0, fi - fu))

                if fu < best:
                    best, best_x = fu, u[:]
                    no_global = 0
                else:
                    no_global += 1
            else:
                no_global += 1

        # update memories (SHADE)
        if SF:
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(SF)] * len(SF)
            else:
                weights = [df / wsum for df in dF]

            num = sum(weights[j] * (SF[j] ** 2) for j in range(len(SF)))
            den = sum(weights[j] * (SF[j]) for j in range(len(SF)))
            MF_new = (num / den) if den > 1e-18 else (sum(SF) / len(SF))
            MCR_new = sum(weights[j] * SCR[j] for j in range(len(SCR)))

            M_F[k_mem] = max(1e-3, min(1.0, MF_new))
            M_CR[k_mem] = max(0.0, min(1.0, MCR_new))
            k_mem = (k_mem + 1) % H

        # population size reduction
        if NP > NP_min and time.time() < deadline:
            target = int(round(NP0 - (NP0 - NP_min) * tfrac))
            target = max(NP_min, min(NP, target))
            if target < NP:
                idx_sorted = sorted(range(NP), key=lambda i: fit[i])
                keep = idx_sorted[:target]
                pop = [pop[i] for i in keep]
                fit = [fit[i] for i in keep]
                NP = len(pop)
                arch_max = NP
                if len(archive) > arch_max:
                    random.shuffle(archive)
                    archive = archive[:arch_max]

        # occasional cheap local refine (very few evals)
        if (gen % refine_every) == 0 and best_x is not None and time.time() < deadline:
            rx, rf = cheap_refine(best_x, best, sweeps=1)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject into worst
                worst = max(range(NP), key=lambda i: fit[i])
                pop[worst] = best_x[:]
                fit[worst] = best
                no_global = 0

        # occasional diagonal-CMA burst around best (especially late)
        if (gen % cma_every) == 0 and best_x is not None and time.time() < deadline:
            # sigma shrinks with time; lam small to keep time low
            base_sigma = max(1e-12, (0.20 * (1.0 - 0.75 * tfrac) + 0.03) * avg_w)
            lam = max(8, min(24, 4 + dim // 2))
            cma_diag = cma_burst(best_x, base_sigma, cma_diag, lam)

        # stagnation handling: immigrants around best + random
        if no_global >= restart_after and time.time() < deadline and best_x is not None:
            no_global = 0
            archive.clear()

            elite_k = max(2, NP // 6)
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            elite = set(idx_sorted[:elite_k])

            # radius schedule: moderate
            g = 0.25 + 0.50 * (1.0 - tfrac) + 0.25 * random.random()
            for i in range(NP):
                if i in elite:
                    continue
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                for d in range(dim):
                    lo, hi = bounds[d]
                    if hi <= lo:
                        x[d] = lo
                    else:
                        if random.random() < 0.20:
                            x[d] = random.uniform(lo, hi)
                        else:
                            sd = max(1e-12, widths[d] * 0.22 * g)
                            x[d] = reflect(x[d] + random.gauss(0.0, sd), lo, hi)
                fx = f(x)
                pop[i] = x
                fit[i] = fx
                if fx < best:
                    best, best_x = fx, x[:]

    return best
