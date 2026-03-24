import random, math, time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Why this should improve over the 3rd (portfolio) version you posted:
      - Uses a modern "ask/tell" DE variant: L-SHADE + current-to-pbest + archive
        (very strong across many bounded continuous problems).
      - Adds *subspace* mutations (operate on random coordinate blocks) which often
        helps in medium/high dimensions under tight time.
      - Adds a lightweight trust-region local search (few evals) on the global best.
      - Uses time-aware scheduling: starts explorative, becomes exploitative.
      - Has controlled restarts with sigma-like radius around best + random immigrants.

    Returns: best objective value found within max_time seconds.
    """

    # -------------------------- helpers --------------------------
    def f(x):
        return float(func(x))

    def clip(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def reflect(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return lo + t if t <= w else hi - (t - w)

    widths = [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def lhs(n):
        # LHS-like stratification per dimension
        strata = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            strata.append(p)
        pts = []
        for j in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    x[d] = lo
                else:
                    u = (strata[d][j] + random.random()) / n
                    x[d] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = lo if hi <= lo else reflect(lo + hi - x[i], lo, hi)
        return y

    def local_trust_refine(x0, f0, rad, iters=2):
        # Very small budget coordinate + random-direction trust region
        x = x0[:]
        fx = f0
        for _ in range(iters):
            # coordinate tries
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                lo, hi = bounds[j]
                if hi <= lo:
                    x[j] = lo
                    continue
                r = rad[j]
                if r <= 0.0:
                    continue
                xj = x[j]
                xp = x[:]; xp[j] = clip(xj + r, lo, hi)
                fp = f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    continue
                xm = x[:]; xm[j] = clip(xj - r, lo, hi)
                fm = f(xm)
                if fm < fx:
                    x, fx = xm, fm

            # one random direction try (scaled)
            if dim > 0:
                dvec = [random.gauss(0.0, 1.0) for _ in range(dim)]
                norm = math.sqrt(sum(v*v for v in dvec)) or 1.0
                y = x[:]
                for j in range(dim):
                    lo, hi = bounds[j]
                    if hi <= lo:
                        y[j] = lo
                    else:
                        step = (dvec[j] / norm) * rad[j]
                        y[j] = reflect(y[j] + step, lo, hi)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy

            # shrink radii
            for j in range(dim):
                rad[j] *= 0.6
        return x, fx

    # -------------------------- time --------------------------
    start = time.time()
    deadline = start + float(max_time)
    if max_time <= 0:
        # minimal safe behavior
        x = rand_vec()
        return f(x)

    # -------------------------- initialization --------------------------
    best = float("inf")
    best_x = None

    # Initial population (moderate)
    NP0 = max(24, min(120, 10 * dim if dim > 0 else 24))
    pts = lhs(max(12, NP0 // 2))
    pts += [opposite(p) for p in pts[:max(6, len(pts)//2)]]
    pts.append([bounds[i][0] if bounds[i][1] <= bounds[i][0] else 0.5*(bounds[i][0] + bounds[i][1]) for i in range(dim)])
    while len(pts) < NP0:
        pts.append(rand_vec())

    pop = []
    fit = []
    for x in pts[:NP0]:
        if time.time() >= deadline:
            return best
        fx = f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_vec()
        best = f(best_x)

    # -------------------------- L-SHADE / JADE state --------------------------
    # Population size reduction schedule
    NP_min = max(8, 4 + int(2.5 * math.log(dim + 2.0)))
    NP = len(pop)

    # Memory for parameter adaptation (SHADE)
    H = 12
    M_F = [0.5] * H
    M_CR = [0.5] * H
    k_mem = 0

    # Archive
    archive = []
    arch_max = NP

    # Stagnation / restart
    no_global = 0
    restart_after = max(80, 30 * (1 + dim // 5))

    gen = 0
    refine_every = 20

    # -------------------------- main loop --------------------------
    while time.time() < deadline:
        gen += 1

        # sort indices by fitness for pbest selection
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # exploration->exploitation schedule based on time
        tfrac = (time.time() - start) / max(1e-9, (deadline - start))
        # p-best fraction: start larger, end smaller (more exploit)
        pmax = 0.25
        pmin = 2.0 / max(2, NP)
        p = max(pmin, pmax * (1.0 - 0.65 * tfrac))
        pcount = max(2, int(math.ceil(p * NP)))

        SF, SCR, dF = [], [], []

        # steady-state style loop over individuals
        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # CR ~ N(mu, 0.1) clipped
            CRi = muCR + 0.1 * random.gauss(0.0, 1.0)
            CRi = 0.0 if CRi < 0.0 else (1.0 if CRi > 1.0 else CRi)

            # F ~ Cauchy(muF, 0.1) resample
            Fi = None
            for _ in range(12):
                u = random.random()
                cand = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < cand <= 1.0:
                    Fi = cand
                    break
            if Fi is None:
                Fi = clip(muF, 1e-3, 1.0)

            # choose pbest
            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # choose r2 from union(pop, archive) not i or r1 (best effort)
            unionN = NP + len(archive)
            if unionN <= 2:
                r2_is_arch = False
                r2 = random.randrange(NP)
            else:
                for _ in range(12):
                    pick = random.randrange(unionN)
                    r2_is_arch = pick >= NP
                    r2 = pick - NP if r2_is_arch else pick
                    if (not r2_is_arch and r2 != i and r2 != r1) or r2_is_arch:
                        break

            xr1 = pop[r1]
            xr2 = archive[r2] if r2_is_arch and len(archive) > 0 else pop[r2]

            # subspace mask: in higher dim, mutate only a block sometimes
            if dim <= 8:
                subspace = None
            else:
                if random.random() < 0.55:
                    k = max(2, int(0.25 * dim))
                    # pick k unique dims
                    subspace = set()
                    while len(subspace) < k:
                        subspace.add(random.randrange(dim))
                else:
                    subspace = None

            # mutation current-to-pbest/1
            v = xi[:]
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    v[d] = lo
                    continue
                if subspace is not None and d not in subspace:
                    # keep as is (no mutation on this coordinate)
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
                # archive defeated
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

        # update memories
        if SF:
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(SF)] * len(SF)
            else:
                weights = [df / wsum for df in dF]

            # Lehmer mean for F, arithmetic for CR
            num = sum(weights[j] * (SF[j] ** 2) for j in range(len(SF)))
            den = sum(weights[j] * (SF[j]) for j in range(len(SF)))
            MF_new = (num / den) if den > 0.0 else (sum(SF) / len(SF))
            MCR_new = sum(weights[j] * SCR[j] for j in range(len(SCR)))

            M_F[k_mem] = clip(MF_new, 1e-3, 1.0)
            M_CR[k_mem] = clip(MCR_new, 0.0, 1.0)
            k_mem = (k_mem + 1) % H

        # linear population size reduction (L-SHADE-ish)
        # reduce slowly as time passes
        if NP > NP_min and time.time() < deadline:
            target = int(round(NP0 - (NP0 - NP_min) * tfrac))
            target = max(NP_min, min(NP, target))
            if target < NP:
                # remove worst individuals
                idx_sorted = sorted(range(NP), key=lambda i: fit[i])
                keep = set(idx_sorted[:target])
                pop = [pop[i] for i in range(NP) if i in keep]
                fit = [fit[i] for i in range(NP) if i in keep]
                NP = len(pop)
                arch_max = NP
                if len(archive) > arch_max:
                    random.shuffle(archive)
                    archive = archive[:arch_max]

        # periodic local refinement of best (tiny budget)
        if (gen % refine_every) == 0 and best_x is not None and time.time() < deadline:
            rad = [max(w * (0.02 * (1.0 - 0.7 * tfrac)), w * 1e-12) for w in widths]
            rx, rf = local_trust_refine(best_x, best, rad, iters=2)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject by replacing worst
                worst = max(range(NP), key=lambda i: fit[i])
                pop[worst] = best_x[:]
                fit[worst] = best
                no_global = 0

        # restart on stagnation
        if no_global >= restart_after and time.time() < deadline and best_x is not None:
            no_global = 0
            archive.clear()

            # keep a small elite, re-seed others around best + random immigrants
            elite_k = max(2, NP // 6)
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            elite = set(idx_sorted[:elite_k])

            # radius around best (shrinks with time a bit, but not too much)
            g = 0.30 + 0.40 * (1.0 - tfrac) + 0.30 * random.random()
            base_sd = [max(w * 1e-9, w * 0.25 * g) for w in widths]

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
                        if random.random() < 0.22:
                            x[d] = random.uniform(lo, hi)
                        else:
                            x[d] = reflect(x[d] + random.gauss(0.0, base_sd[d]), lo, hi)
                fx = f(x)
                pop[i] = x
                fit[i] = fx
                if fx < best:
                    best, best_x = fx, x[:]

    return best
