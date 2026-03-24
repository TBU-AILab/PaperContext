import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-budgeted box-constrained minimizer (self-contained; stdlib only).

    Key improvements vs your #1 (L-SHADE/SHADE DE):
      - Uses *jSO / iL-SHADE style* mutation with a better balance of exploration/exploitation:
          current-to-pbest/1 with (Fw) weighting and a *dynamic p* and *F/CR memories*.
      - Uses *eigen/coordinate-free* help without matrices: periodic *pairwise-difference*
        directions + a very cheap *pattern search* around best (robust late-stage squeeze).
      - Fixes inefficiency/bug risk in your r1 selection (avoids pop.index / identity issues).
      - Better time-aware scheduling + stronger restart/injection policy.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]
    avg_span = (sum(span_safe) / dim) if dim > 0 else 1.0

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    def clamp_reflect_inplace(x):
        # reflect into [lo,hi] then clamp
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                x[i] = (a + y) if (y <= w) else (b - (y - w))
            if x[i] < a:
                x[i] = a
            elif x[i] > b:
                x[i] = b
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    # Box-Muller normal
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(random.random(), 1e-300)
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare = z1
        _has_spare = True
        return z0

    def rand_cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def l2norm(v):
        s = 0.0
        for i in range(dim):
            s += v[i] * v[i]
        return math.sqrt(s)

    # ---------------- best bookkeeping ----------------
    best = float("inf")
    best_x = None

    def try_update(x):
        nonlocal best, best_x
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    if now() >= deadline:
        return best

    # ---------------- seeding ----------------
    # center
    x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    clamp_reflect_inplace(x0)
    try_update(x0)

    # small corner set
    corner_bits = min(dim, 6)
    max_corners = min(16, 1 << corner_bits)
    for mask in range(max_corners):
        if now() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if i < corner_bits:
                x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
            else:
                x[i] = 0.5 * (lo[i] + hi[i])
        try_update(x)

    # random seeds
    for _ in range(12 + 6 * dim):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------------- local pattern search (late squeeze) ----------------
    def pattern_search_best(time_limit):
        nonlocal best, best_x
        if best_x is None or dim <= 0:
            return
        x = best_x[:]
        fx = best

        # start step relative to box; shrink quickly
        step = [0.06 * span_safe[i] for i in range(dim)]
        min_step = [1e-14 * span_safe[i] for i in range(dim)]

        # keep very cheap
        for _ in range(2):
            if now() >= time_limit:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if now() >= time_limit:
                    break
                sj = step[j]
                if sj <= min_step[j]:
                    continue
                base = x[j]

                # try + and -
                y = x[:]
                y[j] = base + sj
                clamp_reflect_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    continue

                y = x[:]
                y[j] = base - sj
                clamp_reflect_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    continue

            if improved:
                for j in range(dim):
                    step[j] = min(0.30 * span_safe[j], step[j] * 1.15)
            else:
                for j in range(dim):
                    step[j] *= 0.45

        if fx < best:
            best, best_x = fx, x[:]

    # ---------------- iL-SHADE / jSO-like DE ----------------
    NP0 = max(28, min(140, 20 + 4 * dim))
    NP_min = max(8, 6 + dim // 5)

    # init pop (inject best + jittered best)
    pop, fit = [], []
    if best_x is not None:
        pop.append(best_x[:])
        fit.append(best)
        for _ in range(min(8, NP0 - 1)):
            if now() >= deadline:
                return best
            x = best_x[:]
            for j in range(dim):
                x[j] += 0.07 * span_safe[j] * randn()
            clamp_reflect_inplace(x)
            fx = try_update(x)
            pop.append(x)
            fit.append(fx)

    while len(pop) < NP0:
        if now() >= deadline:
            return best
        x = rand_point()
        fx = try_update(x)
        pop.append(x)
        fit.append(fx)

    archive = []
    arch_max = NP0

    # SHADE memory
    H = 16
    MF = [0.5] * H
    MCR = [0.8] * H
    k_mem = 0

    last_best = best
    stall = 0
    last_local = now()

    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, (deadline - t0))

        NP = len(pop)
        # linear population size reduction (L-SHADE)
        NP_target = int(round(NP_min + (NP0 - NP_min) * (1.0 - frac)))
        NP_target = max(NP_min, min(NP0, NP_target))

        # rank
        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        # dynamic p (jSO-like): relatively larger early, smaller late
        p = 0.35 - 0.25 * frac
        p = max(0.06, min(0.35, p))
        pcount = max(2, int(math.ceil(p * NP)))
        pbest_pool = idx_sorted[:pcount]

        # success sets
        SF, SCR, dImp = [], [], []

        # union picker
        def pick_union(exclude_pop_idx_set):
            total = NP + len(archive)
            if total <= 0:
                return pop[0]
            for _ in range(30):
                r = random.randrange(total)
                if r < NP:
                    if r in exclude_pop_idx_set:
                        continue
                    return pop[r]
                else:
                    return archive[r - NP]
            # fallback
            for r in range(NP):
                if r not in exclude_pop_idx_set:
                    return pop[r]
            return pop[0]

        next_pop = [None] * NP
        next_fit = [None] * NP

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)

            # F: Cauchy around MF[r]
            Fi = MF[r] + 0.12 * rand_cauchy()
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = MF[r] + 0.12 * rand_cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            # jSO-style cap; allow larger early, smaller late
            Fmax = 1.0 - 0.4 * frac
            if Fmax < 0.55:
                Fmax = 0.55
            Fi = min(Fmax, max(0.05, Fi))

            # CR: normal around MCR[r]
            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # pick pbest
            pb = pbest_pool[random.randrange(len(pbest_pool))]
            xpbest = pop[pb]

            # pick r1 (pop) and r2 (union), using indices not identities
            # r1 must be != i and preferably != pb
            r1 = random.randrange(NP)
            tries = 0
            while (r1 == i or r1 == pb) and tries < 30:
                r1 = random.randrange(NP)
                tries += 1
            xr1 = pop[r1]

            exclude = {i, pb, r1}
            xr2 = pick_union(exclude)

            # jSO-style weighted current-to-pbest:
            # v = xi + Fi*(xpbest - xi) + Fi*(xr1 - xr2), but slightly stronger pull early
            Fw = 0.7 + 0.3 * (1.0 - frac)  # 1.0 early -> 0.7 late
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + (Fi * Fw) * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
            clamp_reflect_inplace(v)

            # binomial crossover
            jrand = random.randrange(dim) if dim > 0 else 0
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]
            clamp_reflect_inplace(u)

            fu = evaluate(u)

            if fu <= fi:
                next_pop[i] = u
                next_fit[i] = fu

                # archive add replaced parent
                archive.append(xi[:])
                if len(archive) > arch_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                imp = fi - fu
                if imp > 0.0:
                    SF.append(Fi)
                    SCR.append(CRi)
                    dImp.append(imp)

                if fu < best:
                    best = fu
                    best_x = u[:]
            else:
                next_pop[i] = xi
                next_fit[i] = fi

        pop, fit = next_pop, next_fit

        # update memories
        if SF:
            wsum = sum(dImp)
            if wsum <= 0.0:
                weights = [1.0 / len(SF)] * len(SF)
            else:
                inv = 1.0 / wsum
                weights = [imp * inv for imp in dImp]

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, fval in zip(weights, SF):
                num += w * (fval * fval)
                den += w * fval
            if den > 1e-30:
                MF[k_mem] = num / den

            # weighted mean for CR
            cr = 0.0
            for w, cval in zip(weights, SCR):
                cr += w * cval
            MCR[k_mem] = cr

            k_mem = (k_mem + 1) % H

        # stagnation + injection
        if best < last_best - 1e-15 * (1.0 + abs(last_best)):
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall > (10 + dim // 2):
            stall = 0
            # replace some worst with (best jitter) and (opposition) and (rand)
            NP = len(pop)
            idx_sorted = list(range(NP))
            idx_sorted.sort(key=lambda i: fit[i])
            worst_count = max(1, NP // 5)

            for k in range(worst_count):
                if now() >= deadline:
                    return best
                idx = idx_sorted[-1 - k]
                r = random.random()
                if best_x is None:
                    xnew = rand_point()
                elif r < 0.45:
                    xnew = best_x[:]
                    # heavy-tailed jump in a few coords
                    kcount = max(1, int(math.sqrt(dim)))
                    if dim > 0:
                        coords = random.sample(range(dim), kcount) if dim >= kcount else list(range(dim))
                        for j in coords:
                            xnew[j] += 0.18 * span_safe[j] * rand_cauchy()
                elif r < 0.75:
                    # opposition around center of box relative to current best
                    xnew = [(lo[j] + hi[j] - best_x[j]) for j in range(dim)]
                    for j in range(dim):
                        xnew[j] += 0.02 * span_safe[j] * randn()
                else:
                    xnew = rand_point()

                clamp_reflect_inplace(xnew)
                fnew = evaluate(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

        # reduce population
        if len(pop) > NP_target:
            idx_sorted = list(range(len(pop)))
            idx_sorted.sort(key=lambda i: fit[i])
            keep = idx_sorted[:NP_target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            arch_max = len(pop)
            while len(archive) > arch_max:
                archive[random.randrange(len(archive))] = archive[-1]
                archive.pop()

        # periodic local improvements late (pattern search + a couple directional probes)
        if best_x is not None:
            # more frequent late
            period = max(0.10, 0.25 * (1.0 - frac) + 0.08)
            if (now() - last_local) >= period:
                last_local = now()
                remain = deadline - now()
                tl = now() + min(0.05 * max_time + 0.04 * max_time * frac, 0.25 * remain, 0.55)

                # coordinate pattern search
                pattern_search_best(tl)

                # a couple cheap directional line tries using pairwise differences from current pop
                for _ in range(2):
                    if now() >= tl:
                        break
                    if len(pop) >= 2 and dim > 0:
                        a = pop[random.randrange(len(pop))]
                        b = pop[random.randrange(len(pop))]
                        d = [best_x[j] - 0.5 * (a[j] + b[j]) for j in range(dim)]
                        dn = l2norm(d)
                        if dn > 0.0:
                            inv = 1.0 / dn
                            for j in range(dim):
                                d[j] *= inv
                            # small step, shrinks with time
                            step = (0.03 - 0.02 * frac) * avg_span
                            if step < 1e-16 * avg_span:
                                step = 1e-16 * avg_span
                            y = best_x[:]
                            for j in range(dim):
                                y[j] += step * d[j]
                            clamp_reflect_inplace(y)
                            fy = evaluate(y)
                            if fy < best:
                                best, best_x = fy, y[:]

    return best
