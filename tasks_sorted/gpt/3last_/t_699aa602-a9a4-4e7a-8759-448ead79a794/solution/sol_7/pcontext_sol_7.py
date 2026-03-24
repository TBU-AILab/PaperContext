import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Hybrid time-budgeted minimizer (self-contained, no external libs).

    Upgrade focus vs your best (#1 JADE-like DE):
      - Keeps a strong DE core, but adds TWO key improvements that often matter a lot:
        (1) L-SHADE-style linear population size reduction (more exploitation later).
        (2) More aggressive "p-best" and success-history adaptation (SHADE memory),
            which is typically more stable than a single mu_F/mu_CR.
      - Adds a cheap surrogate-free "two-point quadratic" line search along
        a few promising directions late in the run (helps squeeze extra improvement).
      - Better restarts/injections using opposition + heavy-tail jitter when stuck.
      - Bound handling: reflect then clamp.

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

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_then_clamp_inplace(x):
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
        return clamp_inplace(x)

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

    # Cauchy (for heavy-tailed mutation factor samples; classic in SHADE/JADE)
    def rand_cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    def l2norm(v):
        s = 0.0
        for i in range(dim):
            s += v[i] * v[i]
        return math.sqrt(s)

    # ---------------- bookkeeping ----------------
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

    # ---------------- initial seeding ----------------
    # center
    x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    reflect_then_clamp_inplace(x0)
    try_update(x0)

    # a handful of corners
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
    for _ in range(10 + 5 * dim):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------------- local intensification tools ----------------
    def quick_coord_refine(x_start, f_start, time_limit):
        x = x_start[:]
        fx = f_start

        step = [0.08 * s for s in span_safe]
        min_step = [1e-14 * s for s in span_safe]

        # keep cheap
        for _ in range(2):
            if now() >= time_limit:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if now() >= time_limit:
                    return x, fx
                sj = step[j]
                if sj <= min_step[j]:
                    continue
                base = x[j]

                y = x[:]
                y[j] = base + sj
                reflect_then_clamp_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    continue

                y = x[:]
                y[j] = base - sj
                reflect_then_clamp_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    continue

            if improved:
                for j in range(dim):
                    step[j] = min(0.35 * span_safe[j], step[j] * 1.20)
            else:
                for j in range(dim):
                    step[j] *= 0.5
        return x, fx

    def quad_line_search(x, fx, d, base_step, time_limit):
        """
        1D refinement along direction d using 2-point quadratic fit around 0:
          f(0)=fx, f(a), f(-a) -> best of {0, +/-a, argmin parabola}
        Very cheap (<= 2-3 evals if time allows).
        """
        if now() >= time_limit:
            return x, fx

        # normalize d
        dn = l2norm(d)
        if dn <= 0.0:
            return x, fx
        inv = 1.0 / dn
        d = [di * inv for di in d]

        a = max(base_step, 1e-16 * avg_span)

        # evaluate at +a and -a
        xp = x[:]
        xm = x[:]
        for j in range(dim):
            xp[j] += a * d[j]
            xm[j] -= a * d[j]
        reflect_then_clamp_inplace(xp)
        reflect_then_clamp_inplace(xm)

        fp = evaluate(xp)
        if fp < fx:
            x, fx = xp, fp

        if now() >= time_limit:
            return x, fx

        fm = evaluate(xm)
        if fm < fx:
            x, fx = xm, fm

        # quadratic fit using f(-a), f(0), f(+a): minimize parabola
        # f(t) = At^2 + Bt + C, with t in {-a,0,+a}
        # A = (fp + fm - 2f0)/(2a^2), B = (fp - fm)/(2a)
        # t* = -B/(2A) (if A>0)
        f0 = fx  # note: fx may have improved vs original; still OK as a local attempt
        A = (fp + fm - 2.0 * f0) / (2.0 * a * a)
        if A > 1e-30:
            B = (fp - fm) / (2.0 * a)
            tstar = -B / (2.0 * A)
            # clamp step to a reasonable bracket
            tstar = max(-2.5 * a, min(2.5 * a, tstar))
            if abs(tstar) > 0.05 * a and now() < time_limit:
                xt = x[:]
                for j in range(dim):
                    xt[j] += tstar * d[j]
                reflect_then_clamp_inplace(xt)
                ft = evaluate(xt)
                if ft < fx:
                    return xt, ft
        return x, fx

    # ---------------- L-SHADE / SHADE-like DE ----------------
    # initial and minimum population sizes
    NP0 = max(24, min(120, 18 + 3 * dim))
    NP_min = max(8, 4 + dim // 4)

    # initialize population (inject best and perturbed best)
    pop = []
    fit = []

    if best_x is not None:
        pop.append(best_x[:])
        fit.append(best)
        for _ in range(min(6, NP0 - 1)):
            if now() >= deadline:
                return best
            x = best_x[:]
            for j in range(dim):
                x[j] += 0.05 * span_safe[j] * randn()
            reflect_then_clamp_inplace(x)
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

    # archive for difference vectors
    archive = []
    arch_max = NP0

    # SHADE memory of successful parameters
    H = 12  # memory size
    MCR = [0.8] * H
    MF = [0.5] * H
    mem_idx = 0

    # schedule
    last_best = best
    stall = 0
    last_local = now()
    last_line = now()

    # main loop
    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, (deadline - t0))

        NP = len(pop)
        if NP <= NP_min:
            NP_target = NP_min
        else:
            # L-SHADE linear reduction
            NP_target = int(round(NP_min + (NP0 - NP_min) * (1.0 - frac)))
            NP_target = max(NP_min, min(NP0, NP_target))

        # rank indices for p-best selection and for pop reduction
        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        # p-best fraction (slightly decreasing p late -> more exploit top few)
        p = 0.30 - 0.20 * frac
        p = max(0.05, min(0.35, p))
        pcount = max(2, int(math.ceil(p * NP)))
        pbest_pool = idx_sorted[:pcount]

        SF = []
        SCR = []
        dF = []

        # to enable population reduction after generation
        next_pop = [None] * NP
        next_fit = [None] * NP

        union_size = NP + len(archive)

        def pick_union(exclude):
            # exclude is a set of pop indices; archive indices are always allowed
            if union_size <= 0:
                return pop[0]
            for _ in range(25):
                r = random.randrange(union_size)
                if r < NP:
                    if r in exclude:
                        continue
                    return pop[r]
                else:
                    return archive[r - NP]
            # fallback
            for r in range(NP):
                if r not in exclude:
                    return pop[r]
            return pop[0]

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            # F: Cauchy around MF[r], resample until >0
            Fi = MF[r] + 0.1 * rand_cauchy()
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = MF[r] + 0.1 * rand_cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            Fi = min(1.0, max(0.05, Fi))

            # CR: normal around MCR[r]
            CRi = MCR[r] + 0.1 * randn()
            CRi = min(1.0, max(0.0, CRi))

            # choose pbest
            pb = pbest_pool[random.randrange(len(pbest_pool))]
            xpbest = pop[pb]

            # r1 from pop (not i), r2 from union (not i, r1 ideally)
            exclude = {i, pb}
            xr1 = pop[random.randrange(NP)]
            tries = 0
            while (xr1 is xi or tries < 10 and pop.index(xr1) in exclude):  # small NP safe-ish
                xr1 = pop[random.randrange(NP)]
                tries += 1

            xr2 = pick_union(exclude)

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
            reflect_then_clamp_inplace(v)

            # binomial crossover
            if dim > 0:
                jrand = random.randrange(dim)
            else:
                jrand = 0
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]
            reflect_then_clamp_inplace(u)

            fu = evaluate(u)

            if fu <= fi:
                # success
                next_pop[i] = u
                next_fit[i] = fu

                archive.append(xi[:])
                if len(archive) > arch_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                imp = max(0.0, fi - fu)
                if imp > 0.0:
                    SF.append(Fi)
                    SCR.append(CRi)
                    dF.append(imp)

                if fu < best:
                    best = fu
                    best_x = u[:]
            else:
                # keep parent
                next_pop[i] = xi
                next_fit[i] = fi

        pop = next_pop
        fit = next_fit

        # update SHADE memories
        if SF:
            wsum = sum(dF)
            if wsum <= 0.0:
                wsum = float(len(SF))
                weights = [1.0 / wsum] * len(SF)
            else:
                weights = [imp / wsum for imp in dF]

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, fval in zip(weights, SF):
                num += w * (fval * fval)
                den += w * fval
            MF[mem_idx] = (num / den) if den > 1e-30 else MF[mem_idx]

            # weighted mean for CR
            cr = 0.0
            for w, cval in zip(weights, SCR):
                cr += w * cval
            MCR[mem_idx] = cr

            mem_idx = (mem_idx + 1) % H

        # stagnation + injections
        if best < last_best - 1e-15 * (1.0 + abs(last_best)):
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall > (10 + dim // 2):
            stall = 0
            # replace a few worst with opposition / jitter around best
            NP = len(pop)
            idx_sorted = list(range(NP))
            idx_sorted.sort(key=lambda i: fit[i])
            worst_count = max(1, NP // 6)
            for k in range(worst_count):
                if now() >= deadline:
                    return best
                idx = idx_sorted[-1 - k]
                if best_x is None:
                    xnew = rand_point()
                else:
                    if random.random() < 0.5:
                        # opposition point around center: x_op = lo+hi - x
                        xnew = [(lo[j] + hi[j] - best_x[j]) for j in range(dim)]
                        # plus small jitter
                        for j in range(dim):
                            xnew[j] += 0.03 * span_safe[j] * randn()
                    else:
                        # heavy-tail jitter
                        xnew = best_x[:]
                        for j in range(dim):
                            xnew[j] += 0.15 * span_safe[j] * rand_cauchy()
                reflect_then_clamp_inplace(xnew)
                fnew = evaluate(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best = fnew
                    best_x = xnew[:]

        # population size reduction (drop worst)
        if len(pop) > NP_target:
            idx_sorted = list(range(len(pop)))
            idx_sorted.sort(key=lambda i: fit[i])
            keep = idx_sorted[:NP_target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            # archive should scale too
            arch_max = len(pop)
            if len(archive) > arch_max:
                # random prune
                while len(archive) > arch_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

        # periodic local refinement (cheap)
        if best_x is not None and (now() - last_local) > max(0.12, 0.06 * max_time * (0.7 + 0.6 * frac)):
            last_local = now()
            remain = deadline - now()
            tl = now() + min(0.05 * max_time + 0.05 * max_time * frac, 0.22 * remain, 0.65)
            xr, fr = quick_coord_refine(best_x, best, tl)
            if fr < best:
                best, best_x = fr, xr[:]
                # inject into population by replacing current worst
                w = max(range(len(pop)), key=lambda i: fit[i])
                pop[w] = best_x[:]
                fit[w] = best

        # late-stage directional quadratic line searches
        if best_x is not None and frac > 0.55 and (now() - last_line) > max(0.18, 0.08 * max_time):
            last_line = now()
            remain = deadline - now()
            tl = now() + min(0.03 * max_time, 0.15 * remain, 0.35)
            # try 2-3 directions: (best - random elite), and a random sparse direction
            for _ in range(2):
                if now() >= tl:
                    break
                d = [0.0] * dim
                if pop and random.random() < 0.7:
                    j = min(len(pop)-1, int(random.random() * len(pop)))
                    # use a random population vector to form direction
                    x2 = pop[j]
                    for k in range(dim):
                        d[k] = best_x[k] - x2[k]
                else:
                    # sparse random direction
                    kcount = max(1, int(math.sqrt(dim)))
                    idxs = random.sample(range(dim), kcount) if dim >= kcount else list(range(dim))
                    for k in idxs:
                        d[k] = randn()
                base_step = 0.02 * avg_span
                xr, fr = quad_line_search(best_x, best, d, base_step, tl)
                if fr < best:
                    best, best_x = fr, xr[:]

    return best
