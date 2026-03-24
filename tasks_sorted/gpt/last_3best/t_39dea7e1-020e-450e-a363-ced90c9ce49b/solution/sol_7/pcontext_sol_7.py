import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only bounded minimizer (anytime).

    Improvements vs prior code:
      - Uses *normalized* coordinates for all evolutionary operators (scale-invariant).
      - Stronger DE core: jDE-style self-adaptive F/CR per individual + current-to-pbest/1
        + optional "rand-to-best/2" injection when stuck.
      - Lightweight trust-region local search around the best: adaptive coordinate search
        + occasional quadratic 1D fit (3-point parabolic step) per coordinate.
      - Better restart logic: multi-start around best + random immigrants + shrinked sigma.
      - Keeps cache (quantized) but makes it safer and less over-colliding.

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ------------------ helpers ------------------
    def safe_float(y):
        try:
            y = float(y)
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == -float("inf"):
            return float("inf")
        return y

    if dim <= 0:
        try:
            return safe_float(func([]))
        except TypeError:
            return safe_float(func())

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_ok = [span[i] > 0.0 for i in range(dim)]

    def clamp01(v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def x_from_z(z):
        x = [0.0] * dim
        for d in range(dim):
            if not span_ok[d]:
                x[d] = lo[d]
            else:
                x[d] = lo[d] + clamp01(z[d]) * span[d]
        return x

    def z_from_x(x):
        z = [0.0] * dim
        for d in range(dim):
            if not span_ok[d]:
                z[d] = 0.0
            else:
                z[d] = (x[d] - lo[d]) / span[d]
                if z[d] < 0.0:
                    z[d] = 0.0
                elif z[d] > 1.0:
                    z[d] = 1.0
        return z

    # quantized cache in normalized coordinates
    # (avoid huge dict, but not too coarse)
    qinv = []
    for d in range(dim):
        if not span_ok[d]:
            qinv.append(0.0)
        else:
            q = 2e-7  # in [0,1] space
            qinv.append(1.0 / q)

    cache = {}

    def eval_z(z):
        # key in [0,1] space
        key = tuple(int(clamp01(z[d]) * qinv[d]) if qinv[d] else 0 for d in range(dim))
        v = cache.get(key)
        if v is not None:
            return v
        x = x_from_z(z)
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        y = safe_float(y)
        cache[key] = y
        return y

    def rand_z():
        return [0.0 if not span_ok[d] else random.random() for d in range(dim)]

    def qlhs_z(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        popz = []
        for i in range(n):
            z = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    z[d] = 0.0
                else:
                    z[d] = (perms[d][i] + random.random()) / n
            popz.append(z)
        return popz

    def opp_z(z):
        oz = z[:]
        for d in range(dim):
            if span_ok[d]:
                oz[d] = 1.0 - clamp01(z[d])
            else:
                oz[d] = 0.0
        return oz

    def gauss():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ------------------ local search ------------------
    def parabolic_1d(z, fz, d, step):
        """Try 3-point quadratic fit along coordinate d: z-step, z, z+step."""
        z0 = z[:]
        a = clamp01(z0[d] - step)
        b = clamp01(z0[d])
        c = clamp01(z0[d] + step)

        # if clamped collapses, bail
        if abs(c - a) < 1e-15:
            return z, fz, 0

        evals = 0
        zA = z0[:]; zA[d] = a
        zC = z0[:]; zC[d] = c

        fA = eval_z(zA); evals += 1
        fB = fz
        fC = eval_z(zC); evals += 1

        # fit parabola through (a,fA),(b,fB),(c,fC); compute vertex
        # Use stable formula in shifted coords
        x1, y1 = a - b, fA - fB
        x2, y2 = c - b, fC - fB
        den = (x1 * x2) * (x1 - x2)
        if abs(den) < 1e-18:
            # fallback: best of endpoints
            if fA < fz and fA <= fC:
                return zA, fA, evals
            if fC < fz and fC < fA:
                return zC, fC, evals
            return z, fz, evals

        # vertex in shifted coordinate t relative to b
        t = (x1 * x1 * y2 - x2 * x2 * y1) / (2.0 * (x1 * y2 - x2 * y1) + 1e-18)
        zv = z0[:]
        zv[d] = clamp01(b + t)
        fv = eval_z(zv); evals += 1

        # take best among A, V, C, B
        bestz, bestf = z, fz
        if fA < bestf:
            bestz, bestf = zA, fA
        if fC < bestf:
            bestz, bestf = zC, fC
        if fv < bestf:
            bestz, bestf = zv, fv
        return bestz, bestf, evals

    def coord_trust_search(bestz, bestf, max_evals):
        """Adaptive coordinate search in normalized space."""
        if max_evals <= 0:
            return bestz, bestf
        z = bestz[:]
        f = bestf
        evals = 0

        # step starts moderate; shrinks quickly if no progress
        step = 0.08
        rounds = 0
        while evals < max_evals and rounds < 6:
            rounds += 1
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for d in order:
                if evals >= max_evals or (not span_ok[d]):
                    continue

                # try +/- step
                base = z[d]
                for sgn in (-1.0, 1.0):
                    if evals >= max_evals:
                        break
                    cand = z[:]
                    cand[d] = clamp01(base + sgn * step)
                    if cand[d] == base:
                        continue
                    fc = eval_z(cand); evals += 1
                    if fc < f:
                        z, f = cand, fc
                        improved = True
                        base = z[d]

                # occasional quadratic refinement if time allows
                if evals + 3 <= max_evals and random.random() < 0.35:
                    candz, candf, used = parabolic_1d(z, f, d, step * 0.8)
                    evals += used
                    if candf < f:
                        z, f = candz, candf
                        improved = True

            if improved:
                step = min(0.20, step * 1.25)
            else:
                step *= 0.35
                if step < 5e-5:
                    break
        return z, f

    # ------------------ time ------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ------------------ init ------------------
    NP = max(20, min(100, 12 * dim + 12))
    popz = qlhs_z(NP)
    popz += [opp_z(z) for z in popz]

    pop = []
    fit = []
    for z in popz:
        if time.time() >= deadline:
            break
        fz = eval_z(z)
        pop.append(z)
        fit.append(fz)

    if not fit:
        return float("inf")

    # fill and trim
    while len(pop) < NP and time.time() < deadline:
        z = rand_z()
        pop.append(z)
        fit.append(eval_z(z))

    if len(pop) > NP:
        idx = sorted(range(len(pop)), key=lambda i: fit[i])[:NP]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    bestz = pop[best_i][:]
    best = fit[best_i]

    # ------------------ jDE/adaptive DE state ------------------
    F_i = [0.6] * NP
    CR_i = [0.5] * NP
    tau1 = 0.10
    tau2 = 0.10
    Fl, Fu = 0.15, 0.95

    # mild archive of replaced solutions (in z-space)
    archive = []
    arch_max = NP

    # restart/stagnation
    last_best = best
    stall = 0
    max_stall = max(18, 5 + 2 * dim)

    gen = 0
    while time.time() < deadline:
        gen += 1
        tl = deadline - time.time()
        frac_left = tl / max(1e-12, float(max_time))
        near_end = frac_left < 0.25
        very_end = frac_left < 0.08

        order = sorted(range(NP), key=lambda i: fit[i])
        p = 0.20 if not near_end else 0.10
        pnum = max(2, int(p * NP))

        new_pop = pop[:]
        new_fit = fit[:]

        # ------------------ DE generation ------------------
        for i in range(NP):
            if time.time() >= deadline:
                break

            # jDE self-adaptation
            if random.random() < tau1:
                F_i[i] = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CR_i[i] = random.random()

            F = F_i[i]
            CR = CR_i[i]

            # choose pbest among top p%
            pbest = order[random.randrange(pnum)]

            # r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 from pop+archive, != i,r1
            pool_n = NP + len(archive)
            if pool_n <= 3:
                r2 = (i + 1) % NP
                z_r2 = pop[r2]
            else:
                while True:
                    r2 = random.randrange(pool_n)
                    if r2 < NP:
                        if r2 != i and r2 != r1:
                            z_r2 = pop[r2]
                            break
                    else:
                        z_r2 = archive[r2 - NP]
                        break

            zi = pop[i]
            zpb = pop[pbest]
            zr1 = pop[r1]

            # mutation: current-to-pbest/1 (in normalized space)
            v = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    v[d] = 0.0
                else:
                    v[d] = zi[d] + F * (zpb[d] - zi[d]) + F * (zr1[d] - z_r2[d])

            # binomial crossover
            u = zi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if (random.random() < CR) or (d == jrand):
                    u[d] = v[d]

            # repair: reflect into [0,1]
            for d in range(dim):
                if not span_ok[d]:
                    u[d] = 0.0
                    continue
                x = u[d]
                if x < 0.0:
                    x = -x
                    if x > 1.0:
                        x = random.random()
                elif x > 1.0:
                    x = 2.0 - x
                    if x < 0.0:
                        x = random.random()
                u[d] = clamp01(x)

            fu = eval_z(u)
            if fu <= fit[i]:
                # archive replaced parent when strictly improved
                if fu < fit[i] and arch_max > 0:
                    if len(archive) < arch_max:
                        archive.append(zi[:])
                    else:
                        archive[random.randrange(len(archive))] = zi[:]

                new_pop[i] = u
                new_fit[i] = fu
                if fu < best:
                    best = fu
                    bestz = u[:]
            else:
                # small chance to accept worsening near end? no (stay conservative)
                pass

        pop, fit = new_pop, new_fit

        # ------------------ exploitation: local trust search ------------------
        if time.time() < deadline and (very_end or (gen % (6 if near_end else 10) == 0)):
            budget = 10 + 3 * dim
            bz, bf = coord_trust_search(bestz, best, budget)
            if bf < best:
                best, bestz = bf, bz[:]

        # ------------------ stagnation & restart ------------------
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0
            # keep elites, inject immigrants around best and random
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(4, NP // 6)

            # shrink archive a bit to avoid stale bias
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif archive:
                for _ in range(max(1, len(archive) // 5)):
                    archive.pop(random.randrange(len(archive)))

            # immigrants
            for k, idx in enumerate(order[keep:]):
                if time.time() >= deadline:
                    break
                if random.random() < 0.70:
                    # around best with gaussian radius
                    z = bestz[:]
                    rad = 0.18 if frac_left > 0.5 else 0.12
                    for d in range(dim):
                        if span_ok[d]:
                            z[d] = clamp01(z[d] + rad * gauss())
                    pop[idx] = z
                else:
                    pop[idx] = rand_z()
                fit[idx] = eval_z(pop[idx])

                # reset individual parameters to encourage exploration
                F_i[idx] = 0.5 + 0.4 * random.random()
                CR_i[idx] = random.random()

            # also do a quick local touch on the best after restart
            if time.time() < deadline:
                bz, bf = coord_trust_search(bestz, best, 6 + dim)
                if bf < best:
                    best, bestz = bf, bz[:]

            best_i = min(range(NP), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                bestz = pop[best_i][:]

    return best
