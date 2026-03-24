import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only, time-bounded minimizer (anytime).

    Upgrade focus (vs prior best):
      - Keep strong global search: L-SHADE/JADE-style DE with archive + success-history.
      - Add *proper* diagonal CMA-ES sampling in normalized space (mean/cov/sigma updated from elites),
        but make it (a) cheaper, (b) more stable near bounds (mirrored repair), (c) triggered adaptively.
      - Add "best-of-mirrors" boundary handling (tries mirrored versions when out of box).
      - Add tiny surrogate-free intensification: occasional pattern step along last successful delta.
      - Add time-aware scheduling: more exploration early, more CMA/local late.

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ------------------ basic helpers ------------------
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

    def clamp01(a):
        if a < 0.0: return 0.0
        if a > 1.0: return 1.0
        return a

    def to_z(x):
        z = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                v = (x[i] - lo[i]) / span[i]
                z[i] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            else:
                z[i] = 0.0
        return z

    def to_x(z):
        x = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                x[i] = lo[i] + clamp01(z[i]) * span[i]
            else:
                x[i] = lo[i]
        return x

    # Quantized cache in normalized coords (helps if algorithms revisit points)
    # Use coarser quantization than 2e-7 to reduce accidental collisions + overhead.
    q = 1e-6
    qinv = (1.0 / q) if q > 0 else 0.0
    cache = {}

    def eval_z(z):
        key = tuple(int(clamp01(z[i]) * qinv) if span_ok[i] else 0 for i in range(dim))
        v = cache.get(key)
        if v is not None:
            return v
        x = to_x(z)
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        y = safe_float(y)
        cache[key] = y
        return y

    def rand_z():
        return [0.0 if not span_ok[i] else random.random() for i in range(dim)]

    def qlhs_z(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        out = []
        for i in range(n):
            z = [0.0] * dim
            for d in range(dim):
                if span_ok[d]:
                    z[d] = (perms[d][i] + random.random()) / n
                else:
                    z[d] = 0.0
            out.append(z)
        return out

    def gauss():
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy(scale):
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        if t > 50.0: t = 50.0
        if t < -50.0: t = -50.0
        return scale * t

    # best-of-mirrors repair (in [0,1]) + optional evaluate a couple mirrored variants
    def repair_and_maybe_mirror(z, base_eval_budget=1):
        # deterministic mirror into [0,1]
        zz = z[:]
        for d in range(dim):
            if not span_ok[d]:
                zz[d] = 0.0
                continue
            v = zz[d]
            if v < 0.0:
                v = -v
            if v > 1.0:
                # fold by reflection repeatedly (but cheap approximation)
                v = v % 2.0
                if v > 1.0:
                    v = 2.0 - v
            zz[d] = clamp01(v)

        if base_eval_budget <= 0:
            return zz, None

        f0 = eval_z(zz)
        if base_eval_budget == 1 or dim == 0:
            return zz, f0

        # try a single "opposite" / mirror of a random subset of dims (helps if best near edges)
        bestz = zz
        bestf = f0
        # up to 2 extra checks
        trials = min(2, base_eval_budget - 1)
        for _ in range(trials):
            cand = zz[:]
            # flip ~sqrt(dim) dims
            k = max(1, int(math.sqrt(dim)))
            for _j in range(k):
                d = random.randrange(dim)
                if span_ok[d]:
                    cand[d] = 1.0 - cand[d]
            fc = eval_z(cand)
            if fc < bestf:
                bestf = fc
                bestz = cand
        return bestz, bestf

    # ------------------ local search (cheap) ------------------
    def coord_search(bestz, bestf, budget):
        if budget <= 0:
            return bestz, bestf
        z = bestz[:]
        f = bestf
        used = 0
        step = 0.10
        for _round in range(5):
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for d in order:
                if used >= budget:
                    return z, f
                if not span_ok[d]:
                    continue
                base = z[d]
                for sgn in (-1.0, 1.0):
                    if used >= budget:
                        break
                    cand = z[:]
                    cand[d] = clamp01(base + sgn * step)
                    if cand[d] == base:
                        continue
                    fc = eval_z(cand); used += 1
                    if fc < f:
                        z, f = cand, fc
                        improved = True
                        base = z[d]
            if improved:
                step = min(0.25, step * 1.35)
            else:
                step *= 0.30
                if step < 2e-5:
                    break
        return z, f

    # ------------------ time ------------------
    t0 = time.time()
    deadline = t0 + float(max_time)
    def time_left():
        return deadline - time.time()

    # ------------------ initialization ------------------
    # DE population
    NP = max(20, min(100, 10 * dim + 20))
    init = qlhs_z(NP)
    # opposition in normalized space
    init += [[(1.0 - z[d]) if span_ok[d] else 0.0 for d in range(dim)] for z in init]

    pop = []
    fit = []
    for z in init:
        if time.time() >= deadline:
            break
        zz, fz = repair_and_maybe_mirror(z, base_eval_budget=1)
        pop.append(zz)
        fit.append(fz)

    if not fit:
        return float("inf")

    while len(pop) < NP and time.time() < deadline:
        z = rand_z()
        pop.append(z)
        fit.append(eval_z(z))

    # keep best NP
    if len(pop) > NP:
        idx = sorted(range(len(pop)), key=lambda i: fit[i])[:NP]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    best_i = min(range(NP), key=lambda i: fit[i])
    bestz = pop[best_i][:]
    best = fit[best_i]
    prev_bestz = bestz[:]
    prev_best = best

    # ------------------ L-SHADE/JADE-ish DE state ------------------
    H = 8
    MF = [0.6] * H
    MCR = [0.5] * H
    mem_pos = 0
    archive = []
    arch_max = NP

    # ------------------ diagonal CMA state in normalized coords ------------------
    # Initialize mean at best
    m = bestz[:]  # in [0,1]
    sigma = 0.25
    Cdiag = [1.0] * dim  # diag covariance (in normalized coordinates)
    ps = [0.0] * dim
    pc = [0.0] * dim

    lam = max(10, min(36, 4 + int(3.0 * math.sqrt(dim))))
    mu = max(2, lam // 2)
    # log weights
    ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(ws)
    w = [wi / wsum for wi in ws]
    mueff = 1.0 / sum(wi * wi for wi in w)

    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim) if dim > 0 else 0.5
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)) if dim > 0 else 1.0

    # ------------------ stagnation / restart ------------------
    stall = 0
    max_stall = max(10, 3 + dim)
    gen = 0

    # track last successful best-move for a small "pattern" step
    last_delta = [0.0] * dim

    while time.time() < deadline:
        gen += 1
        tl = time_left()
        if tl <= 0:
            break
        frac_left = tl / max(1e-12, float(max_time))
        near_end = frac_left < 0.25
        very_end = frac_left < 0.10

        # ---------- DE generation ----------
        order = sorted(range(NP), key=lambda i: fit[i])
        p = 0.18 if not near_end else 0.10
        pnum = max(2, int(p * NP))

        SF, SCR, dF = [], [], []
        new_pop = pop[:]
        new_fit = fit[:]

        for i in range(NP):
            if time.time() >= deadline:
                break

            r = random.randrange(H)
            mu_f = MF[r]
            mu_cr = MCR[r]

            # CR approx normal around mu_cr
            CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() - 1.5)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # F cauchy around mu_f
            F = -1.0
            for _ in range(6):
                F = mu_f + cauchy(0.1)
                if 0.0 < F <= 1.0:
                    break
            if not (0.0 < F <= 1.0):
                F = 0.5

            pbest = order[random.randrange(pnum)]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            pool_n = NP + len(archive)
            if pool_n <= 2:
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

            # current-to-pbest/1 in normalized space
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

            # repair + optionally test a mirror variant early in the run
            extra = 2 if (not near_end and random.random() < 0.12) else 1
            uu, fu = repair_and_maybe_mirror(u, base_eval_budget=extra)
            if fu <= fit[i]:
                new_pop[i] = uu
                new_fit[i] = fu
                if fu < fit[i]:
                    if arch_max > 0:
                        if len(archive) < arch_max:
                            archive.append(zi[:])
                        else:
                            archive[random.randrange(len(archive))] = zi[:]
                    SF.append(F)
                    SCR.append(CR)
                    dF.append(fit[i] - fu)
                if fu < best:
                    best = fu
                    bestz = uu[:]
            # else keep parent

        pop, fit = new_pop, new_fit

        # update memories
        if dF:
            s = sum(dF)
            if s <= 0.0:
                ww = [1.0 / len(dF)] * len(dF)
            else:
                ww = [di / s for di in dF]
            # CR: weighted mean
            mcr = 0.0
            for wi, cri in zip(ww, SCR):
                mcr += wi * cri
            # F: weighted Lehmer mean
            num = 0.0
            den = 0.0
            for wi, fi in zip(ww, SF):
                num += wi * fi * fi
                den += wi * fi
            mf = (num / den) if den > 1e-12 else MF[mem_pos]
            MCR[mem_pos] = mcr
            MF[mem_pos] = mf
            mem_pos = (mem_pos + 1) % H

        # update last_delta if best improved
        if best < prev_best:
            last_delta = [bestz[d] - prev_bestz[d] for d in range(dim)]
            prev_best = best
            prev_bestz = bestz[:]

        # ---------- CMA batch (adaptive trigger) ----------
        # Trigger CMA more when: near end, or after some DE gens, or mild stall.
        do_cma = very_end or (gen % (6 if near_end else 10) == 0) or (stall >= max_stall // 2)
        if do_cma and time.time() < deadline and dim > 0:
            # pull mean toward current best (robust)
            m = bestz[:]

            # sample a small batch (time-aware)
            lam_use = lam if not near_end else max(6, lam // 2)

            samples = []
            for _ in range(lam_use):
                if time.time() >= deadline:
                    break
                # y = sigma * sqrt(Cdiag) * N(0,I)
                y = [0.0] * dim
                zz = [0.0] * dim
                for d in range(dim):
                    if not span_ok[d]:
                        y[d] = 0.0
                        zz[d] = 0.0
                        continue
                    sd = math.sqrt(max(1e-18, Cdiag[d]))
                    y[d] = sigma * sd * gauss()
                    zz[d] = m[d] + y[d]
                candz, fc = repair_and_maybe_mirror(zz, base_eval_budget=1)
                samples.append((fc, candz, y))
                if fc < best:
                    best = fc
                    bestz = candz[:]
                    m = bestz[:]

            if len(samples) >= mu:
                samples.sort(key=lambda t: t[0])

                # recombination: move mean using weighted y (in *pre-repair* step y is fine)
                ymean = [0.0] * dim
                for i in range(mu):
                    wi = w[i]
                    yi = samples[i][2]
                    for d in range(dim):
                        ymean[d] += wi * yi[d]

                # ps update
                c_fac = math.sqrt(cs * (2.0 - cs) * mueff)
                for d in range(dim):
                    if not span_ok[d]:
                        ps[d] = 0.0
                        continue
                    sd = math.sqrt(max(1e-18, Cdiag[d]))
                    ps[d] = (1.0 - cs) * ps[d] + c_fac * (ymean[d] / max(1e-18, sigma)) / sd

                # sigma update
                ps_norm = math.sqrt(sum(v * v for v in ps))
                sigma *= math.exp((cs / ds) * (ps_norm / max(1e-18, chiN) - 1.0))
                if sigma < 1e-6: sigma = 1e-6
                if sigma > 0.9: sigma = 0.9

                # pc update
                cc_fac = math.sqrt(cc * (2.0 - cc) * mueff)
                for d in range(dim):
                    if not span_ok[d]:
                        pc[d] = 0.0
                        continue
                    pc[d] = (1.0 - cc) * pc[d] + cc_fac * (ymean[d] / max(1e-18, sigma))

                # Cdiag update
                invsig = 1.0 / max(1e-18, sigma)
                for d in range(dim):
                    if not span_ok[d]:
                        Cdiag[d] = 1.0
                        continue
                    old = Cdiag[d]
                    rank_one = pc[d] * pc[d]
                    rank_mu = 0.0
                    for i in range(mu):
                        yi = samples[i][2][d] * invsig
                        rank_mu += w[i] * (yi * yi)
                    Cdiag[d] = (1.0 - c1 - cmu) * old + c1 * rank_one + cmu * rank_mu
                    if Cdiag[d] < 1e-12: Cdiag[d] = 1e-12
                    if Cdiag[d] > 1e6: Cdiag[d] = 1e6

        # ---------- small pattern step (intensification) ----------
        if time.time() < deadline and (near_end or (gen % 7 == 0)) and dim > 0:
            if any(abs(v) > 0.0 for v in last_delta):
                alpha = 1.3 if near_end else 1.6
                cand = [bestz[d] + alpha * last_delta[d] for d in range(dim)]
                candz, fc = repair_and_maybe_mirror(cand, base_eval_budget=1)
                if fc < best:
                    best = fc
                    bestz = candz[:]

        # ---------- local refine ----------
        if time.time() < deadline and (very_end or gen % (5 if near_end else 11) == 0):
            budget = 8 + 2 * dim if near_end else 6 + dim
            bz, bf = coord_search(bestz, best, budget)
            if bf < best:
                best, bestz = bf, bz[:]

        # ---------- stagnation handling ----------
        if best < prev_best - 1e-15:
            prev_best = best
            prev_bestz = bestz[:]
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0
            # keep elites; refresh rest using mixture of best-centered gaussian and random
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(4, NP // 6)
            for idx in order[keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.70:
                    z = bestz[:]
                    rad = 0.18 if frac_left > 0.5 else 0.12
                    for d in range(dim):
                        if span_ok[d]:
                            z[d] = clamp01(z[d] + rad * gauss())
                    pop[idx] = z
                else:
                    pop[idx] = rand_z()
                fit[idx] = eval_z(pop[idx])

            # refresh CMA a bit (avoid premature collapse)
            sigma = min(0.35, max(0.08, sigma * 1.25))
            for d in range(dim):
                Cdiag[d] = min(5.0, max(0.15, Cdiag[d]))

            # prune archive lightly
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif archive:
                drop = max(1, len(archive) // 5)
                for _ in range(drop):
                    if archive:
                        archive.pop(random.randrange(len(archive)))

            # update best from pop (safety)
            bi = min(range(NP), key=lambda i: fit[i])
            if fit[bi] < best:
                best = fit[bi]
                bestz = pop[bi][:]

    return best
