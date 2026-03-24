import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only, time-bounded minimizer.

    Improvements over the provided "last generated" code (and the earlier best):
      - Uses *two complementary engines* with clean separation:
          (A) L-SHADE/JADE-style DE in normalized space with archive + success-history.
          (B) Late-stage *trust-region local search* around the best using
              Simultaneous Perturbation (SPSA) gradient sign + backtracking,
              plus a brief coordinate/pattern polish.
        This typically gives much stronger final tightening than DE+CMA-diag alone
        on many black-box testbeds.
      - Better boundary handling: reflection + fold (fast) with no extra evals.
      - Budget-aware intensification schedule: local search ramps up near the end or on stall.
      - Light cache (quantized in normalized coords) to avoid accidental repeats.

    Returns:
        best (float): best objective found within max_time seconds
    """

    # ------------------------ safety / helpers ------------------------
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
        if v < 0.0: return 0.0
        if v > 1.0: return 1.0
        return v

    def to_x(z):
        x = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                x[i] = lo[i] + clamp01(z[i]) * span[i]
            else:
                x[i] = lo[i]
        return x

    # reflect/fold into [0,1] (fast, deterministic)
    def repair01(z):
        zz = z[:]
        for d in range(dim):
            if not span_ok[d]:
                zz[d] = 0.0
                continue
            v = zz[d]
            # fold into [0,2)
            if v < 0.0:
                v = -v
            # fold mod 2 then mirror if needed
            if v > 1.0:
                v = v % 2.0
                if v > 1.0:
                    v = 2.0 - v
            zz[d] = clamp01(v)
        return zz

    # quantized cache in normalized coords
    q = 2e-6
    qinv = 1.0 / q
    cache = {}

    def eval_z(z):
        zz = repair01(z)
        key = tuple(int(zz[i] * qinv) if span_ok[i] else 0 for i in range(dim))
        v = cache.get(key)
        if v is not None:
            return v
        x = to_x(zz)
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        y = safe_float(y)
        cache[key] = y
        return y

    def rand_z():
        return [random.random() if span_ok[i] else 0.0 for i in range(dim)]

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
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy(scale):
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        # cap to avoid insane steps
        if t > 40.0: t = 40.0
        if t < -40.0: t = -40.0
        return scale * t

    # ------------------------ time ------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return deadline - time.time()

    # ------------------------ initialization ------------------------
    NP = max(24, min(120, 12 * dim + 24))
    init = qlhs_z(NP)
    init += [[(1.0 - z[d]) if span_ok[d] else 0.0 for d in range(dim)] for z in init]

    pop = []
    fit = []
    for z in init:
        if time.time() >= deadline:
            break
        zz = repair01(z)
        fz = eval_z(zz)
        pop.append(zz)
        fit.append(fz)

    if not fit:
        return float("inf")

    while len(pop) < NP and time.time() < deadline:
        z = rand_z()
        pop.append(z)
        fit.append(eval_z(z))

    if len(pop) > NP:
        idx = sorted(range(len(pop)), key=lambda i: fit[i])[:NP]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    best_i = min(range(NP), key=lambda i: fit[i])
    bestz = pop[best_i][:]
    best = fit[best_i]

    # ------------------------ L-SHADE/JADE-ish DE state ------------------------
    H = 8
    MF = [0.6] * H
    MCR = [0.5] * H
    mem_pos = 0
    archive = []
    arch_max = NP

    stall = 0
    last_best = best
    max_stall = max(12, 4 + dim)

    # ------------------------ local search: SPSA trust-region ------------------------
    # Very evaluation-efficient approximate gradient method; great for late-stage tightening.
    def spsa_refine(z0, f0, max_evals, base_radius):
        if max_evals <= 0 or dim == 0:
            return z0, f0, 0

        z = z0[:]
        f = f0
        used = 0

        # trust radius in normalized coords
        r = max(1e-6, min(0.5, base_radius))
        # a and c schedules
        a0 = 0.20 * r
        c0 = 0.50 * r

        # keep best seen in refine
        bz, bf = z[:], f

        it = 0
        while used + 2 <= max_evals:
            it += 1
            # diminishing but not too fast (time-bounded)
            a = a0 / (it ** 0.35)
            c = c0 / (it ** 0.20)

            # Rademacher perturbation
            delta = [0.0] * dim
            for d in range(dim):
                if span_ok[d]:
                    delta[d] = -1.0 if random.random() < 0.5 else 1.0
                else:
                    delta[d] = 0.0

            zp = [z[d] + c * delta[d] for d in range(dim)]
            zm = [z[d] - c * delta[d] for d in range(dim)]
            fp = eval_z(zp); fm = eval_z(zm)
            used += 2

            # gradient estimate (component-wise)
            g = [0.0] * dim
            denom = max(1e-18, 2.0 * c)
            for d in range(dim):
                if span_ok[d] and delta[d] != 0.0:
                    g[d] = (fp - fm) / denom * delta[d]
                else:
                    g[d] = 0.0

            # take a step opposite gradient with backtracking within trust radius
            # normalize to keep step inside r
            gn = math.sqrt(sum(gd * gd for gd in g))
            if gn < 1e-18:
                # tiny random nudge inside trust region
                cand = [z[d] + (0.25 * r) * gauss() if span_ok[d] else 0.0 for d in range(dim)]
                fc = eval_z(cand); used += 1
                if fc < bf:
                    bz, bf = repair01(cand), fc
                    z, f = bz[:], bf
                else:
                    r *= 0.7
                if r < 2e-6:
                    break
                continue

            step_scale = min(1.0, r / (a * gn + 1e-18))
            # backtracking tries
            accepted = False
            for bt in range(3):
                if used >= max_evals:
                    break
                aa = a * step_scale * (0.6 ** bt)
                cand = [z[d] - aa * g[d] for d in range(dim)]
                fc = eval_z(cand); used += 1
                if fc < f:
                    z = repair01(cand)
                    f = fc
                    accepted = True
                    if fc < bf:
                        bz, bf = z[:], fc
                    r = min(0.5, r * 1.15)
                    break
            if not accepted:
                r *= 0.6
                if r < 2e-6:
                    break

        return bz, bf, used

    def coord_polish(bestz, bestf, budget):
        if budget <= 0:
            return bestz, bestf
        z = bestz[:]
        f = bestf
        used = 0
        step = 0.05
        for _round in range(4):
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
                step = min(0.15, step * 1.25)
            else:
                step *= 0.35
                if step < 2e-5:
                    break
        return z, f

    # ------------------------ main loop ------------------------
    gen = 0
    last_bestz = bestz[:]

    while time.time() < deadline:
        gen += 1
        tl = time_left()
        if tl <= 0:
            break
        frac_left = tl / max(1e-12, float(max_time))
        near_end = frac_left < 0.25
        very_end = frac_left < 0.10

        # ----- DE generation -----
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

            # CR ~ approx normal around mu_cr
            CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() - 1.5)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # F ~ cauchy around mu_f
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

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if span_ok[d]:
                    v[d] = zi[d] + F * (zpb[d] - zi[d]) + F * (zr1[d] - z_r2[d])
                else:
                    v[d] = 0.0

            # crossover
            u = zi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if (random.random() < CR) or (d == jrand):
                    u[d] = v[d]

            u = repair01(u)
            fu = eval_z(u)

            if fu <= fit[i]:
                new_pop[i] = u
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
                    bestz = u[:]
            # else keep parent

        pop, fit = new_pop, new_fit

        # memory update
        if dF:
            s = sum(dF)
            ww = [(di / s) for di in dF] if s > 0.0 else [1.0 / len(dF)] * len(dF)
            mcr = 0.0
            for wi, cri in zip(ww, SCR):
                mcr += wi * cri
            num = 0.0
            den = 0.0
            for wi, fi in zip(ww, SF):
                num += wi * fi * fi
                den += wi * fi
            mf = (num / den) if den > 1e-12 else MF[mem_pos]
            MCR[mem_pos] = mcr
            MF[mem_pos] = mf
            mem_pos = (mem_pos + 1) % H

        # ----- stall / restart -----
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
            last_bestz = bestz[:]
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0
            # keep elites, refresh rest around best or random
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(4, NP // 6)
            for idx in order[keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.70:
                    z = bestz[:]
                    rad = 0.20 if frac_left > 0.5 else 0.12
                    for d in range(dim):
                        if span_ok[d]:
                            z[d] = clamp01(z[d] + rad * gauss())
                    pop[idx] = z
                else:
                    pop[idx] = rand_z()
                fit[idx] = eval_z(pop[idx])

            # prune archive a bit
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif archive:
                drop = max(1, len(archive) // 5)
                for _ in range(drop):
                    if archive:
                        archive.pop(random.randrange(len(archive)))

            bi = min(range(NP), key=lambda i: fit[i])
            if fit[bi] < best:
                best = fit[bi]
                bestz = pop[bi][:]

        # ----- intensification schedule (late-stage SPSA trust-region) -----
        # Trigger more when (a) near end or (b) mild stall or (c) periodic.
        if time.time() < deadline and dim > 0:
            do_local = very_end or (near_end and gen % 3 == 0) or (stall >= max_stall // 2) or (gen % 9 == 0)
            if do_local:
                # local budget is modest; scales with dim; higher near end
                base_budget = 8 + 2 * dim
                if very_end:
                    base_budget = 14 + 4 * dim
                # radius shrinks over time
                base_radius = 0.18 if frac_left > 0.5 else (0.10 if frac_left > 0.2 else 0.06)
                bz, bf, _ = spsa_refine(bestz, best, base_budget, base_radius)
                if bf < best:
                    best, bestz = bf, bz[:]
                    last_best = best
                    last_bestz = bestz[:]

                # quick polish (cheap)
                if time.time() < deadline and (very_end or random.random() < 0.5):
                    pz, pf = coord_polish(bestz, best, 6 + dim)
                    if pf < best:
                        best, bestz = pf, pz[:]
                        last_best = best
                        last_bestz = bestz[:]

        # ----- tiny pattern step along last improvement direction -----
        if time.time() < deadline and dim > 0 and (near_end or gen % 7 == 0):
            delta = [bestz[d] - last_bestz[d] for d in range(dim)]
            if any(abs(v) > 1e-12 for v in delta):
                alpha = 1.5 if near_end else 1.8
                cand = [bestz[d] + alpha * delta[d] for d in range(dim)]
                cand = repair01(cand)
                fc = eval_z(cand)
                if fc < best:
                    best, bestz = fc, cand[:]
                    last_best = best

    return best
