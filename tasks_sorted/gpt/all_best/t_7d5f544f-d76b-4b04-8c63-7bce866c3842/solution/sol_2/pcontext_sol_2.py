import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libraries):
      - Shade-style Differential Evolution (current-to-pbest/1) with:
          * success-history adaptation of F and CR (memory)
          * external archive (diversity injection)
          * linear population size reduction (LPSR)
      - Oppositional initialization (+ occasional opposition injection)
      - Lightweight local refinement on the incumbent best (coordinate + gaussian)
      - Stagnation handling (partial restart of worst individuals)

    Returns: best (float) found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def clip_vec(v):
        out = v[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if out[i] < lo:
                out[i] = lo
            elif out[i] > hi:
                out[i] = hi
        return out

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite_vec(x):
        # classic opposition point in bounds: x' = lo + hi - x
        out = []
        for i in range(dim):
            lo, hi = bounds[i]
            out.append(clamp(lo + hi - x[i], lo, hi))
        return out

    def safe_eval(x):
        try:
            y = func(x)
            if y is None:
                return float("inf")
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf")
            return y
        except Exception:
            return float("inf")

    # Normal(0,1) via Box-Muller
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Cauchy(0,1) via tan
    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    def argsort(seq):
        return sorted(range(len(seq)), key=lambda i: seq[i])

    # ---------------- settings ----------------
    # Start larger, shrink over time (LPSR). Keep bounded for time-limited runs.
    NP_init = max(18, 10 * dim)
    NP_min = max(8, 4 * dim)

    # SHADE memory size
    H = 6
    M_F = [0.6] * H
    M_CR = [0.9] * H
    k_mem = 0

    # External archive
    archive = []
    # will cap to current NP dynamically

    # p-best selection rate
    p_rate = 0.20

    # local search configuration
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_step = [0.18 * (r if r > 0 else 1.0) for r in ranges]
    base_sigma = [0.12 * (r if r > 0 else 1.0) for r in ranges]

    # stagnation handling
    stagnation = 0
    last_best = float("inf")

    # ---------------- initialization (oppositional) ----------------
    pop = []
    fit = []

    # Build 2*NP_init candidates (x and opposite), keep best NP_init
    cand = []
    for _ in range(NP_init):
        x = rand_vec()
        xo = opposite_vec(x)
        cand.append(x)
        cand.append(xo)

    cand_fit = [safe_eval(x) for x in cand]
    order = sorted(range(len(cand)), key=lambda i: cand_fit[i])
    for i in order[:NP_init]:
        pop.append(cand[i])
        fit.append(cand_fit[i])

    best_idx = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    last_best = best

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # Linear population size reduction target
        time_frac = (now - t0) / max(1e-12, max_time)
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        if NP_target < NP_min:
            NP_target = NP_min

        # Reduce population by removing worst if needed
        while len(pop) > NP_target:
            worst = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(worst)
            fit.pop(worst)

        NP = len(pop)
        if NP < 4:
            # emergency: rebuild
            pop = [rand_vec() for _ in range(max(4, NP_min))]
            fit = [safe_eval(x) for x in pop]
            NP = len(pop)

        # archive cap follows population
        archive_max = NP
        if len(archive) > archive_max:
            random.shuffle(archive)
            archive = archive[:archive_max]

        order = argsort(fit)
        best_idx = order[0]
        if fit[best_idx] < best:
            best = fit[best_idx]
            best_x = pop[best_idx][:]

        # stagnation measure
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # Occasional opposition injection when stuck
        if stagnation in (18, 30, 45) and time.time() < deadline:
            # replace a few worst with opposite of current best + noise
            nrep = max(1, NP // 5)
            worst_indices = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for idx in worst_indices:
                if time.time() >= deadline:
                    return best
                xo = opposite_vec(best_x)
                # small jitter to avoid exact symmetry traps
                for d in range(dim):
                    lo, hi = bounds[d]
                    xo[d] = clamp(xo[d] + 0.02 * (hi - lo) * randn(), lo, hi)
                fo = safe_eval(xo)
                pop[idx] = xo
                fit[idx] = fo
                if fo < best:
                    best = fo
                    best_x = xo[:]

        # Partial restart of worst individuals if very stuck
        if stagnation >= 60 and time.time() < deadline:
            nrep = max(2, NP // 3)
            worst_indices = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for idx in worst_indices:
                if time.time() >= deadline:
                    return best
                x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
            stagnation = 0

        # ---------------- SHADE-like DE generation ----------------
        SF, SCR, dFit = [], [], []

        # top p% for pbest
        pbest_count = max(2, int(math.ceil(p_rate * NP)))
        if pbest_count > NP:
            pbest_count = NP

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # choose memory index r
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample Fi from Cauchy(muF, 0.1) until (0,1]
            Fi = muF + 0.1 * cauchy()
            tries = 0
            while (Fi <= 0.0) and tries < 12:
                Fi = muF + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.2
            if Fi > 1.0:
                Fi = 1.0

            # sample CRi from Normal(muCR, 0.1), clip [0,1]
            CRi = muCR + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # choose pbest from top set
            pbest_idx = order[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            # select r1 from population, distinct from i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # select r2 from population+archive, distinct from i and r1 if in pop
            pool_size = NP + len(archive)
            if pool_size < 3:
                # fallback random trial
                u = rand_vec()
                fu = safe_eval(u)
                if fu <= fi:
                    archive.append(xi[:])
                    pop[i] = u
                    fit[i] = fu
                    if fu < best:
                        best, best_x = fu, u[:]
                continue

            def pool_get(k):
                if k < NP:
                    return pop[k]
                return archive[k - NP]

            while True:
                k2 = random.randrange(pool_size)
                if k2 < NP:
                    if k2 != i and k2 != r1:
                        break
                else:
                    break
            xr1 = pop[r1]
            xr2 = pool_get(k2)

            # current-to-pbest/1 mutation
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]
            u = clip_vec(u)
            fu = safe_eval(u)

            if fu <= fi:
                # archive old
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fit[i] = fu

                imp = fi - fu
                if imp < 0.0:
                    imp = 0.0
                SF.append(Fi)
                SCR.append(CRi)
                dFit.append(imp + 1e-12)

                if fu < best:
                    best = fu
                    best_x = u[:]

        # update memories if there were successes
        if SF:
            wsum = sum(dFit)
            weights = [df / wsum for df in dFit]

            # weighted arithmetic mean for CR, weighted Lehmer mean for F
            cr_new = 0.0
            num = 0.0
            den = 0.0
            for w, fval, crval in zip(weights, SF, SCR):
                cr_new += w * crval
                num += w * (fval * fval)
                den += w * fval

            f_new = (num / den) if den > 0.0 else M_F[k_mem]

            M_F[k_mem] = clamp(f_new, 0.05, 0.95)
            M_CR[k_mem] = clamp(cr_new, 0.0, 1.0)
            k_mem = (k_mem + 1) % H

        # ---------------- local refinement around best ----------------
        # do more late in the run; cheap and effective for many problems
        if (gen % 4 == 0) or (stagnation > 10 and gen % 2 == 0):
            if time.time() >= deadline:
                return best

            now = time.time()
            time_frac = (now - t0) / max(1e-12, max_time)
            shrink = max(0.01, 1.0 - 0.92 * time_frac)

            step = [s * shrink for s in base_step]
            sigma = [s * shrink for s in base_sigma]

            x = best_x[:]
            fx = best

            # coordinate search (1-2 passes)
            passes = 2 if dim <= 12 else 1
            for _ in range(passes):
                improved = False
                for d in range(dim):
                    if time.time() >= deadline:
                        return best
                    lo, hi = bounds[d]
                    cur = x[d]
                    # try both directions; accept first improvement
                    cand = x[:]
                    cand[d] = clamp(cur + step[d], lo, hi)
                    f1 = safe_eval(cand)
                    if f1 < fx:
                        x, fx = cand, f1
                        improved = True
                        continue
                    cand = x[:]
                    cand[d] = clamp(cur - step[d], lo, hi)
                    f2 = safe_eval(cand)
                    if f2 < fx:
                        x, fx = cand, f2
                        improved = True
                if not improved:
                    break

            # gaussian probes around refined x
            probes = max(8, 2 * dim)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                cand = x[:]
                for d in range(dim):
                    cand[d] = cand[d] + sigma[d] * randn()
                cand = clip_vec(cand)
                fc = safe_eval(cand)
                if fc < fx:
                    x, fx = cand, fc

            if fx < best:
                best = fx
                best_x = x[:]
                # inject into population by replacing the worst
                worst = max(range(NP), key=lambda i: fit[i])
                pop[worst] = best_x[:]
                fit[worst] = best

    # (unreachable)
    # return best
