import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained; no external libs).

    Main changes vs your best (plain DE/rand/1 + Gaussian local):
      - Better initialization: quasi-LHS + opposition (stronger coverage early)
      - Stronger DE core: JADE/SHADE-like "current-to-pbest/1" (faster convergence)
      - Success-history adaptation of F and CR (stabilizes across problems)
      - External archive for diversity (prevents premature convergence)
      - Cheap bound handling with reflection (often better than simple clipping)
      - Two-mode local refinement:
          * coordinate/pattern steps (good for separable structure)
          * adaptive Gaussian hillclimb with success-rate sigma control
      - Stagnation response: inject around best + partial restart of worst

    Returns:
      best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_into_bounds(x):
        # Reflection keeps step direction info better than hard clip.
        # Works even if x goes far out of bounds.
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            span = hi - lo
            # reflect repeatedly until inside
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
            # numeric safety
            if v < lo: v = lo
            if v > hi: v = hi
            x[i] = v
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite_vec(x):
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

    # Normal(0,1) Box-Muller
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Cauchy(0,1)
    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def argsort(seq):
        return sorted(range(len(seq)), key=lambda i: seq[i])

    # ---------- problem scale ----------
    ranges = []
    for i in range(dim):
        lo, hi = bounds[i]
        r = hi - lo
        ranges.append(r if r != 0.0 else 1.0)

    # ---------- initialization (quasi-LHS + opposition) ----------
    NP_init = max(20, 10 * dim)
    NP_min = max(10, 4 * dim)

    # quasi-LHS per dimension
    strata = []
    for d in range(dim):
        lo, hi = bounds[d]
        s = []
        for k in range(NP_init):
            a = k / float(NP_init)
            b = (k + 1) / float(NP_init)
            u = a + (b - a) * random.random()
            s.append(lo + u * (hi - lo))
        random.shuffle(s)
        strata.append(s)

    init = []
    for k in range(NP_init):
        init.append([strata[d][k] for d in range(dim)])

    cand = []
    for x in init:
        cand.append(x)
        cand.append(opposite_vec(x))

    cand_fit = [safe_eval(x) for x in cand]
    order = sorted(range(len(cand)), key=lambda i: cand_fit[i])
    pop = [cand[i][:] for i in order[:NP_init]]
    fit = [cand_fit[i] for i in order[:NP_init]]

    bi = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[bi][:]
    best = fit[bi]

    # ---------- SHADE-like memories ----------
    H = 8
    M_F = [0.6] * H
    M_CR = [0.9] * H
    mem_idx = 0

    # JADE-style pbest
    p_rate = 0.15

    # archive for diversity
    archive = []

    # local search step scales
    base_step = [0.12 * r for r in ranges]
    base_sigma = [0.10 * r for r in ranges]
    ls_sigma_scale = 1.0
    ls_succ = 0
    ls_trials = 0

    stagnation = 0
    last_best = best
    gen = 0

    # ---------- main loop ----------
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # population size reduction (gentle linear)
        time_frac = (now - t0) / max(1e-12, max_time)
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        NP_target = max(NP_min, min(NP_init, NP_target))

        # shrink by removing worst
        while len(pop) > NP_target:
            w = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(w)
            fit.pop(w)

        NP = len(pop)
        if NP < 6:
            # emergency rebuild around current best
            pop = [best_x[:]]
            fit = [best]
            while len(pop) < 6:
                x = rand_vec()
                pop.append(x)
                fit.append(safe_eval(x))
            NP = len(pop)

        # cap archive
        if len(archive) > NP:
            random.shuffle(archive)
            archive = archive[:NP]

        # update incumbent
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_x = pop[bi][:]

        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # stagnation: inject around best (replace some worst)
        if stagnation in (12, 24, 36, 48):
            inject = max(2, NP // 6)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:inject]
            shrink = max(0.03, 1.0 - 0.92 * time_frac)
            sig = [base_sigma[d] * shrink for d in range(dim)]
            for idx in worst_idx:
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                for d in range(dim):
                    x[d] = x[d] + sig[d] * randn()
                reflect_into_bounds(x)
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

        # hard stagnation: partial restart of worst third
        if stagnation >= 80:
            nrep = max(2, NP // 3)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for idx in worst_idx:
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

        # ---------- DE generation: current-to-pbest/1 + archive ----------
        order = argsort(fit)
        pbest_count = max(2, int(math.ceil(p_rate * NP)))
        pbest_count = min(NP, pbest_count)

        SF, SCR, dFit = [], [], []

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # F: cauchy around muF, forced into (0,1]
            Fi = muF + 0.1 * cauchy()
            tries = 0
            while Fi <= 0.0 and tries < 12:
                Fi = muF + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.2
            if Fi > 1.0:
                Fi = 1.0

            # CR: normal around muCR, clipped
            CRi = muCR + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            pbest_idx = order[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            # r1 from pop != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 from pop+archive (prefer diversity)
            poolN = NP + len(archive)
            if poolN < 3:
                continue

            def pool_get(k):
                return pop[k] if k < NP else archive[k - NP]

            while True:
                k2 = random.randrange(poolN)
                if k2 < NP:
                    if k2 != i and k2 != r1:
                        break
                else:
                    break

            xr1 = pop[r1]
            xr2 = pool_get(k2)

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]

            reflect_into_bounds(u)
            fu = safe_eval(u)

            if fu <= fi:
                # archive replaced parent
                archive.append(xi[:])
                if len(archive) > NP:
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

        # memory update (SHADE)
        if SF:
            wsum = sum(dFit)
            weights = [df / wsum for df in dFit]

            cr_new = 0.0
            num = 0.0
            den = 0.0
            for w, fval, crval in zip(weights, SF, SCR):
                cr_new += w * crval
                num += w * (fval * fval)
                den += w * fval
            f_new = (num / den) if den > 0.0 else M_F[mem_idx]

            M_F[mem_idx] = clamp(f_new, 0.05, 0.95)
            M_CR[mem_idx] = clamp(cr_new, 0.0, 1.0)
            mem_idx = (mem_idx + 1) % H

        # ---------- Local refinement ----------
        do_local = (gen % 4 == 0) or (stagnation >= 10 and gen % 2 == 0)
        if do_local:
            if time.time() >= deadline:
                return best

            time_frac = (time.time() - t0) / max(1e-12, max_time)
            shrink = max(0.01, 1.0 - 0.93 * time_frac)

            step = [base_step[d] * shrink for d in range(dim)]
            sigma = [base_sigma[d] * shrink * ls_sigma_scale for d in range(dim)]

            x = best_x[:]
            fx = best

            # coordinate/pattern search (very cheap)
            passes = 2 if dim <= 14 else 1
            for _ in range(passes):
                improved = False
                for d in range(dim):
                    if time.time() >= deadline:
                        return best
                    lo, hi = bounds[d]
                    cur = x[d]

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

            # adaptive Gaussian hillclimb probes
            probes = max(10, 3 * dim)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                cand = x[:]
                for d in range(dim):
                    cand[d] = cand[d] + sigma[d] * randn()
                reflect_into_bounds(cand)
                fc = safe_eval(cand)

                ls_trials += 1
                if fc < fx:
                    x, fx = cand, fc
                    ls_succ += 1

            # sigma adaptation (simple success-rate control)
            if ls_trials >= 50:
                rate = ls_succ / float(ls_trials)
                if rate > 0.25:
                    ls_sigma_scale = min(3.0, ls_sigma_scale * 1.25)
                elif rate < 0.15:
                    ls_sigma_scale = max(0.25, ls_sigma_scale * 0.8)
                ls_succ = 0
                ls_trials = 0

            if fx < best:
                best = fx
                best_x = x[:]
                # inject into pop: replace worst
                w = max(range(NP), key=lambda i: fit[i])
                pop[w] = best_x[:]
                fit[w] = best
                stagnation = max(0, stagnation - 10)
