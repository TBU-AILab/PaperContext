import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Upgrades vs the best (DE/rand/1 + Gaussian):
      - Better init: quasi-LHS + opposition selection (fast early wins)
      - Stronger DE: current-to-pbest/1 (JADE-style) with success adaptation of F/CR
      - External archive for diversity (reduces premature convergence)
      - Better bound handling: reflection (often superior to clipping)
      - Budget-aware: light, frequent local search early; stronger late
      - Stagnation control: inject near best + partial restart of worst

    Returns: best (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_vec(x):
        # Reflect into bounds (handles far-out values by repeated reflection).
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
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

    # ---------------- scale ----------------
    ranges = []
    for i in range(dim):
        lo, hi = bounds[i]
        r = hi - lo
        ranges.append(r if r != 0.0 else 1.0)

    # ---------------- initialization: quasi-LHS + opposition ----------------
    NP_init = max(18, 8 * dim)
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

    init = [[strata[d][k] for d in range(dim)] for k in range(NP_init)]

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

    # ---------------- JADE-like adaptation + archive ----------------
    mu_F = 0.6
    mu_CR = 0.9
    c_adapt = 0.10          # smoothing
    p_rate = 0.15           # p-best fraction
    archive = []

    # local search scales
    base_sigma = [0.18 * r for r in ranges]
    base_step = [0.10 * r for r in ranges]
    ls_sigma_scale = 1.0
    ls_succ = 0
    ls_trials = 0

    stagnation = 0
    last_best = best
    gen = 0

    # ---------------- main loop ----------------
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # gentle pop-size reduction with time
        time_frac = (now - t0) / max(1e-12, max_time)
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        NP_target = max(NP_min, min(NP_init, NP_target))

        while len(pop) > NP_target:
            w = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(w)
            fit.pop(w)

        NP = len(pop)
        if NP < 6:
            # emergency rebuild
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

        # incumbent update
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_x = pop[bi][:]

        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # stagnation: inject around best + partial restart worst
        if stagnation in (12, 24, 36):
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
                reflect_vec(x)
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

        if stagnation >= 70:
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

        # ---------------- DE: current-to-pbest/1 + archive ----------------
        order = argsort(fit)
        pbest_count = max(2, int(math.ceil(p_rate * NP)))
        pbest_count = min(NP, pbest_count)

        SF, SCR, dFit = [], [], []
        improved_any = False

        poolN = NP + len(archive)

        def pool_get(k):
            return pop[k] if k < NP else archive[k - NP]

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # CR ~ N(mu_CR, 0.1) clipped
            CRi = mu_CR + 0.1 * randn()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # F ~ Cauchy(mu_F, 0.1) -> (0,1]
            Fi = mu_F + 0.1 * cauchy()
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 12:
                Fi = mu_F + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.3
            if Fi > 1.0:
                Fi = 1.0

            pbest_idx = order[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            if poolN < 3:
                # fallback random trial
                u = rand_vec()
                fu = safe_eval(u)
                if fu <= fi:
                    pop[i] = u
                    fit[i] = fu
                    if fu < best:
                        best = fu
                        best_x = u[:]
                continue

            # pick r2 from pop+archive, avoid i and r1 if from pop
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

            reflect_vec(u)
            fu = safe_eval(u)

            if fu <= fi:
                improved_any = True
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

        # adapt mu_F and mu_CR (JADE-style)
        if SF:
            wsum = sum(dFit)
            weights = [df / wsum for df in dFit]

            cr_mean = 0.0
            num = 0.0
            den = 0.0
            for w, fval, crval in zip(weights, SF, SCR):
                cr_mean += w * crval
                num += w * (fval * fval)   # Lehmer numerator
                den += w * fval            # Lehmer denominator
            lehmerF = (num / den) if den > 0.0 else mu_F

            mu_F = clamp((1.0 - c_adapt) * mu_F + c_adapt * lehmerF, 0.05, 0.95)
            mu_CR = clamp((1.0 - c_adapt) * mu_CR + c_adapt * cr_mean, 0.0, 1.0)
        else:
            # if nothing succeeds, nudge toward exploration slightly
            if stagnation % 10 == 0:
                mu_F = clamp(mu_F + 0.03, 0.05, 0.95)
                mu_CR = clamp(mu_CR * 0.97, 0.05, 1.0)

        # ---------------- local refinement around best ----------------
        # Do it more often later, or when stagnating.
        do_local = (gen % 3 == 0) or (stagnation >= 10 and gen % 2 == 0)
        if do_local:
            if time.time() >= deadline:
                return best

            time_frac = (time.time() - t0) / max(1e-12, max_time)
            shrink = max(0.01, 1.0 - 0.93 * time_frac)

            # adapt gaussian scale by success rate
            if ls_trials >= 40:
                rate = ls_succ / float(ls_trials)
                if rate > 0.25:
                    ls_sigma_scale = min(3.0, ls_sigma_scale * 1.25)
                elif rate < 0.12:
                    ls_sigma_scale = max(0.25, ls_sigma_scale * 0.80)
                ls_succ = 0
                ls_trials = 0

            sigma = [base_sigma[d] * shrink * ls_sigma_scale for d in range(dim)]
            step = [base_step[d] * shrink for d in range(dim)]

            x = best_x[:]
            fx = best

            # cheap coordinate pattern search (1-2 passes)
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

            # gaussian hillclimb probes
            probes = max(12, 3 * dim)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                cand = x[:]
                for d in range(dim):
                    cand[d] = cand[d] + sigma[d] * randn()
                reflect_vec(cand)
                fc = safe_eval(cand)

                ls_trials += 1
                if fc < fx:
                    x, fx = cand, fc
                    ls_succ += 1

            if fx < best:
                best = fx
                best_x = x[:]
                # inject improved best into population (replace worst)
                w = max(range(NP), key=lambda i: fit[i])
                pop[w] = best_x[:]
                fit[w] = best
                stagnation = max(0, stagnation - 10)
