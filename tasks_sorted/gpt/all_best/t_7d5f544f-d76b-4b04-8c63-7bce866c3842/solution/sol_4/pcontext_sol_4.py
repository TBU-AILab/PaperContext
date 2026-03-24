import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Hybrid approach tuned for strong anytime performance:
      1) Coverage-focused init: quasi-LHS + opposition, keep best half
      2) SHADE/JADE-style DE (current-to-pbest/1) with:
           - success-history memories for F and CR
           - external archive for diversity
           - mild population size reduction over time
      3) Fast local improvement on the incumbent best:
           - coordinate pattern search (very cheap)
           - adaptive Gaussian hillclimb (1/5 success-ish sigma adaptation)
      4) Stagnation handling: re-seed worst around best + occasional random reset

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # -------- helpers --------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def clip_vec(v):
        out = v[:]  # list
        for i in range(dim):
            lo, hi = bounds[i]
            if out[i] < lo:
                out[i] = lo
            elif out[i] > hi:
                out[i] = hi
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

    # Normal(0,1): Box-Muller
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Cauchy(0,1)
    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def opposite_vec(x):
        out = []
        for i in range(dim):
            lo, hi = bounds[i]
            out.append(clamp(lo + hi - x[i], lo, hi))
        return out

    def argsort(seq):
        return sorted(range(len(seq)), key=lambda i: seq[i])

    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    ranges_safe = [(r if r > 0.0 else 1.0) for r in ranges]

    # -------- parameters (robust defaults) --------
    NP_init = max(20, 10 * dim)
    NP_min = max(10, 4 * dim)

    H = 8                      # SHADE memory size
    M_F = [0.6] * H
    M_CR = [0.9] * H
    mem_idx = 0

    p_rate = 0.15              # pbest fraction
    archive = []

    # local search step scales
    base_step = [0.12 * r for r in ranges_safe]
    base_sigma = [0.10 * r for r in ranges_safe]
    ls_sigma_scale = 1.0
    ls_succ = 0
    ls_trials = 0

    # -------- initialization: quasi-LHS + opposition --------
    # quasi-LHS: stratify each dim then shuffle
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

    pop0 = []
    for k in range(NP_init):
        pop0.append([strata[d][k] for d in range(dim)])

    cand = []
    for x in pop0:
        cand.append(x)
        cand.append(opposite_vec(x))

    cand_fit = [safe_eval(x) for x in cand]
    order = sorted(range(len(cand)), key=lambda i: cand_fit[i])
    pop = [cand[i][:] for i in order[:NP_init]]
    fit = [cand_fit[i] for i in order[:NP_init]]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    stagnation = 0
    last_best = best
    gen = 0

    # -------- main loop --------
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # population size reduction (gentle)
        time_frac = (now - t0) / max(1e-12, max_time)
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        NP_target = max(NP_min, min(NP_init, NP_target))

        # shrink by removing worst
        while len(pop) > NP_target:
            w = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(w)
            fit.pop(w)

        NP = len(pop)
        if NP < 6:  # safety
            # rebuild small pop around best
            pop = [best_x[:]]
            fit = [best]
            while len(pop) < 6:
                x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
                pop.append(x)
                fit.append(safe_eval(x))
            NP = len(pop)

        # cap archive
        if len(archive) > NP:
            random.shuffle(archive)
            archive = archive[:NP]

        # update best
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_x = pop[bi][:]

        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # stagnation actions: replace a few worst with samples around best
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
                    lo, hi = bounds[d]
                    x[d] = clamp(x[d] + sig[d] * randn(), lo, hi)
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

        if stagnation >= 75:
            # partial random restart of worst third
            nrep = max(2, NP // 3)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for idx in worst_idx:
                if time.time() >= deadline:
                    return best
                x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
            stagnation = 0

        # -------- DE generation: current-to-pbest/1 + archive, SHADE memories --------
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

            # Fi: cauchy around muF
            Fi = muF + 0.1 * cauchy()
            tries = 0
            while Fi <= 0.0 and tries < 12:
                Fi = muF + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.2
            if Fi > 1.0:
                Fi = 1.0

            # CRi: normal around muCR
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

            # r2 from pop+archive, not equal to i/r1 if in pop
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

            # mutation
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]
            u = clip_vec(u)

            fu = safe_eval(u)

            if fu <= fi:
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

        # memory update
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

        # -------- Local search (frequent late, or when stagnating) --------
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

            # coordinate pattern search (accept-improve)
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
            probes = max(10, 3 * dim)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                cand = x[:]
                for d in range(dim):
                    cand[d] = cand[d] + sigma[d] * randn()
                cand = clip_vec(cand)
                fc = safe_eval(cand)

                ls_trials += 1
                if fc < fx:
                    x, fx = cand, fc
                    ls_succ += 1

            # adapt sigma scale occasionally (simple success-rate control)
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
                # inject into population by replacing worst
                w = max(range(NP), key=lambda i: fit[i])
                pop[w] = best_x[:]
                fit[w] = best
                stagnation = max(0, stagnation - 10)
