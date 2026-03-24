import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Hybrid:
      1) Opposition + quasi-LHS style initialization (better coverage)
      2) SHADE/JADE-style DE (current-to-pbest/1) with:
           - success-history memory for F and CR
           - external archive
           - population size reduction over time
      3) Two-phase local refinement on the incumbent best:
           - short coordinate pattern search
           - adaptive (1/5 success) Gaussian hillclimb
      4) Stagnation: inject around best + partial reinit of worst

    Returns:
      best (float): best objective value found within max_time seconds.
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

    # Cauchy(0,1) via tan
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

    # ---------------- initialization (coverage-focused) ----------------
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # Pop size: start larger then shrink
    NP_init = max(20, 10 * dim)
    NP_min = max(10, 4 * dim)

    # quasi-LHS: stratify each dimension across NP_init, then shuffle per dim
    strata = []
    for d in range(dim):
        lo, hi = bounds[d]
        # sample within each stratum
        s = []
        for k in range(NP_init):
            a = k / float(NP_init)
            b = (k + 1) / float(NP_init)
            u = a + (b - a) * random.random()
            s.append(lo + u * (hi - lo))
        random.shuffle(s)
        strata.append(s)

    pop = []
    for k in range(NP_init):
        x = [strata[d][k] for d in range(dim)]
        pop.append(x)

    # opposition augmentation: evaluate both, keep best NP_init
    cand = []
    for x in pop:
        cand.append(x)
        cand.append(opposite_vec(x))

    cand_fit = [safe_eval(x) for x in cand]
    order = sorted(range(len(cand)), key=lambda i: cand_fit[i])
    pop = [cand[i][:] for i in order[:NP_init]]
    fit = [cand_fit[i] for i in order[:NP_init]]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # ---------------- SHADE memory + archive ----------------
    H = 8
    M_F = [0.6] * H
    M_CR = [0.9] * H
    mem_idx = 0

    archive = []

    # exploitation control
    p_rate = 0.15  # top p% as pbest set

    # ---------------- local search params ----------------
    base_step = [0.15 * (r if r > 0 else 1.0) for r in ranges]
    base_sigma = [0.10 * (r if r > 0 else 1.0) for r in ranges]

    # 1/5 success hillclimb params (global, applied around best only)
    ls_sigma_scale = 1.0
    ls_success = 0
    ls_trials = 0

    stagnation = 0
    last_best = best

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # ---- Linear population size reduction ----
        time_frac = (now - t0) / max(1e-12, max_time)
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        if NP_target < NP_min:
            NP_target = NP_min

        # remove worst until target size
        while len(pop) > NP_target:
            w = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(w)
            fit.pop(w)

        NP = len(pop)

        # archive cap
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

        # ---- Stagnation actions ----
        # inject a few samples near best (often very effective)
        if stagnation in (15, 30, 45):
            inject = max(2, NP // 6)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:inject]
            shrink = max(0.05, 1.0 - 0.9 * time_frac)
            sig = [s * shrink for s in base_sigma]
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

        # partial restart if very stuck
        if stagnation >= 70:
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

        # ---- DE generation (current-to-pbest/1 with archive) ----
        order = argsort(fit)
        pbest_count = max(2, int(math.ceil(p_rate * NP)))
        if pbest_count > NP:
            pbest_count = NP

        SF, SCR, dFit = [], [], []

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick memory slot
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample F ~ cauchy(muF,0.1) -> (0,1]
            Fi = muF + 0.1 * cauchy()
            tries = 0
            while Fi <= 0.0 and tries < 12:
                Fi = muF + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.2
            if Fi > 1.0:
                Fi = 1.0

            # sample CR ~ N(muCR,0.1) clipped
            CRi = muCR + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # choose pbest from top set
            pbest_idx = order[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            # r1 from population != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 from population+archive not colliding too much
            poolN = NP + len(archive)
            if poolN < 3:
                # fallback random
                u = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
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
            u = clip_vec(u)
            fu = safe_eval(u)

            if fu <= fi:
                # archive old
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

        # ---- Local refinement (cheap, but frequent late) ----
        # more often later, or when stagnating
        do_local = (gen % 4 == 0) or (stagnation >= 10 and gen % 2 == 0)
        if do_local:
            if time.time() >= deadline:
                return best

            time_frac = (time.time() - t0) / max(1e-12, max_time)
            shrink = max(0.01, 1.0 - 0.93 * time_frac)

            step = [s * shrink for s in base_step]
            sigma = [s * shrink * ls_sigma_scale for s in base_sigma]

            x = best_x[:]
            fx = best

            # short coordinate pattern search (good for separable structure)
            passes = 2 if dim <= 14 else 1
            for _ in range(passes):
                changed = False
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
                        changed = True
                        continue

                    cand = x[:]
                    cand[d] = clamp(cur - step[d], lo, hi)
                    f2 = safe_eval(cand)
                    if f2 < fx:
                        x, fx = cand, f2
                        changed = True
                if not changed:
                    break

            # adaptive gaussian hillclimb around x (1/5 success rule)
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
                    ls_success += 1

            # update sigma scale occasionally
            if ls_trials >= 40:
                rate = ls_success / float(ls_trials)
                # if too successful, enlarge; if failing, shrink
                if rate > 0.25:
                    ls_sigma_scale = min(3.0, ls_sigma_scale * 1.25)
                elif rate < 0.15:
                    ls_sigma_scale = max(0.25, ls_sigma_scale * 0.8)
                ls_success = 0
                ls_trials = 0

            if fx < best:
                best = fx
                best_x = x[:]
                # inject into population: replace worst
                w = max(range(NP), key=lambda i: fit[i])
                pop[w] = best_x[:]
                fit[w] = best
                # also reduce stagnation a bit
                stagnation = max(0, stagnation - 10)
