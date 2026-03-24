import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements over your current best JADE-like DE:
      - Switch DE core to SHADE (success-history memories for F/CR) which is often
        more stable than single mu_F/mu_CR, especially across diverse landscapes.
      - Keep JADE's current-to-pbest/1 + archive, but use:
          * "p" schedule (more exploitation later)
          * linear population size reduction (L-SHADE style)
      - Add a cheap *surrogate directional* local step: use the best of a few
        one-sided coordinate probes to form a descent direction, then do a short
        backtracking line search along it (very effective per evaluation).
      - Keep adaptive Gaussian local search with 1/5-success control, but make it
        budget-aware and triggered by stagnation/time.
      - Stronger stagnation actions: (a) re-seed worst around best with shrinking
        radius, (b) occasional opposition+random mix.

    Returns:
      best (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_vec(x):
        # robust repeated reflection
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                else:
                    v = hi - (v - hi)
            if v < lo: v = lo
            if v > hi: v = hi
            x[i] = v
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite_vec(x):
        o = []
        for i in range(dim):
            lo, hi = bounds[i]
            o.append(clamp(lo + hi - x[i], lo, hi))
        return o

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

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

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
    NP_init = max(24, 10 * dim)
    NP_min = max(10, 4 * dim)

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
    order0 = sorted(range(len(cand)), key=lambda i: cand_fit[i])
    pop = [cand[i][:] for i in order0[:NP_init]]
    fit = [cand_fit[i] for i in order0[:NP_init]]

    bi0 = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[bi0][:]
    best = fit[bi0]

    # ---------------- SHADE memories + archive ----------------
    H = 8
    M_F = [0.6] * H
    M_CR = [0.9] * H
    k_mem = 0

    archive = []

    # local search scales
    base_sigma = [0.16 * r for r in ranges]
    base_step  = [0.08 * r for r in ranges]

    # adaptive gaussian local search
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

        time_frac = (now - t0) / max(1e-12, max_time)

        # L-SHADE-like population size reduction
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        NP_target = max(NP_min, min(NP_init, NP_target))

        while len(pop) > NP_target:
            w = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(w); fit.pop(w)

        NP = len(pop)
        if NP < 6:
            pop = [best_x[:]]
            fit = [best]
            while len(pop) < 6:
                x = rand_vec()
                pop.append(x)
                fit.append(safe_eval(x))
            NP = len(pop)

        # cap archive to NP
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

        # p-best schedule: smaller p later => stronger exploitation
        p_rate = 0.25 - 0.15 * time_frac
        if p_rate < 0.08:
            p_rate = 0.08

        order = argsort(fit)
        pbest_count = max(2, int(math.ceil(p_rate * NP)))
        if pbest_count > NP:
            pbest_count = NP

        # ---------------- stagnation actions ----------------
        if stagnation in (14, 28, 42):
            # reseed a few worst around best / opposite(best) / random
            nrep = max(2, NP // 6)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            shrink = max(0.02, 1.0 - 0.90 * time_frac)
            sig = [base_sigma[d] * shrink for d in range(dim)]
            for t, idx in enumerate(worst_idx):
                if time.time() >= deadline:
                    return best
                mode = t % 3
                if mode == 0:
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += sig[d] * randn()
                    reflect_vec(x)
                elif mode == 1:
                    x = opposite_vec(best_x)
                    for d in range(dim):
                        x[d] += 0.5 * sig[d] * randn()
                    reflect_vec(x)
                else:
                    x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best
                    stagnation = 0

        if stagnation >= 80:
            # partial restart of worst third
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
                    last_best = best
            stagnation = 0

        # ---------------- DE generation: current-to-pbest/1 + archive (SHADE memories) ----------------
        SF, SCR, dFit = [], [], []
        poolN = NP + len(archive)

        def pool_get(k):
            return pop[k] if k < NP else archive[k - NP]

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # Fi ~ Cauchy(muF,0.1) -> (0,1]
            Fi = muF + 0.1 * cauchy()
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 12:
                Fi = muF + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0: Fi = 0.25
            if Fi > 1.0:  Fi = 1.0

            # CRi ~ N(muCR,0.1) clipped
            CRi = muCR + 0.1 * randn()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            pbest_idx = order[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            if poolN < 3:
                # fallback
                u = rand_vec()
                fu = safe_eval(u)
                if fu <= fi:
                    pop[i] = u
                    fit[i] = fu
                    if fu < best:
                        best = fu
                        best_x = u[:]
                continue

            while True:
                k2 = random.randrange(poolN)
                if k2 < NP:
                    if k2 != i and k2 != r1:
                        break
                else:
                    break

            xr1 = pop[r1]
            xr2 = pool_get(k2)

            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]
            reflect_vec(u)

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
                    last_best = best
                    stagnation = 0

        # SHADE memory update
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
            f_new = (num / den) if den > 0.0 else M_F[k_mem]

            M_F[k_mem] = clamp(f_new, 0.05, 0.95)
            M_CR[k_mem] = clamp(cr_new, 0.0, 1.0)
            k_mem = (k_mem + 1) % H

        # ---------------- local refinement around best ----------------
        # A) directional coordinate-probe step + backtracking (very eval-efficient)
        # B) a few Gaussian probes with 1/5 success adaptation
        do_local = (gen % 3 == 0) or (stagnation >= 10 and gen % 2 == 0) or (time_frac > 0.65)
        if do_local:
            if time.time() >= deadline:
                return best

            now2 = time.time()
            time_frac2 = (now2 - t0) / max(1e-12, max_time)
            shrink = max(0.006, 1.0 - 0.95 * time_frac2)

            step = [base_step[d] * shrink for d in range(dim)]
            sigma = [base_sigma[d] * shrink * ls_sigma_scale for d in range(dim)]

            x = best_x[:]
            fx = best

            # (A) build a direction from best one-sided coordinate improvements
            # probe only a subset of dims if very high-dimensional
            max_probe_dims = dim if dim <= 24 else max(24, dim // 3)
            dims = list(range(dim))
            if dim > max_probe_dims:
                random.shuffle(dims)
                dims = dims[:max_probe_dims]

            direction = [0.0] * dim
            improved_any = False
            for d in dims:
                if time.time() >= deadline:
                    return best
                lo, hi = bounds[d]
                cur = x[d]

                # try + and -
                candp = x[:]
                candp[d] = clamp(cur + step[d], lo, hi)
                fp = safe_eval(candp)

                candm = x[:]
                candm[d] = clamp(cur - step[d], lo, hi)
                fm = safe_eval(candm)

                if fp < fx and fp <= fm:
                    direction[d] = candp[d] - cur
                    x = candp
                    fx = fp
                    improved_any = True
                elif fm < fx:
                    direction[d] = candm[d] - cur
                    x = candm
                    fx = fm
                    improved_any = True

            # backtracking line-search along direction (if we got a meaningful direction)
            if improved_any:
                if time.time() >= deadline:
                    return best
                # normalize direction scale a bit
                norm = 0.0
                for d in range(dim):
                    norm += (direction[d] / (ranges[d] if ranges[d] != 0 else 1.0)) ** 2
                if norm > 1e-18:
                    alpha = 1.0
                    for _ in range(6):
                        if time.time() >= deadline:
                            return best
                        cand = x[:]
                        for d in range(dim):
                            cand[d] += alpha * direction[d]
                        reflect_vec(cand)
                        fc = safe_eval(cand)
                        if fc < fx:
                            x, fx = cand, fc
                            alpha *= 1.35
                        else:
                            alpha *= 0.5
                            if alpha < 0.05:
                                break

            # (B) gaussian probes
            if ls_trials >= 50:
                rate = ls_succ / float(ls_trials)
                if rate > 0.22:
                    ls_sigma_scale = min(3.5, ls_sigma_scale * 1.25)
                elif rate < 0.10:
                    ls_sigma_scale = max(0.18, ls_sigma_scale * 0.75)
                ls_trials = 0
                ls_succ = 0

            probes = max(10, 2 * dim) if time_frac2 < 0.7 else max(12, 3 * dim)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                cand = x[:]
                for d in range(dim):
                    cand[d] += sigma[d] * randn()
                reflect_vec(cand)
                fc = safe_eval(cand)
                ls_trials += 1
                if fc < fx:
                    x, fx = cand, fc
                    ls_succ += 1

            if fx < best:
                best = fx
                best_x = x[:]
                last_best = best
                stagnation = 0
                # inject into population: replace worst
                w = max(range(NP), key=lambda j: fit[j])
                pop[w] = best_x[:]
                fit[w] = best

        # occasional elite injection (helps anytime)
        if gen % 9 == 0:
            w = max(range(NP), key=lambda j: fit[j])
            if fit[w] > best:
                pop[w] = best_x[:]
                fit[w] = best
