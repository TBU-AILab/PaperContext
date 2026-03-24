import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvement over the provided best (JADE-like DE + local search):
      - Switch to a faster/stronger DE core: jDE-style *per-individual* self-adaptation
        of F and CR + mixed strategies (current-to-pbest/1 and rand/1).
        This tends to be more robust than only global mu_F/mu_CR updates.
      - Maintain an external archive (JADE) for diversity.
      - Use reflection bounds handling (keeps search directions).
      - Use a cheap but strong local refinement: (1) coordinate pattern search
        (2) adaptive Gaussian hillclimb with 1/5-success sigma control.
      - Smarter restarts: when stagnating, re-seed part of the population from:
           (a) random, (b) around global best, (c) opposition of best.
      - Optional "elite injection": best solution periodically replaces a worst.

    Returns:
      best (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_vec(x):
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            # repeated reflection (handles far outside)
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

    # Normal(0,1) Box-Muller
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def argsort(seq):
        return sorted(range(len(seq)), key=lambda i: seq[i])

    # ---------------- scale ----------------
    ranges = []
    for i in range(dim):
        lo, hi = bounds[i]
        r = hi - lo
        ranges.append(r if r != 0.0 else 1.0)

    # ---------------- initialization (quasi-LHS + opposition) ----------------
    # Slightly larger init helps in short budgets; will shrink later.
    NP_init = max(22, 10 * dim)
    NP_min  = max(10, 4 * dim)

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
    order0 = sorted(range(len(cand)), key=lambda i: cand_fit[i])
    pop = [cand[i][:] for i in order0[:NP_init]]
    fit = [cand_fit[i] for i in order0[:NP_init]]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # ---------------- DE state ----------------
    # jDE parameters (per-individual self-adaptation)
    # Typical jDE: with prob tau1 reset F, with prob tau2 reset CR
    tau1 = 0.1
    tau2 = 0.1
    Fl = 0.1
    Fu = 0.9

    F_i  = [0.5 + 0.3 * random.random() for _ in range(len(pop))]
    CR_i = [0.9 for _ in range(len(pop))]

    # External archive for diversity (JADE)
    archive = []

    # p-best fraction and strategy mixing
    p_rate = 0.15
    # probability of using current-to-pbest/1 vs rand/1 (adapts with time)
    p_cur2pbest0 = 0.75

    # Local search parameters
    base_step = [0.10 * r for r in ranges]
    base_sigma = [0.14 * r for r in ranges]
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

        # Linear population reduction with time (keeps early exploration)
        time_frac = (now - t0) / max(1e-12, max_time)
        NP_target = int(round(NP_init - (NP_init - NP_min) * time_frac))
        NP_target = max(NP_min, min(NP_init, NP_target))

        # shrink population and associated arrays
        while len(pop) > NP_target:
            w = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(w); fit.pop(w)
            F_i.pop(w); CR_i.pop(w)

        NP = len(pop)
        if NP < 6:
            # emergency rebuild around best + random
            pop = [best_x[:]]
            fit = [best]
            F_i = [0.6]
            CR_i = [0.9]
            while len(pop) < 6:
                x = rand_vec()
                pop.append(x)
                fit.append(safe_eval(x))
                F_i.append(0.5 + 0.4 * random.random())
                CR_i.append(random.random())
            NP = len(pop)

        # cap archive
        if len(archive) > NP:
            random.shuffle(archive)
            archive = archive[:NP]

        # update best & stagnation
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_x = pop[bi][:]
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # strategy schedule: more exploitation later
        p_cur2pbest = clamp(p_cur2pbest0 + 0.20 * time_frac, 0.55, 0.95)

        # order for pbest
        order = argsort(fit)
        pbest_count = max(2, int(math.ceil(p_rate * NP)))
        pbest_count = min(NP, pbest_count)

        # ---------------- stagnation handling ----------------
        if stagnation in (18, 36, 54):
            # replace a few worst with mixture around best / opposite(best) / random
            nrep = max(2, NP // 6)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            shrink = max(0.03, 1.0 - 0.90 * time_frac)
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
                        x[d] += (0.5 * sig[d]) * randn()
                    reflect_vec(x)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                F_i[idx] = 0.5 + 0.4 * random.random()
                CR_i[idx] = random.random()
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best
                    stagnation = 0

        if stagnation >= 85:
            # heavier partial restart of worst third
            nrep = max(2, NP // 3)
            worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for idx in worst_idx:
                if time.time() >= deadline:
                    return best
                x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                F_i[idx] = 0.5 + 0.4 * random.random()
                CR_i[idx] = random.random()
                if fx < best:
                    best = fx
                    best_x = x[:]
            stagnation = 0
            last_best = best

        # ---------------- DE generation (mixed strategies + jDE adaptation) ----------------
        # pool for r2 selection: pop + archive
        poolN = NP + len(archive)

        def pool_get(k):
            return pop[k] if k < NP else archive[k - NP]

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # self-adapt F and CR (jDE)
            Fi = F_i[i]
            CRi = CR_i[i]
            if random.random() < tau1:
                Fi = Fl + (Fu - Fl) * random.random()
            if random.random() < tau2:
                CRi = random.random()

            # choose mutation strategy
            use_cur2pbest = (random.random() < p_cur2pbest)

            # pick indices
            if use_cur2pbest:
                # current-to-pbest/1 with archive
                pbest_idx = order[random.randrange(pbest_count)]
                xpbest = pop[pbest_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(NP)

                if poolN < 3:
                    # fallback: random point
                    u = rand_vec()
                    fu = safe_eval(u)
                    if fu <= fi:
                        pop[i] = u; fit[i] = fu
                        F_i[i] = Fi; CR_i[i] = CRi
                        if fu < best:
                            best = fu; best_x = u[:]
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
            else:
                # DE/rand/1 from population only (diversity move)
                a = i
                while a == i:
                    a = random.randrange(NP)
                b = a
                while b == i or b == a:
                    b = random.randrange(NP)
                c = b
                while c == i or c == a or c == b:
                    c = random.randrange(NP)
                xa, xb, xc = pop[a], pop[b], pop[c]
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xa[d] + Fi * (xb[d] - xc[d])

            # crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]
            reflect_vec(u)

            fu = safe_eval(u)

            if fu <= fi:
                # success: archive parent, accept trial, keep new Fi/CRi
                archive.append(xi[:])
                if len(archive) > NP:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fit[i] = fu
                F_i[i] = Fi
                CR_i[i] = CRi

                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_best = best
                    stagnation = 0
            else:
                # failure: still store updated Fi/CRi (jDE usually does)
                F_i[i] = Fi
                CR_i[i] = CRi

        # ---------------- local refinement around best ----------------
        do_local = (gen % 3 == 0) or (stagnation >= 12 and gen % 2 == 0)
        if do_local:
            if time.time() >= deadline:
                return best

            time_frac = (time.time() - t0) / max(1e-12, max_time)
            shrink = max(0.008, 1.0 - 0.94 * time_frac)

            # 1/5-like adaptation of ls_sigma_scale
            if ls_trials >= 50:
                rate = ls_succ / float(ls_trials)
                if rate > 0.22:
                    ls_sigma_scale = min(4.0, ls_sigma_scale * 1.25)
                elif rate < 0.10:
                    ls_sigma_scale = max(0.15, ls_sigma_scale * 0.75)
                ls_trials = 0
                ls_succ = 0

            step = [base_step[d] * shrink for d in range(dim)]
            sigma = [base_sigma[d] * shrink * ls_sigma_scale for d in range(dim)]

            x = best_x[:]
            fx = best

            # (a) coordinate/pattern search
            passes = 2 if dim <= 16 else 1
            for _ in range(passes):
                improved = False
                for d in range(dim):
                    if time.time() >= deadline:
                        return best
                    lo, hi = bounds[d]
                    cur = x[d]

                    cand = x[:]
                    cand[d] = clamp(cur + step[d], lo, hi)
                    fc = safe_eval(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        continue

                    cand = x[:]
                    cand[d] = clamp(cur - step[d], lo, hi)
                    fc = safe_eval(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                if not improved:
                    break

            # (b) gaussian hillclimb
            probes = max(10, 3 * dim)
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
                w = max(range(NP), key=lambda i: fit[i])
                pop[w] = best_x[:]
                fit[w] = best
                F_i[w] = 0.6
                CR_i[w] = 0.9

        # occasional elite injection even without local improvements (anytime boost)
        if gen % 9 == 0:
            w = max(range(NP), key=lambda i: fit[i])
            if fit[w] > best:
                pop[w] = best_x[:]
                fit[w] = best
                F_i[w] = 0.6
                CR_i[w] = 0.9
