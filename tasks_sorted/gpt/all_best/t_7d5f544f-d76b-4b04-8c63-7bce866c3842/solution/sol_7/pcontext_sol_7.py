import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements over the provided best (JADE-like DE + local search):
      - Multi-start / island model: several small evolving islands + periodic migration
        (more robust under multimodality and noisy adaptation)
      - Mixed mutation strategies per individual (current-to-pbest/1, rand/1, best/1)
        chosen adaptively by success
      - Better boundary handling: reflection + occasional randomized repair
      - More disciplined local search: (a) short coordinate search, (b) small
        trust-region Gaussian with 1/5-style adaptation, (c) rare bigger "kick"
      - Stronger stagnation responses per-island and global
    Returns:
      best (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_vec(x):
        # Reflect into bounds (repeated reflection if far outside).
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
            # numeric safety
            if v < lo: v = lo
            if v > hi: v = hi
            x[i] = v
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

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

    def opposite_vec(x):
        out = []
        for i in range(dim):
            lo, hi = bounds[i]
            out.append(clamp(lo + hi - x[i], lo, hi))
        return out

    # ---------------- scale ----------------
    ranges = []
    for i in range(dim):
        lo, hi = bounds[i]
        r = hi - lo
        ranges.append(r if r != 0.0 else 1.0)

    # ---------------- parameters ----------------
    # Islands: more islands for larger dim, but keep cheap.
    islands = 3 if dim <= 12 else 4
    islands = min(6, max(2, islands))

    # Total pop budget scales with dim; split across islands.
    NP_total_init = max(24, 10 * dim)
    NP_total_min = max(12, 5 * dim)

    # Ensure each island has enough members
    per_island_init = max(8, NP_total_init // islands)
    per_island_min = max(6, NP_total_min // islands)

    # pbest rate
    p_rate = 0.15

    # JADE-like global means (kept per island for robustness)
    mu_F0 = 0.6
    mu_CR0 = 0.9
    c_adapt = 0.12

    # Local search base scales (trust-region style)
    base_sigma = [0.14 * r for r in ranges]
    base_step = [0.08 * r for r in ranges]

    # Migration
    migrate_every = 7  # generations
    migrate_count = 2  # individuals moved from best islands to others

    # Strategy mixture: indices 0..2
    # 0: current-to-pbest/1, 1: rand/1, 2: best/1
    strat_probs0 = [0.60, 0.25, 0.15]

    # ---------------- initialization: quasi-LHS + opposition per island ----------------
    # Build a global quasi-LHS pool, then distribute round-robin into islands.
    strata = []
    for d in range(dim):
        lo, hi = bounds[d]
        s = []
        for k in range(per_island_init * islands):
            a = k / float(per_island_init * islands)
            b = (k + 1) / float(per_island_init * islands)
            u = a + (b - a) * random.random()
            s.append(lo + u * (hi - lo))
        random.shuffle(s)
        strata.append(s)

    init_pool = [[strata[d][k] for d in range(dim)] for k in range(per_island_init * islands)]
    # opposition augmentation then keep best half for each island seed pool
    init_cand = []
    for x in init_pool:
        init_cand.append(x)
        init_cand.append(opposite_vec(x))
    init_fit = [safe_eval(x) for x in init_cand]
    init_order = sorted(range(len(init_cand)), key=lambda i: init_fit[i])

    # Start islands with best points but interleaved for diversity
    islands_pop = []
    islands_fit = []
    ptr = 0
    for k in range(islands):
        pop = []
        fit = []
        for _ in range(per_island_init):
            idx = init_order[ptr]
            ptr += 1
            pop.append(init_cand[idx][:])
            fit.append(init_fit[idx])
        islands_pop.append(pop)
        islands_fit.append(fit)

    # per-island state
    mu_F = [mu_F0 for _ in range(islands)]
    mu_CR = [mu_CR0 for _ in range(islands)]
    archive = [[] for _ in range(islands)]

    strat_probs = [strat_probs0[:] for _ in range(islands)]
    strat_succ = [[1.0, 1.0, 1.0] for _ in range(islands)]  # pseudo-counts
    strat_fail = [[1.0, 1.0, 1.0] for _ in range(islands)]

    # local search adaptation (global, around global best)
    ls_sigma_scale = 1.0
    ls_succ = 0
    ls_trials = 0

    # Global best
    best = float("inf")
    best_x = None
    for k in range(islands):
        bi = min(range(len(islands_pop[k])), key=lambda i: islands_fit[k][i])
        if islands_fit[k][bi] < best:
            best = islands_fit[k][bi]
            best_x = islands_pop[k][bi][:]

    stagnation_global = 0
    last_best = best
    gen = 0

    # ---------------- main loop ----------------
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # time-based linear pop reduction (applied per island)
        time_frac = (now - t0) / max(1e-12, max_time)
        per_target = int(round(per_island_init - (per_island_init - per_island_min) * time_frac))
        per_target = max(per_island_min, min(per_island_init, per_target))

        # Update global stagnation
        if best < last_best - 1e-12:
            stagnation_global = 0
            last_best = best
        else:
            stagnation_global += 1

        # Per-island evolution
        for k in range(islands):
            if time.time() >= deadline:
                return best

            pop = islands_pop[k]
            fit = islands_fit[k]

            # shrink island to target by removing worst
            while len(pop) > per_target:
                w = max(range(len(pop)), key=lambda i: fit[i])
                pop.pop(w); fit.pop(w)

            NP = len(pop)
            if NP < 6:
                # emergency rebuild around global best
                pop = [best_x[:]]
                fit = [best]
                while len(pop) < 6:
                    x = rand_vec()
                    pop.append(x)
                    fit.append(safe_eval(x))
                islands_pop[k] = pop
                islands_fit[k] = fit
                NP = len(pop)

            # cap archive to NP
            if len(archive[k]) > NP:
                random.shuffle(archive[k])
                archive[k] = archive[k][:NP]

            # update island best and global best
            bi = min(range(NP), key=lambda i: fit[i])
            if fit[bi] < best:
                best = fit[bi]
                best_x = pop[bi][:]

            # occasional island-level stagnation action (cheap)
            # replace a couple worst with jitter around global best
            if (gen % 11 == 0) or (stagnation_global >= 18 and gen % 5 == 0):
                inject = 1 if NP < 12 else 2
                worst_idx = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:inject]
                shrink = max(0.02, 1.0 - 0.92 * time_frac)
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

            order = argsort(fit)
            pbest_count = max(2, int(math.ceil(p_rate * NP)))
            pbest_count = min(NP, pbest_count)

            SF, SCR, dFit = [], [], []

            poolN = NP + len(archive[k])

            def pool_get(idx):
                return pop[idx] if idx < NP else archive[k][idx - NP]

            # evolve one generation
            for i in range(NP):
                if time.time() >= deadline:
                    return best

                xi = pop[i]
                fi = fit[i]

                # choose strategy according to learned probabilities
                r = random.random()
                p0, p1, p2 = strat_probs[k]
                if r < p0:
                    strat = 0
                elif r < p0 + p1:
                    strat = 1
                else:
                    strat = 2

                # sample CR ~ N(mu_CR, 0.1) clipped
                CRi = mu_CR[k] + 0.10 * randn()
                if CRi < 0.0: CRi = 0.0
                if CRi > 1.0: CRi = 1.0

                # sample F ~ Cauchy(mu_F, 0.1) -> (0,1]
                Fi = mu_F[k] + 0.10 * cauchy()
                tries = 0
                while (Fi <= 0.0 or Fi > 1.0) and tries < 12:
                    Fi = mu_F[k] + 0.10 * cauchy()
                    tries += 1
                if Fi <= 0.0: Fi = 0.35
                if Fi > 1.0: Fi = 1.0

                # pick indices
                # r1 != i
                r1 = i
                while r1 == i:
                    r1 = random.randrange(NP)

                # r2 from pop+archive, avoid i/r1 if from pop
                if poolN < 3:
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

                # choose pbest
                pbest_idx = order[random.randrange(pbest_count)]
                xpbest = pop[pbest_idx]

                # mutation variants
                v = [0.0] * dim
                if strat == 0:
                    # current-to-pbest/1
                    for d in range(dim):
                        v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                elif strat == 1:
                    # rand/1
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
                    for d in range(dim):
                        v[d] = xa[d] + Fi * (xb[d] - xc[d])
                else:
                    # best/1 (use island best)
                    ib = order[0]
                    xb = pop[ib]
                    for d in range(dim):
                        v[d] = xb[d] + Fi * (xr1[d] - xr2[d])

                # crossover
                jrand = random.randrange(dim)
                u = xi[:]
                for d in range(dim):
                    if random.random() < CRi or d == jrand:
                        u[d] = v[d]

                # bounds: mostly reflection; sometimes random repair on persistent out-of-bounds
                reflect_vec(u)
                if (gen % 17 == 0) and random.random() < 0.05:
                    # randomized repair to shake stuck coordinates (rare)
                    for d in range(dim):
                        lo, hi = bounds[d]
                        if u[d] <= lo or u[d] >= hi:
                            u[d] = random.uniform(lo, hi)

                fu = safe_eval(u)

                if fu <= fi:
                    # success
                    strat_succ[k][strat] += 1.0
                    archive[k].append(xi[:])
                    if len(archive[k]) > NP:
                        del archive[k][random.randrange(len(archive[k]))]

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
                else:
                    strat_fail[k][strat] += 1.0

            # adapt mu_F and mu_CR (JADE-style)
            if SF:
                wsum = sum(dFit)
                weights = [df / wsum for df in dFit]

                cr_mean = 0.0
                num = 0.0
                den = 0.0
                for w, fval, crval in zip(weights, SF, SCR):
                    cr_mean += w * crval
                    num += w * (fval * fval)
                    den += w * fval
                lehmerF = (num / den) if den > 0.0 else mu_F[k]

                mu_F[k] = clamp((1.0 - c_adapt) * mu_F[k] + c_adapt * lehmerF, 0.05, 0.95)
                mu_CR[k] = clamp((1.0 - c_adapt) * mu_CR[k] + c_adapt * cr_mean, 0.0, 1.0)
            else:
                # mild exploration nudge
                if gen % 10 == 0:
                    mu_F[k] = clamp(mu_F[k] + 0.02, 0.05, 0.95)
                    mu_CR[k] = clamp(mu_CR[k] * 0.98, 0.05, 1.0)

            # adapt strategy probabilities based on success rates (per island)
            if gen % 4 == 0:
                scores = []
                for s in range(3):
                    suc = strat_succ[k][s]
                    fai = strat_fail[k][s]
                    scores.append(suc / (suc + fai))
                # softmax-ish with floor to keep exploration
                exps = [max(1e-6, sc) for sc in scores]
                sm = exps[0] + exps[1] + exps[2]
                p = [exps[0] / sm, exps[1] / sm, exps[2] / sm]
                floor = 0.08
                p = [max(floor, pi) for pi in p]
                sm2 = p[0] + p[1] + p[2]
                strat_probs[k] = [pi / sm2 for pi in p]

        # ---------------- migration ----------------
        if gen % migrate_every == 0 and islands >= 2:
            # find island bests
            ibest = []
            for k in range(islands):
                pop = islands_pop[k]
                fit = islands_fit[k]
                bi = min(range(len(pop)), key=lambda i: fit[i])
                ibest.append((fit[bi], k, bi))
            ibest.sort()

            # migrate best individuals from top islands to worst islands
            donors = [ibest[i][1] for i in range(min(islands, 2))]
            receivers = [ibest[-i-1][1] for i in range(min(islands, 2))]
            for d_is, r_is in zip(donors, receivers):
                if d_is == r_is:
                    continue
                dpop = islands_pop[d_is]
                dfit = islands_fit[d_is]
                rpop = islands_pop[r_is]
                rfit = islands_fit[r_is]

                dorder = argsort(dfit)
                for m in range(migrate_count):
                    if time.time() >= deadline:
                        return best
                    src = dorder[m % len(dorder)]
                    immigrant = dpop[src][:]

                    # slight jitter to avoid clones
                    time_frac = (time.time() - t0) / max(1e-12, max_time)
                    shrink = max(0.01, 1.0 - 0.90 * time_frac)
                    for j in range(dim):
                        immigrant[j] += (0.02 * ranges[j] * shrink) * randn()
                    reflect_vec(immigrant)
                    fimm = safe_eval(immigrant)

                    # replace worst in receiver
                    w = max(range(len(rpop)), key=lambda i: rfit[i])
                    rpop[w] = immigrant
                    rfit[w] = fimm
                    if fimm < best:
                        best = fimm
                        best_x = immigrant[:]

        # ---------------- local refinement around global best ----------------
        # Frequent but budget-aware.
        do_local = (gen % 3 == 0) or (stagnation_global >= 10 and gen % 2 == 0)
        if do_local:
            if time.time() >= deadline:
                return best

            time_frac = (time.time() - t0) / max(1e-12, max_time)
            shrink = max(0.008, 1.0 - 0.94 * time_frac)

            # adapt gaussian scale by success rate
            if ls_trials >= 50:
                rate = ls_succ / float(ls_trials)
                if rate > 0.25:
                    ls_sigma_scale = min(3.0, ls_sigma_scale * 1.25)
                elif rate < 0.12:
                    ls_sigma_scale = max(0.20, ls_sigma_scale * 0.80)
                ls_succ = 0
                ls_trials = 0

            sigma = [base_sigma[d] * shrink * ls_sigma_scale for d in range(dim)]
            step = [base_step[d] * shrink for d in range(dim)]

            x = best_x[:]
            fx = best

            # (a) small coordinate search (1 pass)
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
                    continue

                cand = x[:]
                cand[d] = clamp(cur - step[d], lo, hi)
                f2 = safe_eval(cand)
                if f2 < fx:
                    x, fx = cand, f2

            # (b) trust-region gaussian probes
            probes = max(10, 2 * dim)
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

            # (c) rare "kick" to escape flat basins late
            if stagnation_global >= 25 and random.random() < 0.15:
                cand = x[:]
                kick = max(0.02, 0.10 * shrink)
                for d in range(dim):
                    cand[d] += (kick * ranges[d]) * randn()
                reflect_vec(cand)
                fc = safe_eval(cand)
                if fc < fx:
                    x, fx = cand, fc

            if fx < best:
                best = fx
                best_x = x[:]
                # inject into each island by replacing its worst (cheap intensification)
                for k in range(islands):
                    pop = islands_pop[k]
                    fit = islands_fit[k]
                    w = max(range(len(pop)), key=lambda i: fit[i])
                    pop[w] = best_x[:]
                    fit[w] = best

        # If very stuck, do a tiny global random restart injection
        if stagnation_global >= 60:
            for k in range(islands):
                if time.time() >= deadline:
                    return best
                pop = islands_pop[k]
                fit = islands_fit[k]
                nrep = 1 if len(pop) < 12 else 2
                worsts = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)[:nrep]
                for idx in worsts:
                    x = rand_vec()
                    fx = safe_eval(x)
                    pop[idx] = x
                    fit[idx] = fx
                    if fx < best:
                        best = fx
                        best_x = x[:]
            stagnation_global = 0
