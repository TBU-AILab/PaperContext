import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded stdlib-only optimizer (no numpy).

    Upgrades vs last version (key practical improvements):
      1) L-SHADE style: linear population size reduction + external archive
      2) Better parameter adaptation (memory of (F, CR) per individual; weighted by improvement)
      3) Stronger bound handling: clip + optional re-sample on huge violations
      4) More robust local polish: multi-scale coordinate search with opportunistic steps
      5) Strict time-awareness everywhere (polish budget scales with remaining time)

    Returns:
      best (float): minimum fitness found within max_time seconds
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ----------------- guards -----------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def clip_vec(v):
        out = v[:]
        for i in range(dim):
            out[i] = clamp(out[i], lows[i], highs[i])
        return out

    # Occasionally re-sample a coordinate if it flies far outside bounds (keeps diversity).
    def repair_vec(v):
        out = v[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                out[i] = lo
                continue
            x = out[i]
            if x < lo:
                # if far outside, re-sample; else clip
                if (lo - x) > 0.5 * (hi - lo):
                    out[i] = lo + random.random() * (hi - lo)
                else:
                    out[i] = lo
            elif x > hi:
                if (x - hi) > 0.5 * (hi - lo):
                    out[i] = lo + random.random() * (hi - lo)
                else:
                    out[i] = hi
        return out

    def rand_vec():
        v = [0.0] * dim
        for i in range(dim):
            if spans[i] <= 0.0:
                v[i] = lows[i]
            else:
                v[i] = lows[i] + random.random() * spans[i]
        return v

    def center_vec():
        return [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def evaluate(v):
        return float(func(v))

    if deadline <= t0:
        return evaluate(center_vec())

    # ----------------- probe eval speed -----------------
    probe_n = 5
    probe_best = float("inf")
    probe_start = time.time()
    for _ in range(probe_n):
        if time.time() >= deadline:
            return probe_best
        x = rand_vec()
        fx = evaluate(x)
        if fx < probe_best:
            probe_best = fx
    probe_dt = max(1e-6, time.time() - probe_start)
    eval_time = probe_dt / float(probe_n)

    # ----------------- initial population size -----------------
    time_left = max(0.0, deadline - time.time())
    approx_evals_left = max(20, int(time_left / max(eval_time, 1e-9)))

    NP0 = max(18, 12 + 6 * dim)     # start pop
    NPmin = max(6, 4 + 2 * dim)     # end pop
    NP0 = min(NP0, max(NPmin, approx_evals_left // 6))
    if NP0 < NPmin:
        NP0 = NPmin

    # LHS init (per-dimension permutation)
    perms = []
    for d in range(dim):
        p = list(range(NP0))
        random.shuffle(p)
        perms.append(p)

    pop = []
    for i in range(NP0):
        v = [0.0] * dim
        for d in range(dim):
            if spans[d] <= 0.0:
                v[d] = lows[d]
            else:
                u = (perms[d][i] + random.random()) / float(NP0)
                v[d] = lows[d] + spans[d] * u
        pop.append(v)

    pop[0] = center_vec()
    for k in range(1, min(1 + max(1, NP0 // 10), NP0)):
        pop[k] = rand_vec()

    fits = [float("inf")] * NP0
    best = float("inf")
    best_x = None

    for i in range(NP0):
        if time.time() >= deadline:
            return best if best < float("inf") else probe_best
        fx = evaluate(pop[i])
        fits[i] = fx
        if fx < best:
            best, best_x = fx, pop[i][:]

    # ----------------- SHADE-like memories -----------------
    H = 8  # memory size
    MCR = [0.85] * H
    MF = [0.6] * H
    k_mem = 0

    # archive for diversity
    archive = []        # list of vectors
    arch_max = NP0      # adaptive cap

    def pick_distinct_indices(n, exclude_set, limit):
        # returns n distinct indices in [0, limit) not in exclude_set
        out = []
        tries = 0
        while len(out) < n and tries < 1000:
            r = random.randrange(limit)
            tries += 1
            if r in exclude_set:
                continue
            ok = True
            for q in out:
                if q == r:
                    ok = False
                    break
            if ok:
                out.append(r)
        if len(out) < n:
            # fallback deterministic fill
            for r in range(limit):
                if r in exclude_set:
                    continue
                if r not in out:
                    out.append(r)
                if len(out) == n:
                    break
        return out

    def cauchy(loc, scale):
        # loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def weighted_lehmer_mean(values, weights):
        num = 0.0
        den = 0.0
        for v, w in zip(values, weights):
            num += w * v * v
            den += w * v
        if den <= 1e-18:
            return None
        return num / den

    def weighted_mean(values, weights):
        s = 0.0
        wsum = 0.0
        for v, w in zip(values, weights):
            s += w * v
            wsum += w
        if wsum <= 1e-18:
            return None
        return s / wsum

    # ----------------- local polish (multi-scale coordinate search) -----------------
    def polish(x0, f0, max_evals):
        x = x0[:]
        fx = f0
        if max_evals <= 0:
            return x, fx, 0

        evals = 0
        # step scales: start moderate, then shrink
        step_levels = [0.2, 0.08, 0.03, 0.012, 0.005]
        # reorder coordinates each round
        idxs = list(range(dim))

        for lvl, step_rel in enumerate(step_levels):
            if evals >= max_evals or time.time() >= deadline:
                break
            random.shuffle(idxs)

            improved_any = False
            for d in idxs:
                if evals >= max_evals or time.time() >= deadline:
                    break
                if spans[d] <= 0.0:
                    continue
                s = spans[d] * step_rel
                if s <= 0.0:
                    continue

                base = x[d]
                # try +/- step, then +/- 2*step if improving
                for mult in (1.0, 2.0):
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    ss = s * mult

                    # minus
                    xm = x[:]
                    xm[d] = clamp(base - ss, lows[d], highs[d])
                    fm = evaluate(xm); evals += 1
                    if fm < fx:
                        x, fx = xm, fm
                        base = x[d]
                        improved_any = True
                        continue

                    if evals >= max_evals or time.time() >= deadline:
                        break

                    # plus
                    xp = x[:]
                    xp[d] = clamp(base + ss, lows[d], highs[d])
                    fp = evaluate(xp); evals += 1
                    if fp < fx:
                        x, fx = xp, fp
                        base = x[d]
                        improved_any = True
                        continue

            # if nothing improved at this scale, continue to smaller; if improved, still continue
            # (smaller scales can further refine)
            _ = improved_any

        return x, fx, evals

    # ----------------- main loop (L-SHADE-ish) -----------------
    gen = 0
    NP = NP0
    evals_done = 0

    # crude evaluation budget estimate used for pop reduction schedule
    # (we only need a monotone "progress" indicator)
    total_budget = max(1, approx_evals_left)

    while time.time() < deadline and NP >= NPmin:
        gen += 1

        # sort for p-best selection
        order = list(range(NP))
        order.sort(key=lambda i: fits[i])
        p = 0.2
        pbest_n = max(2, int(math.ceil(p * NP)))
        top = order[:pbest_n]

        # record successful params and improvement weights
        SF, SCR, dF = [], [], []  # dF weights = improvement amounts

        new_pop = [None] * NP
        new_fits = [None] * NP

        # update archive cap as NP changes
        arch_max = max(NP, NP0)

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            r = random.randrange(H)
            CRm = MCR[r]
            Fm = MF[r]

            # CR from normal, F from cauchy
            CR = clamp(CRm + random.gauss(0.0, 0.1), 0.0, 1.0)

            F = -1.0
            for _ in range(10):
                F = cauchy(Fm, 0.1)
                if F > 0.0:
                    break
            if F <= 0.0:
                F = Fm
            F = clamp(F, 0.05, 1.0)

            pbest = random.choice(top)

            # choose r1 from pop, r2 from pop+archive
            # Ensure distinctness: i, pbest, r1, r2 indices (archive handled separately)
            excl = {i, pbest}
            r1 = pick_distinct_indices(1, excl, NP)[0]
            excl.add(r1)

            use_arch = (len(archive) > 0 and random.random() < 0.5)
            if use_arch:
                # r2 from archive
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_distinct_indices(1, excl, NP)[0]
                xr2 = pop[r2]

            xp = pop[pbest]
            xr1 = pop[r1]

            # current-to-pbest/1
            vi = [0.0] * dim
            for d in range(dim):
                vi[d] = xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # crossover
            ui = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    ui[d] = vi[d]

            ui = repair_vec(ui)
            fui = evaluate(ui)
            evals_done += 1

            if fui <= fi:
                new_pop[i] = ui
                new_fits[i] = fui

                # archive gets replaced parent (as in JADE/SHADE)
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    # random replacement
                    archive[random.randrange(arch_max)] = xi[:]

                if fi - fui > 0.0:
                    SF.append(F)
                    SCR.append(CR)
                    dF.append(fi - fui)
                else:
                    # still "success" but no improvement weight; keep tiny weight
                    SF.append(F)
                    SCR.append(CR)
                    dF.append(1e-12)

                if fui < best:
                    best = fui
                    best_x = ui[:]
            else:
                new_pop[i] = xi
                new_fits[i] = fi

        pop = new_pop
        fits = new_fits

        # update memories with weighted means
        if SF:
            # normalize weights
            wsum = sum(dF)
            if wsum <= 1e-18:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [w / wsum for w in dF]

            mcr = weighted_mean(SCR, weights)
            mf = weighted_lehmer_mean(SF, weights)

            if mcr is not None:
                MCR[k_mem] = clamp(mcr, 0.0, 1.0)
            if mf is not None:
                MF[k_mem] = clamp(mf, 0.05, 0.95)

            k_mem = (k_mem + 1) % H

        # occasional polish near best (time-aware)
        if best_x is not None and (gen % 10) == 0 and time.time() < deadline:
            # give polish a small budget that grows with dimension but limited by remaining time
            rem = deadline - time.time()
            # optimistic remaining evals
            rem_evals = int(rem / max(eval_time, 1e-9))
            polish_evals = min(max(8, 3 * dim), max(8, rem_evals // 20))
            x2, f2, _ = polish(best_x, best, polish_evals)
            if f2 < best:
                best, best_x = f2, x2[:]
                # inject into population by replacing current worst
                worst_i = max(range(NP), key=lambda j: fits[j])
                pop[worst_i] = best_x[:]
                fits[worst_i] = best

        # linear population size reduction (L-SHADE-like)
        # progress ratio based on elapsed time (works without exact eval budget)
        elapsed = time.time() - t0
        T = max(1e-9, deadline - t0)
        ratio = clamp(elapsed / T, 0.0, 1.0)
        target_NP = int(round(NP0 - (NP0 - NPmin) * ratio))
        target_NP = max(NPmin, min(NP0, target_NP))

        if target_NP < NP:
            # remove worst individuals
            order = list(range(NP))
            order.sort(key=lambda i: fits[i])  # best first
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fits = [fits[i] for i in keep]
            NP = target_NP
            # also shrink archive if huge
            if len(archive) > arch_max:
                random.shuffle(archive)
                del archive[arch_max:]

    return best
