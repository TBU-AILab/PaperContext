import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (stdlib-only) with stronger anytime behavior.

    Improvements over the previous version:
      - Adds a *trust-region coordinate search* (very effective for box-bounded problems)
      - Uses *multiple intensification triggers* (best stagnation + periodic)
      - Better evaluation budgeting: estimates eval cost and allocates portions to
        (1) global DE search, (2) local trust-region search, (3) occasional restarts
      - More robust boundary handling: bounce-back + clip
      - Safer with pathological dims/bounds and very small time limits

    Returns:
      best (float): best (minimum) fitness found
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---------- guards ----------
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

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] <= 0.0:
                x[i] = lows[i]
            else:
                x[i] = lows[i] + random.random() * spans[i]
        return x

    def center_vec():
        return [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def repair_bounce(x):
        # Bounce back when out of bounds; if still out due to huge overshoot, clip.
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                y[i] = lo
                continue
            v = y[i]
            if v < lo:
                v = lo + (lo - v)  # reflect
                if v > hi:
                    v = lo + (v - lo) % (hi - lo)
            elif v > hi:
                v = hi - (v - hi)  # reflect
                if v < lo:
                    v = lo + (v - lo) % (hi - lo)
            y[i] = clamp(v, lo, hi)
        return y

    def evaluate(x):
        return float(func(x))

    if time.time() >= deadline:
        return evaluate(center_vec())

    # ---------- estimate evaluation time ----------
    probe_n = 7
    probe_best = float("inf")
    start_probe = time.time()
    for _ in range(probe_n):
        if time.time() >= deadline:
            return probe_best
        fx = evaluate(rand_vec())
        if fx < probe_best:
            probe_best = fx
    eval_time = max(1e-6, (time.time() - start_probe) / float(probe_n))

    # ---------- budget heuristics ----------
    def remaining_evals():
        rem = deadline - time.time()
        if rem <= 0:
            return 0
        return max(0, int(rem / max(eval_time, 1e-12)))

    # ---------- initialization: LHS-ish ----------
    # Choose population size based on time and dimension
    rem0 = max(40, remaining_evals())
    NP0 = max(18, 8 + 6 * dim)
    NPmin = max(8, 4 + 2 * dim)
    NP0 = min(NP0, max(NPmin, rem0 // 6))  # keep several generations
    if NP0 < NPmin:
        NP0 = NPmin

    perms = []
    for d in range(dim):
        p = list(range(NP0))
        random.shuffle(p)
        perms.append(p)

    pop = []
    for i in range(NP0):
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] <= 0.0:
                x[d] = lows[d]
            else:
                u = (perms[d][i] + random.random()) / float(NP0)
                x[d] = lows[d] + spans[d] * u
        pop.append(x)

    # inject center + a few pure randoms
    pop[0] = center_vec()
    for k in range(1, min(NP0, 1 + max(3, NP0 // 10))):
        pop[k] = rand_vec()

    fits = [float("inf")] * NP0
    best = float("inf")
    best_x = None

    for i in range(NP0):
        if time.time() >= deadline:
            return best if best < float("inf") else probe_best
        f = evaluate(pop[i])
        fits[i] = f
        if f < best:
            best = f
            best_x = pop[i][:]

    # ---------- SHADE-like parameter memories ----------
    H = 10
    MCR = [0.85] * H
    MF = [0.6] * H
    k_mem = 0

    archive = []
    arch_max = NP0

    def cauchy(loc, scale):
        # cauchy via inverse CDF
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def weighted_lehmer(values, weights):
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

    def pick_excluding(n, exclude_set):
        for _ in range(20):
            r = random.randrange(n)
            if r not in exclude_set:
                return r
        for r in range(n):
            if r not in exclude_set:
                return r
        return 0

    # ---------- Local search: Trust-region coordinate search ----------
    # This is a robust "anytime" local optimizer for box constraints.
    def trust_region_search(x0, f0, max_evals, init_frac=0.18):
        if max_evals <= 0:
            return x0, f0, 0

        x = x0[:]
        fx = f0
        evals = 0

        # per-dimension radii; start moderate, shrink on failures
        rad = [init_frac * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_rad = [1e-12 + 1e-7 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]

        # prioritize dimensions with larger span, but keep randomness
        dims = list(range(dim))
        dims.sort(key=lambda i: spans[i], reverse=True)

        stall_rounds = 0
        while evals < max_evals and time.time() < deadline:
            improved = False

            # random subspace to reduce cost for large dim
            if dim <= 14:
                sub = dims[:]
            else:
                k = max(8, dim // 3)
                sub = dims[:k]
                random.shuffle(sub)

            for d in sub:
                if evals >= max_evals or time.time() >= deadline:
                    break
                if spans[d] <= 0.0:
                    continue
                if rad[d] <= min_rad[d]:
                    continue

                base = x[d]

                # try +rad and -rad (order randomized)
                if random.random() < 0.5:
                    trials = (+1.0, -1.0)
                else:
                    trials = (-1.0, +1.0)

                for sgn in trials:
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    y = x[:]
                    y[d] = clamp(base + sgn * rad[d], lows[d], highs[d])
                    fy = evaluate(y)
                    evals += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break  # move to next dim

            if improved:
                stall_rounds = 0
                # very slight expand to follow valleys
                for d in sub:
                    rad[d] *= 1.08
            else:
                stall_rounds += 1
                for d in sub:
                    rad[d] *= 0.5

            # termination if radii tiny (or repeated stalls)
            tiny = 0
            for d in sub:
                if rad[d] <= min_rad[d]:
                    tiny += 1
            if tiny >= max(1, len(sub) - 1):
                break
            if stall_rounds >= 3 and dim > 25:
                break

        return x, fx, evals

    # ---------- main loop ----------
    no_improve_gens = 0
    last_best = best
    gen = 0
    NP = NP0

    while time.time() < deadline and NP >= NPmin:
        gen += 1

        # ordering
        order = list(range(NP))
        order.sort(key=lambda i: fits[i])

        # time ratio
        elapsed = time.time() - t0
        T = max(1e-9, deadline - t0)
        ratio = elapsed / T
        if ratio < 0.0: ratio = 0.0
        if ratio > 1.0: ratio = 1.0

        # p-best schedule (broad -> narrow)
        p = 0.30 - 0.22 * ratio   # 0.30 -> 0.08
        pbest_n = max(2, int(math.ceil(p * NP)))
        top = order[:pbest_n]

        SF, SCR, dF = [], [], []
        new_pop = [None] * NP
        new_fits = [None] * NP

        arch_max = max(NP0, NP)

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            r = random.randrange(H)
            CRm = MCR[r]
            Fm = MF[r]

            CR = CRm + random.gauss(0.0, 0.07)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = Fm
            for _ in range(12):
                Fc = cauchy(Fm, 0.07)
                if Fc > 0.0:
                    F = Fc
                    break
            if F < 0.04: F = 0.04
            if F > 1.0: F = 1.0

            pbest = random.choice(top)

            excl = {i, pbest}
            r1 = pick_excluding(NP, excl)
            excl.add(r1)

            use_arch = (archive and random.random() < 0.65)
            if use_arch:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_excluding(NP, excl)
                xr2 = pop[r2]

            xr1 = pop[r1]
            xp = pop[pbest]

            # Strategy mix: more exploration early
            if random.random() < (0.88 - 0.45 * ratio):
                # current-to-pbest/1
                vi = [xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
            else:
                # rand/1
                a = pick_excluding(NP, {i})
                b = pick_excluding(NP, {i, a})
                c = pick_excluding(NP, {i, a, b})
                xa, xb, xc = pop[a], pop[b], pop[c]
                vi = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            # binomial crossover
            ui = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    ui[d] = vi[d]

            ui = repair_bounce(ui)
            fui = evaluate(ui)

            if fui <= fi:
                new_pop[i] = ui
                new_fits[i] = fui

                # archive parent
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                imp = fi - fui
                if imp <= 0.0:
                    imp = 1e-12
                SF.append(F)
                SCR.append(CR)
                dF.append(imp)

                if fui < best:
                    best = fui
                    best_x = ui[:]
            else:
                new_pop[i] = xi
                new_fits[i] = fi

        pop, fits = new_pop, new_fits

        # update memories
        if SF:
            wsum = sum(dF)
            if wsum <= 1e-18:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [w / wsum for w in dF]
            mcr = weighted_mean(SCR, weights)
            mf = weighted_lehmer(SF, weights)
            if mcr is not None:
                if mcr < 0.0: mcr = 0.0
                if mcr > 1.0: mcr = 1.0
                MCR[k_mem] = mcr
            if mf is not None:
                if mf < 0.04: mf = 0.04
                if mf > 0.98: mf = 0.98
                MF[k_mem] = mf
            k_mem = (k_mem + 1) % H

        # opposition/centroid injection (cheap)
        if best_x is not None and (gen % 6) == 0 and time.time() < deadline:
            m = max(3, min(NP, 1 + NP // 6))
            idx = order[:m]
            centroid = [0.0] * dim
            for d in range(dim):
                s = 0.0
                for j in idx:
                    s += pop[j][d]
                centroid[d] = s / float(m)

            cand = [clamp(2.0 * centroid[d] - best_x[d], lows[d], highs[d]) for d in range(dim)]
            fc = evaluate(cand)
            if fc < best:
                best, best_x = fc, cand[:]
                worst_i = max(range(NP), key=lambda j: fits[j])
                pop[worst_i] = best_x[:]
                fits[worst_i] = best

        # stagnation tracking
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # --- Intensification: trust-region search (more aggressive than before) ---
        # Trigger periodically and on stagnation; keep it time-aware.
        if best_x is not None and time.time() < deadline:
            trigger = (gen % 5 == 0) or (no_improve_gens >= 8)
            if trigger:
                rem = remaining_evals()
                # allocate a chunk but don't starve global search
                ls_budget = min(max(12, 3 * dim), max(12, rem // 10))
                if ls_budget > 0:
                    x2, f2, _ = trust_region_search(best_x, best, ls_budget, init_frac=(0.20 - 0.10 * ratio))
                    if f2 < best:
                        best, best_x = f2, x2[:]
                        worst_i = max(range(NP), key=lambda j: fits[j])
                        pop[worst_i] = best_x[:]
                        fits[worst_i] = best
                        no_improve_gens = 0

        # --- restart around best if stuck ---
        if best_x is not None and no_improve_gens >= 14 and time.time() < deadline:
            no_improve_gens = 0
            rad = 0.30 * (1.0 - ratio) + 0.04
            order_worst = list(range(NP))
            order_worst.sort(key=lambda i: fits[i], reverse=True)
            cnt = max(1, NP // 2)
            for k in range(cnt):
                i = order_worst[k]
                x = best_x[:]
                for d in range(dim):
                    if spans[d] <= 0.0:
                        x[d] = lows[d]
                    else:
                        x[d] = clamp(x[d] + random.gauss(0.0, rad) * spans[d], lows[d], highs[d])
                pop[i] = x
                fits[i] = evaluate(x)
                if fits[i] < best:
                    best, best_x = fits[i], pop[i][:]

        # linear population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * ratio))
        if target_NP < NPmin: target_NP = NPmin
        if target_NP > NP0: target_NP = NP0

        if target_NP < NP:
            order_keep = list(range(NP))
            order_keep.sort(key=lambda i: fits[i])
            keep = order_keep[:target_NP]
            pop = [pop[i] for i in keep]
            fits = [fits[i] for i in keep]
            NP = target_NP
            if len(archive) > arch_max:
                random.shuffle(archive)
                del archive[arch_max:]

    return best
