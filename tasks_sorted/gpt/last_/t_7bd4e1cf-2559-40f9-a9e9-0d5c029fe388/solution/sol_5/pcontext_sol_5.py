import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded stdlib-only minimizer.

    Changes vs provided code (focused on practical gains):
      1) Stronger hybridization: L-SHADE-ish DE + explicit "opposition / centroid" injections
      2) Better p-best schedule (decreases p over time) + occasional "DE/rand/1" for exploration
      3) Stagnation detection -> partial restart around best (shrinking radius)
      4) More effective local search: pattern search with adaptive step + random subspace
      5) Tighter time budgeting and fewer wasted evals

    Returns:
      best (float): best (minimum) fitness found
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # -------- guards --------
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
        v = [0.0] * dim
        for i in range(dim):
            if spans[i] <= 0.0:
                v[i] = lows[i]
            else:
                v[i] = lows[i] + random.random() * spans[i]
        return v

    def center_vec():
        return [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def repair(v):
        # clip; if wildly out of range, resample that coordinate
        out = v[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                out[i] = lo
                continue
            x = out[i]
            if x < lo:
                if (lo - x) > 0.75 * (hi - lo):
                    out[i] = lo + random.random() * (hi - lo)
                else:
                    out[i] = lo
            elif x > hi:
                if (x - hi) > 0.75 * (hi - lo):
                    out[i] = lo + random.random() * (hi - lo)
                else:
                    out[i] = hi
        return out

    def evaluate(x):
        return float(func(x))

    if time.time() >= deadline:
        return evaluate(center_vec())

    # -------- probe evaluation speed --------
    probe_n = 5
    probe_best = float("inf")
    probe_start = time.time()
    for _ in range(probe_n):
        if time.time() >= deadline:
            return probe_best
        fx = evaluate(rand_vec())
        if fx < probe_best:
            probe_best = fx
    eval_time = max(1e-6, (time.time() - probe_start) / float(probe_n))

    # -------- population sizing --------
    time_left = max(0.0, deadline - time.time())
    approx_evals_left = max(30, int(time_left / max(eval_time, 1e-12)))

    NP0 = max(24, 10 + 8 * dim)
    NPmin = max(8, 4 + 2 * dim)
    # keep some generations worth of evals
    NP0 = min(NP0, max(NPmin, approx_evals_left // 5))
    if NP0 < NPmin:
        NP0 = NPmin

    # -------- LHS-ish init + extra elite seeds --------
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

    # explicit center & a few randoms
    pop[0] = center_vec()
    for k in range(1, min(NP0, 1 + max(2, NP0 // 12))):
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

    # -------- SHADE memories --------
    H = 10
    MCR = [0.8] * H
    MF = [0.55] * H
    k_mem = 0

    archive = []
    arch_max = NP0

    def cauchy(loc, scale):
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

    def pick_index_excluding(limit, exclude):
        # a few tries then fallback scan
        for _ in range(12):
            r = random.randrange(limit)
            if r not in exclude:
                return r
        for r in range(limit):
            if r not in exclude:
                return r
        return 0

    # -------- local search: adaptive pattern search (random subspace) --------
    def local_search(x0, f0, max_evals):
        if max_evals <= 0:
            return x0, f0, 0
        x = x0[:]
        fx = f0
        evals = 0

        # initial step relative to span; shrink on failure, expand slightly on success
        step = [0.12 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = [1e-12 + 1e-6 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]

        # choose a random subset each round to reduce cost in high-dim
        rounds = 0
        while evals < max_evals and time.time() < deadline:
            rounds += 1
            idxs = list(range(dim))
            random.shuffle(idxs)
            # subspace size
            k = dim if dim <= 12 else max(8, dim // 3)
            idxs = idxs[:k]

            improved = False
            for d in idxs:
                if evals >= max_evals or time.time() >= deadline:
                    break
                if spans[d] <= 0.0:
                    continue
                sd = step[d]
                if sd <= min_step[d]:
                    continue

                base = x[d]

                xm = x[:]
                xm[d] = clamp(base - sd, lows[d], highs[d])
                fm = evaluate(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    continue

                if evals >= max_evals or time.time() >= deadline:
                    break

                xp = x[:]
                xp[d] = clamp(base + sd, lows[d], highs[d])
                fp = evaluate(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

            if improved:
                # mild expansion helps move along valleys
                for d in idxs:
                    step[d] *= 1.12
            else:
                # shrink steps
                for d in idxs:
                    step[d] *= 0.5

            # stop if steps are tiny in subspace
            tiny = 0
            for d in idxs:
                if step[d] <= min_step[d]:
                    tiny += 1
            if tiny >= max(1, len(idxs) - 1):
                break

        return x, fx, evals

    # -------- stagnation / restart controls --------
    no_improve_gens = 0
    last_best = best

    gen = 0
    NP = NP0

    while time.time() < deadline and NP >= NPmin:
        gen += 1

        # sort for p-best and reduction
        order = list(range(NP))
        order.sort(key=lambda i: fits[i])

        # p schedule: start broader, end more exploitative
        elapsed = time.time() - t0
        T = max(1e-9, deadline - t0)
        ratio = clamp(elapsed / T, 0.0, 1.0)
        p = 0.35 - 0.25 * ratio   # 0.35 -> 0.10
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

            CR = clamp(CRm + random.gauss(0.0, 0.08), 0.0, 1.0)

            F = Fm
            for _ in range(12):
                Fc = cauchy(Fm, 0.08)
                if Fc > 0.0:
                    F = Fc
                    break
            F = clamp(F, 0.04, 1.0)

            pbest = random.choice(top)

            # choose r1 from pop, r2 from pop+archive
            excl = {i, pbest}
            r1 = pick_index_excluding(NP, excl)
            excl.add(r1)

            use_arch = (archive and random.random() < 0.6)
            if use_arch:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_index_excluding(NP, excl)
                xr2 = pop[r2]

            xr1 = pop[r1]
            xp = pop[pbest]

            # mutation strategy mix: mostly current-to-pbest/1, sometimes rand/1
            if random.random() < (0.85 - 0.35 * ratio):
                # current-to-pbest/1
                vi = [xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
            else:
                # DE/rand/1 (exploration)
                a = pick_index_excluding(NP, {i})
                b = pick_index_excluding(NP, {i, a})
                c = pick_index_excluding(NP, {i, a, b})
                xa, xb, xc = pop[a], pop[b], pop[c]
                vi = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            # binomial crossover
            ui = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    ui[d] = vi[d]

            ui = repair(ui)
            fui = evaluate(ui)

            if fui <= fi:
                new_pop[i] = ui
                new_fits[i] = fui

                # archive update with parent
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                imp = fi - fui
                SF.append(F)
                SCR.append(CR)
                dF.append(imp if imp > 0.0 else 1e-12)

                if fui < best:
                    best, best_x = fui, ui[:]
            else:
                new_pop[i] = xi
                new_fits[i] = fi

        pop = new_pop
        fits = new_fits

        # update SHADE memories
        if SF:
            wsum = sum(dF)
            weights = [w / wsum for w in dF] if wsum > 1e-18 else [1.0 / len(dF)] * len(dF)
            mcr = weighted_mean(SCR, weights)
            mf = weighted_lehmer(SF, weights)
            if mcr is not None:
                MCR[k_mem] = clamp(mcr, 0.0, 1.0)
            if mf is not None:
                MF[k_mem] = clamp(mf, 0.04, 0.98)
            k_mem = (k_mem + 1) % H

        # opposition/centroid injection (cheap diversification)
        if best_x is not None and (gen % 7) == 0 and time.time() < deadline:
            # compute centroid of top few
            m = max(3, min(NP, 1 + NP // 5))
            idx = order[:m]
            centroid = [0.0] * dim
            for d in range(dim):
                s = 0.0
                for j in idx:
                    s += pop[j][d]
                centroid[d] = s / float(m)

            # opposition of best around centroid (like reflective move)
            opp = [clamp(centroid[d] * 2.0 - best_x[d], lows[d], highs[d]) for d in range(dim)]
            fopp = evaluate(opp)
            if fopp < best:
                best, best_x = fopp, opp[:]
                # replace worst
                worst_i = max(range(NP), key=lambda j: fits[j])
                pop[worst_i] = best_x[:]
                fits[worst_i] = best

        # time-aware local search on best
        if best_x is not None and (gen % 9) == 0 and time.time() < deadline:
            rem = deadline - time.time()
            rem_evals = int(rem / max(eval_time, 1e-12))
            ls_budget = min(max(10, 2 * dim), max(10, rem_evals // 18))
            x2, f2, _ = local_search(best_x, best, ls_budget)
            if f2 < best:
                best, best_x = f2, x2[:]
                worst_i = max(range(NP), key=lambda j: fits[j])
                pop[worst_i] = best_x[:]
                fits[worst_i] = best

        # stagnation check -> partial restart around best with shrinking radius
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if best_x is not None and no_improve_gens >= 12 and time.time() < deadline:
            no_improve_gens = 0
            # radius shrinks with time (more exploitation later)
            rad = 0.35 * (1.0 - ratio) + 0.05
            # reinitialize worst half around best
            order2 = list(range(NP))
            order2.sort(key=lambda i: fits[i], reverse=True)  # worst first
            cnt = max(1, NP // 2)
            for k in range(cnt):
                i = order2[k]
                x = best_x[:]
                for d in range(dim):
                    if spans[d] <= 0:
                        x[d] = lows[d]
                    else:
                        # gaussian perturbation scaled by span
                        x[d] = clamp(x[d] + random.gauss(0.0, rad) * spans[d], lows[d], highs[d])
                pop[i] = x
                fits[i] = evaluate(x)
                if fits[i] < best:
                    best, best_x = fits[i], pop[i][:]

        # linear population reduction by time
        target_NP = int(round(NP0 - (NP0 - NPmin) * ratio))
        target_NP = max(NPmin, min(NP0, target_NP))
        if target_NP < NP:
            order3 = list(range(NP))
            order3.sort(key=lambda i: fits[i])
            keep = order3[:target_NP]
            pop = [pop[i] for i in keep]
            fits = [fits[i] for i in keep]
            NP = target_NP
            if len(archive) > arch_max:
                random.shuffle(archive)
                del archive[arch_max:]

    return best
