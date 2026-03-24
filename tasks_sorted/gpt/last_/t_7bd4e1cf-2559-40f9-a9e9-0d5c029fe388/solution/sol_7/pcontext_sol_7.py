import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib-only).

    Key improvements over the provided code:
      1) Adds a small *surrogate-guided* proposal stream (RBF-like weighted kNN)
         that exploits collected evaluations to suggest promising candidates.
      2) Uses *CMA-ES-like* diagonal sampling around the incumbent (very effective
         for continuous box-bounded problems) as an intensification step.
      3) Stronger restart logic: mixes global (DE), surrogate exploitation,
         and local (coordinate/pattern) search in a time-aware schedule.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ----- guards / bounds -----
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

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

    def repair_reflect(x):
        # reflect then clip (stable for moderate overshoots)
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                y[i] = lo
                continue
            v = y[i]
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            # if still out, modulo wrap then clip
            if v < lo or v > hi:
                w = hi - lo
                if w > 0:
                    v = lo + (v - lo) % w
            y[i] = clamp(v, lo, hi)
        return y

    def evaluate(x):
        return float(func(x))

    if time.time() >= deadline:
        return evaluate(center_vec())

    # ----- estimate eval time -----
    probe_n = 5
    probe_best = float("inf")
    start_probe = time.time()
    for _ in range(probe_n):
        if time.time() >= deadline:
            return probe_best
        fx = evaluate(rand_vec())
        if fx < probe_best:
            probe_best = fx
    eval_time = max(1e-6, (time.time() - start_probe) / float(probe_n))

    def remaining_evals():
        rem = deadline - time.time()
        if rem <= 0:
            return 0
        return max(0, int(rem / max(eval_time, 1e-12)))

    # ----- simple history for surrogate -----
    # store (x, f); keep limited to control overhead
    HIST_MAX = 900
    hist_x = []
    hist_f = []

    def hist_add(x, f):
        hist_x.append(x[:])
        hist_f.append(float(f))
        if len(hist_x) > HIST_MAX:
            # drop some older points (cheap thinning)
            # keep best 1/3 + newest 2/3
            n = len(hist_x)
            idx = list(range(n))
            idx.sort(key=lambda i: hist_f[i])
            keep_best = idx[: max(50, n // 3)]
            keep_new = list(range(max(0, n - (2 * n // 3)), n))
            keep = sorted(set(keep_best + keep_new))
            hx = [hist_x[i] for i in keep]
            hf = [hist_f[i] for i in keep]
            hist_x[:] = hx
            hist_f[:] = hf

    # ----- init population (LHS-ish) -----
    rem0 = max(60, remaining_evals())
    NP0 = max(20, 10 + 6 * dim)
    NPmin = max(10, 5 + 2 * dim)
    NP0 = min(NP0, max(NPmin, rem0 // 6))
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
        hist_add(pop[i], f)
        if f < best:
            best = f
            best_x = pop[i][:]

    # ----- SHADE-like memory for DE parameters -----
    H = 10
    MCR = [0.85] * H
    MF = [0.6] * H
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

    def pick_excluding(n, exclude_set):
        for _ in range(30):
            r = random.randrange(n)
            if r not in exclude_set:
                return r
        for r in range(n):
            if r not in exclude_set:
                return r
        return 0

    # ----- local search: coordinate / pattern (fast, deterministic-ish) -----
    def pattern_search(x0, f0, budget, init_frac):
        if budget <= 0:
            return x0, f0, 0
        x = x0[:]
        fx = f0
        evals = 0

        step = [init_frac * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = [1e-12 + 1e-7 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]

        dims = list(range(dim))
        dims.sort(key=lambda i: spans[i], reverse=True)

        while evals < budget and time.time() < deadline:
            improved = False
            # subset for large dim
            if dim <= 18:
                sub = dims
            else:
                sub = dims[: max(10, dim // 3)]
                random.shuffle(sub)

            for d in sub:
                if evals >= budget or time.time() >= deadline:
                    break
                if spans[d] <= 0.0 or step[d] <= min_step[d]:
                    continue
                base = x[d]
                # try both directions
                for sgn in (1.0, -1.0) if random.random() < 0.5 else (-1.0, 1.0):
                    y = x[:]
                    y[d] = clamp(base + sgn * step[d], lows[d], highs[d])
                    fy = evaluate(y)
                    evals += 1
                    hist_add(y, fy)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break

            if improved:
                for d in sub:
                    step[d] *= 1.12
            else:
                for d in sub:
                    step[d] *= 0.5

            tiny = 0
            for d in sub:
                if step[d] <= min_step[d]:
                    tiny += 1
            if tiny >= max(1, len(sub) - 1):
                break

        return x, fx, evals

    # ----- diagonal CMA-ES-like sampling around best (intensification) -----
    def diag_cma_sample(mu, sigma_vec, lam):
        # returns lam samples
        out = []
        for _ in range(lam):
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            x = [mu[i] + sigma_vec[i] * z[i] for i in range(dim)]
            out.append(repair_reflect(x))
        return out

    # ----- surrogate-guided candidate (RBF-ish kNN, low overhead) -----
    def surrogate_propose(n_cand=24, k=10):
        # If little history, fallback random
        n = len(hist_x)
        if n < max(30, 3 * dim):
            return rand_vec()

        # pick a center: best among a few random history points
        # (bias towards good regions but not always the global best)
        idxs = [random.randrange(n) for _ in range(8)]
        idxs.sort(key=lambda i: hist_f[i])
        base = hist_x[idxs[0]][:]

        # generate candidates around base
        # radius shrinks with time and with history size
        ratio = (time.time() - t0) / max(1e-9, (deadline - t0))
        ratio = 0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else ratio)
        rad = (0.35 * (1.0 - ratio) + 0.06)  # fraction of span
        best_pred = float("inf")
        best_c = None

        for _ in range(max(1, n_cand)):
            c = base[:]
            for d in range(dim):
                if spans[d] > 0:
                    c[d] = clamp(c[d] + random.gauss(0.0, rad) * spans[d], lows[d], highs[d])

            # predict with weighted kNN on normalized distance
            # choose k random points to keep cost low
            kk = min(k, n)
            sel = [random.randrange(n) for _ in range(kk)]
            wsum = 0.0
            psum = 0.0
            for j in sel:
                xj = hist_x[j]
                # normalized squared distance
                ds = 0.0
                for d in range(dim):
                    sp = spans[d]
                    if sp > 0:
                        t = (c[d] - xj[d]) / sp
                        ds += t * t
                # RBF weight
                w = 1.0 / (1e-12 + ds)
                wsum += w
                psum += w * hist_f[j]
            pred = psum / wsum if wsum > 0 else float("inf")

            if pred < best_pred:
                best_pred = pred
                best_c = c

        return best_c if best_c is not None else rand_vec()

    # ----- main loop -----
    no_improve_gens = 0
    last_best = best
    gen = 0
    NP = NP0

    while time.time() < deadline and NP >= NPmin:
        gen += 1

        order = list(range(NP))
        order.sort(key=lambda i: fits[i])

        elapsed = time.time() - t0
        T = max(1e-9, deadline - t0)
        ratio = elapsed / T
        ratio = 0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else ratio)

        # p-best schedule
        p = 0.32 - 0.24 * ratio  # 0.32 -> 0.08
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

            # Occasionally replace a trial with surrogate proposal (especially late)
            use_sur = (random.random() < (0.08 + 0.18 * ratio)) and (len(hist_x) >= 30)
            if use_sur:
                ui = surrogate_propose(n_cand=18, k=10)
                fui = evaluate(ui)
                hist_add(ui, fui)
                if fui <= fi:
                    new_pop[i] = ui
                    new_fits[i] = fui
                    if fui < best:
                        best = fui
                        best_x = ui[:]
                else:
                    new_pop[i] = xi
                    new_fits[i] = fi
                continue

            r = random.randrange(H)
            CRm = MCR[r]
            Fm = MF[r]

            CR = CRm + random.gauss(0.0, 0.07)
            CR = 0.0 if CR < 0.0 else (1.0 if CR > 1.0 else CR)

            F = Fm
            for _ in range(12):
                Fc = cauchy(Fm, 0.07)
                if Fc > 0.0:
                    F = Fc
                    break
            F = 0.04 if F < 0.04 else (1.0 if F > 1.0 else F)

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

            # Strategy mix
            if random.random() < (0.90 - 0.50 * ratio):
                vi = [xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
            else:
                a = pick_excluding(NP, {i})
                b = pick_excluding(NP, {i, a})
                c = pick_excluding(NP, {i, a, b})
                xa, xb, xc = pop[a], pop[b], pop[c]
                vi = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            # crossover
            ui = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    ui[d] = vi[d]
            ui = repair_reflect(ui)

            fui = evaluate(ui)
            hist_add(ui, fui)

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
            weights = [w / wsum for w in dF] if wsum > 1e-18 else [1.0 / len(dF)] * len(dF)
            mcr = weighted_mean(SCR, weights)
            mf = weighted_lehmer(SF, weights)
            if mcr is not None:
                MCR[k_mem] = 0.0 if mcr < 0.0 else (1.0 if mcr > 1.0 else mcr)
            if mf is not None:
                mf = 0.04 if mf < 0.04 else (0.98 if mf > 0.98 else mf)
                MF[k_mem] = mf
            k_mem = (k_mem + 1) % H

        # stagnation tracking
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # intensification schedule: pattern + diagonal-CMA sampling around best
        if best_x is not None and time.time() < deadline:
            rem = remaining_evals()
            if rem > 0 and ((gen % 5) == 0 or no_improve_gens >= 7):
                # pattern search budget
                ps_budget = min(max(12, 2 * dim), max(12, rem // 12))
                if ps_budget > 0:
                    x2, f2, _ = pattern_search(best_x, best, ps_budget, init_frac=(0.18 - 0.10 * ratio))
                    if f2 < best:
                        best, best_x = f2, x2[:]
                        worst_i = max(range(NP), key=lambda j: fits[j])
                        pop[worst_i] = best_x[:]
                        fits[worst_i] = best
                        no_improve_gens = 0

                # diagonal-CMA sampling (very cheap per sample)
                rem = remaining_evals()
                lam = min(max(6, 2 + dim), max(6, rem // 18))
                if lam > 0 and time.time() < deadline:
                    # sigma schedule: shrink over time; keep some floor
                    sigma_frac = (0.22 * (1.0 - ratio) + 0.03)
                    sigma = [(sigma_frac * spans[d] if spans[d] > 0 else 0.0) for d in range(dim)]
                    cands = diag_cma_sample(best_x, sigma, lam)
                    for c in cands:
                        if time.time() >= deadline:
                            break
                        fc = evaluate(c)
                        hist_add(c, fc)
                        if fc < best:
                            best, best_x = fc, c[:]
                            worst_i = max(range(NP), key=lambda j: fits[j])
                            pop[worst_i] = best_x[:]
                            fits[worst_i] = best
                            no_improve_gens = 0

        # restart / partial reinit if stuck
        if best_x is not None and no_improve_gens >= 13 and time.time() < deadline:
            no_improve_gens = 0
            rad = 0.35 * (1.0 - ratio) + 0.06
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
                fi = evaluate(x)
                fits[i] = fi
                hist_add(x, fi)
                if fi < best:
                    best, best_x = fi, x[:]

        # linear population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * ratio))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP > NP0:
            target_NP = NP0

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
