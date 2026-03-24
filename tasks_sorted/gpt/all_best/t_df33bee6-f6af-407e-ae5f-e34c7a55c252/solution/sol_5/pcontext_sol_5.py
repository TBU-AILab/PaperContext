import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no external libs).

    Main upgrades vs your current best:
      - Better use of evaluations: "DE/3-choose-2" -> generate 3 trials but usually
        evaluate only 2 (adaptive), keeping speed while improving quality.
      - Stronger exploitation without gradients: periodic short Nelder–Mead simplex
        around the best (bounded + reflection). Very effective on smooth-ish problems.
      - Smarter local steps: (1+1)-ES with per-dimension scaling + success rule.
      - Diversity control: SHADE-like memory + archive + periodic partial refresh.
      - Robust boundary handling: reflection ("bounce") + occasional random re-entry.

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """

    # -------------------- helpers --------------------
    def eval_f(x):
        try:
            v = float(func(x))
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    def span(i):
        lo, hi = bounds[i]
        s = hi - lo
        return s if s > 0 else 1.0

    def bounce_inplace(x):
        # reflect into [lo, hi] (works even if far outside)
        for i, (lo, hi) in enumerate(bounds):
            w = hi - lo
            if w <= 0.0:
                x[i] = lo
                continue
            xi = x[i]
            if xi < lo or xi > hi:
                y = (xi - lo) % (2.0 * w)
                x[i] = lo + (2.0 * w - y if y > w else y)
        return x

    def reenter_random_dims(x, p=0.02):
        for i, (lo, hi) in enumerate(bounds):
            if random.random() < p:
                x[i] = random.uniform(lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    def randn():
        # approx N(0,1) using CLT (12 uniforms - 6)
        return (random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() - 6.0)

    def stratified_population(n):
        if n <= 0:
            return []
        perms = []
        for d in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            perms.append(idx)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                u = (perms[d][i] + random.random()) / float(n)
                x[d] = lo + u * (hi - lo)
            pop.append(x)
        return pop

    def cauchy_like(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def normal_like(mu, sigma):
        return mu + sigma * randn()

    def pick_index(n, forbid_set):
        j = random.randrange(n)
        tries = 0
        while j in forbid_set and tries < 30:
            j = random.randrange(n)
            tries += 1
        return j

    def argmin(vals):
        bi = 0
        bv = vals[0]
        for i in range(1, len(vals)):
            v = vals[i]
            if v < bv:
                bv = v
                bi = i
        return bi, bv

    # -------- bounded Nelder–Mead (very small budget, around best) --------
    def nelder_mead_local(best_x, best_f, budget_evals, scale):
        # Build simplex: x0 plus coordinate perturbations.
        # scale is relative to span per-dimension.
        if budget_evals <= 0:
            return best_x, best_f, 0

        n = dim
        x0 = best_x[:]
        f0 = best_f

        # Create simplex points
        simplex = [x0]
        fvals = [f0]
        used = 0

        # If dim is huge, limit simplex size to keep it cheap
        m = min(n, 10)  # build only up to 10 extra vertices
        dims = list(range(n))
        random.shuffle(dims)
        dims = dims[:m]

        for d in dims:
            x = x0[:]
            step = scale * span(d)
            if step <= 0.0:
                step = scale
            x[d] += step
            bounce_inplace(x)
            if random.random() < 0.05:
                reenter_random_dims(x, p=0.02)
            fx = eval_f(x)
            used += 1
            simplex.append(x)
            fvals.append(fx)
            if fx < best_f:
                best_x, best_f = x[:], fx
            if used >= budget_evals:
                return best_x, best_f, used

        # NM coefficients
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

        # Iterate a few steps within eval budget
        while used < budget_evals and len(simplex) >= 2:
            # sort
            order = sorted(range(len(simplex)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[0] < best_f:
                best_x, best_f = simplex[0][:], fvals[0]

            # centroid of all but worst
            worst = simplex[-1]
            centroid = [0.0] * dim
            k = len(simplex) - 1
            for i in range(k):
                xi = simplex[i]
                for d in range(dim):
                    centroid[d] += xi[d]
            invk = 1.0 / float(k)
            for d in range(dim):
                centroid[d] *= invk

            def eval_point(x):
                nonlocal used, best_x, best_f
                bounce_inplace(x)
                if random.random() < 0.03:
                    reenter_random_dims(x, p=0.015)
                fx = eval_f(x)
                used += 1
                if fx < best_f:
                    best_x, best_f = x[:], fx
                return fx

            # reflection
            xr = [centroid[d] + alpha * (centroid[d] - worst[d]) for d in range(dim)]
            fr = eval_point(xr)
            if used >= budget_evals:
                break

            if fr < fvals[0]:
                # expansion
                xe = [centroid[d] + gamma * (xr[d] - centroid[d]) for d in range(dim)]
                fe = eval_point(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                # accept reflection
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contraction
                if fr < fvals[-1]:
                    # outside
                    xc = [centroid[d] + rho * (xr[d] - centroid[d]) for d in range(dim)]
                    fc = eval_point(xc)
                    if fc <= fr:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # shrink
                        bestp = simplex[0]
                        for i in range(1, len(simplex)):
                            xs = [bestp[d] + sigma_nm * (simplex[i][d] - bestp[d]) for d in range(dim)]
                            if used >= budget_evals:
                                break
                            fs = eval_point(xs)
                            simplex[i], fvals[i] = xs, fs
                else:
                    # inside
                    xc = [centroid[d] - rho * (centroid[d] - worst[d]) for d in range(dim)]
                    fc = eval_point(xc)
                    if fc < fvals[-1]:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # shrink
                        bestp = simplex[0]
                        for i in range(1, len(simplex)):
                            xs = [bestp[d] + sigma_nm * (simplex[i][d] - bestp[d]) for d in range(dim)]
                            if used >= budget_evals:
                                break
                            fs = eval_point(xs)
                            simplex[i], fvals[i] = xs, fs

        return best_x, best_f, used

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- parameters --------------------
    pop_size = max(20, min(110, 20 + 6 * dim))
    p_rate = 0.22

    # SHADE-like memories
    H = 10
    MCR = [0.85] * H
    MF  = [0.60] * H
    mem_k = 0

    archive = []
    archive_max = pop_size

    # (1+1)-ES global sigma + per-dim scaling
    sigma = 0.11
    sigma_min = 1e-16
    sigma_max = 0.45
    dim_scale = [span(i) for i in range(dim)]

    # coordinate step (for very cheap micro-line searches)
    coord_step = [0.16 * span(i) for i in range(dim)]
    coord_min_step = [1e-16 * span(i) + 1e-18 for i in range(dim)]

    # stall / refresh
    best_t = start
    stall_seconds = max(0.25, 0.13 * max_time)
    last_refresh_t = start
    refresh_period = max(0.35, 0.18 * max_time)

    # -------------------- init: stratified + opposition --------------------
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    # opposition
    for i in range(pop_size):
        if time.time() >= deadline:
            return min(fit)
        xo = opposite(pop[i])
        bounce_inplace(xo)
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i] = xo
            fit[i] = fo

    best_i, best = argmin(fit)
    best_x = pop[best_i][:]

    # Seed a small local cloud (helps early exploitation)
    cloud = min(10, max(3, pop_size // 7))
    for _ in range(cloud):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for d in range(dim):
            x[d] += randn() * (0.04 * span(d))
        bounce_inplace(x)
        f = eval_f(x)
        j = random.randrange(pop_size)
        if j == best_i:
            j = (j + 1) % pop_size
        if f < fit[j]:
            archive.append(pop[j][:])
            if len(archive) > archive_max:
                archive.pop(random.randrange(len(archive)))
            pop[j] = x
            fit[j] = f
            if f < best:
                best, best_x, best_t = f, x[:], time.time()

    # -------------------- main loop --------------------
    while True:
        now = time.time()
        if now >= deadline:
            return best

        tfrac = 1.0 if max_time <= 0 else min(1.0, (now - start) / max_time)

        # p-best pool
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        p_num = max(2, int(math.ceil(p_rate * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        succ_CR, succ_F, succ_w = [], [], []

        union = pop + archive
        ulen = len(union)

        # -------- DE sweep (adaptive 2-of-3 trials) --------
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            r = random.randrange(H)
            mu_cr, mu_f = MCR[r], MF[r]

            CR = normal_like(mu_cr, 0.10 + 0.05 * (1.0 - tfrac))
            CR = 0.0 if CR < 0.0 else (1.0 if CR > 1.0 else CR)

            F = cauchy_like(mu_f, 0.08 + 0.05 * (1.0 - tfrac))
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 12:
                F = cauchy_like(mu_f, 0.08 + 0.05 * (1.0 - tfrac))
                tries += 1
            if F <= 0.0: F = 0.5
            if F > 1.0:  F = 1.0

            # Strategy 1: current-to-pbest/1 (good default)
            pbest = pop[random.choice(pbest_pool)]
            r1 = pick_index(pop_size, {i})
            r2 = pick_index(ulen, {i, r1}) if ulen > 2 else pick_index(pop_size, {i, r1})
            x_r1 = pop[r1]
            x_r2 = union[r2] if ulen > 2 else pop[r2]

            mut1 = [xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d]) for d in range(dim)]
            bounce_inplace(mut1)
            if random.random() < 0.06:
                reenter_random_dims(mut1, p=0.02)

            tr1 = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    tr1[d] = mut1[d]
            f1 = eval_f(tr1)

            # Strategy 2: rand/1 (exploration)
            a = pick_index(pop_size, {i})
            b = pick_index(pop_size, {i, a})
            c = pick_index(ulen, {i, a, b}) if ulen > 3 else pick_index(pop_size, {i, a, b})
            xa, xb = pop[a], pop[b]
            xc = union[c] if ulen > 3 else pop[c]

            mut2 = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
            bounce_inplace(mut2)

            tr2 = xi[:]
            CR2 = min(1.0, max(0.0, CR + 0.12 * (1.0 - tfrac)))
            jrand2 = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR2 or d == jrand2:
                    tr2[d] = mut2[d]
            f2 = eval_f(tr2)

            # Decide whether to spend eval on Strategy 3 (best/2) adaptively:
            # do more later and when already near best (small gap).
            do_third = (tfrac > 0.55) or (f1 < fit[i] and random.random() < (0.15 + 0.35 * tfrac))
            best_trial, f_trial, used_CR = (tr1, f1, CR) if f1 <= f2 else (tr2, f2, CR2)

            if do_third:
                # Strategy 3: best/2 (exploitation)
                r3 = pick_index(pop_size, {i})
                r4 = pick_index(pop_size, {i, r3})
                r5 = pick_index(pop_size, {i, r3, r4})
                r6 = pick_index(pop_size, {i, r3, r4, r5})
                x3, x4, x5, x6 = pop[r3], pop[r4], pop[r5], pop[r6]

                F3 = min(1.0, max(0.10, F * (0.70 + 0.65 * tfrac)))
                mut3 = [best_x[d] + F3 * (x3[d] - x4[d]) + F3 * (x5[d] - x6[d]) for d in range(dim)]
                bounce_inplace(mut3)

                tr3 = xi[:]
                CR3 = min(1.0, max(0.0, 0.55 * CR + 0.35 + 0.20 * tfrac))
                jrand3 = random.randrange(dim)
                for d in range(dim):
                    if random.random() < CR3 or d == jrand3:
                        tr3[d] = mut3[d]
                f3 = eval_f(tr3)

                if f3 < f_trial:
                    best_trial, f_trial, used_CR = tr3, f3, CR3

            if f_trial <= fit[i]:
                old = fit[i]
                archive.append(xi[:])
                if len(archive) > archive_max:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = best_trial
                fit[i] = f_trial

                w = (old - f_trial) if (old != float("inf") and math.isfinite(old)) else 1.0
                if w < 0.0:
                    w = 0.0
                succ_CR.append(used_CR)
                succ_F.append(F)
                succ_w.append(w + 1e-12)

                if f_trial < best:
                    best, best_x, best_t = f_trial, best_trial[:], time.time()

        # -------- memory update --------
        if succ_F:
            wsum = sum(succ_w)
            cr_mean = sum(w * cr for w, cr in zip(succ_w, succ_CR)) / wsum
            num = sum(w * f * f for w, f in zip(succ_w, succ_F))
            den = sum(w * f for w, f in zip(succ_w, succ_F))
            f_mean = (num / den) if den > 0 else MF[mem_k]

            MCR[mem_k] = 0.25 * MCR[mem_k] + 0.75 * cr_mean
            MF[mem_k]  = 0.25 * MF[mem_k]  + 0.75 * f_mean
            mem_k = (mem_k + 1) % H

        # -------- local: (1+1)-ES around best --------
        es_tries = 6 + int(18 * tfrac)
        succ = 0
        att = 0
        # per-dim scaling: slightly smaller in high dims to reduce overshooting
        dim_factor = 1.0 / math.sqrt(max(1.0, dim))
        for _ in range(es_tries):
            if time.time() >= deadline:
                return best
            att += 1
            cand = best_x[:]
            # mix isotropic + sparse kick to handle separable-ish functions
            if random.random() < 0.55:
                for d in range(dim):
                    cand[d] += randn() * (sigma * dim_factor * dim_scale[d])
            else:
                k = 1 + int(2 * random.random() * math.sqrt(max(1, dim)))
                for _k in range(k):
                    d = random.randrange(dim)
                    cand[d] += randn() * (sigma * 1.6 * dim_factor * dim_scale[d])
            bounce_inplace(cand)
            f = eval_f(cand)
            if f < best:
                best, best_x, best_t = f, cand[:], time.time()
                succ += 1

        if att:
            rate = succ / float(att)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.18)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.74)

        # -------- micro coordinate tweaks (cheap) --------
        if tfrac > 0.25:
            k = min(dim, 2 + int(4 * tfrac))
            for _ in range(k):
                if time.time() >= deadline:
                    return best
                d = random.randrange(dim)
                sd = coord_step[d]
                if sd <= coord_min_step[d]:
                    continue
                x0 = best_x[d]
                improved = False
                for xd in (x0 + sd, x0 - sd, x0 + 0.5 * sd, x0 - 0.5 * sd):
                    cand = best_x[:]
                    cand[d] = xd
                    bounce_inplace(cand)
                    f = eval_f(cand)
                    if f < best:
                        best, best_x, best_t = f, cand[:], time.time()
                        coord_step[d] = min(0.35 * span(d), coord_step[d] * 1.12)
                        improved = True
                        break
                if not improved:
                    coord_step[d] *= 0.62

        # -------- periodic Nelder–Mead burst around best --------
        # Keep rare and short (good payoff on many continuous objectives).
        if tfrac > 0.15 and (time.time() - best_t) < 0.35 * stall_seconds:
            # run occasionally; more often late
            if random.random() < (0.06 + 0.10 * tfrac):
                # allocate a tiny evaluation budget
                nm_budget = 6 + int(10 * tfrac)
                # scale shrinks over time
                nm_scale = (0.10 * (1.0 - tfrac) + 0.02)  # relative to span
                best_x, best, _ = nelder_mead_local(best_x, best, nm_budget, nm_scale)
                best_t = time.time()

        # -------- periodic partial refresh (aging) --------
        now = time.time()
        if now - last_refresh_t > refresh_period and tfrac > 0.12:
            elite_k = max(3, pop_size // 8)
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elites = set(idx_sorted[:elite_k])

            repl = max(2, pop_size // 6)
            worst = idx_sorted[-repl:]
            for wi in worst:
                if time.time() >= deadline:
                    return best
                if wi in elites:
                    continue
                if random.random() < 0.55:
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += randn() * (max(0.05, sigma) * 0.20 * span(d))
                    bounce_inplace(x)
                else:
                    x = rand_vec()
                    if random.random() < 0.50:
                        x = opposite(x)
                        bounce_inplace(x)
                f = eval_f(x)
                pop[wi] = x
                fit[wi] = f
                if f < best:
                    best, best_x, best_t = f, x[:], time.time()

            archive = []
            last_refresh_t = time.time()

        # -------- full restart on stagnation --------
        if time.time() - best_t > stall_seconds:
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elite_k = max(4, pop_size // 7)
            elites = [pop[i][:] for i in idx_sorted[:elite_k]]
            elites_fit = [fit[i] for i in idx_sorted[:elite_k]]

            new_pop = elites[:]
            new_fit = elites_fit[:]

            remain = pop_size - elite_k
            n_global = remain // 3
            n_opp = remain // 3
            n_local = remain - n_global - n_opp

            for x in stratified_population(n_global):
                new_pop.append(x)
                new_fit.append(eval_f(x))

            for _ in range(n_opp):
                x = rand_vec()
                xo = opposite(x)
                bounce_inplace(xo)
                fx = eval_f(x)
                fo = eval_f(xo)
                if fo < fx:
                    new_pop.append(xo); new_fit.append(fo)
                else:
                    new_pop.append(x);  new_fit.append(fx)

            local_sigma = max(0.08, sigma)
            for _ in range(n_local):
                x = best_x[:]
                for d in range(dim):
                    x[d] += randn() * (local_sigma * 0.22 * span(d))
                bounce_inplace(x)
                new_pop.append(x)
                new_fit.append(eval_f(x))

            pop, fit = new_pop, new_fit
            bi, bv = argmin(fit)
            if bv < best:
                best, best_x = bv, pop[bi][:]
            best_t = time.time()
            last_refresh_t = best_t

            archive = []
            # small memory perturbation
            for k in range(H):
                MCR[k] = min(0.95, max(0.05, MCR[k] + 0.035 * randn()))
                MF[k]  = min(0.95, max(0.05, MF[k]  + 0.035 * randn()))
            sigma = max(0.10, sigma)
            coord_step = [max(0.10 * span(i), coord_step[i]) for i in range(dim)]
