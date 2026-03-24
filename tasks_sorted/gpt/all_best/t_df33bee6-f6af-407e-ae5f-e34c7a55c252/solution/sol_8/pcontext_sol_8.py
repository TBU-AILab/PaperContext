import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (self-contained; no external libs).

    Improvements vs the previous best (DE + ES + occasional NM):
      1) Evaluation cache (rounded keys): avoids wasting time re-evaluating near-identical points.
      2) Better local refinement: adds a cheap bounded quadratic line-search on a few random
         directions and coordinates (often stronger than short Nelder–Mead bursts).
      3) Keeps a good global engine: SHADE-like DE (current-to-pbest/1 + archive) with
         adaptive 2-of-3 trial evaluation and a light exploration arm.
      4) Smarter restarts: trigger depends on BOTH time-since-improvement and low diversity.
      5) More stable boundary handling: reflection + rare random re-entry to escape edge traps.

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """

    start = time.time()
    deadline = start + float(max_time)

    # -------------------- helpers --------------------
    def span(i):
        lo, hi = bounds[i]
        s = hi - lo
        return s if s > 0.0 else 1.0

    spans = [span(i) for i in range(dim)]
    inv_spans = [1.0 / (spans[i] + 1e-300) for i in range(dim)]

    def bounce_inplace(x):
        # reflect into [lo, hi] even if far outside
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

    def reenter_random_dims(x, p=0.015):
        for i, (lo, hi) in enumerate(bounds):
            if random.random() < p:
                x[i] = random.uniform(lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    def randn():
        # approx N(0,1) via CLT
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
        while j in forbid_set and tries < 40:
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

    def normed_l2_sq(a, b):
        s = 0.0
        for i in range(dim):
            d = (a[i] - b[i]) * inv_spans[i]
            s += d * d
        return s

    def diversity(pop):
        # mean normalized L1 distance to centroid
        if not pop:
            return 0.0
        cen = [0.0] * dim
        invn = 1.0 / float(len(pop))
        for x in pop:
            for d in range(dim):
                cen[d] += x[d]
        for d in range(dim):
            cen[d] *= invn
        tot = 0.0
        for x in pop:
            s = 0.0
            for d in range(dim):
                s += abs(x[d] - cen[d]) * inv_spans[d]
            tot += s
        return tot * invn / max(1, dim)

    # -------------------- evaluation cache --------------------
    cache = {}
    # quantization ~ 1e-12 of span (per dim) to deduplicate near-equal points
    q = [max(1e-12 * spans[i], 1e-18) for i in range(dim)]

    def key_of(x):
        ks = []
        for i in range(dim):
            lo = bounds[i][0]
            ks.append(int((x[i] - lo) / q[i]))
        return tuple(ks)

    def eval_f(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        try:
            v = float(func(x))
            if not math.isfinite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v

    # -------------------- parameters --------------------
    pop_size = max(22, min(120, 22 + 6 * dim))
    p_rate = 0.22

    # SHADE memories
    H = 10
    MCR = [0.85] * H
    MF  = [0.60] * H
    mem_k = 0

    archive = []
    archive_max = pop_size

    # ES sigma
    sigma = 0.11
    sigma_min = 1e-18
    sigma_max = 0.45

    # coordinate step for micro-search + quadratic fit
    coord_step = [0.16 * spans[i] for i in range(dim)]
    coord_min_step = [1e-18 * spans[i] + 1e-18 for i in range(dim)]

    # direction set for cheap line-searches
    dirs = []
    # a few coordinate directions (up to 10)
    for d in range(min(dim, 10)):
        u = [0.0] * dim
        u[d] = 1.0
        dirs.append(u)
    # and a few random directions
    for _ in range(6):
        u = [randn() for _ in range(dim)]
        dirs.append(u)

    # stall/restart controls
    best_t = start
    last_refresh_t = start
    stall_seconds = max(0.25, 0.13 * max_time)
    refresh_period = max(0.35, 0.18 * max_time)

    # -------------------- init --------------------
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    # opposition pass
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

    # small local cloud around best
    cloud = min(12, max(4, pop_size // 7))
    for _ in range(cloud):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for d in range(dim):
            x[d] += randn() * (0.04 * spans[d])
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

    # -------------------- local search primitives --------------------
    def quad_line_search(x0, f0, u, step_abs, budget):
        # Evaluate at +-step, then quadratic fit (3 points) and optionally evaluate minimizer.
        # Returns (best_x, best_f, used_evals).
        used = 0

        # normalize direction by max-abs (avoid sqrt)
        m = 0.0
        for d in range(dim):
            ad = abs(u[d])
            if ad > m:
                m = ad
        if m <= 0.0 or step_abs <= 0.0 or budget <= 0:
            return x0, f0, 0
        invm = 1.0 / m
        uu = [u[d] * invm for d in range(dim)]

        def at(alpha):
            xx = [x0[d] + alpha * uu[d] for d in range(dim)]
            bounce_inplace(xx)
            if random.random() < 0.03:
                reenter_random_dims(xx, p=0.01)
            return xx, eval_f(xx)

        # center is known
        bestx, bestf = x0, f0

        if used >= budget:
            return bestx, bestf, used

        x1, f1 = at(+step_abs); used += 1
        if f1 < bestf:
            bestx, bestf = x1, f1
        if used >= budget:
            return bestx, bestf, used

        x2, f2 = at(-step_abs); used += 1
        if f2 < bestf:
            bestx, bestf = x2, f2
        if used >= budget:
            return bestx, bestf, used

        # quadratic fit through (-h,f2),(0,f0),(+h,f1)
        # denom = f(-h) - 2f(0) + f(+h)
        denom = (f2 - 2.0 * f0 + f1)
        if abs(denom) > 1e-18 and math.isfinite(denom):
            h = step_abs
            b = (f1 - f2) / (2.0 * h)
            a = denom / (2.0 * h * h)
            # minimizer alpha* = -b/(2a)
            alpha = -b / (2.0 * a)
            # trust region clamp
            alpha = max(-1.6 * h, min(1.6 * h, alpha))
            x3, f3 = at(alpha); used += 1
            if f3 < bestf:
                bestx, bestf = x3, f3

        return bestx, bestf, used

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

        # -------- DE sweep (adaptive 2-of-3) --------
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mu_cr, mu_f = MCR[r], MF[r]

            CR = normal_like(mu_cr, 0.10 + 0.05 * (1.0 - tfrac))
            CR = 0.0 if CR < 0.0 else (1.0 if CR > 1.0 else CR)

            F = cauchy_like(mu_f, 0.08 + 0.05 * (1.0 - tfrac))
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 12:
                F = cauchy_like(mu_f, 0.08 + 0.05 * (1.0 - tfrac))
                tries += 1
            if F <= 0.0:
                F = 0.5
            if F > 1.0:
                F = 1.0

            pbest = pop[random.choice(pbest_pool)]

            # Trial 1: current-to-pbest/1 + archive
            r1 = pick_index(pop_size, {i})
            r2 = pick_index(ulen, {i, r1}) if ulen > 2 else pick_index(pop_size, {i, r1})
            x_r1 = pop[r1]
            x_r2 = union[r2] if ulen > 2 else pop[r2]

            mut1 = [xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d]) for d in range(dim)]
            bounce_inplace(mut1)
            if random.random() < 0.05:
                reenter_random_dims(mut1, p=0.02)

            tr1 = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    tr1[d] = mut1[d]
            f1 = eval_f(tr1)

            best_trial, f_trial, used_CR = tr1, f1, CR

            # Trial 2: rand/1 exploration (often helps)
            do_second = (f1 >= fi) or (random.random() < (0.22 + 0.30 * (1.0 - tfrac)))
            if do_second:
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
                if f2 < f_trial:
                    best_trial, f_trial, used_CR = tr2, f2, CR2

            # Trial 3: "triangular" current-to-best/pbest (cheap exploitation), only sometimes
            do_third = (tfrac > 0.55 and random.random() < 0.35) or (fi < best and random.random() < 0.20)
            if do_third:
                mut3 = [xi[d] + F * (best_x[d] - xi[d]) + 0.5 * F * (pbest[d] - xi[d]) for d in range(dim)]
                bounce_inplace(mut3)
                tr3 = xi[:]
                CR3 = min(1.0, max(0.0, 0.40 + 0.35 * CR))
                jrand3 = random.randrange(dim)
                for d in range(dim):
                    if random.random() < CR3 or d == jrand3:
                        tr3[d] = mut3[d]
                f3 = eval_f(tr3)
                if f3 < f_trial:
                    best_trial, f_trial, used_CR = tr3, f3, CR3

            # selection
            if f_trial <= fi:
                old = fi
                archive.append(xi[:])
                if len(archive) > archive_max:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = best_trial
                fit[i] = f_trial

                w = (old - f_trial) if (math.isfinite(old) and old != float("inf")) else 1.0
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
            f_mean = (num / den) if den > 0.0 else MF[mem_k]

            MCR[mem_k] = 0.25 * MCR[mem_k] + 0.75 * cr_mean
            MF[mem_k]  = 0.25 * MF[mem_k]  + 0.75 * f_mean
            mem_k = (mem_k + 1) % H

        # -------- local: (1+1)-ES around best --------
        es_tries = 6 + int(18 * tfrac)
        succ = 0
        att = 0
        dim_factor = 1.0 / math.sqrt(max(1.0, dim))
        for _ in range(es_tries):
            if time.time() >= deadline:
                return best
            att += 1
            cand = best_x[:]
            if random.random() < 0.55:
                for d in range(dim):
                    cand[d] += randn() * (sigma * dim_factor * spans[d])
            else:
                k = 1 + int(2 * random.random() * math.sqrt(max(1, dim)))
                for _k in range(k):
                    d = random.randrange(dim)
                    cand[d] += randn() * (sigma * 1.6 * dim_factor * spans[d])
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

        # -------- cheap quadratic local searches (directions + coords) --------
        # replaces Nelder–Mead with fewer overhead evaluations, usually better ROI.
        if tfrac > 0.12:
            # a small eval budget each iteration
            budget = 3 + int(6 * tfrac)
            used = 0

            # try 1-2 random directions
            tries_dirs = 1 if tfrac < 0.6 else 2
            for _ in range(tries_dirs):
                if time.time() >= deadline or used >= budget:
                    break
                u = random.choice(dirs)
                # step based on spans
                step_abs = (0.08 * (1.0 - tfrac) + 0.02) * max(spans)
                bx, bf, uev = quad_line_search(best_x, best, u, step_abs, budget - used)
                used += uev
                if bf < best:
                    best, best_x, best_t = bf, bx[:], time.time()
                    # add displacement direction to dirs
                    disp = [best_x[d] - pop[idx_sorted[0]][d] for d in range(dim)]
                    if normed_l2_sq(disp, [0.0]*dim) > 1e-24:
                        dirs = [disp] + dirs
                        if len(dirs) > 30:
                            dirs = dirs[:30]

            # try a couple coordinate quadratic fits
            if used < budget:
                k = 1 + int(2 * tfrac)
                for _ in range(k):
                    if time.time() >= deadline or used >= budget:
                        break
                    d = random.randrange(dim)
                    sd = coord_step[d]
                    if sd <= coord_min_step[d]:
                        continue
                    # coordinate direction
                    u = [0.0] * dim
                    u[d] = 1.0
                    bx, bf, uev = quad_line_search(best_x, best, u, sd, budget - used)
                    used += uev
                    if bf < best:
                        best, best_x, best_t = bf, bx[:], time.time()
                        coord_step[d] = min(0.35 * spans[d], coord_step[d] * 1.10)
                    else:
                        coord_step[d] *= 0.70

        # -------- periodic partial refresh --------
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
                    rad = (max(0.05, sigma) * 0.20)
                    for d in range(dim):
                        x[d] += randn() * (rad * spans[d])
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

        # -------- full restart on stagnation OR low diversity --------
        div = diversity(pop)
        if (time.time() - best_t > stall_seconds) or (tfrac > 0.20 and div < 0.015 and time.time() - best_t > 0.35 * stall_seconds):
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
                fx = eval_f(x)
                new_pop.append(x)
                new_fit.append(fx)

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
                    x[d] += randn() * (local_sigma * 0.22 * spans[d])
                bounce_inplace(x)
                fx = eval_f(x)
                new_pop.append(x)
                new_fit.append(fx)

            pop, fit = new_pop, new_fit
            bi, bv = argmin(fit)
            if bv < best:
                best, best_x = bv, pop[bi][:]

            best_t = time.time()
            last_refresh_t = best_t
            archive = []

            # memory perturbation
            for k in range(H):
                MCR[k] = min(0.95, max(0.05, MCR[k] + 0.035 * randn()))
                MF[k]  = min(0.95, max(0.05, MF[k]  + 0.035 * randn()))
            sigma = max(0.10, sigma)
            coord_step = [max(0.10 * spans[i], coord_step[i]) for i in range(dim)]
