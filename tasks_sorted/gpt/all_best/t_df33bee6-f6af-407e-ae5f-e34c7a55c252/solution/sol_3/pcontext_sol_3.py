import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (no external libraries).

    Improvements over your best (JADE/SHADE-ish DE + (1+1)-ES + restart):
      - Opposition + stratified init (often gives a big early jump)
      - Dual-DE generation per individual: pbest/1 AND rand/1, pick best trial
      - Two-arm local search: (1+1)-ES + accelerated coordinate search around best
      - Smarter boundary handling: "bounce-back" reflection (better than hard clip)
      - Budget-aware scheduling + stagnation restarts with mixed global/opposition/local

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
        # reflection boundary handling (keeps diversity better than pure clipping)
        for i, (lo, hi) in enumerate(bounds):
            xi = x[i]
            if xi < lo or xi > hi:
                w = hi - lo
                if w <= 0:
                    x[i] = lo
                    continue
                # reflect into [lo,hi] even if far out of bounds
                y = (xi - lo) % (2.0 * w)
                x[i] = lo + (2.0 * w - y if y > w else y)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    # CLT normal-ish: mean 0, var ~1
    def randn():
        return (random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() - 6.0)

    def stratified_population(n):
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

    # choose index != forbidden set (small sets, so simple loop is fine)
    def pick_index(n, forbid):
        j = random.randrange(n)
        tries = 0
        while j in forbid and tries < 20:
            j = random.randrange(n)
            tries += 1
        return j

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- parameters --------------------
    pop_size = max(18, min(90, 18 + 5 * dim))
    p_rate = 0.22
    archive = []
    archive_max = pop_size

    # SHADE-like memories
    H = 8
    MCR = [0.85] * H
    MF  = [0.60] * H
    mem_k = 0

    # local search controls
    sigma = 0.12
    sigma_min = 1e-14
    sigma_max = 0.40

    # coordinate local step
    coord_step = [0.18 * span(i) for i in range(dim)]
    coord_min_step = [1e-14 * span(i) + 1e-18 for i in range(dim)]

    # stagnation / restart
    best_t = start
    stall_seconds = max(0.25, 0.14 * max_time)

    # -------------------- init: stratified + opposition --------------------
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    # evaluate opposites for a subset (cheap but very effective)
    # do it for all if time allows early; otherwise still fine
    for i in range(pop_size):
        if time.time() >= deadline:
            return min(fit)
        xo = opposite(pop[i])
        bounce_inplace(xo)
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i] = xo
            fit[i] = fo

    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # -------------------- main loop --------------------
    while True:
        now = time.time()
        if now >= deadline:
            return best

        t = 1.0 if max_time <= 0 else min(1.0, (now - start) / max_time)

        # p-best pool
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        p_num = max(2, int(math.ceil(p_rate * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        succ_CR, succ_F, succ_w = [], [], []

        # -------- DE generation (two strategies, best-of-two) --------
        # Strategy A: current-to-pbest/1 with archive (exploit + diversity)
        # Strategy B: rand/1 with archive (explore; helps avoid premature convergence)
        union = pop + archive
        ulen = len(union)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            r = random.randrange(H)
            mu_cr, mu_f = MCR[r], MF[r]

            CR = normal_like(mu_cr, 0.12)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = cauchy_like(mu_f, 0.10)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 10:
                F = cauchy_like(mu_f, 0.10)
                tries += 1
            if F <= 0.0: F = 0.5
            if F > 1.0:  F = 1.0

            # indices and vectors
            pbest = pop[random.choice(pbest_pool)]

            r1 = pick_index(pop_size, {i})
            r2 = pick_index(ulen, {i, r1}) if ulen > 2 else pick_index(pop_size, {i, r1})
            x_r1 = pop[r1]
            x_r2 = union[r2] if ulen > 2 else pop[r2]

            # ---- trial A: current-to-pbest/1 ----
            mutantA = [0.0] * dim
            for d in range(dim):
                mutantA[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d])
            bounce_inplace(mutantA)

            trialA = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    trialA[d] = mutantA[d]
            fA = eval_f(trialA)

            # ---- trial B: rand/1 ----
            a = pick_index(pop_size, {i})
            b = pick_index(pop_size, {i, a})
            c = pick_index(ulen, {i, a, b}) if ulen > 3 else pick_index(pop_size, {i, a, b})
            xa, xb = pop[a], pop[b]
            xc = union[c] if ulen > 3 else pop[c]

            mutantB = [0.0] * dim
            for d in range(dim):
                mutantB[d] = xa[d] + F * (xb[d] - xc[d])
            bounce_inplace(mutantB)

            trialB = xi[:]
            jrand2 = random.randrange(dim)
            # slightly higher crossover on exploratory arm early
            CRb = min(1.0, max(0.0, CR + (0.10 * (1.0 - t))))
            for d in range(dim):
                if random.random() < CRb or d == jrand2:
                    trialB[d] = mutantB[d]
            fB = eval_f(trialB)

            # pick better trial among A and B
            if fB < fA:
                trial, f_trial = trialB, fB
                used_CR = CRb
            else:
                trial, f_trial = trialA, fA
                used_CR = CR

            if f_trial <= fit[i]:
                old = fit[i]
                # archive store replaced
                archive.append(xi[:])
                if len(archive) > archive_max:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = trial
                fit[i] = f_trial

                w = (old - f_trial) if (old != float("inf") and math.isfinite(old)) else 1.0
                if w < 0.0:
                    w = 0.0
                succ_CR.append(used_CR)
                succ_F.append(F)
                succ_w.append(w + 1e-12)

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
                    best_t = time.time()

        # memory update
        if succ_F:
            wsum = sum(succ_w)
            cr_mean = sum(w * cr for w, cr in zip(succ_w, succ_CR)) / wsum

            num = sum(w * f * f for w, f in zip(succ_w, succ_F))
            den = sum(w * f for w, f in zip(succ_w, succ_F))
            f_mean = (num / den) if den > 0 else MF[mem_k]

            # mild damping toward old values improves stability
            MCR[mem_k] = 0.2 * MCR[mem_k] + 0.8 * cr_mean
            MF[mem_k]  = 0.2 * MF[mem_k]  + 0.8 * f_mean
            mem_k = (mem_k + 1) % H

        # -------- Local refinement: (1+1)-ES around best --------
        es_tries = 6 + int(16 * t)
        succ = 0
        att = 0
        for _ in range(es_tries):
            if time.time() >= deadline:
                return best
            att += 1
            cand = best_x[:]
            for d in range(dim):
                cand[d] += randn() * (sigma * span(d))
            bounce_inplace(cand)
            f_c = eval_f(cand)
            if f_c < best:
                best = f_c
                best_x = cand
                best_t = time.time()
                succ += 1

        if att:
            rate = succ / float(att)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.20)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.72)

        # -------- Fast coordinate acceleration around best (late-biased) --------
        if t > 0.30:
            # keep it cheap: a few dims each time, not all dims
            k = min(dim, 2 + int(4 * t))
            for _ in range(k):
                if time.time() >= deadline:
                    return best
                d = random.randrange(dim)
                sd = coord_step[d]
                if sd <= coord_min_step[d]:
                    continue

                x0 = best_x[d]
                # try a small line search stencil
                candidates = [x0 + sd, x0 - sd, x0 + 0.5 * sd, x0 - 0.5 * sd]
                improved = False
                for xd in candidates:
                    cand = best_x[:]
                    cand[d] = xd
                    bounce_inplace(cand)
                    f_c = eval_f(cand)
                    if f_c < best:
                        best = f_c
                        best_x = cand
                        best_t = time.time()
                        improved = True
                        break
                if improved:
                    coord_step[d] = min(0.35 * span(d), coord_step[d] * 1.12)
                else:
                    coord_step[d] = coord_step[d] * 0.60

        # -------- Restart on stagnation --------
        if time.time() - best_t > stall_seconds:
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elite_k = max(3, pop_size // 7)
            elites = [pop[i][:] for i in idx_sorted[:elite_k]]
            elites_fit = [fit[i] for i in idx_sorted[:elite_k]]

            new_pop = elites[:]
            new_fit = elites_fit[:]

            remain = pop_size - elite_k
            # mix: stratified global + opposition + local cloud
            n_global = remain // 3
            n_opp = remain // 3
            n_local = remain - n_global - n_opp

            # global stratified
            for x in stratified_population(n_global):
                new_pop.append(x)
                new_fit.append(eval_f(x))

            # opposition points (from random points)
            for _ in range(n_opp):
                x = rand_vec()
                xo = opposite(x)
                bounce_inplace(xo)
                fx = eval_f(x)
                fo = eval_f(xo)
                if fo < fx:
                    new_pop.append(xo)
                    new_fit.append(fo)
                else:
                    new_pop.append(x)
                    new_fit.append(fx)

            # local cloud around best
            local_sigma = max(0.08, sigma)
            for _ in range(n_local):
                x = best_x[:]
                for d in range(dim):
                    x[d] += randn() * (local_sigma * 0.20 * span(d))
                bounce_inplace(x)
                new_pop.append(x)
                new_fit.append(eval_f(x))

            pop, fit = new_pop, new_fit

            # refresh best
            best_i = min(range(pop_size), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                best_x = pop[best_i][:]
            best_t = time.time()

            # reset / perturb memories and archive to avoid lock-in
            archive = []
            for k in range(H):
                MCR[k] = min(0.95, max(0.05, MCR[k] + 0.04 * randn()))
                MF[k]  = min(0.95, max(0.05, MF[k]  + 0.04 * randn()))
            sigma = max(0.10, sigma)
            coord_step = [max(0.12 * span(i), coord_step[i]) for i in range(dim)]
