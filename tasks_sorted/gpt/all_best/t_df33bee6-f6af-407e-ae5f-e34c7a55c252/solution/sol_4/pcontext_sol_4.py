import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (self-contained; no external libs).

    Improvements vs the provided best:
      - Better early progress: stratified init + opposition + small "radius" cloud
      - Stronger global search: JADE/SHADE-like DE (current-to-pbest/1 + archive)
        + *per-individual* choice among 3 trial generators (pbest, rand/1, best/2)
      - More robust boundary handling: reflection ("bounce") + rare random re-entry
      - More reliable exploitation: short trust-region coordinate search around best
        with adaptive radius + (1+1)-ES with success rule
      - Population "aging": periodic partial refresh while preserving elites (less stall)
      - Stagnation restart: multi-source rebuild (elites + stratified + opposition + local)

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
        # reflection boundary handling
        for i, (lo, hi) in enumerate(bounds):
            w = hi - lo
            if w <= 0:
                x[i] = lo
                continue
            xi = x[i]
            if xi < lo or xi > hi:
                y = (xi - lo) % (2.0 * w)
                x[i] = lo + (2.0 * w - y if y > w else y)
        return x

    def reenter_random_dim(x, p=0.02):
        # rare random re-entry for stubborn dimensions after reflection
        for i, (lo, hi) in enumerate(bounds):
            if random.random() < p:
                x[i] = random.uniform(lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    def randn():
        # approx N(0,1) using CLT: sum of 12 U(0,1) - 6
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

    def pick_index(n, forbid):
        j = random.randrange(n)
        tries = 0
        while j in forbid and tries < 30:
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

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- parameters --------------------
    # Population sizing: slightly larger than before for better coverage, but bounded.
    pop_size = max(22, min(110, 22 + 6 * dim))

    # JADE/SHADE-ish
    p_rate = 0.20
    archive = []
    archive_max = pop_size

    H = 10
    MCR = [0.85] * H
    MF  = [0.60] * H
    mem_k = 0

    # Local search controls
    sigma = 0.10
    sigma_min = 1e-16
    sigma_max = 0.45

    # Trust-region coordinate search radius
    tr = [0.12 * span(i) for i in range(dim)]
    tr_min = [1e-16 * span(i) + 1e-18 for i in range(dim)]
    tr_max = [0.40 * span(i) for i in range(dim)]

    # Stagnation / refresh
    best_t = start
    stall_seconds = max(0.25, 0.13 * max_time)
    last_refresh_t = start
    refresh_period = max(0.35, 0.18 * max_time)  # partial refresh even without full stall

    # -------------------- init: stratified + opposition + local cloud --------------------
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    # opposition (often yields a big early jump)
    for i in range(pop_size):
        if time.time() >= deadline:
            _, b = argmin(fit)
            return b
        xo = opposite(pop[i])
        bounce_inplace(xo)
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i] = xo
            fit[i] = fo

    best_i, best = argmin(fit)
    best_x = pop[best_i][:]

    # small local cloud around best to seed exploitation early
    cloud_n = min(pop_size // 6, 10)
    for _ in range(cloud_n):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for d in range(dim):
            x[d] += randn() * (0.05 * span(d))
        bounce_inplace(x)
        f = eval_f(x)
        # replace a random non-best if improved vs that slot
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
                best = f
                best_x = x[:]
                best_t = time.time()

    # -------------------- main loop --------------------
    while True:
        now = time.time()
        if now >= deadline:
            return best

        t = 1.0 if max_time <= 0 else min(1.0, (now - start) / max_time)

        # sort for p-best
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        p_num = max(2, int(math.ceil(p_rate * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        succ_CR, succ_F, succ_w = [], [], []

        union = pop + archive
        ulen = len(union)

        # ---- DE sweep ----
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            r = random.randrange(H)
            mu_cr, mu_f = MCR[r], MF[r]

            CR = normal_like(mu_cr, 0.10 + 0.05 * (1.0 - t))
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = cauchy_like(mu_f, 0.08 + 0.06 * (1.0 - t))
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 12:
                F = cauchy_like(mu_f, 0.08 + 0.06 * (1.0 - t))
                tries += 1
            if F <= 0.0: F = 0.5
            if F > 1.0:  F = 1.0

            # build 3 candidate trials, pick best among them (extra evals but higher quality)
            # Strategy 1: current-to-pbest/1 with archive
            pbest = pop[random.choice(pbest_pool)]
            r1 = pick_index(pop_size, {i})
            r2 = pick_index(ulen, {i, r1}) if ulen > 2 else pick_index(pop_size, {i, r1})
            x_r1 = pop[r1]
            x_r2 = union[r2] if ulen > 2 else pop[r2]

            mut1 = [0.0] * dim
            for d in range(dim):
                mut1[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d])
            bounce_inplace(mut1)
            if random.random() < 0.07:
                reenter_random_dim(mut1, p=0.03)

            tr1 = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    tr1[d] = mut1[d]
            f1 = eval_f(tr1)

            # Strategy 2: rand/1 with archive (exploration)
            a = pick_index(pop_size, {i})
            b = pick_index(pop_size, {i, a})
            c = pick_index(ulen, {i, a, b}) if ulen > 3 else pick_index(pop_size, {i, a, b})
            xa, xb = pop[a], pop[b]
            xc = union[c] if ulen > 3 else pop[c]

            mut2 = [0.0] * dim
            for d in range(dim):
                mut2[d] = xa[d] + F * (xb[d] - xc[d])
            bounce_inplace(mut2)

            tr2 = xi[:]
            CR2 = min(1.0, max(0.0, CR + 0.12 * (1.0 - t)))
            jrand2 = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR2 or d == jrand2:
                    tr2[d] = mut2[d]
            f2 = eval_f(tr2)

            # Strategy 3: best/2 (strong exploitation; late-biased)
            # Use it more later; still always evaluated but will often lose early.
            r3 = pick_index(pop_size, {i})
            r4 = pick_index(pop_size, {i, r3})
            r5 = pick_index(pop_size, {i, r3, r4})
            r6 = pick_index(pop_size, {i, r3, r4, r5})
            x3, x4 = pop[r3], pop[r4]
            x5, x6 = pop[r5], pop[r6]

            F3 = min(1.0, max(0.15, F * (0.75 + 0.60 * t)))  # slightly more conservative
            mut3 = [0.0] * dim
            for d in range(dim):
                mut3[d] = best_x[d] + F3 * (x3[d] - x4[d]) + F3 * (x5[d] - x6[d])
            bounce_inplace(mut3)

            tr3 = xi[:]
            CR3 = min(1.0, max(0.0, 0.65 * CR + 0.30 + 0.15 * t))
            jrand3 = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR3 or d == jrand3:
                    tr3[d] = mut3[d]
            f3 = eval_f(tr3)

            # pick best trial
            if f2 < f1:
                best_trial, f_trial, used_CR = tr2, f2, CR2
            else:
                best_trial, f_trial, used_CR = tr1, f1, CR
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
                    best = f_trial
                    best_x = best_trial[:]
                    best_t = time.time()

        # ---- memory update (damped) ----
        if succ_F:
            wsum = sum(succ_w)
            cr_mean = sum(w * cr for w, cr in zip(succ_w, succ_CR)) / wsum
            num = sum(w * f * f for w, f in zip(succ_w, succ_F))
            den = sum(w * f for w, f in zip(succ_w, succ_F))
            f_mean = (num / den) if den > 0 else MF[mem_k]

            MCR[mem_k] = 0.25 * MCR[mem_k] + 0.75 * cr_mean
            MF[mem_k]  = 0.25 * MF[mem_k]  + 0.75 * f_mean
            mem_k = (mem_k + 1) % H

        # ---- local: (1+1)-ES around best ----
        es_tries = 5 + int(18 * t)
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
            f = eval_f(cand)
            if f < best:
                best = f
                best_x = cand
                best_t = time.time()
                succ += 1

        if att:
            rate = succ / float(att)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.18)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.75)

        # ---- local: trust-region coordinate search (cheap, consistent) ----
        # evaluates 2-4 points in a few dimensions each iteration
        k = min(dim, 2 + int(5 * t))
        for _ in range(k):
            if time.time() >= deadline:
                return best
            d = random.randrange(dim)
            rd = tr[d]
            if rd <= tr_min[d]:
                continue

            x0 = best_x[d]
            # try +/- rd then +/- rd/2 (in this order)
            for xd in (x0 + rd, x0 - rd, x0 + 0.5 * rd, x0 - 0.5 * rd):
                cand = best_x[:]
                cand[d] = xd
                bounce_inplace(cand)
                f = eval_f(cand)
                if f < best:
                    best = f
                    best_x = cand
                    best_t = time.time()
                    tr[d] = min(tr_max[d], tr[d] * 1.15)
                    break
            else:
                tr[d] = tr[d] * 0.62

        # ---- periodic partial refresh ("aging") ----
        # helps avoid slow lock-in even if not fully stalled
        now = time.time()
        if now - last_refresh_t > refresh_period and t > 0.15:
            elite_k = max(3, pop_size // 8)
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elites = set(idx_sorted[:elite_k])

            # replace a slice of worst individuals
            repl = max(2, pop_size // 6)
            worst = idx_sorted[-repl:]
            for wi in worst:
                if time.time() >= deadline:
                    return best
                if wi in elites:
                    continue
                # mix local + opposition-global
                if random.random() < 0.55:
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += randn() * (max(0.06, sigma) * 0.22 * span(d))
                    bounce_inplace(x)
                else:
                    x = rand_vec()
                    if random.random() < 0.5:
                        x = opposite(x)
                        bounce_inplace(x)
                f = eval_f(x)
                pop[wi] = x
                fit[wi] = f
                if f < best:
                    best = f
                    best_x = x[:]
                    best_t = time.time()

            archive = []
            last_refresh_t = time.time()

        # ---- full restart on stagnation ----
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
                    x[d] += randn() * (local_sigma * 0.25 * span(d))
                bounce_inplace(x)
                new_pop.append(x)
                new_fit.append(eval_f(x))

            pop, fit = new_pop, new_fit
            bi, bv = argmin(fit)
            if bv < best:
                best = bv
                best_x = pop[bi][:]
            best_t = time.time()
            last_refresh_t = best_t

            archive = []
            # small memory perturbation to avoid repeating the same dynamics
            for k in range(H):
                MCR[k] = min(0.95, max(0.05, MCR[k] + 0.035 * randn()))
                MF[k]  = min(0.95, max(0.05, MF[k]  + 0.035 * randn()))
            sigma = max(0.10, sigma)
            for d in range(dim):
                tr[d] = max(0.10 * span(d), tr[d])
