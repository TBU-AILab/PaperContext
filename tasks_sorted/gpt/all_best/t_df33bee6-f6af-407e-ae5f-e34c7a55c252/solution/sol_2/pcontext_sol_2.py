import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries).

    Key upgrades vs the provided DE+pattern-search:
      - Better initialization: Latin-hypercube-like stratified sampling
      - Stronger global search: Differential Evolution "current-to-pbest/1" with
        p-best selection + archive (JADE/SHADE-inspired) for diversity
      - Self-adaptation: per-individual sampling of F and CR from memories
      - Robust local refinement: (1+1)-ES around best with 1/5 success rule
      - Restart on stagnation: rebuild population from elites + stratified + local cloud

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """

    # -------------------- helpers --------------------
    def is_finite(v):
        return v is not None and isinstance(v, (int, float)) and math.isfinite(v)

    def eval_f(x):
        try:
            v = func(x)
            v = float(v)
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    def clip_inplace(x):
        for i, (lo, hi) in enumerate(bounds):
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def span(i):
        lo, hi = bounds[i]
        s = hi - lo
        return s if s > 0 else 1.0

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # CLT normal-ish (mean 0, var ~1) without numpy
    def randn():
        return (random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() - 6.0)

    # Stratified sampling for quick coverage
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
        # Heavy-tailed step: tan(pi*(u-0.5)) ~ Cauchy(0,1)
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def normal_like(mu, sigma):
        return mu + sigma * randn()

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- parameters --------------------
    pop_size = max(14, min(70, 16 + 4 * dim))

    # JADE-ish settings
    p_rate = 0.18  # fraction for p-best
    archive = []   # stores replaced solutions (for mutation diversity)
    archive_max = pop_size

    # Parameter memories (SHADE-style, small)
    H = 6
    MCR = [0.85] * H
    MF  = [0.60] * H
    mem_k = 0

    # Local ES parameters
    sigma = 0.10
    sigma_min = 1e-14
    sigma_max = 0.35

    # Stagnation/restart
    best_t = start
    stall_seconds = max(0.30, 0.18 * max_time)

    # -------------------- init --------------------
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # -------------------- main loop --------------------
    while True:
        now = time.time()
        if now >= deadline:
            return best

        # Time fraction (0..1)
        t = 1.0 if max_time <= 0 else min(1.0, (now - start) / max_time)

        # Sort indices by fitness for p-best selection
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        p_num = max(2, int(math.ceil(p_rate * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        # Track successful parameters for memory update
        succ_CR = []
        succ_F = []
        succ_w = []

        # -------- DE generation (current-to-pbest/1 with archive) --------
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # sample memory index
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # sample CR from normal, F from heavy tail
            CR = normal_like(mu_cr, 0.12)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = cauchy_like(mu_f, 0.10)
            # resample F until in (0,1]
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 8:
                F = cauchy_like(mu_f, 0.10)
                tries += 1
            if F <= 0.0: F = 0.5
            if F > 1.0:  F = 1.0

            # choose pbest
            pbest = pop[random.choice(pbest_pool)]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            x_r1 = pop[r1]

            # choose r2 from pop U archive, distinct from i and r1
            union = pop + archive
            if len(union) < 2:
                union = pop
            # try a few draws to avoid equality; if not, accept
            x_r2 = union[random.randrange(len(union))]
            tries = 0
            while (x_r2 is xi or x_r2 is x_r1) and tries < 10:
                x_r2 = union[random.randrange(len(union))]
                tries += 1

            # mutation: current-to-pbest/1
            mutant = [0.0] * dim
            for d in range(dim):
                mutant[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d])
            clip_inplace(mutant)

            # binomial crossover
            trial = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    trial[d] = mutant[d]

            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                # store replaced to archive
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # random removal keeps it simple and effective
                    archive.pop(random.randrange(len(archive)))

                # selection
                old = fit[i]
                pop[i] = trial
                fit[i] = f_trial

                # record success for memory update (weight by improvement)
                if old != float("inf"):
                    w = max(0.0, old - f_trial)
                else:
                    w = 1.0
                succ_CR.append(CR)
                succ_F.append(F)
                succ_w.append(w + 1e-12)

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
                    best_t = time.time()

        # update parameter memories (weighted means)
        if succ_F:
            wsum = sum(succ_w)
            # weighted arithmetic mean for CR, weighted Lehmer mean for F
            cr_mean = sum(w * cr for w, cr in zip(succ_w, succ_CR)) / wsum

            num = sum(w * f * f for w, f in zip(succ_w, succ_F))
            den = sum(w * f for w, f in zip(succ_w, succ_F))
            f_mean = (num / den) if den > 0 else MF[mem_k]

            MCR[mem_k] = cr_mean
            MF[mem_k] = f_mean
            mem_k = (mem_k + 1) % H

        # -------- Local refinement: (1+1)-ES around best --------
        # allocate more local tries later in budget
        es_tries = 4 + int(14 * t)
        succ = 0
        att = 0
        for _ in range(es_tries):
            if time.time() >= deadline:
                return best
            att += 1
            cand = best_x[:]
            for d in range(dim):
                cand[d] += randn() * (sigma * span(d))
            clip_inplace(cand)
            f_c = eval_f(cand)
            if f_c < best:
                best = f_c
                best_x = cand
                best_t = time.time()
                succ += 1

        # 1/5 success rule (loose thresholds)
        if att > 0:
            rate = succ / float(att)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.22)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.72)

        # -------- Restart on stagnation --------
        if time.time() - best_t > stall_seconds:
            # keep a few elites, rebuild rest
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elite_k = max(3, pop_size // 7)
            elites = [pop[i][:] for i in idx_sorted[:elite_k]]
            elites_fit = [fit[i] for i in idx_sorted[:elite_k]]

            new_pop = elites[:]
            new_fit = elites_fit[:]

            remain = pop_size - elite_k
            n_global = remain // 2
            n_local = remain - n_global

            # global stratified
            for x in stratified_population(n_global):
                new_pop.append(x)
                new_fit.append(eval_f(x))

            # local cloud around best
            local_sigma = max(0.08, sigma)
            for _ in range(n_local):
                x = best_x[:]
                for d in range(dim):
                    x[d] += randn() * (local_sigma * 0.22 * span(d))
                clip_inplace(x)
                new_pop.append(x)
                new_fit.append(eval_f(x))

            pop, fit = new_pop, new_fit

            # refresh best
            best_i = min(range(pop_size), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                best_x = pop[best_i][:]
            best_t = time.time()

            # also refresh archive and memories slightly to avoid lock-in
            archive = []
            MCR = [min(0.95, max(0.05, m + randn() * 0.03)) for m in MCR]
            MF = [min(0.95, max(0.05, m + randn() * 0.03)) for m in MF]
            sigma = max(0.10, sigma)
