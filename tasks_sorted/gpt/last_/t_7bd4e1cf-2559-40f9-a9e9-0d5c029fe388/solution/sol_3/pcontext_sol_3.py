import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger stdlib-only bounded optimizer (no numpy).

    Improvements vs previous:
      - DE with better initialization (proper per-dimension stratification; no shuffling bug)
      - p-best/1 + occasional current-to-best + rand/1 mixing (JADE-ish behavior)
      - Success-based adaptation of F and CR (lightweight, robust)
      - Restart / re-diversification on stagnation
      - Polishing local search using randomized coordinate + 1D quadratic fit fallback
      - Strict time-aware evaluation budgeting

    Returns:
      best (float): minimum fitness found within max_time seconds
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ----------------- basic guards -----------------
    if dim <= 0:
        # convention: evaluate empty vector
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect_1d(x, lo, hi):
        if hi <= lo:
            return lo
        # reflect into [lo, hi]
        w = hi - lo
        # fast reflection using modulo on a doubled interval
        # map to [0, 2w)
        y = (x - lo) % (2.0 * w)
        if y < 0.0:
            y += 2.0 * w
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def reflect_vec(v):
        out = v[:]
        for i in range(dim):
            out[i] = reflect_1d(out[i], lows[i], highs[i])
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

    # evaluation wrapper (keep it tiny)
    def evaluate(v):
        return float(func(v))

    if deadline <= t0:
        x = center_vec()
        return evaluate(x)

    # ----------------- estimate eval speed -----------------
    # Use a few random points to estimate time per evaluation.
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

    # ----------------- choose population size -----------------
    time_left = max(0.0, deadline - time.time())
    approx_evals_left = max(10, int(time_left / max(eval_time, 1e-9)))

    # want enough generations; constrain pop
    pop_min = max(12, 5 * dim)
    pop_max = max(pop_min, min(120, max(12, approx_evals_left // 8)))
    pop_size = int(min(max(12, 10 * dim), pop_max))
    pop_size = max(pop_size, pop_min)
    # if time is extremely tight, shrink more
    if approx_evals_left < pop_size + 5:
        pop_size = max(6, min(pop_size, max(6, approx_evals_left - 2)))

    # ----------------- stratified init (true per-dimension LHS) -----------------
    # Create a different permutation per dimension for stratification.
    perms = []
    for d in range(dim):
        p = list(range(pop_size))
        random.shuffle(p)
        perms.append(p)

    pop = []
    for i in range(pop_size):
        v = [0.0] * dim
        for d in range(dim):
            if spans[d] <= 0.0:
                v[d] = lows[d]
            else:
                # stratified bin with jitter
                u = (perms[d][i] + random.random()) / float(pop_size)
                v[d] = lows[d] + spans[d] * u
        pop.append(v)

    # ensure center and a few pure random injections
    if pop_size > 0:
        pop[0] = center_vec()
    for k in range(1, min(1 + max(1, pop_size // 10), pop_size)):
        pop[k] = rand_vec()

    # ----------------- evaluate initial population -----------------
    fits = [float("inf")] * pop_size
    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if time.time() >= deadline:
            return best if best < float("inf") else probe_best
        fx = evaluate(pop[i])
        fits[i] = fx
        if fx < best:
            best = fx
            best_x = pop[i][:]

    # ----------------- DE parameters / adaptation -----------------
    # JADE-ish:
    mu_F = 0.6
    mu_CR = 0.85
    c_adapt = 0.1

    # mixing probabilities
    p_ctb = 0.15  # current-to-best component
    p_rand1 = 0.20  # pure rand/1 to maintain exploration

    # p-best fraction
    pbest_frac = 0.2  # choose pbest among top p%

    # stagnation / restart
    best_at_restart = best
    no_improve_gens = 0
    stagn_gen_limit = 30

    # local polish config
    polish_every = 12
    polish_budget = max(10, 4 * dim)

    def pick_indices_excluding(exclude, n):
        # returns n distinct indices not containing 'exclude'
        chosen = []
        while len(chosen) < n:
            r = random.randrange(pop_size)
            if r == exclude:
                continue
            ok = True
            for c in chosen:
                if c == r:
                    ok = False
                    break
            if ok:
                chosen.append(r)
        return chosen

    def lehmer_mean(vals):
        # for F adaptation; avoid division by zero
        num = 0.0
        den = 0.0
        for x in vals:
            num += x * x
            den += x
        if den <= 1e-12:
            return None
        return num / den

    # ----------------- local polish (coordinate + quadratic probe) -----------------
    def polish(x0, f0):
        x = x0[:]
        fx = f0
        # start step relative to span
        step_rel = 0.12
        step_min = 1e-8
        used = 0

        idxs = list(range(dim))
        random.shuffle(idxs)

        while used < polish_budget and step_rel > step_min and time.time() < deadline:
            improved = False
            random.shuffle(idxs)
            for d in idxs:
                if used >= polish_budget or time.time() >= deadline:
                    break
                s = spans[d] * step_rel
                if s <= 0.0:
                    continue

                x_d = x[d]

                # Evaluate three points: -s, 0, +s for quadratic fit along this coordinate
                # Ensure inside bounds via reflection.
                x1 = x[:]
                x2 = x[:]
                x1[d] = reflect_1d(x_d - s, lows[d], highs[d])
                x2[d] = reflect_1d(x_d + s, lows[d], highs[d])

                f1 = evaluate(x1); used += 1
                if f1 < fx:
                    x, fx = x1, f1
                    improved = True
                    x_d = x[d]  # update
                    # continue to next dim; we already improved
                    continue
                if used >= polish_budget or time.time() >= deadline:
                    break

                f2 = evaluate(x2); used += 1
                if f2 < fx:
                    x, fx = x2, f2
                    improved = True
                    continue

                # Quadratic fit if we have f1, f0, f2 around x_d (original center at x)
                # Here "center" is current x, but we evaluated around previous x_d; ok as a heuristic:
                # If no direct improvement, try estimated minimizer in [-s, s].
                # Use formula for parabola through (-s,f1), (0,fx), (+s,f2).
                denom = (f1 - 2.0 * fx + f2)
                if abs(denom) > 1e-12:
                    t = 0.5 * (f1 - f2) / denom  # minimizer in units of s
                    t = clamp(t, -1.0, 1.0)
                    xq = x[:]
                    xq[d] = reflect_1d(x_d + t * s, lows[d], highs[d])
                    fq = evaluate(xq); used += 1
                    if fq < fx:
                        x, fx = xq, fq
                        improved = True

            if improved:
                # keep step, slightly enlarge a bit (but capped)
                step_rel = min(0.35, step_rel * 1.25)
            else:
                step_rel *= 0.5

        return x, fx

    # ----------------- main loop -----------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # rank population indices by fitness
        order = list(range(pop_size))
        order.sort(key=lambda i: fits[i])
        pbest_n = max(2, int(math.ceil(pbest_frac * pop_size)))
        top = order[:pbest_n]

        # success memories
        succ_F = []
        succ_CR = []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # sample CR ~ N(mu_CR, 0.1), clipped
            CR = mu_CR + random.gauss(0.0, 0.1)
            CR = clamp(CR, 0.0, 1.0)

            # sample F ~ Cauchy(mu_F, 0.1) via tangent method, resample if <=0
            # stdlib has no cauchy, implement via tan(pi*(u-0.5))
            F = -1.0
            tries = 0
            while F <= 0.0 and tries < 8:
                u = random.random()
                F = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            if F <= 0.0:
                F = mu_F
            F = clamp(F, 0.05, 1.0)

            # pick pbest
            pbest = random.choice(top)

            # pick r1,r2 distinct and not i and not pbest (as much as possible)
            r1, r2 = pick_indices_excluding(i, 2)
            if r1 == pbest:
                r1 = pick_indices_excluding(i, 1)[0]
            if r2 == pbest or r2 == r1:
                r2 = pick_indices_excluding(i, 1)[0]
                if r2 == r1:
                    r2 = (r2 + 1) % pop_size
                    if r2 == i:
                        r2 = (r2 + 1) % pop_size

            x = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # choose strategy mix
            u = random.random()
            mutant = [0.0] * dim
            if u < p_rand1:
                # rand/1: xr1 + F*(xp - xr2)  (use pbest as "b")
                for d in range(dim):
                    mutant[d] = xr1[d] + F * (xp[d] - xr2[d])
            elif u < p_rand1 + p_ctb and best_x is not None:
                # current-to-best/1 + difference
                # x + F*(best-x) + F*(xr1-xr2)
                for d in range(dim):
                    mutant[d] = x[d] + F * (best_x[d] - x[d]) + F * (xr1[d] - xr2[d])
            else:
                # pbest/1: x + F*(xp-x) + F*(xr1-xr2)
                for d in range(dim):
                    mutant[d] = x[d] + F * (xp[d] - x[d]) + F * (xr1[d] - xr2[d])

            # binomial crossover
            trial = x[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = mutant[d]

            trial = reflect_vec(trial)
            ftrial = evaluate(trial)

            if ftrial <= fits[i]:
                pop[i] = trial
                fits[i] = ftrial
                succ_F.append(F)
                succ_CR.append(CR)
                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

        # adapt mu_F and mu_CR
        if succ_F:
            mF = lehmer_mean(succ_F)
            if mF is not None:
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * mF
                mu_F = clamp(mu_F, 0.05, 0.95)
        if succ_CR:
            mCR = sum(succ_CR) / float(len(succ_CR))
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * mCR
            mu_CR = clamp(mu_CR, 0.0, 1.0)

        # stagnation tracking + occasional restart injection
        if best < best_at_restart - 1e-12:
            best_at_restart = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= stagn_gen_limit and time.time() < deadline:
            # replace worst ~30% with random points + keep best
            order = list(range(pop_size))
            order.sort(key=lambda i: fits[i], reverse=True)  # worst first
            nrep = max(1, pop_size // 3)
            for k in range(nrep):
                ii = order[k]
                if best_x is not None and pop[ii] == best_x:
                    continue
                if time.time() >= deadline:
                    break
                v = rand_vec()
                fv = evaluate(v)
                pop[ii] = v
                fits[ii] = fv
                if fv < best:
                    best = fv
                    best_x = v[:]
            no_improve_gens = 0
            # slightly increase exploration after restart
            mu_F = clamp(mu_F + 0.05, 0.05, 0.95)
            mu_CR = clamp(mu_CR - 0.05, 0.0, 1.0)

        # occasional polish
        if best_x is not None and (gen % polish_every) == 0 and time.time() < deadline:
            x2, f2 = polish(best_x, best)
            if f2 < best:
                best, best_x = f2, x2
                # inject improved best into population (replace worst)
                worst_i = max(range(pop_size), key=lambda i: fits[i])
                pop[worst_i] = best_x[:]
                fits[worst_i] = best

    return best
