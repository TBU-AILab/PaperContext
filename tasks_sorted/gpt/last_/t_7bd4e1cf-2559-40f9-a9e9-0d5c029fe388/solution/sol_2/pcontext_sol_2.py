import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved stdlib-only optimizer for bounded continuous domains.

    Key upgrades vs previous:
      - Differential Evolution (DE/rand/1/bin) style global search (very strong baseline)
      - Lightweight local refinement on the current best (adaptive coordinate pattern search)
      - Time-aware population sizing and evaluation budgeting
      - Robust bound handling via reflection

    Returns:
      best (float): minimum fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ----------------- helpers -----------------
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect_1d(x, lo, hi):
        if lo == hi:
            return lo
        # reflect until inside
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        return clamp(x, lo, hi)

    def reflect_vec(v):
        out = v[:]
        for i in range(dim):
            out[i] = reflect_1d(out[i], lows[i], highs[i])
        return out

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(v):
        return float(func(v))

    # quick guard
    if max_time <= 0:
        # best effort single evaluation at center
        x = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
        return float(func(x))

    # ----------------- estimate evaluation speed -----------------
    # Do a few evaluations to estimate how many we can afford.
    probe_x = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    n_probe = 3
    probe_start = time.time()
    probe_best = float("inf")
    for _ in range(n_probe):
        if time.time() >= deadline:
            return probe_best if probe_best != float("inf") else float("inf")
        fx = evaluate(probe_x)
        if fx < probe_best:
            probe_best = fx
        # tiny perturbation for next probe
        for i in range(dim):
            if spans[i] > 0:
                probe_x[i] = reflect_1d(probe_x[i] + (random.random() - 0.5) * 0.01 * spans[i], lows[i], highs[i])
    probe_dt = max(1e-6, time.time() - probe_start)
    eval_time = probe_dt / float(n_probe)

    # Choose population size based on dim and time budget; keep it sensible.
    # DE works well with ~8D..20D, but we must be time-aware.
    time_left = max(0.0, deadline - time.time())
    approx_evals_left = max(20, int(time_left / max(eval_time, 1e-6)))

    # Let pop be at most a fraction of remaining evals so we can do generations.
    base_pop = 10 * dim
    pop_min = max(12, 4 * dim)
    pop_max = max(pop_min, min(80, approx_evals_left // 6))  # leave room for generations + local search
    pop_size = int(min(max(base_pop, pop_min), pop_max))

    # ----------------- initialize population (LHS-ish) -----------------
    # stratify each dimension to avoid clumping
    pop = []
    for k in range(pop_size):
        v = [0.0] * dim
        for i in range(dim):
            if spans[i] <= 0:
                v[i] = lows[i]
            else:
                # stratified bin with jitter; permute implicitly by using (k + rand)/pop
                u = (k + random.random()) / float(pop_size)
                v[i] = lows[i] + spans[i] * u
        random.shuffle(v)  # weak mixing across dims
        pop.append(reflect_vec(v))

    # also inject a few purely random points + center for robustness
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    inject = min(5, pop_size)
    for j in range(inject):
        pop[j] = rand_vec()
    if pop_size > 0:
        pop[0] = center[:]

    # evaluate initial pop
    fits = []
    best = float("inf")
    best_x = None
    for v in pop:
        if time.time() >= deadline:
            return best if best != float("inf") else probe_best
        fv = evaluate(v)
        fits.append(fv)
        if fv < best:
            best = fv
            best_x = v[:]

    # ----------------- DE main loop -----------------
    # Adaptive-ish parameters
    F = 0.6
    CR = 0.9
    # occasional "current-to-best" helps exploitation while retaining exploration
    p_current_to_best = 0.25

    # local search parameters
    # coordinate pattern search around best with decaying step
    local_every = 10  # generations between local refinements (time permitting)
    local_rel0 = 0.15
    local_rel_min = 1e-6
    local_shrink = 0.5

    gen = 0
    while True:
        if time.time() >= deadline:
            return best

        gen += 1

        # mild parameter jitter to avoid stagnation
        if (gen % 7) == 0:
            F = clamp(F + random.gauss(0.0, 0.08), 0.25, 0.95)
            CR = clamp(CR + random.gauss(0.0, 0.07), 0.05, 0.98)

        # one DE generation: pop_size trials (or fewer if nearly out of time)
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # pick r1,r2,r3 distinct and != i
            # (simple rejection sampling; pop_size is small-ish)
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)
            r3 = i
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(pop_size)

            x = pop[i]
            a = pop[r1]
            b = pop[r2]
            c = pop[r3]

            use_ctb = (best_x is not None) and (random.random() < p_current_to_best)

            # mutation
            mutant = [0.0] * dim
            if use_ctb:
                # current-to-best/1: v = x + F*(best-x) + F*(b-c)
                for d in range(dim):
                    mutant[d] = x[d] + F * (best_x[d] - x[d]) + F * (b[d] - c[d])
            else:
                # rand/1: v = a + F*(b-c)
                for d in range(dim):
                    mutant[d] = a[d] + F * (b[d] - c[d])

            # crossover (binomial)
            trial = x[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = mutant[d]

            trial = reflect_vec(trial)
            ftrial = evaluate(trial)

            # selection
            if ftrial <= fits[i]:
                pop[i] = trial
                fits[i] = ftrial
                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

        # ----------------- occasional local refinement on best -----------------
        if best_x is not None and (gen % local_every) == 0 and (time.time() < deadline):
            # Coordinate pattern search with adaptive step
            step_rel = local_rel0
            x = best_x[:]
            fx = best
            # cap local work to a small evaluation budget
            local_budget = max(8, 3 * dim)
            used = 0

            while step_rel >= local_rel_min and used < local_budget and time.time() < deadline:
                improved = False
                # random order of coordinates
                idx = list(range(dim))
                random.shuffle(idx)

                for d in idx:
                    if used >= local_budget or time.time() >= deadline:
                        break
                    sc = spans[d] * step_rel
                    if sc <= 0:
                        continue

                    # try + and -
                    for sgn in (1.0, -1.0):
                        if used >= local_budget or time.time() >= deadline:
                            break
                        cand = x[:]
                        cand[d] = cand[d] + sgn * sc
                        cand = reflect_vec(cand)
                        fc = evaluate(cand)
                        used += 1
                        if fc < fx:
                            x, fx = cand, fc
                            improved = True

                if improved:
                    best_x, best = x[:], fx
                    # slightly expand to keep moving if improvement continues
                    step_rel = min(0.5, step_rel / local_shrink)
                else:
                    step_rel *= local_shrink

            # reinsert improved best into population replacing worst (helps DE converge)
            worst_i = 0
            worst_f = fits[0]
            for i in range(1, pop_size):
                if fits[i] > worst_f:
                    worst_f = fits[i]
                    worst_i = i
            pop[worst_i] = best_x[:]
            fits[worst_i] = best

        # ----------------- diversity maintenance if stagnating -----------------
        # If population is too similar or best hasn't improved in a while, inject randomness.
        # Cheap heuristic: occasionally replace a few worst with random points.
        if (gen % 25) == 0 and time.time() < deadline:
            # replace up to ~10% worst
            krep = max(1, pop_size // 10)
            # get indices of worst
            worst = sorted(range(pop_size), key=lambda ii: fits[ii], reverse=True)[:krep]
            for ii in worst:
                if time.time() >= deadline:
                    break
                v = rand_vec()
                fv = evaluate(v)
                pop[ii] = v
                fits[ii] = fv
                if fv < best:
                    best = fv
                    best_x = v[:]
