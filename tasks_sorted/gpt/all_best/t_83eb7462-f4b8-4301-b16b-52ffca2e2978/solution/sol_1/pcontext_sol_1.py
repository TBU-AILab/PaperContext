import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Hybrid approach:
      1) Differential Evolution (DE/current-to-pbest/1 + archive, jDE-style parameter adaptation)
      2) Lightweight local search around the incumbent (coordinate + stochastic steps)
      3) Occasional partial restart with mixture of global + best-centered sampling

    Returns
    -------
    best : float
        Best (minimum) fitness found within the time budget.
    """

    # ------------------------- helpers -------------------------
    def clip_inplace(x):
        for i in range(dim):
            lo, hi = bounds[i]
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def pick_distinct_excluding(n, exclude_set):
        # returns an int in [0,n) not in exclude_set
        while True:
            r = random.randrange(n)
            if r not in exclude_set:
                return r

    # approximate scale of each dimension for step sizes
    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # in case of degenerate bounds
    widths = [w if w > 0 else 1.0 for w in widths]

    # ------------------------- time -------------------------
    start = time.perf_counter()
    deadline = start + float(max_time)

    # ------------------------- DE setup -------------------------
    # Slightly larger than previous for better coverage, but still fast.
    pop_size = max(12, min(60, 14 + 3 * dim))

    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # jDE-like self-adaptive F and CR per individual
    F = [random.uniform(0.4, 0.9) for _ in range(pop_size)]
    CR = [random.uniform(0.2, 0.95) for _ in range(pop_size)]

    # external archive (JADE-style) to increase diversity
    archive = []          # list of vectors
    archive_max = pop_size

    # p-best fraction (current-to-pbest/1)
    p_best_frac = 0.2

    # restart / intensification controls
    no_improve_iters = 0
    last_best = best_val
    restart_after = max(30, 12 * dim)  # iterations without improvement triggers partial restart

    # local-search scheduling
    next_local_search = 0
    local_search_period = 3  # do a small local-search every few DE generations

    # ------------------------- local search -------------------------
    def local_improve(x0, f0, budget_steps, scale):
        """
        Very cheap derivative-free improvement:
        mixture of coordinate steps + random direction steps.
        """
        x = x0[:]
        f = f0

        # coordinate-wise step sizes (relative to bounds width)
        # scale is in [0.5..0.02] roughly depending on time/progress.
        steps = [max(1e-12, scale * w) for w in widths]

        for _ in range(budget_steps):
            if time.perf_counter() >= deadline:
                break

            if dim == 1 or random.random() < 0.65:
                # coordinate move
                j = random.randrange(dim)
                trial = x[:]
                # symmetric step; sometimes bigger jump
                s = steps[j] * (1.0 if random.random() < 0.8 else 3.0)
                trial[j] = trial[j] + (s if random.random() < 0.5 else -s)
                clip_inplace(trial)
            else:
                # random direction move
                trial = x[:]
                for j in range(dim):
                    # gaussian step per coordinate
                    trial[j] += random.gauss(0.0, steps[j])
                clip_inplace(trial)

            ft = safe_eval(trial)
            if ft < f:
                x, f = trial, ft
                # mild step increase upon success
                if random.random() < 0.2:
                    steps = [s * 1.2 for s in steps]
            else:
                # mild step decay
                if random.random() < 0.35:
                    steps = [s * 0.9 for s in steps]
        return x, f

    # ------------------------- main loop -------------------------
    gen = 0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            return best_val

        gen += 1

        # progress in [0,1]
        t = (now - start) / max(1e-12, (deadline - start))
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

        # p-best set size
        pcount = max(2, int(math.ceil(p_best_frac * pop_size)))

        # indices sorted by fitness once per generation (cheap)
        order = sorted(range(pop_size), key=lambda i: fit[i])

        improved = False

        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                return best_val

            # jDE parameter update (with small probability)
            if random.random() < 0.1:
                # bias F slightly down later in time for fine-tuning
                F[i] = 0.1 + random.random() * (0.9 - 0.5 * t)
                if F[i] > 1.0:
                    F[i] = 1.0
            if random.random() < 0.1:
                # CR tends to be higher early, slightly lower late
                base = 0.9 - 0.4 * t
                CR[i] = min(1.0, max(0.0, random.gauss(base, 0.1)))

            Fi = F[i]
            CRi = CR[i]

            # choose pbest from top pcount
            pbest = order[random.randrange(pcount)]

            # choose r1 from population, distinct
            r1 = pick_distinct_excluding(pop_size, {i, pbest})

            # choose r2 from population U archive, distinct
            pool_size = pop_size + len(archive)
            # map index to vector
            while True:
                r2_idx = random.randrange(pool_size) if pool_size > 0 else random.randrange(pop_size)
                if r2_idx < pop_size:
                    if r2_idx not in (i, pbest, r1):
                        x_r2 = pop[r2_idx]
                        break
                else:
                    # archive index
                    aidx = r2_idx - pop_size
                    if 0 <= aidx < len(archive):
                        x_r2 = archive[aidx]
                        # archive vectors are not any current index, so only need to check vs None
                        break

            xi = pop[i]
            x_pbest = pop[pbest]
            x_r1 = pop[r1]

            # DE/current-to-pbest/1:
            # v = xi + Fi*(x_pbest - xi) + Fi*(x_r1 - x_r2)
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (x_pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])
                # bounce-back boundary handling for better behavior than plain clip
                lo, hi = bounds[j]
                if vj < lo:
                    vj = lo + (lo - vj) * 0.5
                    if vj > hi:
                        vj = lo
                elif vj > hi:
                    vj = hi - (vj - hi) * 0.5
                    if vj < lo:
                        vj = hi
                v[j] = vj

            # binomial crossover
            trial = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]

            # selection
            ft = safe_eval(trial)
            if ft <= fit[i]:
                # successful: move current into archive
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # random removal maintains diversity
                    del archive[random.randrange(len(archive))]

                pop[i] = trial
                fit[i] = ft

                if ft < best_val:
                    best_val = ft
                    best_x = trial[:]
                    improved = True

        # small local improvement every few gens (time-aware)
        if gen >= next_local_search and time.perf_counter() < deadline:
            next_local_search = gen + local_search_period

            # scale decreases with time: more exploration early, more exploitation late
            scale = 0.25 * (1.0 - t) + 0.02 * t

            # keep it cheap: a few steps, proportional to dim but bounded
            steps = max(6, min(40, 4 * dim))
            bx, bv = local_improve(best_x, best_val, steps, scale)
            if bv < best_val:
                best_x, best_val = bx, bv
                improved = True

        # update stagnation detection
        if best_val < last_best - 1e-12:
            last_best = best_val
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        # partial restart if stagnating
        if no_improve_iters >= restart_after and time.perf_counter() < deadline:
            no_improve_iters = 0

            # keep elites, re-seed others (mix of best-centered + global)
            elite = max(2, pop_size // 5)
            elites = order[:elite]

            new_pop = [pop[idx][:] for idx in elites]
            new_fit = [fit[idx] for idx in elites]
            new_F = [F[idx] for idx in elites]
            new_CR = [CR[idx] for idx in elites]

            # restart sampling scale (shrinks over time)
            center_scale = 0.45 * (1.0 - t) + 0.05 * t

            while len(new_pop) < pop_size and time.perf_counter() < deadline:
                if random.random() < 0.7:
                    # best-centered gaussian
                    x = []
                    for d in range(dim):
                        lo, hi = bounds[d]
                        s = center_scale * widths[d]
                        val = random.gauss(best_x[d], s)
                        if val < lo:
                            val = lo
                        elif val > hi:
                            val = hi
                        x.append(val)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                new_pop.append(x)
                new_fit.append(fx)
                new_F.append(random.uniform(0.35, 0.9))
                new_CR.append(random.uniform(0.1, 0.95))

                if fx < best_val:
                    best_val = fx
                    best_x = x[:]

            pop, fit, F, CR = new_pop, new_fit, new_F, new_CR
            # refresh archive to avoid dragging old stuff
            archive = []
            archive_max = pop_size
