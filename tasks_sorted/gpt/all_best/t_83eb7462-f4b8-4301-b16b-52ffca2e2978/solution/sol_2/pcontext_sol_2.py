import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    Hybrid:
      1) JADE-like Differential Evolution: current-to-pbest/1 + external archive
         with per-individual self-adaptation of F and CR
      2) Lightweight trust-region local search around the incumbent (few evals)
      3) Stagnation-triggered partial restart (elite preserve + mixed resampling)

    Returns
    -------
    best : float
        Best (minimum) fitness found within time budget.
    """

    # -------------------- helpers --------------------
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

    def rand_cauchy(mu, gamma):
        # simple Cauchy sampler, no external libs
        u = random.random() - 0.5
        return mu + gamma * math.tan(math.pi * u)

    def pick_distinct(n, banned):
        while True:
            r = random.randrange(n)
            if r not in banned:
                return r

    widths = []
    for i in range(dim):
        lo, hi = bounds[i]
        w = hi - lo
        widths.append(w if w > 0 else 1.0)

    # -------------------- time --------------------
    start = time.perf_counter()
    deadline = start + float(max_time)

    def time_frac():
        now = time.perf_counter()
        if now <= start:
            return 0.0
        den = max(1e-12, deadline - start)
        t = (now - start) / den
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t

    # -------------------- initialization --------------------
    # A bit larger than plain DE for robustness, but still capped for speed.
    pop_size = max(14, min(64, 16 + 3 * dim))

    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # Per-individual control parameters (JADE/jDE flavor)
    F = [random.uniform(0.35, 0.9) for _ in range(pop_size)]
    CR = [random.uniform(0.1, 0.95) for _ in range(pop_size)]

    # JADE-style archive
    archive = []
    archive_max = pop_size

    # JADE means for control parameters (learned from successes)
    mu_F = 0.6
    mu_CR = 0.7
    c = 0.1  # learning rate

    # p-best fraction for current-to-pbest/1
    p_best_frac = 0.2

    # Stagnation handling
    no_improve_gens = 0
    last_best = best_val
    restart_after = max(25, 10 * dim)

    # Local search state (tiny trust region around best)
    tr_scale = 0.25  # relative to widths; will adapt

    # -------------------- local search --------------------
    def local_search(best_x, best_val, eval_budget, scale):
        """
        Cheap local improvement with a tiny trust region.
        Uses mostly coordinate moves + occasional random direction.
        """
        x = best_x[:]
        f = best_val

        # per-dim step sizes
        steps = [max(1e-12, scale * w) for w in widths]

        for _ in range(eval_budget):
            if time.perf_counter() >= deadline:
                break

            trial = x[:]
            if dim == 1 or random.random() < 0.75:
                j = random.randrange(dim)
                s = steps[j] * (1.0 if random.random() < 0.85 else 2.5)
                trial[j] += s if random.random() < 0.5 else -s
            else:
                for j in range(dim):
                    trial[j] += random.gauss(0.0, steps[j])

            clip_inplace(trial)
            ft = safe_eval(trial)

            if ft < f:
                x, f = trial, ft
                # expand a bit on success
                if random.random() < 0.3:
                    steps = [s * 1.15 for s in steps]
            else:
                # contract a bit on failure
                if random.random() < 0.4:
                    steps = [s * 0.9 for s in steps]

        return x, f

    # -------------------- main loop --------------------
    gen = 0
    while True:
        if time.perf_counter() >= deadline:
            return best_val

        gen += 1
        t = time_frac()

        # sort once per generation
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p_best_frac * pop_size)))

        # collect successful parameter values (for JADE updates)
        succ_F = []
        succ_CR = []
        succ_df = []  # fitness improvements for weighted mean

        improved = False

        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                return best_val

            # --- sample/adapt control parameters ---
            # CR from normal around mu_CR; clamp
            CRi = random.gauss(mu_CR, 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # F from Cauchy around mu_F; resample if <= 0, clamp to 1
            Fi = rand_cauchy(mu_F, 0.1)
            tries = 0
            while Fi <= 0.0 and tries < 5:
                Fi = rand_cauchy(mu_F, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            # time-bias: slightly smaller F late for fine-tuning
            Fi = max(0.05, min(1.0, Fi * (1.0 - 0.25 * t)))

            # store per-individual (not strictly needed, but can help persistence)
            F[i] = Fi
            CR[i] = CRi

            # choose pbest among top pcount
            pbest = order[random.randrange(pcount)]

            # choose r1 != i, pbest
            r1 = pick_distinct(pop_size, {i, pbest})

            # choose r2 from pop U archive, distinct from i,pbest,r1 if from pop
            pool_size = pop_size + len(archive)
            while True:
                if pool_size <= 0:
                    r2_vec = pop[pick_distinct(pop_size, {i, pbest, r1})]
                    break
                k = random.randrange(pool_size)
                if k < pop_size:
                    if k not in (i, pbest, r1):
                        r2_vec = pop[k]
                        break
                else:
                    r2_vec = archive[k - pop_size]
                    break

            xi = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]
            xr2 = r2_vec

            # --- mutation: current-to-pbest/1 ---
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

                # bounce-back boundary handling (often better than hard clip)
                lo, hi = bounds[j]
                if vj < lo:
                    vj = lo + 0.5 * (lo - vj)
                    if vj > hi:
                        vj = lo
                elif vj > hi:
                    vj = hi - 0.5 * (vj - hi)
                    if vj < lo:
                        vj = hi
                v[j] = vj

            # --- crossover (binomial) ---
            trial = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]

            ft = safe_eval(trial)

            # --- selection ---
            if ft <= fit[i]:
                # add replaced to archive
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                df = max(0.0, fit[i] - ft)

                pop[i] = trial
                fit[i] = ft

                # record successful params (for JADE mean update)
                succ_F.append(Fi)
                succ_CR.append(CRi)
                succ_df.append(df if df > 0 else 1e-12)

                if ft < best_val:
                    best_val = ft
                    best_x = trial[:]
                    improved = True

        # --- update JADE means ---
        if succ_F:
            # weighted Lehmer mean for F, weighted arithmetic for CR
            wsum = sum(succ_df)
            if wsum <= 0.0:
                wsum = float(len(succ_df))
                weights = [1.0 / wsum] * len(succ_df)
            else:
                weights = [d / wsum for d in succ_df]

            # Lehmer mean: sum(w*F^2) / sum(w*F)
            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for w, fval, crval in zip(weights, succ_F, succ_CR):
                num += w * fval * fval
                den += w * fval
                cr_mean += w * crval
            if den > 1e-12:
                mu_F = (1.0 - c) * mu_F + c * (num / den)
            mu_CR = (1.0 - c) * mu_CR + c * cr_mean

            # clamp means
            mu_F = max(0.05, min(0.95, mu_F))
            mu_CR = max(0.0, min(1.0, mu_CR))

        # --- local refinement occasionally (time-aware) ---
        # More often later, but still cheap.
        if time.perf_counter() < deadline and (gen % (3 if t < 0.7 else 2) == 0):
            # trust region shrinks over time; also adapts with success
            base = 0.22 * (1.0 - t) + 0.015 * t
            ls_budget = max(6, min(45, 3 * dim + 8))
            bx, bv = local_search(best_x, best_val, ls_budget, max(1e-6, tr_scale * base))
            if bv < best_val:
                best_x, best_val = bx, bv
                improved = True
                tr_scale = min(2.5, tr_scale * 1.15)
            else:
                tr_scale = max(0.2, tr_scale * 0.92)

        # --- stagnation / restart logic ---
        if best_val < last_best - 1e-12:
            last_best = best_val
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= restart_after and time.perf_counter() < deadline:
            no_improve_gens = 0

            # keep elites
            order = sorted(range(pop_size), key=lambda i: fit[i])
            elite = max(2, pop_size // 5)

            new_pop = [pop[idx][:] for idx in order[:elite]]
            new_fit = [fit[idx] for idx in order[:elite]]

            # time-dependent reseed: mix best-centered + global
            center_scale = 0.5 * (1.0 - t) + 0.06 * t

            while len(new_pop) < pop_size and time.perf_counter() < deadline:
                if random.random() < 0.75:
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

                if fx < best_val:
                    best_val = fx
                    best_x = x[:]

            pop = new_pop
            fit = new_fit
            archive = []
            mu_F = 0.6
            mu_CR = 0.7
            tr_scale = 0.25

    # unreachable
