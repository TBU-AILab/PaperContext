import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: Multi-strategy Differential Evolution (JADE-ish current-to-pbest)
    + archive + occasional restart + coordinate/local pattern search.

    - No external libraries.
    - func(params)->float where params is list/array-like length dim.
    - Respects bounds.
    - Returns best (float).

    Key upgrades vs the prior DE:
      * current-to-pbest/1 mutation (strong exploitation without premature collapse)
      * external archive to re-inject diversity (classic JADE idea)
      * per-individual adaptive F and CR sampling
      * occasional partial restart on stagnation
      * cheap coordinate-wise/pattern local search around incumbent best
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def clip_vec(v):
        out = v[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if out[i] < lo: out[i] = lo
            elif out[i] > hi: out[i] = hi
        return out

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(x):
        try:
            y = func(x)
            if y is None:
                return float("inf")
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf")
            return y
        except Exception:
            return float("inf")

    # Normal(0,1) via Box-Muller
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Cauchy(loc, scale): loc + scale * tan(pi*(u-0.5))
    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # choose an index by sorting fitness (returns list of indices)
    def argsort(seq):
        return sorted(range(len(seq)), key=lambda i: seq[i])

    # ---------------- initialization ----------------
    # population size tuned for time-bounded runs
    pop_size = max(16, 8 * dim)
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    order = argsort(fit)
    best_idx = order[0]
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # archive for diversity
    archive = []          # stores vectors only
    archive_max = pop_size

    # parameter adaptation memory (JADE-style global means)
    mu_F = 0.6
    mu_CR = 0.9

    # ranges for local steps
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_step = [0.20 * (r if r > 0 else 1.0) for r in ranges]

    stagnation = 0
    last_best = best

    # control knobs
    p_best_rate = 0.2              # top p% candidates for "pbest"
    min_pbest = 2
    max_pbest = max(min_pbest, int(p_best_rate * pop_size))
    # local search cadence
    local_every = 6

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1

        # Update ordering once per generation
        order = argsort(fit)

        # Detect stagnation
        if best < last_best - 1e-12:
            last_best = best
            stagnation = 0
        else:
            stagnation += 1

        # Partial restart if stuck: reinitialize worst half, keep best half + archive
        if stagnation >= 25 and time.time() < deadline:
            half = pop_size // 2
            worst = order[half:]  # indices of worse half
            for idx in worst:
                if time.time() >= deadline:
                    return best
                pop[idx] = rand_vec()
                fit[idx] = safe_eval(pop[idx])
            # trim archive too
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]
            stagnation = 0
            # refresh best
            order = argsort(fit)
            best_idx = order[0]
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = pop[best_idx][:]

        SF = []   # successful F
        SCR = []  # successful CR
        dF = []   # fitness improvements for weighting

        improved_any = False

        # Create a combined pool for r2 selection: population + archive
        # We'll index it on the fly to avoid copies.
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick pbest from top set
            pbest_count = max(min_pbest, min(max_pbest, pop_size))
            pbest_idx = order[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            # sample CR ~ N(mu_CR, 0.1), clipped
            CRi = clamp(mu_CR + 0.1 * randn(), 0.0, 1.0)

            # sample F from Cauchy(mu_F, 0.1) until in (0,1]
            Fi = rand_cauchy(mu_F, 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = rand_cauchy(mu_F, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            # select r1 != i from population
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # select r2 != i and != r1 from population+archive
            total_pool = pop_size + len(archive)
            # guarantee we can pick distinct
            if total_pool < 3:
                # fallback: random mutation
                trial = rand_vec()
                f_trial = safe_eval(trial)
                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
                if f_trial <= fi:
                    pop[i] = trial
                    fit[i] = f_trial
                continue

            def get_pool_vec(k):
                if k < pop_size:
                    return pop[k]
                return archive[k - pop_size]

            # choose r2 distinct from i and r1 if those are in population range
            while True:
                r2k = random.randrange(total_pool)
                # if r2 from population ensure not i/r1
                if r2k < pop_size:
                    if r2k != i and r2k != r1:
                        break
                else:
                    # archive always distinct from population indices
                    break

            xr1 = pop[r1]
            xr2 = get_pool_vec(r2k)

            # current-to-pbest/1:
            # v = xi + Fi*(xpbest - xi) + Fi*(xr1 - xr2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # Binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]  # trial
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]
            u = clip_vec(u)

            fu = safe_eval(u)

            # Selection + archive update
            if fu <= fi:
                # push replaced vector into archive
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # drop random elements to keep size
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fit[i] = fu
                improved_any = True

                # for parameter adaptation (success history)
                # weight by improvement magnitude
                imp = (fi - fu)
                if imp < 0.0:
                    imp = 0.0
                SF.append(Fi)
                SCR.append(CRi)
                dF.append(imp + 1e-12)

                if fu < best:
                    best = fu
                    best_x = u[:]

        # Adapt mu_F and mu_CR (JADE-style)
        if SF:
            # weighted Lehmer mean for F; weighted arithmetic mean for CR
            wsum = sum(dF)
            if wsum <= 0.0:
                wsum = float(len(dF))
                weights = [1.0 / wsum] * len(dF)
            else:
                weights = [df / wsum for df in dF]

            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for w, fval, crval in zip(weights, SF, SCR):
                num += w * (fval * fval)
                den += w * fval
                cr_mean += w * crval

            if den > 0.0:
                lehmerF = num / den
                # smoothing
                mu_F = clamp(0.9 * mu_F + 0.1 * lehmerF, 0.05, 0.95)
            mu_CR = clamp(0.9 * mu_CR + 0.1 * cr_mean, 0.0, 1.0)

        # ----------- occasional local search around best -----------
        # coordinate/pattern search + small gaussian probes, shrinking with time
        if gen % local_every == 0 and time.time() < deadline:
            time_frac = (time.time() - t0) / max(1e-9, max_time)
            shrink = max(0.01, 1.0 - 0.9 * time_frac)
            step = [bs * shrink for bs in base_step]

            x = best_x[:]
            fx = best

            # coordinate search passes
            passes = 1 + (1 if dim <= 10 else 0)
            for _ in range(passes):
                improved = False
                for d in range(dim):
                    if time.time() >= deadline:
                        return best
                    lo, hi = bounds[d]

                    # try +step then -step (or vice versa)
                    for direction in (1.0, -1.0):
                        cand = x[:]
                        cand[d] = clamp(cand[d] + direction * step[d], lo, hi)
                        fc = safe_eval(cand)
                        if fc < fx:
                            x, fx = cand, fc
                            improved = True
                            break
                if not improved:
                    break

            # a few gaussian probes around refined x
            probes = max(6, 2 * dim)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                cand = x[:]
                for d in range(dim):
                    cand[d] = cand[d] + step[d] * 0.5 * randn()
                cand = clip_vec(cand)
                fc = safe_eval(cand)
                if fc < fx:
                    x, fx = cand, fc

            if fx < best:
                best = fx
                best_x = x[:]
                # inject into population by replacing current worst
                worst_idx = max(range(pop_size), key=lambda k: fit[k])
                pop[worst_idx] = best_x[:]
                fit[worst_idx] = best

        # If nothing improved for a while, slightly increase exploration by lowering mu_CR
        if not improved_any and stagnation % 10 == 0:
            mu_CR = clamp(mu_CR * 0.95, 0.05, 1.0)
            mu_F = clamp(mu_F + 0.02, 0.05, 0.95)
