import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs):
      - Differential Evolution with SHADE-style adaptive memories for F and CR (very robust on many black-box tasks)
      - current-to-pbest/1 mutation (good balance of exploration/exploitation)
      - External archive (JADE/SHADE idea) to maintain diversity
      - Budgeted, greedy pattern/coordinate local search around global best
      - Reflection bounds handling (stable) + occasional mild restarts on stagnation

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    # --- utilities ---
    def time_left():
        return time.time() < deadline

    def safe_eval(x):
        try:
            v = func(x)
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def reflect_into_bounds(x):
        # reflect repeatedly (handles overshoot); keeps values in [lo, hi]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            # avoid pathological infinite loops with huge values
            # by folding using span
            span = hi - lo
            if span <= 0.0:
                x[i] = lo
                continue
            if v < lo or v > hi:
                # map to [0, 2*span) then reflect
                u = (v - lo) % (2.0 * span)
                if u < 0.0:
                    u += 2.0 * span
                if u > span:
                    u = 2.0 * span - u
                v = lo + u
            x[i] = v
        return x

    def clamp01(v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    # Normal sampler (Box-Muller), avoiding random.gauss dependence variability
    _has_spare = [False]
    _spare = [0.0]
    def randn():
        if _has_spare[0]:
            _has_spare[0] = False
            return _spare[0]
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _spare[0] = z1
        _has_spare[0] = True
        return z0

    # Cauchy sampler for F (heavy-tailed); resample until within (0,1]
    def cauchy(loc, scale):
        # loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # --- parameters ---
    # population sizing: keep moderate; too large wastes time on slow funcs
    pop_size = max(18, min(70, 8 * dim + 10))

    # SHADE memory size
    H = max(5, min(25, pop_size // 2))
    MCR = [0.9 for _ in range(H)]
    MF = [0.5 for _ in range(H)]
    mem_idx = 0

    # p-best selection fraction (current-to-pbest/1)
    p_min, p_max = 0.08, 0.25
    p = p_min + (p_max - p_min) * 0.5

    # archive (stores replaced individuals)
    archive = []
    archive_max = pop_size

    # stagnation controls
    stagnation = 0
    last_best = float("inf")
    restart_limit = 40  # generations w/o improvement
    # local search frequency
    ls_every = 6

    # --- init ---
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [float("inf")] * pop_size

    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if not time_left():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # --- local search around global best: greedy coordinate/pattern ---
    def local_search(x0, f0, eval_budget):
        if x0 is None:
            return x0, f0
        x = x0[:]
        f = f0
        # start step relative to span; shrink when no improvements
        step = [0.15 * spans[j] for j in range(dim)]
        noimp = 0
        for _ in range(eval_budget):
            if not time_left():
                break
            j = random.randrange(dim)
            if step[j] <= 1e-14 * (spans[j] + 1.0):
                continue

            base = x[j]
            amp = step[j] * (0.5 + random.random())
            cand1 = x[:]
            cand1[j] = base + amp
            reflect_into_bounds(cand1)
            f1 = safe_eval(cand1)

            if f1 < f:
                x, f = cand1, f1
                noimp = 0
                continue

            cand2 = x[:]
            cand2[j] = base - amp
            reflect_into_bounds(cand2)
            f2 = safe_eval(cand2)

            if f2 < f:
                x, f = cand2, f2
                noimp = 0
                continue

            # no improvement => shrink selected coordinate, sometimes global shrink
            step[j] *= 0.75
            noimp += 1
            if noimp >= max(10, dim):
                # occasional global shrink
                for k in range(dim):
                    step[k] *= 0.9
                noimp = 0
        return x, f

    gen = 0
    while time_left():
        gen += 1

        # rank indices by fitness for p-best and best tracking
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        if fit[idx_sorted[0]] < best:
            best = fit[idx_sorted[0]]
            best_x = pop[idx_sorted[0]][:]
        # stagnation update
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # occasional local improvement
        if best_x is not None and (gen % ls_every == 0):
            ls_budget = max(6, min(30, 2 * dim + 6))
            bx, bf = local_search(best_x, best, ls_budget)
            if bf < best:
                best, best_x = bf, bx[:]
                stagnation = 0
                last_best = best

        # mild restart if stagnating: re-seed some worst around best + random
        if stagnation >= restart_limit:
            # reseed bottom fraction
            frac = 0.25
            k = max(2, int(pop_size * frac))
            worst = idx_sorted[-k:]
            for wi in worst:
                if not time_left():
                    return best
                if best_x is not None and random.random() < 0.65:
                    # sample around best with decreasing radius
                    x = best_x[:]
                    rad = 0.25 * (0.5 ** min(6, stagnation // restart_limit))
                    for d in range(dim):
                        x[d] = x[d] + rad * spans[d] * randn()
                    reflect_into_bounds(x)
                else:
                    x = rand_vec()
                pop[wi] = x
                fit[wi] = safe_eval(x)
            archive.clear()
            stagnation = 0

        # choose p based on progress (slightly more exploit later)
        # progress proxy: generations elapsed -> increase exploitation a bit
        if gen < 30:
            p = 0.22
        elif gen < 80:
            p = 0.16
        else:
            p = 0.10

        # generation accumulators for memory update
        SCR = []
        SF = []
        dF = []  # fitness improvements

        # prebuild union pool indices for mutation (pop + archive)
        # We'll sample vectors by taking actual vectors from list references
        union = pop + archive
        union_size = len(union)

        for i in range(pop_size):
            if not time_left():
                return best

            xi = pop[i]

            # pick memory slot
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # sample CR ~ N(mu_cr, 0.1)
            CR = clamp01(mu_cr + 0.1 * randn())

            # sample F ~ Cauchy(mu_f, 0.1), retry
            F = cauchy(mu_f, 0.1)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 12:
                F = cauchy(mu_f, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.5
            if F > 1.0:
                F = 1.0

            # choose pbest from top p%
            pcount = max(2, int(math.ceil(p * pop_size)))
            pbest_idx = idx_sorted[random.randrange(pcount)]
            x_pbest = pop[pbest_idx]

            # choose r1 from population != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            x_r1 = pop[r1]

            # choose r2 from union != i and != r1 and != pbest_idx (indexing differs for archive)
            # We'll just reject by object identity (list) to stay simple.
            x_r2 = None
            if union_size <= 2:
                x_r2 = pop[random.randrange(pop_size)]
            else:
                for _ in range(30):
                    cand = union[random.randrange(union_size)]
                    if cand is xi or cand is x_r1 or cand is x_pbest:
                        continue
                    x_r2 = cand
                    break
                if x_r2 is None:
                    x_r2 = union[random.randrange(union_size)]

            # current-to-pbest/1: v = xi + F*(x_pbest - xi) + F*(x_r1 - x_r2)
            donor = [0.0] * dim
            for j in range(dim):
                donor[j] = xi[j] + F * (x_pbest[j] - xi[j]) + F * (x_r1[j] - x_r2[j])

            # binomial crossover
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]
            reflect_into_bounds(trial)

            ftrial = safe_eval(trial)
            fi = fit[i]

            if ftrial <= fi:
                # success: add replaced to archive
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    # random replacement
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = trial
                fit[i] = ftrial

                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

                # memory update stats
                df = fi - ftrial
                if df < 0.0:
                    df = 0.0
                SCR.append(CR)
                SF.append(F)
                dF.append(df)

        # update memories (weighted Lehmer mean for F, weighted mean for CR)
        if dF:
            wsum = sum(dF)
            if wsum <= 0.0:
                # if all equal improvements, fallback to uniform weights
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / wsum for di in dF]

            # weighted mean for CR
            mcr = 0.0
            for wi, cri in zip(w, SCR):
                mcr += wi * cri

            # weighted Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, SF):
                num += wi * fi * fi
                den += wi * fi
            mf = (num / den) if den > 1e-12 else 0.5

            MCR[mem_idx] = clamp01(mcr)
            MF[mem_idx] = min(1.0, max(1e-6, mf))
            mem_idx = (mem_idx + 1) % H

    return best
