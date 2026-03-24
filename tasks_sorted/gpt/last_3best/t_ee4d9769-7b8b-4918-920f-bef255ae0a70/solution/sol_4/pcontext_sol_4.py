import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Hybrid: SHADE-style Differential Evolution (global, robust) + short
    pattern-search refinements (local). This generally dominates plain DE and
    simplified CMA variants under tight time limits.

    Key features:
    - Good init: stratified LHS + opposition + a few greedy “best-of-k” samples
    - Main engine: SHADE-like DE/current-to-pbest/1 with an archive (JADE/SHADE family)
      * per-individual F sampled from Cauchy around historical mean
      * per-individual CR sampled from Normal around historical mean
      * successful parameters update memory (Lehmer mean for F)
    - Bound handling: reflection (works well with vector differentials)
    - Time-aware: always checks deadline; adapts p-best fraction over time
    - Occasional local search around best with step shrinking

    Returns:
        best fitness (float)
    """

    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def reflect_scalar(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect until in range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def reflect_vec(x):
        return [reflect_scalar(x[i], i) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def lhs_points(n):
        # cheap LHS-like stratification
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for j in range(dim):
                u = (perms[j][k] + random.random()) / n
                x[j] = lows[j] + u * spans[j]
            pts.append(x)
        return pts

    # --------- small utilities for SHADE ----------
    def clamp01(v):
        if v < 0.0: return 0.0
        if v > 1.0: return 1.0
        return v

    def rand_cauchy(loc, scale):
        # Cauchy: loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def sample_F(muF):
        # Keep drawing until positive, then cap to 1
        for _ in range(12):
            f = rand_cauchy(muF, 0.1)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        # fallback
        return max(0.1, min(1.0, muF))

    def sample_CR(muCR):
        cr = random.gauss(muCR, 0.1)
        return clamp01(cr)

    # --------- choose population size time-aware ----------
    # For tight budgets, moderate population gives more generations.
    pop_size = 12 + 6 * dim
    if pop_size < 24: pop_size = 24
    if pop_size > 96: pop_size = 96
    if max_time <= 0.35:
        pop_size = min(pop_size, 36)
    elif max_time <= 0.8:
        pop_size = min(pop_size, 60)

    # --------- initialization ----------
    best = float("inf")
    best_x = None

    pop = []
    fit = []

    # Spend limited evaluations on diversified init
    init_n = pop_size
    # LHS half
    pts = lhs_points(max(4, init_n // 2))
    for x in pts:
        if now() >= deadline:
            return best
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

        if len(pop) >= init_n:
            break

        # opposition point
        ox = reflect_vec(opposite_point(x))
        if now() >= deadline:
            return best
        fox = eval_f(ox)
        pop.append(ox); fit.append(fox)
        if fox < best:
            best, best_x = fox, ox[:]

        if len(pop) >= init_n:
            break

    # fill remainder with "best-of-k random" (a bit greedier than pure random)
    while len(pop) < pop_size:
        if now() >= deadline:
            return best
        k = 3 if dim <= 10 else 2
        bx, bf = None, float("inf")
        for _ in range(k):
            x = rand_point()
            fx = eval_f(x)
            if fx < bf:
                bx, bf = x, fx
            if fx < best:
                best, best_x = fx, x[:]
            if now() >= deadline:
                return best
        pop.append(bx); fit.append(bf)

    # --------- SHADE memory + archive ----------
    H = max(6, min(25, 2 * int(math.sqrt(dim + 1)) + 6))
    M_F = [0.5] * H
    M_CR = [0.8] * H
    k_mem = 0

    archive = []
    archive_max = pop_size

    def get_from_pool(idx):
        # pool = pop + archive
        if idx < pop_size:
            return pop[idx]
        return archive[idx - pop_size]

    def pick_distinct(exclude, count, pool_n):
        chosen = set()
        while len(chosen) < count:
            r = random.randrange(pool_n)
            if r == exclude:
                continue
            chosen.add(r)
        return list(chosen)

    # --------- local refinement (pattern search) ----------
    def local_refine(x0, f0, time_cap):
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        # start with moderate step, shrink on failure
        step = [0.08 * spans[i] for i in range(dim)]
        min_step = [max(1e-12, 1e-9 * spans[i]) for i in range(dim)]

        no_improve_rounds = 0
        while now() < t_end:
            improved = False

            # coordinate probes (random order)
            order = list(range(dim))
            random.shuffle(order)
            probes = min(dim, 14)
            for jj in range(probes):
                if now() >= t_end:
                    break
                j = order[jj]
                sj = step[j]
                if sj < min_step[j]:
                    continue
                base = x[j]
                for sign in (1.0, -1.0):
                    cand = x[:]
                    cand[j] = reflect_scalar(base + sign * sj, j)
                    fc = eval_f(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        break

            # one random-direction probe
            if now() < t_end:
                v = [random.gauss(0.0, 1.0) for _ in range(dim)]
                nrm = math.sqrt(sum(vi * vi for vi in v)) or 1.0
                scale = (sum(step) / dim) * (0.4 + 0.7 * random.random())
                cand = x[:]
                for i in range(dim):
                    cand[i] = reflect_scalar(cand[i] + scale * (v[i] / nrm), i)
                fc = eval_f(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True

            if improved:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                # shrink steps
                shrink = 0.6 if no_improve_rounds < 3 else 0.5
                for i in range(dim):
                    step[i] *= shrink
                if all(step[i] < min_step[i] for i in range(dim)):
                    break

        return x, fx

    # --------- main loop ----------
    last_refine = start
    refine_interval = max(0.12, 0.06 * max_time)

    while True:
        t = now()
        if t >= deadline:
            return best

        # occasional local refinement near current best
        if best_x is not None and (t - last_refine) >= refine_interval:
            remaining = deadline - t
            if remaining > 0.05:
                cap = min(0.07 * max_time, 0.22 * remaining, 0.30)
                rx, rf = local_refine(best_x, best, cap)
                if rf < best:
                    best, best_x = rf, rx[:]
                    # inject into worst to propagate
                    worst = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst] = best_x[:]
                    fit[worst] = best
                last_refine = now()

        # progress-dependent pbest fraction (more exploit later)
        prog = (t - start) / (max_time if max_time > 1e-12 else 1e-12)
        p = 0.28 - 0.18 * max(0.0, min(1.0, prog))  # 0.28 -> 0.10
        pbest_count = max(2, int(p * pop_size))

        # rank indices by fitness
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        S_F = []
        S_CR = []
        dF = []  # fitness improvements for weighting

        # one generation
        for i in range(pop_size):
            if now() >= deadline:
                return best

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]
            Fi = sample_F(muF)
            CRi = sample_CR(muCR)

            # choose pbest from top set
            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            pool_n = pop_size + len(archive)
            r1, r2 = pick_distinct(exclude=i, count=2, pool_n=pool_n)
            x_r1 = get_from_pool(r1)
            x_r2 = get_from_pool(r2)

            # DE/current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect_scalar(v[d], d)

            fu = eval_f(u)

            if fu <= fit[i]:
                # archive the replaced parent (for diversity)
                if len(archive) < archive_max:
                    archive.append(pop[i])
                else:
                    archive[random.randrange(archive_max)] = pop[i]

                # record successful params
                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(max(0.0, fit[i] - fu))

                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best, best_x = fu, u[:]

        # update SHADE memory
        if S_F:
            # weights proportional to improvement (or uniform if zero)
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / wsum for di in dF]

            # Lehmer mean for F: sum(w * F^2) / sum(w * F)
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            new_MF = (num / den) if den > 1e-18 else M_F[k_mem]

            # weighted mean for CR
            new_MCR = 0.0
            for w, cr in zip(weights, S_CR):
                new_MCR += w * cr

            # mild smoothing/robustness
            M_F[k_mem] = min(1.0, max(0.05, new_MF))
            M_CR[k_mem] = clamp01(new_MCR)

            k_mem = (k_mem + 1) % H

        # keep archive size bounded
        if len(archive) > archive_max:
            archive = archive[-archive_max:]
