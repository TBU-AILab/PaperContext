import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Upgrade vs prior SHADE+pattern:
    - Adds a true "memetic" local optimizer: Adaptive Coordinate Descent + 2-point quadratic
      interpolation per coordinate (very cheap, strong for separable/weakly-coupled problems).
    - Uses a restart-on-stagnation schedule + shrinking trust region around best (intensification).
    - Keeps SHADE-like DE/current-to-pbest/1 + archive, but with:
        * sinusoidal p-best pressure schedule (helps escape plateaus)
        * occasional "best/2" mutant injection when diversity collapses
    - More evaluation-efficient initialization: LHS + elite resampling around best.

    Returns:
        best fitness (float)
    """
    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def now():
        return time.time()

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

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

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def lhs_points(n):
        # Simple stratified sampling (LHS-like)
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

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def sample_F(muF):
        for _ in range(16):
            f = rand_cauchy(muF, 0.12)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return max(0.15, min(1.0, muF))

    def sample_CR(muCR):
        return clamp01(random.gauss(muCR, 0.12))

    # ---------- sizing ----------
    pop_size = 14 + 6 * dim
    if pop_size < 26: pop_size = 26
    if pop_size > 110: pop_size = 110
    if max_time <= 0.35:
        pop_size = min(pop_size, 40)
    elif max_time <= 0.8:
        pop_size = min(pop_size, 70)

    # ---------- initialization ----------
    best = float("inf")
    best_x = None

    pop, fit = [], []

    init_n = pop_size
    n_lhs = max(6, init_n // 2)
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return best
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        if len(pop) >= init_n:
            break

        ox = [reflect_scalar(opposite_point(x)[i], i) for i in range(dim)]
        if now() >= deadline:
            return best
        fox = eval_f(ox)
        pop.append(ox); fit.append(fox)
        if fox < best:
            best, best_x = fox, ox[:]
        if len(pop) >= init_n:
            break

    # Greedy fill: best-of-k random (k small to save evals)
    while len(pop) < pop_size:
        if now() >= deadline:
            return best
        k = 3 if dim <= 12 else 2
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

    # ---------- SHADE memory + archive ----------
    H = max(6, min(28, 2 * int(math.sqrt(dim + 1)) + 8))
    M_F = [0.55] * H
    M_CR = [0.85] * H
    k_mem = 0

    archive = []
    archive_max = pop_size

    def get_from_pool(idx):
        if idx < pop_size:
            return pop[idx]
        return archive[idx - pop_size]

    def pick_distinct(exclude, count, pool_n):
        chosen = set()
        # try bounded loops (avoid worst-case)
        while len(chosen) < count:
            r = random.randrange(pool_n)
            if r == exclude:
                continue
            chosen.add(r)
        return list(chosen)

    # ---------- strong local search (adaptive coordinate + quadratic) ----------
    def local_memetic(x0, f0, time_cap, radius_scale):
        """
        Coordinate-wise search with:
        - +/- step probes
        - optional 2-point quadratic interpolation along coordinate if both sides exist
        - adaptive per-coordinate step sizes
        """
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        # trust-region-ish step sizes
        step = [max(1e-12, radius_scale * 0.15 * spans[i]) for i in range(dim)]
        min_step = [max(1e-12, 1e-10 * spans[i]) for i in range(dim)]

        stall = 0
        while now() < t_end:
            improved_any = False
            order = list(range(dim))
            random.shuffle(order)

            # try limited number of coordinates per round (keeps it cheap in high dim)
            probes = min(dim, 18)
            for jj in range(probes):
                if now() >= t_end:
                    break
                j = order[jj]
                sj = step[j]
                if sj < min_step[j]:
                    continue

                xj0 = x[j]
                # evaluate + and -
                xp = x[:]
                xm = x[:]
                xp[j] = reflect_scalar(xj0 + sj, j)
                xm[j] = reflect_scalar(xj0 - sj, j)

                fp = eval_f(xp)
                if now() >= t_end:
                    break
                fm = eval_f(xm)

                best_cand = x
                best_fc = fx

                if fp < best_fc:
                    best_cand, best_fc = xp, fp
                if fm < best_fc:
                    best_cand, best_fc = xm, fm

                # Quadratic interpolation along coordinate if it looks promising
                # Fit parabola through (x-s, fm), (x, fx), (x+s, fp)
                # Only if points are distinct and curvature reasonable.
                if now() < t_end:
                    denom = (fp - 2.0 * fx + fm)
                    if abs(denom) > 1e-18:
                        # minimizer offset from center
                        delta = 0.5 * sj * (fm - fp) / denom
                        # restrict delta within [-1.5*s, 1.5*s] to be safe
                        if delta > 1.5 * sj: delta = 1.5 * sj
                        if delta < -1.5 * sj: delta = -1.5 * sj
                        xq = x[:]
                        xq[j] = reflect_scalar(xj0 + delta, j)
                        fq = eval_f(xq)
                        if fq < best_fc:
                            best_cand, best_fc = xq, fq

                if best_fc < fx:
                    x, fx = best_cand, best_fc
                    improved_any = True
                    # mildly increase step on success (speeds up on smooth valleys)
                    step[j] = min(0.5 * spans[j], step[j] * 1.25 + 1e-18)
                else:
                    # shrink step if nothing helped on this coordinate
                    step[j] *= 0.65

            if improved_any:
                stall = 0
            else:
                stall += 1
                # shrink all steps globally if a full round didn't help
                for j in range(dim):
                    step[j] *= 0.7
                if stall >= 3 and all(step[j] < min_step[j] for j in range(dim)):
                    break

        return x, fx

    # ---------- main loop controls ----------
    last_refine = start
    last_improve_t = start
    best_at_last_improve = best

    # diversity / stagnation tracking
    def pop_diversity():
        # cheap: average normalized L1 distance to best on a small sample
        if best_x is None:
            return 1.0
        m = min(pop_size, 12)
        idxs = random.sample(range(pop_size), m)
        s = 0.0
        for i in idxs:
            xi = pop[i]
            s += sum(abs(xi[d] - best_x[d]) / spans[d] for d in range(dim)) / dim
        return s / m

    # ---------- evolution ----------
    while True:
        t = now()
        if t >= deadline:
            return best

        # progress in [0,1]
        prog = (t - start) / (max_time if max_time > 1e-12 else 1e-12)
        if prog < 0.0: prog = 0.0
        if prog > 1.0: prog = 1.0

        # adaptive p-best fraction with mild oscillation (helps avoid long plateaus)
        base_p = 0.26 - 0.16 * prog   # 0.26 -> 0.10
        osc = 0.04 * math.sin(2.0 * math.pi * (prog * 1.35 + 0.1))
        p = max(0.08, min(0.32, base_p + osc))
        pbest_count = max(2, int(p * pop_size))

        # rank
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        # occasional memetic refine around best (more towards the end)
        refine_period = max(0.10, 0.055 * max_time)
        if best_x is not None and (t - last_refine) >= refine_period:
            remaining = deadline - t
            if remaining > 0.04:
                # allocate slightly more later; also shrink radius over time
                rad = max(0.06, 0.55 * (1.0 - prog) + 0.08)
                cap = min(0.06 * max_time + 0.10 * prog, 0.25 * remaining, 0.35)
                rx, rf = local_memetic(best_x, best, cap, rad)
                if rf < best:
                    best, best_x = rf, rx[:]
                    last_improve_t = now()
                    best_at_last_improve = best
                    # inject into worst
                    worst = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst] = best_x[:]
                    fit[worst] = best
                last_refine = now()

        # stagnation / restart logic
        # If no improvement for a while and diversity low, partially restart around best + random
        if best_x is not None:
            if best < best_at_last_improve - 1e-14:
                best_at_last_improve = best
                last_improve_t = t

            remaining = deadline - t
            if remaining > 0.08:
                div = pop_diversity()
                no_imp = (t - last_improve_t)
                # threshold scales with max_time
                if no_imp > max(0.25, 0.22 * max_time) and div < 0.10:
                    # partial restart: keep elites, re-seed others
                    elites = max(2, pop_size // 6)
                    elite_idxs = idx_sorted[:elites]
                    new_pop = [pop[i][:] for i in elite_idxs]
                    new_fit = [fit[i] for i in elite_idxs]

                    # seed around best with shrinking Gaussian radius + some global random
                    sigma = (0.20 * (1.0 - prog) + 0.04)  # normalized
                    while len(new_pop) < pop_size and now() < deadline:
                        if random.random() < 0.65:
                            x = best_x[:]
                            for d in range(dim):
                                x[d] = reflect_scalar(x[d] + random.gauss(0.0, sigma) * spans[d], d)
                        else:
                            x = rand_point()
                        fx = eval_f(x)
                        new_pop.append(x); new_fit.append(fx)
                        if fx < best:
                            best, best_x = fx, x[:]
                            best_at_last_improve = best
                            last_improve_t = now()

                    pop, fit = new_pop, new_fit
                    archive = []
                    # reset memories mildly
                    for h in range(H):
                        M_F[h] = 0.55
                        M_CR[h] = 0.85
                    k_mem = 0

        # one generation
        S_F, S_CR, dF = [], [], []

        # Diversity-collapse injection: occasionally use best/2 mutation for a few trials
        div_now = pop_diversity() if best_x is not None else 1.0
        use_best2 = (div_now < 0.06 and prog > 0.25 and random.random() < 0.35)

        for i in range(pop_size):
            if now() >= deadline:
                return best

            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            x_i = pop[i]

            # pbest
            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            x_pbest = pop[pbest_idx]

            pool_n = pop_size + len(archive)
            r1, r2 = pick_distinct(exclude=i, count=2, pool_n=pool_n)
            x_r1 = get_from_pool(r1)
            x_r2 = get_from_pool(r2)

            if use_best2 and best_x is not None and pool_n >= 4:
                # best/2: best + F*(r1-r2 + r3-r4)
                r3, r4 = pick_distinct(exclude=i, count=2, pool_n=pool_n)
                x_r3 = get_from_pool(r3)
                x_r4 = get_from_pool(r4)
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = best_x[d] + Fi * ((x_r1[d] - x_r2[d]) + (x_r3[d] - x_r4[d]))
            else:
                # current-to-pbest/1
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # crossover
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect_scalar(v[d], d)

            fu = eval_f(u)

            if fu <= fit[i]:
                # archive parent
                if len(archive) < archive_max:
                    archive.append(pop[i])
                else:
                    archive[random.randrange(archive_max)] = pop[i]

                imp = fit[i] - fu
                if imp < 0.0:
                    imp = 0.0
                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(imp)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = now()
                    best_at_last_improve = best

        # update memories (SHADE)
        if S_F:
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / wsum for di in dF]

            # Lehmer mean for F
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

            # smoothing + keep in bounds
            M_F[k_mem] = min(1.0, max(0.08, 0.85 * M_F[k_mem] + 0.15 * new_MF))
            M_CR[k_mem] = clamp01(0.85 * M_CR[k_mem] + 0.15 * new_MCR)
            k_mem = (k_mem + 1) % H

        # trim archive
        if len(archive) > archive_max:
            archive = archive[-archive_max:]
