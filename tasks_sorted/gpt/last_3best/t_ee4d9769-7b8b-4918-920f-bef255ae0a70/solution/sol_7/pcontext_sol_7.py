import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Main improvements vs your last (DE + Powell line-search) version:
    1) Replace Powell (often eval-hungry and brittle under tight budgets) with a much more
       evaluation-efficient local stage:
         - adaptive coordinate search with quadratic interpolation (cheap)
         - occasional random-subspace directions (helps coupling)
         - trust-region style step control
    2) Keep a strong global engine: SHADE/JADE-style DE/current-to-pbest/1 + archive,
       but tuned to spend fewer evaluations on overhead and more on useful candidates.
    3) Better restart logic: triggered by both stagnation AND collapsed diversity.

    Returns:
        best fitness (float)
    """

    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def reflect_scalar(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect until in range (better than clamp for DE steps)
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
        # simple LHS-like stratification without numpy
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

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # ---------------- SHADE-ish sampling ----------------
    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def sample_F(muF):
        for _ in range(16):
            f = rand_cauchy(muF, 0.12)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return max(0.10, min(1.0, muF))

    def sample_CR(muCR):
        return clamp01(random.gauss(muCR, 0.12))

    # ---------------- sizing ----------------
    # Favor more generations over big pops under tight time.
    pop_size = 12 + 6 * dim
    pop_size = max(24, min(96, pop_size))
    if max_time <= 0.35:
        pop_size = min(pop_size, 36)
    elif max_time <= 0.8:
        pop_size = min(pop_size, 60)

    # ---------------- initialization ----------------
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

    # fill remainder: best-of-k random (cheap greedy)
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

    # ---------------- DE memory + archive ----------------
    H = max(6, min(26, 2 * int(math.sqrt(dim + 1)) + 8))
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
        while len(chosen) < count:
            r = random.randrange(pool_n)
            if r != exclude:
                chosen.add(r)
        return list(chosen)

    # ---------------- local search: adaptive coordinate + quad + subspace dirs ----------------
    def local_refine(x0, f0, time_cap, radius_scale, subspace_k):
        """
        Very evaluation-efficient local improvement:
        - per-coordinate +/- step
        - 3-point quadratic interpolation when informative
        - occasional random direction in a small subspace to handle coupling
        """
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        # trust region-ish step per coordinate
        step = [max(1e-12, radius_scale * 0.18 * spans[i]) for i in range(dim)]
        min_step = [max(1e-12, 1e-10 * spans[i]) for i in range(dim)]

        stall_rounds = 0
        while now() < t_end:
            improved_any = False

            order = list(range(dim))
            random.shuffle(order)

            probes = min(dim, 20)
            for jj in range(probes):
                if now() >= t_end:
                    break
                j = order[jj]
                sj = step[j]
                if sj < min_step[j]:
                    continue

                xj0 = x[j]

                xp = x[:]; xp[j] = reflect_scalar(xj0 + sj, j)
                fp = eval_f(xp)
                if now() >= t_end:
                    break

                xm = x[:]; xm[j] = reflect_scalar(xj0 - sj, j)
                fm = eval_f(xm)
                if now() >= t_end:
                    break

                best_cand = None
                best_fc = fx

                if fp < best_fc:
                    best_cand, best_fc = xp, fp
                if fm < best_fc:
                    best_cand, best_fc = xm, fm

                # quadratic interpolation around center (x-s, fm), (x, fx), (x+s, fp)
                denom = (fp - 2.0 * fx + fm)
                if abs(denom) > 1e-18:
                    delta = 0.5 * sj * (fm - fp) / denom
                    # keep it conservative
                    if delta > 1.5 * sj: delta = 1.5 * sj
                    if delta < -1.5 * sj: delta = -1.5 * sj
                    if abs(delta) > 1e-15:
                        xq = x[:]
                        xq[j] = reflect_scalar(xj0 + delta, j)
                        fq = eval_f(xq)
                        if fq < best_fc:
                            best_cand, best_fc = xq, fq

                if best_cand is not None and best_fc < fx:
                    x, fx = best_cand, best_fc
                    improved_any = True
                    # increase step on success (but bound it)
                    step[j] = min(0.5 * spans[j], step[j] * 1.25 + 1e-18)
                else:
                    step[j] *= 0.65

            # random subspace direction (helps non-separable problems)
            if now() < t_end and subspace_k > 0:
                idxs = random.sample(range(dim), min(dim, subspace_k))
                v = [0.0] * dim
                for j in idxs:
                    v[j] = random.gauss(0.0, 1.0)
                nrm = math.sqrt(sum(vj * vj for vj in v)) or 1.0
                avg_step = sum(step[j] for j in idxs) / float(len(idxs))
                scale = avg_step * (0.6 + 0.8 * random.random())
                cand = x[:]
                for j in idxs:
                    cand[j] = reflect_scalar(cand[j] + scale * (v[j] / nrm), j)
                fc = eval_f(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved_any = True

            if improved_any:
                stall_rounds = 0
            else:
                stall_rounds += 1
                for j in range(dim):
                    step[j] *= 0.75
                if stall_rounds >= 3 and all(step[j] < min_step[j] for j in range(dim)):
                    break

        return x, fx

    # ---------------- diversity estimate ----------------
    def diversity_to_best(sample_n=12):
        if best_x is None:
            return 1.0
        m = min(pop_size, sample_n)
        idxs = random.sample(range(pop_size), m)
        s = 0.0
        for i in idxs:
            xi = pop[i]
            s += sum(abs(xi[d] - best_x[d]) / spans[d] for d in range(dim)) / dim
        return s / m

    # ---------------- main loop ----------------
    last_best = best
    last_improve_t = start
    last_local_t = start

    # local search cadence and budget
    local_every = max(0.10, 0.05 * max_time)

    while True:
        t = now()
        if t >= deadline:
            return best

        prog = (t - start) / (max_time if max_time > 1e-12 else 1e-12)
        prog = 0.0 if prog < 0.0 else (1.0 if prog > 1.0 else prog)

        # ---- local refinement around best (memetic step)
        if best_x is not None and (t - last_local_t) >= local_every:
            remaining = deadline - t
            if remaining > 0.04:
                # shrink radius as time progresses
                rad = max(0.05, 0.55 * (1.0 - prog) + 0.07)
                # small-subspace dimension: grows mildly with dim, capped
                sub_k = 0 if dim <= 2 else min(dim, max(3, int(math.sqrt(dim) + 2)))
                cap = min(0.06 + 0.10 * prog, 0.24 * remaining, 0.32)
                rx, rf = local_refine(best_x, best, cap, rad, sub_k)
                if rf < best:
                    best, best_x = rf, rx[:]
                    last_improve_t = now()
                    # inject into worst to propagate
                    worst = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst] = best_x[:]
                    fit[worst] = best
                last_local_t = now()

        # ---- stagnation tracking
        if best < last_best - 1e-14:
            last_best = best
            last_improve_t = t

        # ---- partial restart if stagnant + low diversity
        if best_x is not None and (deadline - t) > 0.10:
            div = diversity_to_best()
            if (t - last_improve_t) > max(0.25, 0.22 * max_time) and div < 0.10:
                idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
                elites = max(2, pop_size // 6)
                new_pop = [pop[i][:] for i in idx_sorted[:elites]]
                new_fit = [fit[i] for i in idx_sorted[:elites]]

                sigma = (0.20 * (1.0 - prog) + 0.04)
                while len(new_pop) < pop_size and now() < deadline:
                    if random.random() < 0.70:
                        x = best_x[:]
                        for d in range(dim):
                            x[d] = reflect_scalar(x[d] + random.gauss(0.0, sigma) * spans[d], d)
                    else:
                        x = rand_point()
                    fx = eval_f(x)
                    new_pop.append(x); new_fit.append(fx)
                    if fx < best:
                        best, best_x = fx, x[:]
                        last_best = best
                        last_improve_t = now()

                pop, fit = new_pop, new_fit
                archive = []
                for h in range(H):
                    M_F[h] = 0.55
                    M_CR[h] = 0.85
                k_mem = 0
                last_local_t = now()
                continue

        # ---- DE generation
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        # p-best fraction schedule (more exploit later, but not too aggressive)
        p = 0.28 - 0.18 * prog  # 0.28 -> 0.10
        if p < 0.08: p = 0.08
        pbest_count = max(2, int(p * pop_size))

        S_F, S_CR, dF = [], [], []

        pool_n = pop_size + len(archive)

        for i in range(pop_size):
            if now() >= deadline:
                return best

            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            pool_n = pop_size + len(archive)
            r1, r2 = pick_distinct(i, 2, pool_n)
            x_r1 = get_from_pool(r1)
            x_r2 = get_from_pool(r2)

            # current-to-pbest/1
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
                    last_best = best
                    last_improve_t = now()

        # update parameter memories (SHADE)
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

            # smooth for stability
            M_F[k_mem] = min(1.0, max(0.08, 0.85 * M_F[k_mem] + 0.15 * new_MF))
            M_CR[k_mem] = clamp01(0.85 * M_CR[k_mem] + 0.15 * new_MCR)
            k_mem = (k_mem + 1) % H

        if len(archive) > archive_max:
            archive = archive[-archive_max:]
