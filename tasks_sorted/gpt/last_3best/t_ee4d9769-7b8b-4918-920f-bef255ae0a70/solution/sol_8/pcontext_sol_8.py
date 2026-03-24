import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvement over the provided DE+local versions:
    - Uses a *multi-armed bandit* to adaptively choose among several DE mutation strategies:
        1) current-to-pbest/1 (JADE/SHADE workhorse)
        2) rand/1 (exploration)
        3) best/1 (exploitation when safe)
        4) current-to-rand/1 (diversity-preserving)
      The bandit rewards strategies by achieved fitness improvement per trial, so time is
      spent on what works for the current function.
    - Adds *cheap surrogate-like "try many, evaluate few"* in local refinement:
        * stochastic coordinate groups + opportunistic 1D quadratic step
      but strictly keeps evaluation count low.
    - Stronger restart: detects BOTH stagnation and radius collapse; reseeds using a
      mixture of: around-best, opposition(best), and global random.
    - Budget-aware: converts time into an estimated evaluation budget online and throttles
      expensive steps when evaluations are "slow".

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

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def reflect_scalar(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect until inside (more DE-friendly than clamp)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        # final safety
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
        # lightweight LHS-like stratification
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

    # ---- SHADE-ish parameter sampling
    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def sample_F(muF):
        for _ in range(16):
            f = rand_cauchy(muF, 0.12)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return max(0.08, min(1.0, muF))

    def sample_CR(muCR):
        return clamp01(random.gauss(muCR, 0.12))

    # ---------------- sizing ----------------
    # Moderate population works best under time limits.
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

    # Track eval speed to adapt local-search usage
    evals = 0
    eval_time_acc = 0.0

    def timed_eval(x):
        nonlocal evals, eval_time_acc, best, best_x
        t0 = now()
        fx = eval_f(x)
        dt = now() - t0
        evals += 1
        eval_time_acc += dt
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    # LHS + opposition
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return best
        fx = timed_eval(x)
        pop.append(x); fit.append(fx)
        if len(pop) >= init_n:
            break

        ox = [reflect_scalar(opposite_point(x)[i], i) for i in range(dim)]
        if now() >= deadline:
            return best
        fox = timed_eval(ox)
        pop.append(ox); fit.append(fox)
        if len(pop) >= init_n:
            break

    # Fill remainder with best-of-k random
    while len(pop) < pop_size:
        if now() >= deadline:
            return best
        k = 3 if dim <= 12 else 2
        bx, bf = None, float("inf")
        for _ in range(k):
            x = rand_point()
            fx = timed_eval(x)
            if fx < bf:
                bx, bf = x, fx
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

    # ---------------- bandit over mutation strategies ----------------
    # 0: current-to-pbest/1
    # 1: rand/1
    # 2: best/1
    # 3: current-to-rand/1
    n_strat = 4
    bandit_score = [1.0] * n_strat
    bandit_uses = [1] * n_strat

    def pick_strategy(prog):
        # epsilon decreases over time (explore early)
        eps = 0.25 * (1.0 - prog) + 0.05
        if random.random() < eps:
            return random.randrange(n_strat)
        # UCB-ish
        total = sum(bandit_uses)
        best_s = 0
        best_ucb = -1e100
        for s in range(n_strat):
            mean = bandit_score[s] / bandit_uses[s]
            ucb = mean + 0.60 * math.sqrt(math.log(total + 1.0) / bandit_uses[s])
            if ucb > best_ucb:
                best_ucb = ucb
                best_s = s
        return best_s

    def reward_strategy(s, improvement):
        # improvement is positive if child better than parent
        bandit_uses[s] += 1
        bandit_score[s] += max(0.0, improvement)

    # ---------------- local refine (cheap, evaluation-thrifty) ----------------
    def local_refine_grouped(x0, f0, time_cap, radius_scale):
        """
        Grouped coordinate search + opportunistic quadratic step on a few coords.
        Uses very few evaluations per call.
        """
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        # choose small active set size
        group = min(dim, max(2, int(math.sqrt(dim) + 2)))
        step0 = radius_scale * 0.16
        step = [max(1e-12, step0 * spans[i]) for i in range(dim)]
        min_step = [max(1e-12, 1e-10 * spans[i]) for i in range(dim)]

        stall = 0
        while now() < t_end:
            improved = False

            # pick a random subset of coordinates to work on
            idxs = list(range(dim))
            random.shuffle(idxs)
            idxs = idxs[:group]

            for j in idxs:
                if now() >= t_end:
                    break
                sj = step[j]
                if sj < min_step[j]:
                    continue

                xj = x[j]
                xp = x[:]; xp[j] = reflect_scalar(xj + sj, j)
                fp = timed_eval(xp)
                if now() >= t_end:
                    break

                xm = x[:]; xm[j] = reflect_scalar(xj - sj, j)
                fm = timed_eval(xm)
                if now() >= t_end:
                    break

                best_cand = None
                best_fc = fx
                if fp < best_fc:
                    best_cand, best_fc = xp, fp
                if fm < best_fc:
                    best_cand, best_fc = xm, fm

                # quadratic step (only if curvature information usable)
                denom = (fp - 2.0 * fx + fm)
                if abs(denom) > 1e-18:
                    delta = 0.5 * sj * (fm - fp) / denom
                    if delta > 1.25 * sj: delta = 1.25 * sj
                    if delta < -1.25 * sj: delta = -1.25 * sj
                    if abs(delta) > 1e-15 and now() < t_end:
                        xq = x[:]
                        xq[j] = reflect_scalar(xj + delta, j)
                        fq = timed_eval(xq)
                        if fq < best_fc:
                            best_cand, best_fc = xq, fq

                if best_cand is not None and best_fc < fx:
                    x, fx = best_cand, best_fc
                    improved = True
                    step[j] = min(0.5 * spans[j], step[j] * 1.30 + 1e-18)
                else:
                    step[j] *= 0.60

            if improved:
                stall = 0
            else:
                stall += 1
                for j in idxs:
                    step[j] *= 0.75
                if stall >= 3 and all(step[j] < min_step[j] for j in idxs):
                    break

        return x, fx

    # ---------------- diversity + restart ----------------
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

    last_improve_t = start
    last_best = best
    last_local_t = start

    # ---------------- main loop ----------------
    while True:
        t = now()
        if t >= deadline:
            return best

        prog = (t - start) / (max_time if max_time > 1e-12 else 1e-12)
        prog = 0.0 if prog < 0.0 else (1.0 if prog > 1.0 else prog)

        # estimate eval cost; reduce local-search if expensive
        avg_eval = (eval_time_acc / evals) if evals > 0 else 1e-6

        # ---- periodic local refine around best
        # do it more later, but only if evals are reasonably fast
        local_every = max(0.10, 0.055 * max_time)
        if best_x is not None and (t - last_local_t) >= local_every:
            remaining = deadline - t
            if remaining > 3.0 * avg_eval:
                # allocate a tiny time slice, adaptive
                rad = max(0.05, 0.60 * (1.0 - prog) + 0.06)
                cap = min(0.05 + 0.10 * prog, 0.20 * remaining, 0.28)
                rx, rf = local_refine_grouped(best_x, best, cap, rad)
                if rf < best:
                    best, best_x = rf, rx[:]
                    last_improve_t = now()
                    # inject into worst
                    worst = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst] = best_x[:]
                    fit[worst] = best
                last_local_t = now()

        # ---- stagnation tracking
        if best < last_best - 1e-14:
            last_best = best
            last_improve_t = t

        # ---- restart if stagnant and collapsed
        if best_x is not None and (deadline - t) > 0.10:
            div = diversity_to_best()
            if (t - last_improve_t) > max(0.22, 0.20 * max_time) and div < 0.09:
                idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
                elites = max(2, pop_size // 6)

                new_pop = [pop[i][:] for i in idx_sorted[:elites]]
                new_fit = [fit[i] for i in idx_sorted[:elites]]

                # reseed mixture: around-best, opposition(best), random
                sigma = (0.22 * (1.0 - prog) + 0.05)
                opp_best = opposite_point(best_x)
                while len(new_pop) < pop_size and now() < deadline:
                    r = random.random()
                    if r < 0.55:
                        x = best_x[:]
                        for d in range(dim):
                            x[d] = reflect_scalar(x[d] + random.gauss(0.0, sigma) * spans[d], d)
                    elif r < 0.75:
                        x = opp_best[:]
                        for d in range(dim):
                            x[d] = reflect_scalar(x[d] + random.gauss(0.0, 0.35 * sigma) * spans[d], d)
                    else:
                        x = rand_point()
                    fx = timed_eval(x)
                    new_pop.append(x); new_fit.append(fx)

                pop, fit = new_pop, new_fit
                archive = []
                # soften memory reset (not full wipe)
                for h in range(H):
                    M_F[h] = 0.60 * M_F[h] + 0.40 * 0.55
                    M_CR[h] = 0.60 * M_CR[h] + 0.40 * 0.85
                k_mem = 0
                # reset bandit slightly
                for s in range(n_strat):
                    bandit_score[s] = 1.0
                    bandit_uses[s] = 1
                last_local_t = now()
                last_improve_t = now()
                continue

        # ---- DE generation
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        # p-best pressure schedule
        p = 0.30 - 0.20 * prog  # 0.30 -> 0.10
        if p < 0.08: p = 0.08
        pbest_count = max(2, int(p * pop_size))

        S_F, S_CR, dF = [], [], []

        for i in range(pop_size):
            if now() >= deadline:
                return best

            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            strat = pick_strategy(prog)

            x_i = pop[i]
            x_best = best_x
            x_pbest = pop[idx_sorted[random.randrange(pbest_count)]]

            pool_n = pop_size + len(archive)

            # build mutant vector v depending on strategy
            if strat == 0:
                # current-to-pbest/1
                r1, r2 = pick_distinct(i, 2, pool_n)
                x_r1 = get_from_pool(r1)
                x_r2 = get_from_pool(r2)
                v = [x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d]) for d in range(dim)]
            elif strat == 1:
                # rand/1
                r0, r1, r2 = pick_distinct(i, 3, pool_n)
                x_r0 = get_from_pool(r0)
                x_r1 = get_from_pool(r1)
                x_r2 = get_from_pool(r2)
                v = [x_r0[d] + Fi * (x_r1[d] - x_r2[d]) for d in range(dim)]
            elif strat == 2 and x_best is not None:
                # best/1 (strong exploit)
                r1, r2 = pick_distinct(i, 2, pool_n)
                x_r1 = get_from_pool(r1)
                x_r2 = get_from_pool(r2)
                v = [x_best[d] + Fi * (x_r1[d] - x_r2[d]) for d in range(dim)]
            else:
                # current-to-rand/1 (diversity)
                r0, r1, r2 = pick_distinct(i, 3, pool_n)
                x_r0 = get_from_pool(r0)
                x_r1 = get_from_pool(r1)
                x_r2 = get_from_pool(r2)
                K = random.random()
                v = [x_i[d] + K * (x_r0[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d]) for d in range(dim)]

            # crossover
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect_scalar(v[d], d)

            fu = timed_eval(u)

            if fu <= fit[i]:
                # archive parent
                if len(archive) < archive_max:
                    archive.append(pop[i])
                else:
                    archive[random.randrange(archive_max)] = pop[i]

                improvement = fit[i] - fu
                if improvement < 0.0:
                    improvement = 0.0

                reward_strategy(strat, improvement)

                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(improvement)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_best = best
                    last_improve_t = now()
            else:
                # still count usage (already counted), but no reward beyond 0
                reward_strategy(strat, 0.0)

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

            # smoothing for stability
            M_F[k_mem] = min(1.0, max(0.08, 0.85 * M_F[k_mem] + 0.15 * new_MF))
            M_CR[k_mem] = clamp01(0.85 * M_CR[k_mem] + 0.15 * new_MCR)
            k_mem = (k_mem + 1) % H

        if len(archive) > archive_max:
            archive = archive[-archive_max:]
