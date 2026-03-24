import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (no external libs).

    Improved vs your last bandit-DE version:
    - Uses a stronger DE backbone: *jDE* self-adaptation per individual (F, CR mutate),
      which is very evaluation-efficient under unknown landscapes/time limits.
    - Adds *stochastic 2-phase mutation*: rand/1 early (explore) -> current-to-best/1 later (exploit),
      plus occasional best/1 when diversity is healthy.
    - Adds *true “best-first” memetic local search* near the end using
      (a) grouped coordinate search + quadratic step (cheap) and
      (b) 2-direction random subspace steps (helps non-separability),
      but strictly time-sliced.
    - Improves restart: keeps elites, reseeds with mixture around best / opposite(best) / global.
    - Keeps overhead low (no sorting in inner loops except once per generation).

    Returns: best fitness (float)
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
        # reflect until in bounds
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Simple LHS-like initialization (cheap, no numpy)
    def lhs_points(n):
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

    best = float("inf")
    best_x = None

    evals = 0
    eval_time_acc = 0.0

    def timed_eval(x):
        nonlocal best, best_x, evals, eval_time_acc
        t0 = now()
        fx = float(func(x))
        dt = now() - t0
        evals += 1
        eval_time_acc += dt
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    # Population size tuned for time-bounded use
    pop_size = 14 + 6 * dim
    pop_size = max(24, min(90, pop_size))
    if max_time <= 0.35:
        pop_size = min(pop_size, 34)
    elif max_time <= 0.8:
        pop_size = min(pop_size, 56)

    # --- init population: LHS + opposition + greedy fill
    pop, fit = [], []
    n0 = pop_size
    n_lhs = max(6, n0 // 2)

    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return best
        fx = timed_eval(x)
        pop.append(x); fit.append(fx)
        if len(pop) >= n0:
            break

        ox = [reflect_scalar(opposite_point(x)[i], i) for i in range(dim)]
        if now() >= deadline:
            return best
        fox = timed_eval(ox)
        pop.append(ox); fit.append(fox)
        if len(pop) >= n0:
            break

    while len(pop) < pop_size:
        if now() >= deadline:
            return best
        # best-of-k random (small k)
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

    # --- jDE parameters per individual (self-adaptive)
    F = [0.5 + 0.3 * random.random() for _ in range(pop_size)]
    CR = [0.6 + 0.3 * random.random() for _ in range(pop_size)]
    tau1 = 0.10  # prob to resample F
    tau2 = 0.10  # prob to resample CR

    # --- diversity/stagnation helpers
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
    last_restart_t = start

    # --- cheap memetic local search (time-sliced)
    def local_memetic(x0, f0, time_cap, radius_scale):
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        # grouped coordinate search
        group = min(dim, max(2, int(math.sqrt(dim) + 2)))
        step = [max(1e-12, radius_scale * 0.18 * spans[i]) for i in range(dim)]
        min_step = [max(1e-12, 1e-10 * spans[i]) for i in range(dim)]

        stall = 0
        while now() < t_end:
            improved = False

            # coords subset
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

                # 3-point quadratic step if curvature is usable
                denom = (fp - 2.0 * fx + fm)
                if abs(denom) > 1e-18 and now() < t_end:
                    delta = 0.5 * sj * (fm - fp) / denom
                    if delta > 1.25 * sj: delta = 1.25 * sj
                    if delta < -1.25 * sj: delta = -1.25 * sj
                    if abs(delta) > 1e-15:
                        xq = x[:]
                        xq[j] = reflect_scalar(xj + delta, j)
                        fq = timed_eval(xq)
                        if fq < best_fc:
                            best_cand, best_fc = xq, fq

                if best_cand is not None and best_fc < fx:
                    x, fx = best_cand, best_fc
                    improved = True
                    step[j] = min(0.5 * spans[j], step[j] * 1.25 + 1e-18)
                else:
                    step[j] *= 0.65

            # two random subspace direction probes (helps coupling)
            if now() < t_end and dim > 1:
                subk = min(dim, max(2, int(math.sqrt(dim))))
                for _ in range(2):
                    if now() >= t_end:
                        break
                    idxs2 = random.sample(range(dim), subk)
                    v = [0.0] * dim
                    for j in idxs2:
                        v[j] = random.gauss(0.0, 1.0)
                    nrm = math.sqrt(sum(vj * vj for vj in v)) or 1.0
                    avg_step = sum(step[j] for j in idxs2) / float(len(idxs2))
                    scale = avg_step * (0.6 + 0.8 * random.random())
                    cand = x[:]
                    for j in idxs2:
                        cand[j] = reflect_scalar(cand[j] + scale * (v[j] / nrm), j)
                    fc = timed_eval(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True

            if improved:
                stall = 0
            else:
                stall += 1
                for j in idxs:
                    step[j] *= 0.75
                if stall >= 3 and all(step[j] < min_step[j] for j in idxs):
                    break

        return x, fx

    # --- main loop
    gen = 0
    while True:
        t = now()
        if t >= deadline:
            return best

        prog = (t - start) / (max_time if max_time > 1e-12 else 1e-12)
        if prog < 0.0: prog = 0.0
        if prog > 1.0: prog = 1.0

        avg_eval = (eval_time_acc / evals) if evals > 0 else 1e-6

        # periodic memetic refine (more likely late, but only if time remains)
        if best_x is not None and (deadline - t) > 4.0 * avg_eval:
            # do it not every gen; gate by time and progress
            if (gen % (3 if dim <= 10 else 4) == 0) and (prog > 0.35):
                cap = min(0.06 + 0.12 * prog, 0.22 * (deadline - t), 0.35)
                rad = max(0.04, 0.55 * (1.0 - prog) + 0.06)
                rx, rf = local_memetic(best_x, best, cap, rad)
                if rf < best:
                    best, best_x = rf, rx[:]
                    # inject into worst
                    worst = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst] = best_x[:]
                    fit[worst] = best

        # stagnation bookkeeping
        if best < last_best - 1e-14:
            last_best = best
            last_improve_t = t

        # restart if stagnant + collapsed diversity, but not too frequently
        if best_x is not None and (deadline - t) > 0.12 and (t - last_restart_t) > 0.12:
            div = diversity_to_best()
            if (t - last_improve_t) > max(0.22, 0.20 * max_time) and div < 0.08:
                # keep elites
                idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
                elites = max(2, pop_size // 6)
                new_pop = [pop[i][:] for i in idx_sorted[:elites]]
                new_fit = [fit[i] for i in idx_sorted[:elites]]
                new_F = [F[i] for i in idx_sorted[:elites]]
                new_CR = [CR[i] for i in idx_sorted[:elites]]

                sigma = 0.22 * (1.0 - prog) + 0.05
                opp = opposite_point(best_x)

                while len(new_pop) < pop_size and now() < deadline:
                    r = random.random()
                    if r < 0.55:
                        x = best_x[:]
                        for d in range(dim):
                            x[d] = reflect_scalar(x[d] + random.gauss(0.0, sigma) * spans[d], d)
                    elif r < 0.75:
                        x = opp[:]
                        for d in range(dim):
                            x[d] = reflect_scalar(x[d] + random.gauss(0.0, 0.35 * sigma) * spans[d], d)
                    else:
                        x = rand_point()

                    fx = timed_eval(x)
                    new_pop.append(x); new_fit.append(fx)
                    new_F.append(0.45 + 0.5 * random.random())
                    new_CR.append(random.random())

                pop, fit, F, CR = new_pop, new_fit, new_F, new_CR
                last_restart_t = now()
                last_improve_t = now()
                gen += 1
                continue

        # --- one DE generation (jDE self-adaptation + time-varying strategy)
        # Choose best index once per generation
        best_idx = min(range(pop_size), key=lambda i: fit[i])

        for i in range(pop_size):
            if now() >= deadline:
                return best

            # jDE parameter adaptation
            Fi = F[i]
            CRi = CR[i]
            if random.random() < tau1:
                Fi = 0.1 + 0.9 * random.random()
            if random.random() < tau2:
                CRi = random.random()

            # mutation strategy schedule:
            # early: more rand/1, late: more current-to-best/1
            r = random.random()
            if prog < 0.45:
                # mostly exploration
                strat = 0 if r < 0.80 else 1  # 0: rand/1, 1: current-to-best/1
            else:
                # mostly exploitation, but keep some exploration
                strat = 1 if r < 0.75 else (2 if r < 0.88 else 0)  # 2: best/1 sometimes

            # pick distinct indices
            def pick_excluding(k):
                while True:
                    j = random.randrange(pop_size)
                    if j != i and j not in k:
                        return j

            used = set()
            a = pick_excluding(used); used.add(a)
            b = pick_excluding(used); used.add(b)
            c = pick_excluding(used); used.add(c)

            x_i = pop[i]
            if strat == 0:
                # rand/1: x_a + F*(x_b - x_c)
                base = pop[a]
                xb = pop[b]
                xc = pop[c]
                v = [base[d] + Fi * (xb[d] - xc[d]) for d in range(dim)]
            elif strat == 2:
                # best/1: x_best + F*(x_b - x_c)
                base = pop[best_idx]
                xb = pop[b]
                xc = pop[c]
                v = [base[d] + Fi * (xb[d] - xc[d]) for d in range(dim)]
            else:
                # current-to-best/1: x_i + F*(x_best - x_i) + F*(x_b - x_c)
                base = pop[best_idx]
                xb = pop[b]
                xc = pop[c]
                v = [x_i[d] + Fi * (base[d] - x_i[d]) + Fi * (xb[d] - xc[d]) for d in range(dim)]

            # binomial crossover + reflection
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect_scalar(v[d], d)

            fu = timed_eval(u)

            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                F[i] = Fi
                CR[i] = CRi

        gen += 1
