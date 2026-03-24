import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Hybrid optimizer (self-contained):
    - Multi-start Sobol-like (actually: hashed LHS) + opposition init
    - Differential Evolution with jDE-style self-adaptation of F/CR per individual
    - "current-to-pbest/1" (JADE-like) mutation for faster convergence
    - Archive of replaced solutions to increase diversity
    - Occasional lightweight local search (random+coordinate pattern) near best
    Returns: best fitness (float)
    """

    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # --- helpers ---
    def now():
        return time.time()

    def reflect(xi, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        v = xi
        # reflect repeatedly until in range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def clip_vec(x):
        return [reflect(x[i], i) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def hashed_lhs(n):
        # Stratified per-dim bins with independent shuffles
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for k in range(n):
            x = []
            for j in range(dim):
                bin_id = perms[j][k]
                u = (bin_id + random.random()) / n
                x.append(lows[j] + u * spans[j])
            pts.append(x)
        return pts

    # --- time-aware sizing ---
    # Keep pop moderate; too big wastes time on evaluations.
    pop_size = 10 + 5 * dim
    if pop_size < 24: pop_size = 24
    if pop_size > 80: pop_size = 80
    if max_time <= 0.4:
        pop_size = min(pop_size, 32)

    # --- initialization (LHS + opposition + a few pure random) ---
    pop = []
    fit = []

    best = float("inf")
    best_x = None

    init_n = pop_size
    pts = hashed_lhs(init_n // 2)
    for x in pts:
        if now() >= deadline:
            return best
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if fx < best:
            best = fx; best_x = x[:]

        if len(pop) >= init_n:
            break
        ox = opposite_point(x)
        if now() >= deadline:
            return best
        fox = eval_f(ox)
        pop.append(ox); fit.append(fox)
        if fox < best:
            best = fox; best_x = ox[:]

        if len(pop) >= init_n:
            break

    while len(pop) < pop_size:
        if now() >= deadline:
            return best
        x = rand_point()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if fx < best:
            best = fx; best_x = x[:]

    # --- per-individual self-adaptive parameters (jDE style) ---
    # Each individual carries its own F and CR
    F_i = [0.5 + 0.3 * random.random() for _ in range(pop_size)]  # [0.5,0.8]
    CR_i = [0.7 + 0.25 * random.random() for _ in range(pop_size)] # [0.7,0.95]
    tau1, tau2 = 0.1, 0.1  # adaptation probabilities

    # --- archive for diversity ---
    archive = []
    archive_max = pop_size

    def pick_indices(exclude, n):
        # pick n distinct indices from pop+archive pools
        # pool indices: 0..pop_size-1 for pop, pop_size..pop_size+len(archive)-1 for archive
        pool_n = pop_size + len(archive)
        chosen = set()
        while len(chosen) < n:
            r = random.randrange(pool_n)
            if r == exclude:
                continue
            chosen.add(r)
        return list(chosen)

    def get_vec(idx):
        if idx < pop_size:
            return pop[idx]
        return archive[idx - pop_size]

    # --- local search around best ---
    def local_search(x0, f0, time_cap):
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        step = [0.08 * spans[i] for i in range(dim)]
        min_step = [max(1e-12, 1e-9 * spans[i]) for i in range(dim)]

        # mix of coordinate and random-direction probes
        while now() < t_end:
            improved = False

            # a few coordinate tries
            for _ in range(min(dim, 12)):
                if now() >= t_end:
                    break
                j = random.randrange(dim)
                sj = step[j]
                if sj < min_step[j]:
                    continue
                for sign in (1.0, -1.0):
                    if now() >= t_end:
                        break
                    cand = x[:]
                    cand[j] = reflect(cand[j] + sign * sj * (0.6 + 0.8 * random.random()), j)
                    fc = eval_f(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        break

            # one random-direction try
            if now() < t_end:
                v = [random.gauss(0.0, 1.0) for _ in range(dim)]
                nrm = math.sqrt(sum(vi * vi for vi in v)) or 1.0
                mean_step = sum(step) / dim
                scale = mean_step * (0.4 + 0.8 * random.random())
                cand = x[:]
                for i in range(dim):
                    cand[i] = reflect(cand[i] + scale * (v[i] / nrm), i)
                fc = eval_f(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True

            if not improved:
                # shrink steps
                for i in range(dim):
                    step[i] *= 0.6
                if all(step[i] < min_step[i] for i in range(dim)):
                    break

        return x, fx

    # --- main loop (JADE-like pbest + jDE adaption) ---
    last_refine = start
    refine_interval = max(0.12, 0.07 * max_time)

    while True:
        if now() >= deadline:
            return best

        # occasional local refinement
        if best_x is not None and (now() - last_refine) >= refine_interval:
            remaining = deadline - now()
            if remaining > 0.04:
                cap = min(0.06 * max_time, 0.22 * remaining, 0.30)
                rx, rf = local_search(best_x, best, cap)
                if rf < best:
                    best = rf
                    best_x = rx[:]
                    # inject into worst
                    worst = max(range(pop_size), key=lambda k: fit[k])
                    pop[worst] = best_x[:]
                    fit[worst] = best
                last_refine = now()

        # compute p-best pool size (top p fraction)
        p = 0.2
        pbest_count = max(2, int(p * pop_size))

        # rank indices by fitness
        idx_sorted = sorted(range(pop_size), key=lambda k: fit[k])

        for i in range(pop_size):
            if now() >= deadline:
                return best

            # jDE adaptation
            Fi = F_i[i]
            CRi = CR_i[i]
            if random.random() < tau1:
                Fi = 0.1 + 0.9 * random.random()  # (0.1,1.0)
            if random.random() < tau2:
                CRi = random.random()  # (0,1)
            # keep in reasonable range
            if Fi < 0.15: Fi = 0.15
            if Fi > 0.95: Fi = 0.95

            # choose pbest from top set
            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            # choose r1, r2 from pop+archive
            r1, r2 = pick_indices(exclude=i, n=2)
            x_r1 = get_vec(r1)
            x_r2 = get_vec(r2)

            # current-to-pbest/1:
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect(v[d], d)

            fu = eval_f(u)

            # selection + archive update
            if fu <= fit[i]:
                # add replaced parent to archive
                if len(archive) < archive_max:
                    archive.append(pop[i])
                else:
                    archive[random.randrange(archive_max)] = pop[i]

                pop[i] = u
                fit[i] = fu
                F_i[i] = Fi
                CR_i[i] = CRi

                if fu < best:
                    best = fu
                    best_x = u[:]
            else:
                # still update parameters sometimes to avoid stagnation
                F_i[i] = 0.9 * F_i[i] + 0.1 * Fi
                CR_i[i] = 0.9 * CR_i[i] + 0.1 * CRi

        # slight archive trimming if needed (safety)
        if len(archive) > archive_max:
            archive = archive[-archive_max:]
