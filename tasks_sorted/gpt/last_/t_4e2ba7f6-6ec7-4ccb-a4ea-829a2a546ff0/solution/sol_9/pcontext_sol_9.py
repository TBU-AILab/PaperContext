import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improved vs previous:
      - Uses a stronger hybrid: CMA-ES-like diagonal adaptation (simple, cheap) +
        Differential Evolution + opportunistic local pattern/coordinate search.
      - Better bound handling (reflect) and safer evaluation wrapper.
      - More systematic restarts with shrinking global scale.
      - Maintains an elite set to guide sampling (cross-entropy flavor).

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    if any(s <= 0.0 for s in spans):
        x = [lows[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # -------------------- helpers --------------------
    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect into [lo,hi], robust even for big steps
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            else:
                v = hi - (v - hi)
        return clamp(v, lo, hi)

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def randn():
        # Box-Muller
        u1 = max(1e-16, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2.0 * math.pi * u2)

    def eval_f(x):
        # keep it lean; assume caller handles time
        try:
            y = func(x)
            return float(y)
        except Exception:
            return float("inf")

    # -------------------- local search: pattern + coordinate hybrid --------------------
    def local_search(x0, f0, budget_evals=50):
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0
        evals = 0

        # per-dimension step sizes
        step = [0.15 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] + 1e-15 for i in range(dim)]

        # coordinate exploration, then a pattern step if improved
        while evals < budget_evals and time.time() < deadline:
            improved_any = False
            base = x[:]
            base_f = fx

            order = list(range(dim))
            random.shuffle(order)

            for j in order:
                if time.time() >= deadline or evals >= budget_evals:
                    break
                if step[j] <= min_step[j]:
                    continue

                sj = step[j]
                xj = x[j]

                xp = x[:]
                xp[j] = reflect(xj + sj, lows[j], highs[j])
                fp = eval_f(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved_any = True
                    step[j] = min(spans[j], sj * 1.35)
                    continue

                if time.time() >= deadline or evals >= budget_evals:
                    break

                xm = x[:]
                xm[j] = reflect(xj - sj, lows[j], highs[j])
                fm = eval_f(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved_any = True
                    step[j] = min(spans[j], sj * 1.35)
                    continue

                step[j] = sj * 0.55

            # pattern move: x <- x + (x - base) if we improved
            if improved_any and time.time() < deadline and evals < budget_evals:
                xp = [0.0] * dim
                for j in range(dim):
                    xp[j] = reflect(x[j] + 0.9 * (x[j] - base[j]), lows[j], highs[j])
                fp = eval_f(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    # slightly expand steps
                    for j in range(dim):
                        step[j] = min(spans[j], step[j] * 1.1)
            else:
                # global shrink; stop if all tiny
                tiny = True
                for j in range(dim):
                    step[j] *= 0.7
                    if step[j] > min_step[j]:
                        tiny = False
                if tiny:
                    break

            # if we got worse somehow, revert
            if fx > base_f:
                x, fx = base, base_f

        return x, fx

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None

    # more diverse init with a few corner-ish points + random
    init_n = max(12, min(60, 6 * dim + 12))
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        if k < 2 * dim:
            # axis-biased sample
            x = rand_vec()
            j = k % dim
            x[j] = lows[j] if (k // dim) == 0 else highs[j]
        else:
            x = rand_vec()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    # quick local touch
    if best_x is not None and time.time() < deadline:
        bx, bf = local_search(best_x, best, budget_evals=30 if dim <= 20 else 20)
        if bf < best:
            best, best_x = bf, bx[:]

    # -------------------- population for DE / elite --------------------
    NP = max(8, min(40, 8 + 2 * dim))
    pop = []
    fit = []

    # seed: half around best, half random
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if best_x is not None and i < NP // 2:
            x = [reflect(best_x[j] + 0.25 * spans[j] * randn(), lows[j], highs[j]) for j in range(dim)]
        else:
            x = rand_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # -------------------- diagonal "CMA-ish" sampler state --------------------
    # mean at best, sigma per dimension; updated from elites (cheap, robust)
    mean = best_x[:] if best_x is not None else rand_vec()
    sigma = [0.35 * spans[i] for i in range(dim)]
    sigma_min = [1e-9 * spans[i] + 1e-12 for i in range(dim)]
    sigma_max = [0.8 * spans[i] for i in range(dim)]

    # -------------------- main loop --------------------
    CR = 0.9
    Fmin, Fmax = 0.35, 0.95

    gen = 0
    no_improve = 0
    last_best = best
    restart_scale = 1.0

    while time.time() < deadline:
        gen += 1

        # ----- elite update (cross-entropy flavor) -----
        # pick top mu individuals as elites
        idx = list(range(NP))
        idx.sort(key=lambda i: fit[i])
        mu = max(2, NP // 5)
        elites = idx[:mu]

        # update mean and sigma from elites, with smoothing
        # compute elite mean
        new_mean = [0.0] * dim
        for i in elites:
            xi = pop[i]
            for j in range(dim):
                new_mean[j] += xi[j]
        inv_mu = 1.0 / float(mu)
        for j in range(dim):
            new_mean[j] *= inv_mu

        # elite std (diagonal)
        new_sig = [0.0] * dim
        for i in elites:
            xi = pop[i]
            for j in range(dim):
                d = xi[j] - new_mean[j]
                new_sig[j] += d * d
        for j in range(dim):
            new_sig[j] = math.sqrt(new_sig[j] * inv_mu + 1e-30)

        # smooth updates; shrink with progress, expand on restarts
        a_m = 0.2
        a_s = 0.25
        for j in range(dim):
            mean[j] = (1.0 - a_m) * mean[j] + a_m * new_mean[j]
            s = (1.0 - a_s) * sigma[j] + a_s * new_sig[j]
            # keep some exploration floor related to restart_scale
            floor = max(sigma_min[j], 0.02 * spans[j] * restart_scale)
            sigma[j] = clamp(s, floor, sigma_max[j])

        # ----- occasional local polishing on incumbent -----
        if best_x is not None and (gen % 7 == 0 or no_improve >= 10) and time.time() < deadline:
            bx, bf = local_search(best_x, best, budget_evals=60 if dim <= 15 else 35)
            if bf < best:
                best, best_x = bf, bx[:]
                # pull mean to best strongly
                for j in range(dim):
                    mean[j] = 0.7 * mean[j] + 0.3 * best_x[j]
            no_improve = 0

        # ----- one generation: mix DE moves and Gaussian sampling -----
        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # with probability, do DE; otherwise do "CMA-ish" sample around mean
            if random.random() < 0.6:
                # DE current-to-best/1/bin
                r1 = i
                while r1 == i:
                    r1 = random.randrange(NP)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(NP)

                xr1 = pop[r1]
                xr2 = pop[r2]
                xb = best_x if best_x is not None else pop[random.randrange(NP)]
                F = Fmin + (Fmax - Fmin) * random.random()

                v = [0.0] * dim
                for j in range(dim):
                    v[j] = xi[j] + F * (xb[j] - xi[j]) + F * (xr1[j] - xr2[j])

                u = xi[:]
                jrand = random.randrange(dim)
                for j in range(dim):
                    if j == jrand or random.random() < CR:
                        u[j] = reflect(v[j], lows[j], highs[j])
            else:
                # diagonal Gaussian sample around mean (exploitation/exploration)
                u = [0.0] * dim
                for j in range(dim):
                    u[j] = reflect(mean[j] + sigma[j] * randn(), lows[j], highs[j])

            fu = eval_f(u)
            if fu <= fi:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best, best_x = fu, u[:]

        # ----- stagnation tracking & restarts -----
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            no_improve = 0
            restart_scale = max(0.35, restart_scale * 0.92)
        else:
            no_improve += 1

        if no_improve >= 22 and time.time() < deadline:
            # restart: keep best, repopulate with broader sigma, inject diversity
            no_improve = 0
            restart_scale = min(2.0, restart_scale * 1.25)

            # recenter mean at best (or random if unknown)
            if best_x is not None:
                mean = best_x[:]
            else:
                mean = rand_vec()

            # widen sigma a bit
            for j in range(dim):
                sigma[j] = clamp(max(sigma[j], 0.12 * spans[j] * restart_scale), sigma_min[j], sigma_max[j])

            # reinitialize most of population
            keep = max(1, NP // 6)
            # keep a few best individuals
            idx = list(range(NP))
            idx.sort(key=lambda i: fit[i])
            keep_idx = set(idx[:keep])

            for i in range(NP):
                if time.time() >= deadline:
                    return best
                if i in keep_idx:
                    continue
                if best_x is not None and random.random() < 0.7:
                    x = [reflect(best_x[j] + (0.6 * restart_scale) * spans[j] * randn(), lows[j], highs[j])
                         for j in range(dim)]
                else:
                    x = rand_vec()
                fx = eval_f(x)
                pop[i] = x
                fit[i] = fx
                if fx < best:
                    best, best_x = fx, x[:]

    return best
