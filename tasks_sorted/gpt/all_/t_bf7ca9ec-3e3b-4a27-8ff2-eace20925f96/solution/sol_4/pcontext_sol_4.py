import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (standard library only).

    Algorithm: JADE-like Differential Evolution (current-to-pbest/1) with:
      - external archive (for diversity),
      - adaptive parameter memories for F and CR (per JADE spirit),
      - optional "opposition" injection for quick coverage,
      - periodic lightweight local coordinate polish on the best,
      - stagnation-triggered partial restart.

    Returns
    -------
    best : float
        Best (minimum) fitness found within the time budget.
    """
    t0 = time.time()
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # --- time helpers ---
    def time_up():
        return (time.time() - t0) >= max_time

    def now():
        return time.time()

    # --- bound handling (reflection) ---
    def reflect(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            w = hi - lo
            v = y[i] - lo
            m = v % (2.0 * w)
            if m <= w:
                y[i] = lo + m
            else:
                y[i] = lo + (2.0 * w - m)
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        # reflection across center: x_op = lo+hi - x
        y = [lows[i] + highs[i] - x[i] for i in range(dim)]
        return reflect(y)

    def eval_f(x):
        return float(func(x))

    # --- quick distribution samplers ---
    def cauchy(loc, scale):
        # inverse CDF of Cauchy: loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def normal(mu, sigma):
        return random.gauss(mu, sigma)

    # Population size: modest; better than too-small DE in rugged landscapes
    NP = max(14, min(60, 10 + 4 * dim))

    # Initialize population (with some opposition sampling to cover space)
    pop = []
    fit = []
    for _ in range(NP):
        x = rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if time_up():
            return min(fit) if fit else float("inf")

    # one pass of opposition injection for some individuals (cheap coverage gain)
    opp_trials = max(2, NP // 5)
    for _ in range(opp_trials):
        if time_up():
            return min(fit)
        i = random.randrange(NP)
        xo = opposite_point(pop[i])
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i] = xo
            fit[i] = fo

    best_i = min(range(NP), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # External archive (stores replaced parents), bounded size
    archive = []
    archive_max = NP

    # JADE-style parameter memories
    # mu_F, mu_CR updated from successful trials
    mu_F = 0.6
    mu_CR = 0.6
    c = 0.1  # learning rate

    # p-best fraction for current-to-pbest
    p = 0.2

    # Stagnation control
    last_improve = now()
    patience = max(0.12 * max_time, 0.8)

    # Periodic local polish of best
    next_polish = now() + max(0.15 * max_time, 0.25)
    polish_budget = max(12, 4 * dim)

    def local_polish(x0, f0):
        x = x0[:]
        fx = f0
        steps = [0.05 * spans[i] for i in range(dim)]
        min_steps = [1e-12 * spans[i] for i in range(dim)]

        for _ in range(polish_budget):
            if time_up():
                break
            i = random.randrange(dim)
            s = steps[i]
            if s < min_steps[i]:
                continue

            xp = x[:]
            xp[i] += s
            xp = reflect(xp)
            fp = eval_f(xp)
            if time_up():
                return x, fx

            xm = x[:]
            xm[i] -= s
            xm = reflect(xm)
            fm = eval_f(xm)

            if fp < fx or fm < fx:
                if fp <= fm:
                    x, fx = xp, fp
                else:
                    x, fx = xm, fm
                steps[i] = min(0.25 * spans[i], steps[i] * 1.20)
            else:
                steps[i] = max(min_steps[i], steps[i] * 0.60)
        return x, fx

    # Utility: choose indices distinct from i
    def pick_two_distinct(exclude_i, n):
        # picks 2 distinct indices in [0,n) excluding exclude_i
        a = exclude_i
        while a == exclude_i:
            a = random.randrange(n)
        b = a
        while b == exclude_i or b == a:
            b = random.randrange(n)
        return a, b

    # Main loop
    while not time_up():
        # periodic polish
        if now() >= next_polish and not time_up():
            bx, bf = local_polish(best_x, best)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve = now()
            # more frequent later
            rem = max(0.0, max_time - (now() - t0))
            next_polish = now() + max(0.10, 0.06 * rem)

        # stagnation: partial restart (reseed worst around best / random)
        if (now() - last_improve) >= patience and not time_up():
            k = max(2, NP // 4)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                if time_up():
                    break
                if random.random() < 0.45:
                    x = rand_point()
                else:
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += 0.10 * spans[d] * math.tan(math.pi * (random.random() - 0.5))
                    x = reflect(x)
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve = now()
            # clear archive a bit (fresh dynamics)
            if len(archive) > 0:
                archive = archive[len(archive)//2:]
            last_improve = now()

        # generation successes for adapting mu_F, mu_CR
        S_F = []
        S_CR = []
        S_df = []

        # prepare ranking for p-best selection
        # sort indices by fitness ascending
        order = sorted(range(NP), key=lambda i: fit[i])
        pnum = max(2, int(math.ceil(p * NP)))

        for i in range(NP):
            if time_up():
                break

            xi = pop[i]
            fi = fit[i]

            # sample CR ~ N(mu_CR, 0.1), clip to [0,1]
            CRi = normal(mu_CR, 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # sample F ~ Cauchy(mu_F, 0.1) until >0, then clip to <=1
            Fi = cauchy(mu_F, 0.1)
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(mu_F, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            # choose pbest from top pnum
            pbest_idx = order[random.randrange(pnum)]
            xpbest = pop[pbest_idx]

            # choose r1 from pop, r2 from pop U archive (both distinct from i and pbest)
            r1 = i
            while r1 == i or r1 == pbest_idx:
                r1 = random.randrange(NP)

            use_archive = (len(archive) > 0 and random.random() < (len(archive) / float(len(archive) + NP)))
            if use_archive:
                # pick from archive, but ensure not same vector object; no index constraints
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == pbest_idx or r2 == r1:
                    r2 = random.randrange(NP)
                xr2 = pop[r2]

            xr1 = pop[r1]

            # current-to-pbest/1: v = x + F*(pbest-x) + F*(r1-r2)
            v = [xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d]) for d in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]

            u = reflect(u)
            fu = eval_f(u)

            if fu <= fi:
                # success: archive parent
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # random removal
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                pop[i] = u
                fit[i] = fu

                # parameter adaptation data
                S_F.append(Fi)
                S_CR.append(CRi)
                df = abs(fi - fu)
                S_df.append(df if df > 0.0 else 1e-12)

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = now()

        # adapt mu_F, mu_CR from successes
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = float(len(S_df))

            # weighted arithmetic mean for CR
            mean_CR = 0.0
            for cr, w in zip(S_CR, S_df):
                mean_CR += cr * (w / wsum)

            # weighted Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_df):
                wf = w / wsum
                num += wf * f * f
                den += wf * f
            mean_F = (num / den) if den > 0.0 else mu_F

            mu_CR = (1.0 - c) * mu_CR + c * mean_CR
            mu_F = (1.0 - c) * mu_F + c * mean_F

            # keep in sane ranges
            if mu_CR < 0.05:
                mu_CR = 0.05
            elif mu_CR > 0.95:
                mu_CR = 0.95
            if mu_F < 0.10:
                mu_F = 0.10
            elif mu_F > 0.95:
                mu_F = 0.95

    return best
