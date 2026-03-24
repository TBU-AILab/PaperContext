import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (standard library only).

    Algorithm: Differential Evolution "current-to-best/1" + simple success-based
    adaptation + periodic local coordinate polish + stagnation-triggered partial
    restart. Designed to be robust under unknown landscapes and tight time limits.

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

    def time_left():
        return max_time - (time.time() - t0)

    def time_up():
        return (time.time() - t0) >= max_time

    def reflect(x):
        # reflection handling (better than clamp for search dynamics)
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
            # numerical safety
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # Population size (small for speed, but scales with dim)
    NP = max(10, min(50, 8 + 3 * dim))

    # Initialize
    pop = [rand_point() for _ in range(NP)]
    fit = [eval_f(ind) for ind in pop]
    best_i = min(range(NP), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # DE parameters (start exploratory; will adapt)
    F = 0.65
    CR = 0.9

    # Success-based adaptation bookkeeping
    succ = 0
    att = 0
    adapt_every = 25

    # Stagnation handling
    last_improve = time.time()
    patience = max(0.10 * max_time, 0.7)

    # Local polish scheduling
    next_polish = time.time() + max(0.12 * max_time, 0.25)
    polish_tries = max(12, 4 * dim)

    def local_polish(x0, f0):
        x = x0[:]
        fx = f0
        # start with modest coordinate steps; shrink on failures
        steps = [0.06 * spans[i] for i in range(dim)]
        min_steps = [1e-12 * spans[i] for i in range(dim)]
        for _ in range(polish_tries):
            if time_up():
                break
            i = random.randrange(dim)
            s = steps[i]
            if s < min_steps[i]:
                continue

            # best of (+s, -s)
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

            if fp <= fx or fm <= fx:
                if fp <= fm:
                    x, fx = xp, fp
                else:
                    x, fx = xm, fm
                steps[i] = min(0.25 * spans[i], steps[i] * 1.20)
            else:
                steps[i] = max(min_steps[i], steps[i] * 0.60)
        return x, fx

    # Main loop
    while not time_up():
        # Periodic local polish on incumbent best
        if time.time() >= next_polish and not time_up():
            bx, bf = local_polish(best_x, best)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve = time.time()
            # more frequent later
            rem = max(0.0, time_left())
            next_polish = time.time() + max(0.12, 0.06 * rem)

        # Stagnation: re-seed worst quarter around best / random
        if (time.time() - last_improve) >= patience and not time_up():
            k = max(1, NP // 4)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                if random.random() < 0.5:
                    pop[idx] = rand_point()
                else:
                    y = best_x[:]
                    for j in range(dim):
                        # heavy-tailed-ish kick around best
                        u = random.random()
                        c = math.tan(math.pi * (u - 0.5))
                        y[j] += 0.08 * spans[j] * c
                    pop[idx] = reflect(y)
                fit[idx] = eval_f(pop[idx])
                if fit[idx] < best:
                    best = fit[idx]
                    best_x = pop[idx][:]
                    last_improve = time.time()
            last_improve = time.time()
            # slightly re-randomize controls
            F = 0.55 + 0.35 * random.random()
            CR = 0.75 + 0.20 * random.random()

        # One generation
        for i in range(NP):
            if time_up():
                break

            # pick r1,r2 distinct and != i
            idxs = list(range(NP))
            idxs.remove(i)
            r1, r2 = random.sample(idxs, 2)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # jittered parameters per trial
            Fj = min(0.95, max(0.10, F + random.uniform(-0.20, 0.20)))
            CRj = min(0.98, max(0.05, CR + random.uniform(-0.12, 0.12)))

            # current-to-best/1: v = x + F*(best-x) + F*(r1-r2)
            v = [xi[d] + Fj * (best_x[d] - xi[d]) + Fj * (xr1[d] - xr2[d]) for d in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CRj:
                    u[d] = v[d]
            u = reflect(u)

            fu = eval_f(u)

            att += 1
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                succ += 1
                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_improve = time.time()

            # success-based adaptation (cheap, stable)
            if att >= adapt_every:
                rate = succ / float(att)
                # if too few successes -> reduce F/CR a bit; if many -> increase slightly
                if rate < 0.15:
                    F = max(0.20, F * 0.93)
                    CR = max(0.55, CR * 0.97)
                elif rate > 0.30:
                    F = min(0.90, F * 1.04)
                    CR = min(0.98, CR * 1.02)
                else:
                    # small random walk to avoid freezing
                    F = min(0.90, max(0.20, F + random.uniform(-0.02, 0.02)))
                    CR = min(0.98, max(0.55, CR + random.uniform(-0.02, 0.02)))
                succ = 0
                att = 0

    return best
