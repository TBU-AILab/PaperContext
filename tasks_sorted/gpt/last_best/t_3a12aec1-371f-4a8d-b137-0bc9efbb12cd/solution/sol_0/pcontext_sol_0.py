import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimization (minimization) using a self-contained
    Differential Evolution (DE/rand/1/bin) core with:
      - opposition-based initialization (helps early progress)
      - periodic random immigrants (helps escape stagnation)
      - safe bound handling (clamp)

    Returns:
        best (float): best (minimum) objective value found within max_time
    """
    t0 = time.time()

    # ---- helpers ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # avoid degenerate spans
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clamp_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def opposite(x):
        # opposition point within bounds
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def safe_eval(x):
        # Evaluate func; if it errors or returns non-finite, treat as very bad.
        try:
            v = func(x)
            if v is None:
                return float('inf')
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float('inf')
            return v
        except Exception:
            return float('inf')

    # ---- parameter choices (robust defaults) ----
    pop_size = max(12, 8 * dim)          # small-but-decent population
    F = 0.7                              # differential weight
    CR = 0.9                             # crossover rate
    immigrant_period = 25                # generations between immigrant injections
    immigrant_frac = 0.15                # replace this fraction of worst individuals
    jitter_prob = 0.02                   # tiny chance to jitter a coordinate in trial

    # ---- initialization with opposition ----
    pop = []
    fit = []
    best = float('inf')

    for _ in range(pop_size):
        x = rand_vec()
        xo = opposite(x)

        fx = safe_eval(x)
        fo = safe_eval(xo)

        if fo < fx:
            pop.append(xo)
            fit.append(fo)
            if fo < best:
                best = fo
        else:
            pop.append(x)
            fit.append(fx)
            if fx < best:
                best = fx

        if time.time() - t0 >= max_time:
            return best

    # ---- main loop ----
    gen = 0
    while True:
        if time.time() - t0 >= max_time:
            return best

        gen += 1

        # Determine current best (for tracking only; DE variant uses rand/1)
        # (We still update 'best' continuously.)
        for i in range(pop_size):
            if fit[i] < best:
                best = fit[i]

        # Periodic immigrants: replace worst fraction with new random points
        if immigrant_period > 0 and (gen % immigrant_period == 0):
            k = max(1, int(pop_size * immigrant_frac))
            # indices of individuals sorted by fitness descending (worst first)
            worst = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]
            for idx in worst:
                if time.time() - t0 >= max_time:
                    return best
                xnew = rand_vec()
                fnew = safe_eval(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best = fnew

        # One DE generation
        for i in range(pop_size):
            if time.time() - t0 >= max_time:
                return best

            # choose r1,r2,r3 distinct and != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = random.choice(idxs); idxs.remove(r1)
            r2 = random.choice(idxs); idxs.remove(r2)
            r3 = random.choice(idxs)

            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            target = pop[i]

            # mutation: v = x1 + F*(x2-x3)
            donor = [x1[j] + F * (x2[j] - x3[j]) for j in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            trial = target[:]  # copy
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]

                # rare coordinate jitter to add exploration
                if random.random() < jitter_prob:
                    trial[j] = lows[j] + random.random() * spans[j]

            clamp_vec(trial)

            ftrial = safe_eval(trial)

            # selection
            if ftrial <= fit[i]:
                pop[i] = trial
                fit[i] = ftrial
                if ftrial < best:
                    best = ftrial
