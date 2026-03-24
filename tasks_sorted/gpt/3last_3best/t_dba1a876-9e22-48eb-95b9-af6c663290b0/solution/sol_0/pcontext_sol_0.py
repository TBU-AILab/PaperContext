import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Differential Evolution (DE)
    with random-restart + local refinement (coordinate/pattern search).
    Returns the best (minimum) fitness found within max_time seconds.

    func: callable(params_list)->float
    dim: int
    bounds: list of (low, high) pairs, length=dim
    max_time: seconds (int/float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        # clip in-place and return x
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Cache evaluations for exact repeats (rare but cheap to keep)
    cache = {}

    def evaluate(x):
        # Ensure hashable key; rounding reduces floating noise without hurting much
        key = tuple(round(v, 12) for v in x)
        if key in cache:
            return cache[key]
        val = float(func(list(x)))
        cache[key] = val
        return val

    # --- DE parameters (adaptive-ish) ---
    # Population size: modest to work in limited time
    pop_size = max(8, min(30, 10 * dim))
    # Mutation factor and crossover rate ranges
    F_min, F_max = 0.4, 0.95
    CR_min, CR_max = 0.1, 0.95

    # --- initialization ---
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [evaluate(ind) for ind in pop]
    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # --- local refinement: coordinate/pattern search around an incumbent ---
    def local_refine(x0, f0, time_limit):
        x = x0[:]
        fx = f0
        # Initial step sizes per coordinate
        steps = [0.2 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = 1e-12

        # Simple pattern/coordinate search with step halving
        while time.time() < time_limit:
            improved = False
            for i in range(dim):
                if time.time() >= time_limit:
                    break
                si = steps[i]
                if si <= min_step:
                    continue

                # try +step
                xp = x[:]
                xp[i] += si
                clip(xp)
                fp = evaluate(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                # try -step
                xm = x[:]
                xm[i] -= si
                clip(xm)
                fm = evaluate(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    continue

            if not improved:
                # reduce steps
                for i in range(dim):
                    steps[i] *= 0.5
                # stop if all steps tiny
                if max(steps) <= min_step:
                    break
        return x, fx

    # --- main loop ---
    gen = 0
    # How often to attempt local search (in generations)
    local_every = max(5, int(10 + dim / 2))

    while time.time() < deadline:
        gen += 1

        # mild adaptation of F, CR over time/gen
        # Bias exploration early and exploitation later
        frac = min(1.0, (time.time() - t0) / max(1e-9, (deadline - t0)))
        F = F_max - (F_max - F_min) * frac
        CR = CR_min + (CR_max - CR_min) * (1.0 - frac)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose three distinct indices a,b,c != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation: v = xa + F*(xb-xc)
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]

            # binomial crossover to form trial u
            u = pop[i][:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    u[j] = v[j]

            clip(u)
            fu = evaluate(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # occasional local refinement of current best
        if gen % local_every == 0 and time.time() < deadline:
            # allocate a small slice of remaining time to local search
            remaining = deadline - time.time()
            ls_budget = min(0.15 * max_time, 0.25 * remaining)
            if ls_budget > 0:
                x_ls, f_ls = local_refine(best_x, best, time.time() + ls_budget)
                if f_ls < best:
                    best = f_ls
                    best_x = x_ls[:]

        # stagnation handling: partial random restart
        # If population becomes too similar, inject randomness.
        if gen % (local_every * 2) == 0 and time.time() < deadline:
            # compute rough diversity as average normalized std per dimension
            means = [0.0] * dim
            for ind in pop:
                for j in range(dim):
                    means[j] += ind[j]
            means = [m / pop_size for m in means]

            vars_ = [0.0] * dim
            for ind in pop:
                for j in range(dim):
                    d = (ind[j] - means[j]) / (spans[j] if spans[j] != 0 else 1.0)
                    vars_[j] += d * d
            std_norm = math.sqrt(sum(vars_) / (pop_size * max(1, dim)))

            if std_norm < 1e-3:
                # reinitialize worst half, keep best half
                order = sorted(range(pop_size), key=lambda k: fit[k])
                keep = pop_size // 2
                for k in order[keep:]:
                    pop[k] = rand_vec()
                    fit[k] = evaluate(pop[k])
                best_idx = min(range(pop_size), key=lambda k: fit[k])
                if fit[best_idx] < best:
                    best = fit[best_idx]
                    best_x = pop[best_idx][:]

    return best
