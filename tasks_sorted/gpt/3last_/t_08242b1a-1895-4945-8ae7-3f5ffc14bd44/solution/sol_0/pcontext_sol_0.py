import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained variant of Differential Evolution
    with periodic restarts and occasional local refinement (coordinate search).
    No external libraries required.

    Args:
        func: callable(list[float]) -> float
        dim: int
        bounds: list[tuple(float,float)] length=dim
        max_time: seconds (int/float)

    Returns:
        best (float): best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x, i):
        lo, hi = lows[i], highs[i]
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_vec(x):
        # Ensure within bounds (defensive)
        xx = [clip(x[i], i) for i in range(dim)]
        return func(xx)

    def coordinate_refine(x, fx, time_budget_frac=0.08):
        """
        Simple pattern/coordinate search around current best.
        Runs only for a small fraction of remaining time.
        """
        t_end = min(deadline, time.time() + max(0.0, (deadline - time.time()) * time_budget_frac))
        step = [0.1 * s if s > 0 else 1.0 for s in spans]  # initial step sizes
        xbest = x[:]
        fbest = fx

        while time.time() < t_end:
            improved = False
            for i in range(dim):
                if time.time() >= t_end:
                    break

                # try +step
                xp = xbest[:]
                xp[i] = clip(xp[i] + step[i], i)
                fp = eval_vec(xp)
                if fp < fbest:
                    xbest, fbest = xp, fp
                    improved = True
                    continue

                # try -step
                xm = xbest[:]
                xm[i] = clip(xm[i] - step[i], i)
                fm = eval_vec(xm)
                if fm < fbest:
                    xbest, fbest = xm, fm
                    improved = True
                    continue

                # no improvement on this coordinate -> shrink step a bit
                step[i] *= 0.7

            # if no coordinate improved, shrink all steps
            if not improved:
                for i in range(dim):
                    step[i] *= 0.85

            # stop if steps are tiny
            if max(step) < 1e-12 * (max(spans) if max(spans) > 0 else 1.0):
                break

        return xbest, fbest

    # --- algorithm parameters (adaptive / conservative) ---
    # population size: small for speed, but at least 8
    pop_size = max(8, min(40, 10 * dim))
    # Differential evolution controls
    F_base = 0.55
    CR_base = 0.85

    best = float("inf")
    best_x = None

    # Restart logic
    stagnation_limit = 25 * pop_size  # number of trials without global improvement
    trials_since_improve = 0

    # Main loop with restarts
    while time.time() < deadline:
        # --- initialize population ---
        pop = [rand_vec() for _ in range(pop_size)]
        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = eval_vec(x)
            fit.append(fx)
            if fx < best:
                best, best_x = fx, x[:]
                trials_since_improve = 0

        # --- evolve ---
        idx_best = min(range(pop_size), key=lambda i: fit[i])
        if fit[idx_best] < best:
            best, best_x = fit[idx_best], pop[idx_best][:]
            trials_since_improve = 0

        while time.time() < deadline:
            # mild adaptation / jitter to avoid getting stuck
            F = min(0.95, max(0.1, random.gauss(F_base, 0.08)))
            CR = min(1.0, max(0.0, random.gauss(CR_base, 0.08)))

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                # pick 3 distinct indices != i
                a = b = c = i
                while a == i:
                    a = random.randrange(pop_size)
                while b == i or b == a:
                    b = random.randrange(pop_size)
                while c == i or c == a or c == b:
                    c = random.randrange(pop_size)

                xa, xb, xc = pop[a], pop[b], pop[c]
                xi = pop[i]

                # mutation: v = xa + F*(xb - xc)
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xa[d] + F * (xb[d] - xc[d])

                # crossover (binomial), ensure at least one dimension from v
                jrand = random.randrange(dim)
                u = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        u[d] = clip(v[d], d)

                fu = eval_vec(u)
                trials_since_improve += 1

                # selection
                if fu <= fit[i]:
                    pop[i] = u
                    fit[i] = fu

                    if fu < best:
                        best = fu
                        best_x = u[:]
                        trials_since_improve = 0

            # If stagnating, attempt a quick local refine on current best, then restart
            if trials_since_improve >= stagnation_limit:
                if best_x is not None and time.time() < deadline:
                    best_x, best = coordinate_refine(best_x, best, time_budget_frac=0.10)
                trials_since_improve = 0
                break  # restart population

            # Occasionally do a short refinement if we have time (helps smooth problems)
            if best_x is not None and random.random() < 0.03:
                if time.time() < deadline:
                    best_x, best = coordinate_refine(best_x, best, time_budget_frac=0.03)

    return best
