import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved derivative-free minimizer (self-contained, no external libs).

    Main changes vs prior version:
    - Population-based search (DE: Differential Evolution) for stronger global exploration.
    - Time-aware: keeps evaluating until deadline; adapts population size to dim/time.
    - Occasional local refinement around current best using coordinate search + step halving.
    - Boundary handling via reflection (often better than hard clipping for DE).
    - Early cheap "opposition" points to widen initial coverage.
    Returns: best fitness (float).
    """

    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    eps_span = [s if s > 0.0 else 1.0 for s in spans]

    # --- helpers ---
    def reflect(val, i):
        lo = lows[i]
        hi = highs[i]
        if lo == hi:
            return lo
        # Reflect repeatedly until inside bounds
        while val < lo or val > hi:
            if val < lo:
                val = lo + (lo - val)
            if val > hi:
                val = hi - (val - hi)
        # Numerical safety
        if val < lo: val = lo
        if val > hi: val = hi
        return val

    def rand_point():
        return [lows[i] + random.random() * eps_span[i] for i in range(dim)]

    def opposite_point(x):
        # "Opposition-based" point across the center of bounds
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # --- choose population size ---
    # DE typically uses ~5..10*dim; keep bounded to preserve time.
    # Also use a small minimum for low dim.
    pop_size = 6 * dim + 10
    if pop_size < 20:
        pop_size = 20
    if pop_size > 80:
        pop_size = 80

    # If time is very small, shrink population to avoid wasting budget on init
    if max_time <= 0.25:
        pop_size = min(pop_size, 24)
    elif max_time <= 0.75:
        pop_size = min(pop_size, 36)

    # --- initialize population with random + opposition points ---
    pop = []
    fit = []

    best_x = None
    best = float("inf")

    # create half random, half opposite (paired)
    n_pairs = pop_size // 2
    for _ in range(n_pairs):
        if time.time() >= deadline:
            return best
        x = rand_point()
        ox = opposite_point(x)

        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x[:]

        if time.time() >= deadline:
            return best
        fox = eval_f(ox)
        if fox < best:
            best = fox
            best_x = ox[:]

        pop.append(x); fit.append(fx)
        if len(pop) < pop_size:
            pop.append(ox); fit.append(fox)

    while len(pop) < pop_size:
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if fx < best:
            best = fx; best_x = x[:]

    # --- DE parameters ---
    # Use a little randomness in F and CR to reduce tuning sensitivity
    F_base = 0.7
    CR_base = 0.9

    # --- local refinement (coordinate search) ---
    def local_refine(x0, f0, time_cap):
        # Simple deterministic-ish coordinate pattern search with step halving.
        # Limited by time_cap seconds.
        t_end = min(deadline, time.time() + time_cap)
        x = x0[:]
        fx = f0

        # initial step sizes relative to spans
        step = [0.1 * eps_span[i] for i in range(dim)]
        min_step = [max(1e-12, 1e-8 * eps_span[i]) for i in range(dim)]

        improved_any = True
        while time.time() < t_end and improved_any:
            improved_any = False
            for j in range(dim):
                if time.time() >= t_end:
                    break

                sj = step[j]
                if sj < min_step[j]:
                    continue

                # try + and -
                for sign in (1.0, -1.0):
                    if time.time() >= t_end:
                        break
                    cand = x[:]
                    cand[j] = reflect(cand[j] + sign * sj, j)
                    fc = eval_f(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved_any = True
                        break  # move to next coordinate after improvement

            # if no improvement, shrink steps
            if not improved_any:
                for j in range(dim):
                    step[j] *= 0.5
                # stop if all tiny
                if all(step[j] < min_step[j] for j in range(dim)):
                    break

        return x, fx

    # --- main loop: DE + occasional refinement ---
    iter_count = 0
    last_refine = start

    while True:
        now = time.time()
        if now >= deadline:
            return best

        iter_count += 1

        # Time-adaptive: occasionally refine the current best (not too often)
        # Spend a small fraction of time, more as time grows.
        if (now - last_refine) > max(0.15, 0.08 * max_time):
            # only refine if we have enough time left
            remaining = deadline - now
            if remaining > 0.05:
                cap = min(0.08 * max_time, 0.25 * remaining, 0.35)  # bounded
                rx, rf = local_refine(best_x, best, cap)
                if rf < best:
                    best, best_x = rf, rx[:]
                    # inject refined best into worst individual to spread improvement
                    worst_i = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst_i] = best_x[:]
                    fit[worst_i] = best
                last_refine = time.time()

        # One DE generation (or partial if close to deadline)
        # Use "current-to-best/1" mix sometimes for exploitation.
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # Choose mutation strategy based on progress fraction
            tfrac = (time.time() - start) / max(1e-12, max_time)

            # Randomize parameters slightly
            F = min(0.95, max(0.2, random.gauss(F_base, 0.08)))
            CR = min(0.98, max(0.05, random.gauss(CR_base, 0.08)))

            # pick 3 distinct indices != i
            a = b = c = i
            while a == i:
                a = random.randrange(pop_size)
            while b == i or b == a:
                b = random.randrange(pop_size)
            while c == i or c == a or c == b:
                c = random.randrange(pop_size)

            xi = pop[i]
            xa = pop[a]
            xb = pop[b]
            xc = pop[c]

            # mutation
            # early: classic rand/1
            # later: current-to-best/1 for faster convergence
            if tfrac < 0.55 or best_x is None:
                v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
            else:
                v = [xi[d] + F * (best_x[d] - xi[d]) + F * (xb[d] - xc[d]) for d in range(dim)]

            # crossover (binomial); ensure at least one dimension from mutant
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = reflect(v[d], d)

            fu = eval_f(u)
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # mild parameter drift (helps on some problems)
        # as time progresses, lower F and CR slightly to stabilize
        if iter_count % 5 == 0:
            prog = (time.time() - start) / max(1e-12, max_time)
            F_base = 0.75 - 0.25 * min(1.0, prog)   # 0.75 -> 0.50
            CR_base = 0.95 - 0.20 * min(1.0, prog)  # 0.95 -> 0.75
