import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: Differential Evolution + best-so-far tracking.
    - No external libraries required.
    - Works on continuous bounds; robust on many black-box functions.

    Returns:
        best (float): fitness of the best found solution.
    """
    t0 = time.time()

    # --- helpers ---
    def clip(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi:
            return hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def ensure_bounds(vec):
        return [clip(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(vec):
        # func expects an "array-like"; list is typically acceptable.
        return float(func(vec))

    # --- edge cases ---
    if dim <= 0:
        # no parameters: evaluate once if possible
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # --- Differential Evolution parameters (adapted to dim) ---
    pop_size = max(8, min(40, 10 * dim))  # small-to-moderate population for speed
    F = 0.6                               # differential weight
    CR = 0.9                              # crossover prob

    # --- initialize population ---
    pop = []
    fit = []
    best = float("inf")

    for _ in range(pop_size):
        if time.time() - t0 >= max_time:
            return best
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx

    # --- main loop ---
    # We keep mutating until time runs out.
    while True:
        if time.time() - t0 >= max_time:
            return best

        for i in range(pop_size):
            if time.time() - t0 >= max_time:
                return best

            # pick 3 distinct indices a,b,c != i
            # (simple rejection sampling; pop_size is small)
            a = i
            while a == i:
                a = random.randrange(pop_size)
            b = a
            while b == i or b == a:
                b = random.randrange(pop_size)
            c = b
            while c == i or c == a or c == b:
                c = random.randrange(pop_size)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation: v = xa + F*(xb - xc)
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]
            v = ensure_bounds(v)

            # crossover (binomial); ensure at least one dimension from v
            j_rand = random.randrange(dim)
            u = []
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    u.append(v[j])
                else:
                    u.append(pop[i][j])

            fu = eval_f(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu

        # light parameter jitter to escape stagnation (very cheap)
        # adjust only if we still have time
        if time.time() - t0 < max_time:
            # small random walk on a random individual
            k = random.randrange(pop_size)
            x = pop[k][:]
            scale = 0.01  # 1% of range
            for j in range(dim):
                lo, hi = bounds[j]
                step = (hi - lo) * scale * (2.0 * random.random() - 1.0)
                x[j] = clip(x[j] + step, lo, hi)
            fx = eval_f(x)
            if fx <= fit[k]:
                pop[k] = x
                fit[k] = fx
                if fx < best:
                    best = fx
