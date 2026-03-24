import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Differential Evolution (DE)
    with occasional random-restarts and an "elite" archive.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a sequence (length dim)
    dim : int
        Dimensionality of x
    bounds : list of (low, high)
        Bounds per dimension
    max_time : int/float
        Time budget in seconds

    Returns
    -------
    best : float
        Best (minimum) fitness found within time
    """
    t0 = time.time()

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def timed_out():
        return (time.time() - t0) >= max_time

    # --- DE parameters (chosen to be robust across many problems) ---
    # Population size: small but not tiny; scale with dim.
    pop_size = max(12, 6 * dim)
    # Limit pop size a bit to keep evaluations manageable in high dimension.
    pop_size = min(pop_size, 80)

    # Archive size for elites (used to add extra candidate vectors)
    elite_size = max(4, pop_size // 5)

    # Initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [None] * pop_size

    best = float("inf")
    best_x = None

    # Evaluate initial population
    for i in range(pop_size):
        if timed_out():
            return best
        f = float(func(pop[i]))
        fit[i] = f
        if f < best:
            best = f
            best_x = pop[i][:]

    # Elite archive (store best vectors seen)
    elites = []
    elites_fit = []

    def try_add_elite(x, fx):
        nonlocal elites, elites_fit
        # Insert sorted by fitness, keep unique-ish by fitness tolerance
        # (simple filter; avoids archive filling with near-identical entries)
        tol = 1e-14
        for ef in elites_fit:
            if abs(ef - fx) <= tol * (1.0 + abs(fx)):
                break
        else:
            # insert
            j = 0
            while j < len(elites_fit) and elites_fit[j] <= fx:
                j += 1
            elites.insert(j, x[:])
            elites_fit.insert(j, fx)
            if len(elites) > elite_size:
                elites.pop()
                elites_fit.pop()

    for i in range(pop_size):
        try_add_elite(pop[i], fit[i])

    # --- main loop ---
    # Stagnation tracking -> random partial restart
    no_improve = 0
    restart_after = max(30, 10 * dim)

    while not timed_out():
        # Occasionally vary DE parameters (jitter improves robustness)
        F = 0.5 + 0.3 * (random.random() - 0.5)    # ~ [0.35, 0.65]
        CR = 0.8 + 0.2 * (random.random() - 0.5)   # ~ [0.7, 0.9]
        CR = min(0.95, max(0.05, CR))

        improved_this_gen = False

        # Create one generation
        for i in range(pop_size):
            if timed_out():
                return best

            # Choose base and difference vectors.
            # Prefer "current-to-best/1" style when we have best_x, else rand/1.
            # Pick r1, r2 distinct and not i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = random.choice(idxs)
            idxs.remove(r1)
            r2 = random.choice(idxs)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # Base: with some probability use global best or an elite
            use_best = (best_x is not None and random.random() < 0.6)
            if use_best:
                base = best_x
            elif elites and random.random() < 0.6:
                base = elites[random.randrange(len(elites))]
            else:
                # random base different from i
                rb = random.randrange(pop_size)
                if rb == i:
                    rb = (rb + 1) % pop_size
                base = pop[rb]

            # Mutation: base + F*(xr1 - xr2) + small "current-to-base" pull
            # Pull term helps exploitation without killing exploration.
            pull = 0.2
            v = [0.0] * dim
            for d in range(dim):
                v[d] = base[d] + F * (xr1[d] - xr2[d]) + pull * (base[d] - xi[d])

            clip_inplace(v)

            # Crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]

            # Optional small local perturbation around best (rare, but useful)
            if best_x is not None and random.random() < 0.03:
                # scale perturbation with span
                for d in range(dim):
                    if random.random() < 0.25:
                        u[d] = best_x[d] + (random.random() * 2.0 - 1.0) * 0.02 * spans[d]
                clip_inplace(u)

            fu = float(func(u))

            # Selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                try_add_elite(u, fu)
                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_this_gen = True

        if improved_this_gen:
            no_improve = 0
        else:
            no_improve += 1

        # Partial restart on stagnation: re-randomize worst fraction
        if no_improve >= restart_after and not timed_out():
            no_improve = 0
            # Determine how many to restart (20% worst)
            k = max(1, pop_size // 5)
            # Indices sorted by descending fitness (worst first)
            worst = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]
            for idx in worst:
                if timed_out():
                    return best
                pop[idx] = rand_vec()
                fit[idx] = float(func(pop[idx]))
                try_add_elite(pop[idx], fit[idx])
                if fit[idx] < best:
                    best = fit[idx]
                    best_x = pop[idx][:]

    return best
