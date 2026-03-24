import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid algorithm:
    - Latin-ish random initialization
    - (mu + lambda) evolution strategy
    - Adaptive step-size (1/5 success rule style)
    - Occasional random restarts
    Returns: best (float) = minimum function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # Basic checks / setup
    if dim <= 0:
        return float("inf")
    if bounds is None or len(bounds) != dim:
        raise ValueError("bounds must be a list of (low, high) pairs, one per dimension")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if not (s > 0.0):
            raise ValueError("Each bound must satisfy high > low")

    def clamp(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        # func expects an array-like; we pass list.
        # If func errors, treat as very bad.
        try:
            v = func(x)
            # Protect against NaN/inf
            if v is None:
                return float("inf")
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return float("inf")
                return v
            # If it returns something numeric-like
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Strategy parameters (kept small so it works under tight time budgets)
    mu = max(4, min(20, 2 * dim))           # parents
    lam = max(10, min(60, 5 * dim))         # offspring
    elite_keep = max(1, mu // 2)            # elites
    restart_after = 200 + 20 * dim          # stagnation iterations before restart

    # Initial step sizes per dimension (fraction of span)
    base_sigma = [0.15 * spans[i] for i in range(dim)]

    # Create initial population
    pop = []
    for _ in range(mu):
        x = rand_vec()
        f = safe_eval(x)
        pop.append([f, x])

    pop.sort(key=lambda t: t[0])
    best = pop[0][0]
    best_x = pop[0][1][:]

    # Adaptive state
    sigma_scale = 1.0
    success_counter = 0
    trial_counter = 0
    no_improve_iters = 0

    # Helper: recombine two parents (intermediate recombination)
    def recombine(a, b):
        w = random.random()
        return [w * a[i] + (1.0 - w) * b[i] for i in range(dim)]

    # Helper: mutate
    def mutate(x):
        y = x[:]
        # Log-normal global sigma adjustment via sigma_scale; keep bounded
        for i in range(dim):
            step = random.gauss(0.0, base_sigma[i] * sigma_scale)
            y[i] += step
        return clamp(y)

    # Main loop
    while time.time() < deadline:
        # Produce offspring
        offspring = []

        # Keep elites to preserve best solutions
        pop.sort(key=lambda t: t[0])
        elites = pop[:elite_keep]

        # Offspring creation
        for _ in range(lam):
            # Parent selection: tournament
            def pick():
                k = 3
                cand = [pop[random.randrange(len(pop))] for _ in range(k)]
                cand.sort(key=lambda t: t[0])
                return cand[0][1]
            p1 = pick()
            p2 = pick()

            child = recombine(p1, p2)

            # With some probability, use a pure random restart individual
            if random.random() < 0.05:
                child = rand_vec()
            else:
                child = mutate(child)

            f = safe_eval(child)
            offspring.append([f, child])

            # Time check inside tight loop
            if time.time() >= deadline:
                break

        # Combine and select next generation (mu + lambda)
        combined = elites + offspring
        combined.sort(key=lambda t: t[0])
        pop = combined[:mu]

        # Update global best
        if pop[0][0] < best:
            best = pop[0][0]
            best_x = pop[0][1][:]
            success_counter += 1
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        trial_counter += 1

        # Step-size adaptation every few iterations (1/5 success rule heuristic)
        if trial_counter >= 10:
            rate = success_counter / float(trial_counter)
            # If too successful, increase step; if not, decrease
            if rate > 0.2:
                sigma_scale *= 1.2
            else:
                sigma_scale *= 0.82
            # Keep sigma in reasonable range
            sigma_scale = max(1e-6, min(5.0, sigma_scale))
            success_counter = 0
            trial_counter = 0

        # Restart if stagnating
        if no_improve_iters >= restart_after:
            # Re-seed population around best_x plus noise, and some random points
            new_pop = [[best, best_x[:]]]
            for i in range(1, mu):
                if i < mu // 2:
                    # local samples around best
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.25 * spans[d] * sigma_scale)
                    x = clamp(x)
                else:
                    x = rand_vec()
                f = safe_eval(x)
                new_pop.append([f, x])
            new_pop.sort(key=lambda t: t[0])
            pop = new_pop
            sigma_scale = 1.0
            no_improve_iters = 0

    # return fitness of the best found solution
    return best
