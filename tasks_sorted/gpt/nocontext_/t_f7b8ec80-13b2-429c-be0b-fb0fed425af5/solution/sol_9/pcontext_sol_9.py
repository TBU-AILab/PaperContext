import random
import math
import time

def run(func, dim, bounds, max_time):
    # Self-contained stochastic optimizer: hybrid of
    # 1) Latin-ish random exploration
    # 2) Local coordinate perturbations
    # 3) Simulated annealing style acceptance
    # 4) Occasional restarts with shrinking radius around best

    start = time.time()
    deadline = start + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        return [max(lows[i], min(highs[i], x[i])) for i in range(dim)]

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; we pass a list
        return float(func(x))

    # --- initialization ---
    best_x = rand_point()
    best = evaluate(best_x)

    # Current state for annealing walk
    x = list(best_x)
    fx = best

    # Step radius (fraction of span), will adapt
    base_radius = 0.25
    radius = base_radius

    # For adaptation
    no_improve = 0
    accepted = 0
    tried = 0

    # Temperature schedule parameters
    # Start with a temperature tied to scale of objective changes (estimated online)
    T = 1.0
    T_min = 1e-12

    # Estimate typical delta by sampling a few points quickly
    # (kept small to respect time)
    for _ in range(5):
        if time.time() >= deadline:
            return best
        y = rand_point()
        fy = evaluate(y)
        if fy < best:
            best, best_x = fy, y
            x, fx = list(best_x), best
        # rough scale
        T = max(T, abs(fy - fx) + 1e-9)

    # --- main loop ---
    while time.time() < deadline:
        # Update temperature (exponential cooling)
        # Tie cooling to progress in time
        t = (time.time() - start) / max(1e-9, max_time)
        T = max(T_min, (1.0 - t) * T)

        # Generate candidate by perturbing a subset of coordinates
        # More local as radius shrinks.
        k = 1 if dim == 1 else (1 + (random.randrange(dim) if random.random() < 0.35 else 0))
        idxs = random.sample(range(dim), k)

        cand = list(x)
        for i in idxs:
            # heavy-tailed step (Cauchy-like) for occasional long jumps
            u = random.random() - 0.5
            step = math.tan(math.pi * u)  # heavy tail
            # scale and squash
            step = max(-10.0, min(10.0, step))
            cand[i] = cand[i] + step * radius * spans[i]

        cand = clip(cand)
        f_cand = evaluate(cand)

        tried += 1
        delta = f_cand - fx

        # Accept if better, or sometimes if worse (annealing)
        accept = False
        if delta <= 0:
            accept = True
        else:
            # SA acceptance
            prob = math.exp(-delta / max(T_min, T))
            if random.random() < prob:
                accept = True

        if accept:
            x, fx = cand, f_cand
            accepted += 1

            # Track best
            if f_cand < best:
                best, best_x = f_cand, list(cand)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Adapt radius based on recent acceptance ratio
        if tried >= 50:
            acc_rate = accepted / float(tried)
            # If accepting too often, steps may be too small -> enlarge
            if acc_rate > 0.45:
                radius = min(0.5, radius * 1.25)
            # If accepting rarely, steps too big -> shrink
            elif acc_rate < 0.15:
                radius = max(1e-6, radius * 0.7)
            accepted = 0
            tried = 0

        # Occasional restart around best if stagnating
        if no_improve > 200:
            # shrink radius to intensify around best, but sometimes broaden
            if random.random() < 0.8:
                radius = max(1e-6, radius * 0.5)
                x = list(best_x)
                fx = best
            else:
                radius = base_radius
                x = rand_point()
                fx = evaluate(x)
                if fx < best:
                    best, best_x = fx, list(x)
            no_improve = 0

    # return fitness of the best found solution
    return best
