import random
import math
import time

def run(func, dim, bounds, max_time):
    # Self-contained optimization: randomized local search with adaptive step sizes,
    # occasional restarts, and a lightweight coordinate/gradient-free refinement.
    start = time.time()
    deadline = start + max_time

    # --- helpers (no external libs) ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi: 
            return hi
        return x

    def project(vec):
        return [clamp(vec[i], lows[i], highs[i]) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; list is fine for most call sites.
        return float(func(x))

    # --- initialization ---
    # A few initial samples to find a decent starting point.
    best_x = rand_vec()
    best = evaluate(best_x)

    # Adaptive step sizes per dimension (start at 20% span, decay on failures).
    step = [0.2 * s if s > 0 else 1.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # Bookkeeping for restart logic
    no_improve = 0
    # This threshold adapts to dimension; higher dim => allow more attempts
    restart_patience = max(50, 30 * dim)

    # --- main loop ---
    while time.time() < deadline:
        # 1) Try a local move around current best (Gaussian perturbation).
        cand = best_x[:]
        # Perturb a subset of dimensions for efficiency
        k = 1 + int(random.random() * min(dim, 8))  # perturb up to 8 dims
        for _ in range(k):
            j = random.randrange(dim)
            # Gaussian step; scale with current step size
            cand[j] += random.gauss(0.0, step[j])
        cand = project(cand)

        f = evaluate(cand)
        if f <= best:
            best, best_x = f, cand
            no_improve = 0
            # Slightly increase steps when improving to speed exploration
            for j in range(dim):
                step[j] *= 1.02
            continue
        else:
            no_improve += 1
            # Reduce step sizes when not improving
            for j in range(dim):
                step[j] *= 0.98
                if step[j] < min_step[j]:
                    step[j] = min_step[j]

        if time.time() >= deadline:
            break

        # 2) Lightweight coordinate refinement occasionally (pattern search-like).
        # Only do it sometimes to control function evaluations.
        if no_improve % (10 + dim) == 0 and time.time() < deadline:
            improved = False
            for j in range(dim):
                if time.time() >= deadline:
                    break

                # Try positive direction
                x1 = best_x[:]
                x1[j] = clamp(x1[j] + step[j], lows[j], highs[j])
                f1 = evaluate(x1)

                if f1 <= best:
                    best, best_x = f1, x1
                    improved = True
                    continue

                # Try negative direction
                x2 = best_x[:]
                x2[j] = clamp(x2[j] - step[j], lows[j], highs[j])
                f2 = evaluate(x2)

                if f2 <= best:
                    best, best_x = f2, x2
                    improved = True

            if improved:
                no_improve = 0
                # Expand a bit after successful refinement
                for j in range(dim):
                    step[j] *= 1.05

        # 3) Random restart if stuck: sample globally, then continue local search.
        if no_improve >= restart_patience and time.time() < deadline:
            no_improve = 0
            # Keep best, but jump to a new region with a random candidate
            x0 = rand_vec()
            f0 = evaluate(x0)
            if f0 <= best:
                best, best_x = f0, x0
            else:
                # Even if not better, restart local exploration around a mixture
                # of best and random to diversify.
                mix = best_x[:]
                alpha = 0.5 + 0.5 * random.random()
                for j in range(dim):
                    mix[j] = clamp(alpha * best_x[j] + (1 - alpha) * x0[j], lows[j], highs[j])
                fm = evaluate(mix)
                if fm <= best:
                    best, best_x = fm, mix

            # Reset steps to moderately large to explore anew
            for j in range(dim):
                step[j] = 0.2 * spans[j] if spans[j] > 0 else 1.0
                if step[j] < min_step[j]:
                    step[j] = min_step[j]

    # return fitness of the best found solution
    return best
