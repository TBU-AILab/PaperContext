import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a compact hybrid:
      - Latin-hypercube-like diversified initial sampling
      - Best-so-far tracking
      - Adaptive coordinate/local search around the incumbent (pattern search)
      - Occasional global restarts with shrinking radius

    Returns:
      best (float): fitness of the best found solution
    """
    t0 = time.time()

    def time_left():
        return max_time - (time.time() - t0)

    # --- Helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_vec(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; we pass a list (works with most funcs)
        return float(func(x))

    # Robust epsilon based on scale
    span_max = max(spans) if dim > 0 else 1.0
    eps = 1e-12 * (1.0 + span_max)

    # --- Initial best ---
    best_x = rand_vec()
    best = eval_f(best_x)

    # If time is extremely small, return quickly
    if time_left() <= 0:
        return best

    # --- Diversified initial sampling (stratified per dimension) ---
    # Build K points where each dimension is stratified into K bins and permuted.
    # K chosen to be modest and time-safe.
    K = max(8, min(40, 4 * dim + 8))
    perms = []
    for j in range(dim):
        p = list(range(K))
        random.shuffle(p)
        perms.append(p)

    for i in range(K):
        if time_left() <= 0:
            return best
        x = []
        for j in range(dim):
            # sample within bin perms[j][i]
            bin_idx = perms[j][i]
            u = (bin_idx + random.random()) / K
            x.append(lows[j] + u * spans[j])
        f = eval_f(x)
        if f < best:
            best = f
            best_x = x

    # --- Main loop: adaptive local/pattern search with restarts ---
    # Initial step sizes as fractions of span
    base_step = [0.2 * s if s > 0 else 1.0 for s in spans]
    min_step = [1e-9 * (1.0 + s) for s in spans]

    step = base_step[:]  # mutable
    no_improve = 0
    iter_count = 0

    # Probability of a "global" perturbation
    p_global = 0.15

    while time_left() > 0:
        iter_count += 1

        # Occasionally perform a global move (restart-like) around best or random
        if random.random() < p_global:
            if random.random() < 0.7:
                # perturb around best with radius proportional to current step
                x0 = best_x[:]
                x = []
                for j in range(dim):
                    r = step[j]
                    # heavy-tailed-ish perturbation: sign * r * (u/(1-u))
                    u = random.random()
                    tail = (u / max(1e-12, 1.0 - u))
                    delta = (1.0 if random.random() < 0.5 else -1.0) * r * min(5.0, tail)
                    x.append(x0[j] + delta)
                x = clip_vec(x)
            else:
                x = rand_vec()
        else:
            # Local coordinate / pattern search around best
            x = best_x[:]

            # Randomize coordinate order to avoid bias
            coords = list(range(dim))
            random.shuffle(coords)

            improved = False
            for j in coords:
                if time_left() <= 0:
                    return best

                # Try plus step
                cand = x[:]
                cand[j] += step[j]
                cand = clip_vec(cand)
                f1 = eval_f(cand)
                if f1 + eps < best:
                    best = f1
                    best_x = cand
                    x = cand
                    improved = True
                    continue

                # Try minus step
                cand = x[:]
                cand[j] -= step[j]
                cand = clip_vec(cand)
                f2 = eval_f(cand)
                if f2 + eps < best:
                    best = f2
                    best_x = cand
                    x = cand
                    improved = True
                    continue

            if improved:
                no_improve = 0
            else:
                no_improve += 1

                # Shrink step sizes when stuck
                if no_improve % 3 == 0:
                    for j in range(dim):
                        step[j] *= 0.6
                        if step[j] < min_step[j]:
                            step[j] = min_step[j]

                # If we're very stuck, do a stronger restart and reset some step sizes
                if no_improve >= 15:
                    no_improve = 0
                    # reset steps partially
                    for j in range(dim):
                        step[j] = max(step[j], 0.05 * spans[j] if spans[j] > 0 else step[j])

                    # jump to a new region but keep best
                    x = rand_vec()
                    f = eval_f(x)
                    if f < best:
                        best = f
                        best_x = x

        # Optional gentle annealing of global probability over time
        # (slightly less global search later)
        if iter_count % 50 == 0:
            p_global = max(0.05, p_global * 0.98)

    return best
