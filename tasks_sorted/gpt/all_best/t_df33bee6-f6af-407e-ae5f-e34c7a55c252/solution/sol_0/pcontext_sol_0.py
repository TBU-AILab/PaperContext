import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - global exploration with Differential Evolution style mutations
    - local refinement with coordinate-wise pattern search
    - lightweight restarts if progress stalls

    Returns:
        best (float): best (minimum) fitness value found within max_time seconds.
    """
    # ---------- helpers ----------
    def clip_vec(x):
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        lo, hi = bounds[i]
        return hi - lo

    # Safe evaluation wrapper (keeps algorithm running even if func errors)
    def eval_f(x):
        try:
            v = func(x)
            # Guard against NaN / non-finite
            if v is None:
                return float("inf")
            v = float(v)
            if not math.isfinite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ---------- initialization ----------
    start = time.time()
    deadline = start + float(max_time)

    # Population size (small enough to be fast; large enough for diversity)
    pop_size = max(8, min(40, 10 + 2 * dim))
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [eval_f(x) for x in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # Differential Evolution parameters (adapted mildly over time)
    F = 0.7     # mutation scale
    CR = 0.9    # crossover probability

    # Local search step sizes (relative to bounds)
    step = [0.15 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]
    min_step = [1e-12 * (span(i) if span(i) > 0 else 1.0) + 1e-15 for i in range(dim)]

    # Stall control / restarts
    last_improve_t = start
    stall_seconds = max(0.25, 0.15 * max_time)  # if no improvement for this long -> diversify
    evals = 0

    # ---------- main loop ----------
    while time.time() < deadline:
        # --- global search: DE-like generation ---
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # Choose distinct indices a,b,c != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # Mutant: xa + F*(xb-xc)
            mutant = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]
            clip_vec(mutant)

            # Binomial crossover
            trial = pop[i][:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    trial[j] = mutant[j]

            # Evaluate and select
            f_trial = eval_f(trial); evals += 1
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
                    last_improve_t = time.time()

        # Mild parameter adaptation (keeps search dynamic, still deterministic-free)
        # Decrease F slowly for more local exploitation; keep CR high-ish
        elapsed = time.time() - start
        if max_time > 0:
            t = min(1.0, elapsed / max_time)
            F = 0.85 - 0.45 * t   # from ~0.85 to ~0.40
            CR = 0.95 - 0.20 * t  # from ~0.95 to ~0.75

        # --- local refinement around current best: coordinate pattern search ---
        # A few quick coordinate tweaks each outer iteration
        for _ in range(2):  # small fixed budget
            if time.time() >= deadline:
                return best

            improved = False
            x = best_x[:]
            for d in range(dim):
                if time.time() >= deadline:
                    return best

                sd = step[d]
                if sd <= min_step[d]:
                    continue

                # Try +sd then -sd
                for direction in (+1.0, -1.0):
                    cand = x[:]
                    cand[d] += direction * sd
                    clip_vec(cand)
                    f_c = eval_f(cand); evals += 1
                    if f_c < best:
                        best = f_c
                        best_x = cand
                        x = cand
                        improved = True
                        last_improve_t = time.time()
                        break

            # If no coordinate improved, shrink steps (like pattern search)
            if not improved:
                step = [s * 0.5 for s in step]
            else:
                # If improved, slightly expand steps to move faster along descent
                step = [min(0.25 * span(i) if span(i) > 0 else s, s * 1.2) for i, s in enumerate(step)]

        # --- stall handling / restart ---
        if time.time() - last_improve_t > stall_seconds:
            # diversify: re-seed a fraction of the population around best + random points
            k = max(2, pop_size // 3)
            for _ in range(k):
                if time.time() >= deadline:
                    return best
                idx = random.randrange(pop_size)
                if idx == best_idx:
                    continue

                # Mix: half near-best gaussian-ish, half purely random
                if random.random() < 0.6:
                    cand = best_x[:]
                    for j in range(dim):
                        r = span(j)
                        # bounded "gaussian-like" via sum of uniforms (CLT) without numpy
                        g = (random.random() + random.random() + random.random() + random.random() - 2.0)
                        cand[j] += g * 0.10 * r
                    clip_vec(cand)
                else:
                    cand = rand_vec()

                f_c = eval_f(cand); evals += 1
                pop[idx] = cand
                fit[idx] = f_c
                if f_c < best:
                    best = f_c
                    best_x = cand[:]
                    last_improve_t = time.time()

            # Reset steps a bit after a restart to allow movement again
            step = [max(s, 0.08 * span(i) if span(i) > 0 else s) for i, s in enumerate(step)]
            last_improve_t = time.time()  # prevent repeated immediate restarts

        # Update best_idx
        best_idx = min(range(pop_size), key=lambda ii: fit[ii])

    return best
