import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer: Differential Evolution (DE/rand/1/bin)
    + occasional local coordinate refinement + soft restarts.

    Why this is typically stronger than the previous Halton+pattern search:
      - DE uses population-based recombination -> much better global search on
        rugged / rotated / non-separable landscapes.
      - Binomial crossover + mutation gives efficient exploration.
      - Lightweight local refinement exploits good candidates near the end.

    Returns:
        best (float): best fitness found within max_time seconds
    """

    # ---------------------------
    # Helpers (no external libs)
    # ---------------------------
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def eval_f(x):
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        try:
            y = float(y)
        except Exception:
            y = float("inf")
        # guard against NaN
        if y != y:
            return float("inf")
        return y

    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # Avoid zeros (degenerate bounds)
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 0.0

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def jitter_around(x, scale):
        # scale in [0,1] relative to span
        v = x[:]
        for i in range(dim):
            if spans[i] > 0:
                v[i] = clamp(v[i] + random.uniform(-1.0, 1.0) * spans[i] * scale,
                             bounds[i][0], bounds[i][1])
            else:
                v[i] = bounds[i][0]
        return v

    # ---------------------------
    # Time bookkeeping
    # ---------------------------
    t0 = time.time()
    deadline = t0 + max_time
    def time_left():
        return deadline - time.time()

    # ---------------------------
    # Differential Evolution setup
    # ---------------------------
    # Population size: small but effective; scale with dim
    # (kept modest to reduce evaluation cost)
    NP = max(12, min(60, 8 * dim))

    # Initialize population
    pop = [rand_vec() for _ in range(NP)]
    fit = [eval_f(x) for x in pop]

    best_idx = min(range(NP), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # Parameter ranges (self-adapted per individual like "jDE"-style sampling)
    # We'll resample F, CR frequently to avoid stagnation.
    F_lo, F_hi = 0.35, 0.95
    CR_lo, CR_hi = 0.05, 0.95

    # Restart / diversity control
    stall = 0
    last_best = best
    max_stall = 25  # generations without improvement before partial reinit

    # Local refinement controls (very lightweight)
    def local_refine(x, fx, budget_evals):
        # Coordinate-wise small pattern refinement with shrinking steps.
        # Uses relative steps per dimension, starting modest.
        center = x[:]
        f_center = fx

        # Initial step per dimension
        steps = []
        for i in range(dim):
            if spans[i] > 0:
                steps.append(0.08 * spans[i])
            else:
                steps.append(0.0)

        evals = 0
        # A few passes, shrinking on no improvement
        for _ in range(3):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if evals >= budget_evals or time.time() >= deadline:
                    return center, f_center, evals
                step = steps[j]
                if step <= 0:
                    continue
                base = center[j]
                # try +/- step
                for sgn in (-1.0, 1.0):
                    if evals >= budget_evals or time.time() >= deadline:
                        return center, f_center, evals
                    cand = center[:]
                    cand[j] = clamp(base + sgn * step, bounds[j][0], bounds[j][1])
                    if cand[j] == base:
                        continue
                    fc = eval_f(cand)
                    evals += 1
                    if fc < f_center:
                        center, f_center = cand, fc
                        improved = True
                        base = center[j]
            if improved:
                # slightly increase successful steps
                for j in range(dim):
                    steps[j] = min(steps[j] * 1.15, spans[j] * 0.25 if spans[j] > 0 else 0.0)
            else:
                # shrink steps
                for j in range(dim):
                    steps[j] *= 0.35
        return center, f_center, evals

    # ---------------------------
    # Main loop (generational DE)
    # ---------------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # Optional: bias a bit more exploitation near the end
        tl = time_left()
        near_end = (tl < 0.25 * max_time)

        # For each target vector
        for i in range(NP):
            if time.time() >= deadline:
                break

            # Random DE parameters (helps across problem types)
            F = random.uniform(F_lo, F_hi)
            CR = random.uniform(CR_lo, CR_hi)

            # Mutation: pick r1,r2,r3 distinct and != i
            # (classic DE/rand/1)
            idxs = list(range(NP))
            idxs.remove(i)
            r1, r2, r3 = random.sample(idxs, 3)

            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            target = pop[i]

            # Mutant
            mutant = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    mutant[d] = bounds[d][0]
                else:
                    mutant[d] = x1[d] + F * (x2[d] - x3[d])

            # Crossover (binomial)
            trial = target[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if dim == 0:
                    break
                if random.random() < CR or d == jrand:
                    trial[d] = mutant[d]

            # Bound handling: clamp + slight random reset if clamped too much
            # (keeps diversity and avoids edge-sticking)
            for d in range(dim):
                lo, hi = bounds[d]
                td = trial[d]
                if td < lo:
                    # bounce-in
                    trial[d] = lo + random.random() * (hi - lo) * 0.1 if hi > lo else lo
                elif td > hi:
                    trial[d] = hi - random.random() * (hi - lo) * 0.1 if hi > lo else hi

            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]

        # Stall / restart logic
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            # Partial restart: reinitialize worst portion, keep elites
            stall = 0
            # Keep top k
            order = sorted(range(NP), key=lambda k: fit[k])
            k_keep = max(3, NP // 5)
            keep = set(order[:k_keep])
            # Reinit the rest (some around best, some random)
            for idx in order[k_keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.65:
                    pop[idx] = jitter_around(best_x, scale=0.25)
                else:
                    pop[idx] = rand_vec()
                fit[idx] = eval_f(pop[idx])

            best_idx = min(range(NP), key=lambda ii: fit[ii])
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = pop[best_idx][:]

        # A small local refinement near the end or occasionally after improvement
        if dim > 0 and best_x is not None and (near_end or (gen % 10 == 0 and time_left() > 0.05)):
            # tiny budget to avoid stealing time from DE
            budget = 12 + 2 * dim
            x2, f2, _ = local_refine(best_x, best, budget)
            if f2 < best:
                best, best_x = f2, x2

    return best
