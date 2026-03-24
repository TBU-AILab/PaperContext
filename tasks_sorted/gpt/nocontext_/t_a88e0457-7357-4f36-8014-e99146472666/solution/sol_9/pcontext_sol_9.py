import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: Differential Evolution + occasional local refinement.
    - func: callable(list[float]) -> float
    - dim: int
    - bounds: list[(low, high)] length == dim
    - max_time: seconds (int/float)
    Returns: best (float) fitness found within time limit
    """

    start = time.time()
    deadline = start + float(max_time)

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        # Ensure float output; treat non-finite as very bad.
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # --- hyperparameters (adaptive where possible) ---
    # Keep population modest to work under tight time budgets.
    pop_size = max(8, min(30, 10 + 2 * dim))
    F_min, F_max = 0.4, 0.95
    CR_min, CR_max = 0.1, 0.95

    # Initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(ind) for ind in pop]
    best = min(fit) if fit else float("inf")
    best_idx = fit.index(best) if fit else 0
    best_x = pop[best_idx][:]

    # If max_time is extremely small, just return whatever we got
    if time.time() >= deadline:
        return best

    # --- main loop: Differential Evolution (DE/rand/1/bin) with occasional local search ---
    gen = 0
    no_improve = 0

    while time.time() < deadline:
        gen += 1

        # Simple schedule: more exploration early, more exploitation later
        t = (time.time() - start) / max(1e-12, (deadline - start))
        t = max(0.0, min(1.0, t))
        F = F_max - (F_max - F_min) * t
        CR = CR_min + (CR_max - CR_min) * (1.0 - t)

        improved_this_gen = False

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # pick r1,r2,r3 distinct and != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = random.sample(idxs, 3)

            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            target = pop[i]

            # mutation
            mutant = [x1[j] + F * (x2[j] - x3[j]) for j in range(dim)]
            mutant = clip_vec(mutant)

            # binomial crossover
            jrand = random.randrange(dim)
            trial = target[:]
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    trial[j] = mutant[j]

            # selection
            f_trial = safe_eval(trial)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
                    improved_this_gen = True

        if improved_this_gen:
            no_improve = 0
        else:
            no_improve += 1

        # --- occasional local refinement around current best (coordinate search) ---
        # Trigger when stagnating a bit, but keep it cheap.
        if no_improve >= 5 and time.time() < deadline:
            x = best_x[:]
            fx = best
            # step sizes start moderately, then shrink
            base = 0.05 * (1.0 - 0.7 * t)  # relative to span
            step = [max(1e-12, base * spans[j]) for j in range(dim)]

            # a few passes of coordinate tweaks
            for _ in range(2):
                if time.time() >= deadline:
                    return best
                for j in range(dim):
                    if time.time() >= deadline:
                        return best

                    # try +/- step in coordinate j
                    cand1 = x[:]
                    cand1[j] = min(highs[j], cand1[j] + step[j])
                    f1 = safe_eval(cand1)

                    cand2 = x[:]
                    cand2[j] = max(lows[j], cand2[j] - step[j])
                    f2 = safe_eval(cand2)

                    if f1 < fx or f2 < fx:
                        if f1 <= f2:
                            x, fx = cand1, f1
                        else:
                            x, fx = cand2, f2

                # shrink steps
                step = [s * 0.5 for s in step]

            if fx < best:
                best = fx
                best_x = x[:]
                improved_this_gen = True
                no_improve = 0
            else:
                # mild restart pressure: replace a few worst individuals with random ones
                # to escape local minima.
                worst_k = max(1, pop_size // 5)
                worst = sorted(range(pop_size), key=lambda k: fit[k], reverse=True)[:worst_k]
                for k in worst:
                    if time.time() >= deadline:
                        return best
                    pop[k] = rand_vec()
                    fit[k] = safe_eval(pop[k])
                    if fit[k] < best:
                        best = fit[k]
                        best_x = pop[k][:]

    return best
