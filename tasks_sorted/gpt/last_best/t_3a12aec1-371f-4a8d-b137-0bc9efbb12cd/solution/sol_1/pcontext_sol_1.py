import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer:
      - Differential Evolution with mixed strategies:
          * current-to-best/1/bin (fast exploitation)
          * rand/1/bin (exploration fallback)
      - jDE-style self-adaptation of F and CR per-individual
      - lightweight local search around the best (stochastic coordinate / pattern search)
      - boundary handling by reflection (usually better than clamping)
      - occasional partial reinitialization when stagnating

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    def time_left():
        return time.time() < deadline

    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def reflect_into_bounds(x):
        # Reflect coordinates into [low, high] (can handle overshoot repeatedly)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            while x[i] < lo or x[i] > hi:
                if x[i] < lo:
                    x[i] = lo + (lo - x[i])
                if x[i] > hi:
                    x[i] = hi - (x[i] - hi)
        return x

    # --- population sizing: moderate, but not huge (time-bounded) ---
    pop_size = max(20, min(80, 10 * dim))  # capped to keep evaluations manageable

    # jDE parameters
    tau1 = 0.1   # probability to resample F
    tau2 = 0.1   # probability to resample CR
    Fl, Fu = 0.1, 0.9

    # Initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [float("inf")] * pop_size
    F_i = [0.5] * pop_size
    CR_i = [0.9] * pop_size

    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if not time_left():
            return best
        f = safe_eval(pop[i])
        fit[i] = f
        if f < best:
            best = f
            best_x = pop[i][:]

    # --- local search around best (cheap, derivative-free) ---
    def local_search(best_x, best_f, budget_evals):
        # Stochastic coordinate search with shrinking step
        if best_x is None:
            return best_x, best_f
        x = best_x[:]
        f = best_f
        # initial step relative to span
        step = [0.2 * spans[j] for j in range(dim)]
        for _ in range(budget_evals):
            if not time_left():
                break
            j = random.randrange(dim)
            if step[j] <= 1e-12 * (spans[j] + 1.0):
                continue
            # try +/- move
            improved = False
            for sgn in (1.0, -1.0):
                y = x[:]
                y[j] = y[j] + sgn * step[j] * (0.5 + random.random())
                reflect_into_bounds(y)
                fy = safe_eval(y)
                if fy < f:
                    x, f = y, fy
                    improved = True
                    break
            if not improved:
                # shrink that coordinate step
                step[j] *= 0.75
        return x, f

    stagnation = 0
    last_best = best

    gen = 0
    while time_left():
        gen += 1

        # Occasionally run a small local search on the current best
        # (kept small so it doesn't dominate budget)
        if gen % 7 == 0 and best_x is not None:
            # budget depends lightly on dimension
            ls_budget = max(4, min(20, 2 * dim))
            bx, bf = local_search(best_x, best, ls_budget)
            if bf < best:
                best = bf
                best_x = bx[:]

        # Identify best index in population (for current-to-best strategy)
        best_idx = min(range(pop_size), key=lambda k: fit[k])
        if fit[best_idx] < best:
            best = fit[best_idx]
            best_x = pop[best_idx][:]

        # Stagnation tracking
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1

        # Partial restart if stagnating
        if stagnation >= 25:
            # reinit a fraction of worst individuals
            k = max(2, pop_size // 5)
            worst = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]
            for idx in worst:
                if not time_left():
                    return best
                pop[idx] = rand_vec()
                fit[idx] = safe_eval(pop[idx])
                F_i[idx] = 0.5
                CR_i[idx] = 0.9
                if fit[idx] < best:
                    best = fit[idx]
                    best_x = pop[idx][:]
            stagnation = 0

        # One DE generation
        for i in range(pop_size):
            if not time_left():
                return best

            # jDE self-adaptation
            if random.random() < tau1:
                F_i[i] = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CR_i[i] = random.random()

            Fi = F_i[i]
            CR = CR_i[i]

            # pick distinct indices r1,r2,r3 != i
            # (avoid allocations)
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)
            r3 = i
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(pop_size)

            xi = pop[i]
            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            xbest = pop[best_idx]

            # Mix strategies: mostly exploit current-to-best, sometimes explore rand/1
            if random.random() < 0.7:
                # current-to-best/1: v = xi + Fi*(xbest-xi) + Fi*(x2-x3)
                donor = [xi[j] + Fi * (xbest[j] - xi[j]) + Fi * (x2[j] - x3[j]) for j in range(dim)]
            else:
                # rand/1: v = x1 + Fi*(x2-x3)
                donor = [x1[j] + Fi * (x2[j] - x3[j]) for j in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            trial = xi[:]  # start from target
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]

            reflect_into_bounds(trial)
            ftrial = safe_eval(trial)

            # selection
            if ftrial <= fit[i]:
                pop[i] = trial
                fit[i] = ftrial
                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

    return best
