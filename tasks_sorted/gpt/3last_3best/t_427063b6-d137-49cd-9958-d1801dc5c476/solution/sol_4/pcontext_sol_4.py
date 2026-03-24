import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer (self-contained; no external libraries).

    Strategy:
      - Random initialization
      - Differential Evolution style mutation/crossover
      - Elitist selection
      - Occasional local random perturbation around current best
    Returns:
      - best (float): fitness of best found solution
    """
    start = time.time()

    # --- helpers ---
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(x):
        # func expects an array-like; we pass list
        try:
            v = func(x)
            # Ensure it's a float-like number
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return float('inf')
            return float(v)
        except Exception:
            return float('inf')

    # --- parameters (lightweight defaults) ---
    pop_size = max(8, min(30, 10 * dim))
    F = 0.7
    CR = 0.9
    p_local = 0.15  # probability of local perturbation step

    # --- init population ---
    pop = [rand_vec() for _ in range(pop_size)]
    # FIXED SYNTAX ERROR: closing bracket must be ']'
    fit = [safe_eval(x) for x in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best = fit[best_idx]
    best_x = pop[best_idx][:]

    # --- main loop ---
    while True:
        if time.time() - start >= max_time:
            return best

        for i in range(pop_size):
            if time.time() - start >= max_time:
                return best

            # choose 3 distinct indices different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation: v = a + F*(b-c)
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]

            # crossover (binomial)
            j_rand = random.randrange(dim)
            trial = []
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    val = v[j]
                else:
                    val = pop[i][j]
                trial.append(clip(val, bounds[j][0], bounds[j][1]))

            # occasional local step around best
            if random.random() < p_local:
                # small gaussian-like step using sum of uniforms (no external libs)
                local = []
                for j in range(dim):
                    lo, hi = bounds[j]
                    span = hi - lo
                    # approx normal(0,1): sum of 12 uniforms - 6
                    g = sum(random.random() for _ in range(12)) - 6.0
                    step = 0.05 * span * g
                    local.append(clip(best_x[j] + step, lo, hi))
                # pick better of trial and local
                f_trial = safe_eval(trial)
                f_local = safe_eval(local)
                if f_local < f_trial:
                    trial, f_trial = local, f_local
            else:
                f_trial = safe_eval(trial)

            # selection
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
