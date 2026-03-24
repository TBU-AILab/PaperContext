import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a simple but strong approach:
    Differential Evolution (DE) + occasional restarts + boundary handling.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a sequence of length dim
    dim : int
        dimensionality
    bounds : list of (low, high)
        bounds per dimension
    max_time : int or float
        time budget in seconds

    Returns
    -------
    best : float
        fitness of best found solution
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def ensure_bounds(x):
        # "reflect" back into bounds (works better than hard clamp in practice)
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if y[i] < lo or y[i] > hi:
                if spans[i] <= 0:
                    y[i] = lo
                else:
                    # reflect repeatedly if far outside
                    while y[i] < lo or y[i] > hi:
                        if y[i] < lo:
                            y[i] = lo + (lo - y[i])
                        if y[i] > hi:
                            y[i] = hi - (y[i] - hi)
                    # numerical safety
                    if y[i] < lo: y[i] = lo
                    if y[i] > hi: y[i] = hi
        return y

    def safe_eval(x):
        # robust evaluation (avoid crashing on NaNs/errors)
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

    # --- Differential Evolution parameters (adaptive-ish) ---
    # Population size: moderate; scales with dimension.
    pop_size = max(12, min(60, 10 * dim))
    # Mutation factor and crossover rate ranges
    F_min, F_max = 0.45, 0.95
    CR_min, CR_max = 0.05, 0.95

    # --- initialize population ---
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best = fit[best_idx]
    best_x = pop[best_idx][:]

    # restart control
    last_improve_t = time.time()
    stall_restart_seconds = max(0.25, 0.25 * float(max_time))  # restart if no improve for this long

    gen = 0
    while True:
        if time.time() >= deadline:
            return best

        gen += 1

        # mild parameter drift over time: more exploration early, more exploitation later
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / max(1e-9, float(max_time)))
        # shrink F and CR slightly toward later iterations
        F_base = F_max - frac * (F_max - F_min)
        CR_base = CR_max - frac * (CR_max - CR_min)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose distinct indices a,b,c != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            # jitter parameters per-individual
            F = max(0.1, min(1.2, random.gauss(F_base, 0.1)))
            CR = max(0.0, min(1.0, random.gauss(CR_base, 0.1)))

            xa, xb, xc = pop[a], pop[b], pop[c]
            xi = pop[i]

            # DE/rand/1 mutation: v = xa + F*(xb - xc)
            v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
            v = ensure_bounds(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = []
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    u.append(v[d])
                else:
                    u.append(xi[d])

            u = ensure_bounds(u)
            fu = safe_eval(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_improve_t = time.time()

        # --- occasional local-ish refinement around best (cheap stochastic hillclimb) ---
        if time.time() < deadline and dim > 0 and (gen % 5 == 0):
            # step size decreases with time
            step_scale = (1.0 - frac) ** 2
            tries = 4
            for _ in range(tries):
                if time.time() >= deadline:
                    return best
                cand = best_x[:]
                # perturb a few coordinates
                k = 1 if dim == 1 else random.randint(1, max(1, dim // 3))
                for _j in range(k):
                    d = random.randrange(dim)
                    # gaussian step proportional to span
                    cand[d] += random.gauss(0.0, 0.15 * spans[d] * step_scale)
                cand = ensure_bounds(cand)
                fc = safe_eval(cand)
                if fc < best:
                    best = fc
                    best_x = cand[:]
                    last_improve_t = time.time()

        # --- restart if stalled: keep best, re-seed others ---
        if time.time() - last_improve_t > stall_restart_seconds:
            # keep a few elites including best
            elite_n = max(1, pop_size // 6)
            elite_idxs = sorted(range(pop_size), key=lambda k: fit[k])[:elite_n]
            elites = [pop[k][:] for k in elite_idxs]
            elites_fit = [fit[k] for k in elite_idxs]

            pop = elites[:]
            fit = elites_fit[:]

            # re-seed remainder with a mix of random and around-best
            while len(pop) < pop_size and time.time() < deadline:
                if random.random() < 0.6:
                    x = rand_vec()
                else:
                    # sample around best with broad noise
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.35 * spans[d])
                    x = ensure_bounds(x)
                pop.append(x)
                fit.append(safe_eval(x))

            best_idx = min(range(pop_size), key=lambda ii: fit[ii])
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = pop[best_idx][:]
            last_improve_t = time.time()
#
#
