import random
import time
import math

def run(func, dim, bounds, max_time):
    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def clip(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # -----------------------------
    # Phase 1: random sampling to seed elites
    # -----------------------------
    best = float("inf")
    best_x = None

    elite_k = max(6, 2 * dim)         # keep a small elite pool
    elites = []                        # list of (f, x)

    # time-aware init budget
    init_n = max(32, 10 * dim)
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = rand_vec()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x
        elites.append((fx, x))
    elites.sort(key=lambda t: t[0])
    elites = elites[:elite_k]

    # Estimate an eval time to set internal micro-budgets
    # (safe: if too small, we just do fewer steps)
    if time.time() >= deadline:
        return best

    # -----------------------------
    # Phase 2: hybrid search
    # - CMA-ish cross-entropy sampling from elites (diagonal covariance)
    # - periodic local pattern search refinement on the current best
    # -----------------------------
    # Per-dimension sigma derived from elite spread; fallback to wide exploration
    def compute_mu_sigma(elites_list):
        k = len(elites_list)
        mu = [0.0] * dim
        for _, x in elites_list:
            for i in range(dim):
                mu[i] += x[i]
        invk = 1.0 / k
        for i in range(dim):
            mu[i] *= invk

        # variance
        var = [0.0] * dim
        for _, x in elites_list:
            for i in range(dim):
                d = x[i] - mu[i]
                var[i] += d * d
        for i in range(dim):
            var[i] = var[i] * invk

        sigma = [math.sqrt(v) for v in var]
        # floor/ceiling to avoid collapse or too-wide steps
        for i in range(dim):
            sigma[i] = max(1e-9 * spans[i], min(0.5 * spans[i], sigma[i] if sigma[i] > 0 else 0.25 * spans[i]))
        return mu, sigma

    # Local search: coordinate pattern search with adaptive step
    def local_refine(x0, f0, time_budget):
        endt = min(deadline, time.time() + time_budget)
        x = x0[:]
        f = f0
        step = [0.15 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]
        # try greedy coordinate moves; shrink if stuck
        while time.time() < endt:
            improved = False
            idxs = list(range(dim))
            random.shuffle(idxs)
            for i in idxs:
                if time.time() >= endt:
                    break
                if step[i] <= min_step[i]:
                    continue
                xi = x[i]
                # try +/- step
                for sgn in (1.0, -1.0):
                    cand = x[:]
                    cand[i] = clip(xi + sgn * step[i], lows[i], highs[i])
                    if cand[i] == xi:
                        continue
                    fc = eval_f(cand)
                    if fc < f:
                        x, f = cand, fc
                        improved = True
                        break
            if improved:
                for i in range(dim):
                    step[i] = min(spans[i], step[i] * 1.08)
            else:
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.5)
                # if all steps tiny, stop
                if max(step) <= max(min_step) * 10.0:
                    break
        return x, f

    # Main loop parameters
    mu, sigma = compute_mu_sigma(elites)
    temperature = 1.0                 # controls exploration; anneal down slowly
    stall = 0
    last_best = best

    # determine cadence for local refinement: more often in low dimensions
    refine_every = max(25, 10 * dim)
    it = 0

    while time.time() < deadline:
        it += 1

        # Occasionally refine the current best locally (strong exploitation)
        if best_x is not None and (it % refine_every == 0):
            # keep it short so we don't miss global improvements
            budget = 0.06 * max_time
            bx, bf = local_refine(best_x, best, budget)
            if bf < best:
                best, best_x = bf, bx
                elites.append((bf, bx))
                elites.sort(key=lambda t: t[0])
                elites = elites[:elite_k]
                mu, sigma = compute_mu_sigma(elites)

        if time.time() >= deadline:
            break

        # Sample around a random elite (mixture model) + occasional pure random
        if random.random() < 0.12:
            x = rand_vec()
        else:
            base = elites[random.randrange(len(elites))][1]
            x = [0.0] * dim
            for i in range(dim):
                # gaussian step with annealing temperature
                z = random.gauss(0.0, 1.0)
                step = (0.35 * sigma[i] + 0.65 * abs(base[i] - mu[i]) + 1e-12 * spans[i])
                step *= max(0.15, temperature)
                x[i] = clip(base[i] + z * step, lows[i], highs[i])

        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

        # Maintain elites
        # accept into elite set if good enough, otherwise sometimes accept for diversity
        if len(elites) < elite_k or fx <= elites[-1][0] or random.random() < 0.03:
            elites.append((fx, x))
            elites.sort(key=lambda t: t[0])
            elites = elites[:elite_k]
            mu, sigma = compute_mu_sigma(elites)

        # Simple stall/anneal control
        if best < last_best - 1e-12 * (1.0 + abs(last_best)):
            last_best = best
            stall = 0
            temperature = max(0.35, temperature * 0.97)
        else:
            stall += 1
            # if stalled, reheat a bit and inject diversity by widening sigmas
            if stall > 40 + 5 * dim:
                stall = 0
                temperature = min(1.4, temperature * 1.15)
                # widen sigma to escape local minima
                for i in range(dim):
                    sigma[i] = min(0.6 * spans[i], sigma[i] * 1.35)

    return best
