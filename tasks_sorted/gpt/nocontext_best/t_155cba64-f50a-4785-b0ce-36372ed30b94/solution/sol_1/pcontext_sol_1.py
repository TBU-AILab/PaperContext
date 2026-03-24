import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (no external libs).
    Hybrid:
      1) scrambled low-discrepancy init (Halton) + a few random points
      2) (mu,lambda)-ES with self-adaptive step size (log-normal)
      3) occasional local coordinate refinement around the incumbent
      4) restarts with shrinking global sigma if stuck

    Calls func(list_of_floats) -> float
    Returns best (minimum) fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ----------------- helpers -----------------
    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def eval_x(x):
        # ensure bounds
        xx = [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
        v = func(xx)
        return float(v), xx

    # Halton sequence (bases = first primes)
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def halton_index(i, base):
        # i >= 1
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(max(1, dim))

    def halton_point(k):
        # k >= 1
        x = []
        for j in range(dim):
            u = halton_index(k, primes[j])
            lo, hi = bounds[j]
            x.append(lo + (hi - lo) * u)
        return x

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # robust scale for bounds
    ranges = [(bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 1.0 for i in range(dim)]
    avg_range = sum(ranges) / float(dim) if dim > 0 else 1.0

    # ----------------- initialization -----------------
    best = float("inf")
    best_x = None

    # Number of init points: small but useful; time-bounded anyway
    n_init = max(16, min(120, 12 * dim))
    # scramble by skipping some indices
    halton_skip = 7

    k = 1 + halton_skip
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        x = halton_point(k)
        k += 1
        v, xx = eval_x(x)
        if v < best:
            best, best_x = v, xx

    # add a few purely random points (helps if Halton aligns poorly)
    for _ in range(max(4, dim // 2)):
        if time.time() >= deadline:
            return best
        v, xx = eval_x(rand_point())
        if v < best:
            best, best_x = v, xx

    if best_x is None:
        v, best_x = eval_x(rand_point())
        best = v

    # ----------------- ES parameters -----------------
    # population sizes
    lam = max(12, 6 * dim)          # offspring
    mu = max(4, lam // 4)           # parents used for recombination

    # recombination weights (log)
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    # self-adaptation params
    # classical choices:
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(dim + 1.0))
    tau0 = 1.0 / math.sqrt(2.0 * (dim + 1.0))

    # start sigma as fraction of scale
    sigma = 0.25 * avg_range
    sigma_min = 1e-12 * (avg_range if avg_range > 0 else 1.0)

    # parents (start from best and small cloud)
    parents = [best_x[:]]
    parent_vals = [best]

    # generate a few additional parents around best
    for _ in range(mu - 1):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for j in range(dim):
            amp = 0.05 * ranges[j]
            x[j] = clip(x[j] + random.uniform(-amp, amp), bounds[j][0], bounds[j][1])
        v, xx = eval_x(x)
        parents.append(xx)
        parent_vals.append(v)
        if v < best:
            best, best_x = v, xx

    # sort parents
    idx = sorted(range(len(parents)), key=lambda i: parent_vals[i])
    parents = [parents[i] for i in idx]
    parent_vals = [parent_vals[i] for i in idx]
    parents = parents[:mu]
    parent_vals = parent_vals[:mu]

    # local coordinate refinement around incumbent (cheap hill-climb)
    def local_refine(x0, v0, budget_evals=40):
        x = x0[:]
        bestv = v0
        steps = [0.1 * r for r in ranges]
        min_steps = [max(1e-12, 1e-9 * (r if r > 0 else 1.0)) for r in ranges]
        grow = 1.25
        shrink = 0.5

        for _ in range(budget_evals):
            if time.time() >= deadline:
                break
            j = random.randrange(dim)
            if steps[j] < min_steps[j]:
                continue
            improved = False
            s = steps[j]
            for d in (1.0, -1.0):
                if time.time() >= deadline:
                    break
                trial = x[:]
                trial[j] = clip(trial[j] + d * s, bounds[j][0], bounds[j][1])
                vv, tt = eval_x(trial)
                if vv < bestv:
                    x, bestv = tt, vv
                    steps[j] = s * grow
                    improved = True
                    break
            if not improved:
                steps[j] = s * shrink
        return bestv, x

    # ----------------- main loop -----------------
    stuck = 0
    last_best = best
    gen = 0

    while time.time() < deadline:
        gen += 1

        # Recombine parents to mean (weighted)
        mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            pi = parents[i]
            for j in range(dim):
                mean[j] += w * pi[j]

        # produce offspring
        offspring = []
        offspring_vals = []

        # global noise terms for sigma adaptation
        global_n = random.gauss(0.0, 1.0)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # log-normal sigma' (per-individual)
            # include per-dim noise for richer adaptation even without covariance matrix
            local_n = [random.gauss(0.0, 1.0) for _ in range(dim)]
            sig_i = sigma * math.exp(tau0 * global_n + tau * (sum(local_n) / math.sqrt(dim)))
            if sig_i < sigma_min:
                sig_i = sigma_min

            child = mean[:]
            for j in range(dim):
                child[j] = clip(child[j] + sig_i * local_n[j], bounds[j][0], bounds[j][1])

            v, cc = eval_x(child)
            offspring.append((v, cc, sig_i))
            if v < best:
                best, best_x = v, cc

        # select next parents from offspring (plus elitism of current best parent)
        offspring.sort(key=lambda t: t[0])
        parents = [offspring[i][1] for i in range(mu)]
        parent_vals = [offspring[i][0] for i in range(mu)]

        # update sigma towards successful offspring (median of top mu sigmas)
        top_sigmas = sorted(offspring[i][2] for i in range(mu))
        sigma = top_sigmas[mu // 2]

        # occasional local refine when improvement slows
        if gen % 8 == 0 and time.time() < deadline:
            v0, x0 = best, best_x
            v1, x1 = local_refine(x0, v0, budget_evals=30 + 2 * dim)
            if v1 < best:
                best, best_x = v1, x1
                parents[0] = best_x[:]
                parent_vals[0] = best

        # stagnation detection and restart
        if best < last_best - 1e-12 * (1.0 + abs(last_best)):
            last_best = best
            stuck = 0
        else:
            stuck += 1

        if stuck >= 12:
            # restart around best with reduced sigma and some random immigrants
            stuck = 0
            sigma = max(sigma * 0.4, 0.02 * avg_range, sigma_min)

            # rebuild parents: best + random immigrants + near-best perturbations
            new_parents = [best_x[:]]
            new_vals = [best]

            # near-best perturbations
            for _ in range(max(1, mu // 2)):
                if time.time() >= deadline:
                    break
                x = best_x[:]
                for j in range(dim):
                    amp = 0.15 * ranges[j]
                    x[j] = clip(x[j] + random.gauss(0.0, 1.0) * amp, bounds[j][0], bounds[j][1])
                v, xx = eval_x(x)
                new_parents.append(xx); new_vals.append(v)
                if v < best:
                    best, best_x = v, xx

            # random immigrants
            while len(new_parents) < mu and time.time() < deadline:
                v, xx = eval_x(rand_point())
                new_parents.append(xx); new_vals.append(v)
                if v < best:
                    best, best_x = v, xx

            idx = sorted(range(len(new_parents)), key=lambda i: new_vals[i])
            parents = [new_parents[i] for i in idx[:mu]]
            parent_vals = [new_vals[i] for i in idx[:mu]]

    return best
