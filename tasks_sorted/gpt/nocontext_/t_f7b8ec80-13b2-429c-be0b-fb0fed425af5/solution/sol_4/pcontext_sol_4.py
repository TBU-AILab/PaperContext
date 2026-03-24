import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Evolution Strategy with:
      - Latin-hypercube-like initialization (stratified per-dimension sampling)
      - (mu, lambda) evolution strategy
      - 1/5 success rule step-size adaptation
      - occasional random-restarts to escape stagnation

    Returns:
      best (float): fitness of the best found solution
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def gaussian():
        # Box-Muller transform: standard normal
        u1 = random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(max(1e-12, u1)))
        theta = 2.0 * math.pi * u2
        return r * math.cos(theta)

    def eval_func(x):
        # func expects "array-like"; we avoid external libs and pass list
        return float(func(x))

    # --- initialization (stratified per-dimension) ---
    # Keep modest sizes for speed and robustness.
    mu = max(4, min(20, 2 * dim))
    lam = max(10, min(60, 6 * dim))

    # stratified samples: for each dim, permute bins to reduce clustering
    # create mu candidates
    bins = list(range(mu))
    per_dim_perm = []
    for _ in range(dim):
        b = bins[:]
        random.shuffle(b)
        per_dim_perm.append(b)

    pop = []
    for k in range(mu):
        x = []
        for i in range(dim):
            # sample within the assigned stratum
            u = (per_dim_perm[i][k] + random.random()) / float(mu)
            x.append(lows[i] + u * spans[i])
        pop.append(x)

    best = float("inf")
    best_x = None

    fitnesses = []
    for x in pop:
        if time.time() >= deadline:
            return best
        f = eval_func(x)
        fitnesses.append(f)
        if f < best:
            best = f
            best_x = x[:]

    # --- strategy parameters ---
    # initial global step: 20% of average span (scaled down with dim)
    avg_span = sum(spans) / float(dim) if dim > 0 else 1.0
    sigma = 0.2 * avg_span / max(1.0, math.sqrt(dim))

    # success tracking for 1/5th rule
    succ = 0
    trials = 0

    # stagnation / restart control
    no_improve = 0
    restart_after = 30  # generations without improvement triggers restart

    gen = 0
    while time.time() < deadline:
        gen += 1

        # sort parents by fitness (lower is better)
        idx = list(range(len(pop)))
        idx.sort(key=lambda i: fitnesses[i])
        pop = [pop[i] for i in idx[:mu]]
        fitnesses = [fitnesses[i] for i in idx[:mu]]

        # update best
        if fitnesses[0] < best:
            best = fitnesses[0]
            best_x = pop[0][:]
            no_improve = 0
        else:
            no_improve += 1

        # restart on stagnation (keeps best)
        if no_improve >= restart_after and time.time() < deadline:
            no_improve = 0
            # reinit around best and some random points
            pop = []
            for k in range(mu):
                if k < max(1, mu // 3) and best_x is not None:
                    # sample near best
                    x = [best_x[i] + sigma * gaussian() for i in range(dim)]
                    pop.append(clip_vec(x))
                else:
                    pop.append(rand_vec())
            fitnesses = []
            for x in pop:
                if time.time() >= deadline:
                    return best
                f = eval_func(x)
                fitnesses.append(f)
                if f < best:
                    best = f
                    best_x = x[:]
            # slightly increase exploration after restart
            sigma = min(avg_span, sigma * 1.5)
            continue

        # produce offspring
        offspring = []
        off_fit = []
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # select parent (tournament of 2 for speed)
            a = random.randrange(mu)
            b = random.randrange(mu)
            parent = pop[a] if fitnesses[a] < fitnesses[b] else pop[b]

            # mutate
            x = [parent[i] + sigma * gaussian() for i in range(dim)]
            x = clip_vec(x)

            f = eval_func(x)
            offspring.append(x)
            off_fit.append(f)

            # success statistic vs current best parent
            trials += 1
            if f < fitnesses[0]:
                succ += 1

            if f < best:
                best = f
                best_x = x[:]
                no_improve = 0

        # select next parents from offspring (mu best)
        oidx = list(range(len(offspring)))
        oidx.sort(key=lambda i: off_fit[i])
        pop = [offspring[i] for i in oidx[:mu]]
        fitnesses = [off_fit[i] for i in oidx[:mu]]

        # 1/5 success rule adaptation every few generations
        if trials >= 5 * lam:
            rate = succ / float(trials)
            # If success > 1/5 increase sigma, else decrease
            if rate > 0.2:
                sigma *= 1.2
            else:
                sigma *= 0.82
            # keep sigma in reasonable bounds
            sigma = max(1e-12, min(avg_span, sigma))
            succ = 0
            trials = 0

    return best
