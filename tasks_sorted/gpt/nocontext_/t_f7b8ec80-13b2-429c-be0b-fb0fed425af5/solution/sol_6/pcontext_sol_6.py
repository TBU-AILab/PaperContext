import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like stratified initialization
      - (mu+lambda) evolution strategy with self-adapting step sizes
      - Bound handling by reflection (keeps search continuous near edges)

    Returns: best (float) = best objective value found.
    """
    t0 = time.time()

    # ----------------------- helpers -----------------------
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # avoid degenerate bounds
    spans = [s if s != 0.0 else 1.0 for s in spans]

    def time_left():
        return max_time - (time.time() - t0)

    def reflect_into_bounds(x):
        # Reflect coordinate-wise into [low, high]
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflection loop (fast in practice; step sizes are limited)
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                if v > hi:
                    v = hi - (v - hi)
            y[i] = v
        return y

    def rand_uniform():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like
        return float(func(x))

    # ------------------- initialization --------------------
    # Population size: small but scales with dimension
    mu = max(5, 4 + int(2 * math.sqrt(dim)))
    lam = max(10, 4 * mu)

    # Initial global step size proportional to average span
    avg_span = sum(spans) / float(dim)
    sigma0 = 0.2 * avg_span

    # Latin-hypercube-like stratified samples for a better first coverage
    # Create per-dimension permutations of bins
    bins = list(range(mu))
    perms = []
    for _ in range(dim):
        p = bins[:]
        random.shuffle(p)
        perms.append(p)

    population = []
    best = float("inf")

    # Each individual has (x, sigma_vector, fitness)
    for k in range(mu):
        x = []
        for i in range(dim):
            # sample from bin perms[i][k]
            b = perms[i][k]
            u = (b + random.random()) / mu  # in (0,1)
            x.append(lows[i] + u * spans[i])
        sig = [sigma0 for _ in range(dim)]
        f = evaluate(x)
        if f < best:
            best = f
        population.append([x, sig, f])

    # Track best point for gentle restarts
    best_x = min(population, key=lambda ind: ind[2])[0][:]

    # Self-adaptation parameters (log-normal mutation of sigmas)
    # Standard ES settings
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(dim))
    tau0 = 1.0 / math.sqrt(2.0 * dim)

    # To avoid too small / too large steps
    sigma_min = 1e-12
    sigma_max = 0.5 * avg_span if avg_span > 0 else 1.0

    # Stagnation control
    no_improve = 0
    last_best = best
    stagnation_limit = 30  # generations

    # --------------------- main loop -----------------------
    gen = 0
    while time_left() > 0:
        gen += 1

        # Sort population by fitness (ascending)
        population.sort(key=lambda ind: ind[2])
        if population[0][2] < best:
            best = population[0][2]
            best_x = population[0][0][:]
        # stagnation detection
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # Recombination: use mean of top parents
        parents = population[:mu]
        mean_x = [0.0] * dim
        mean_sig = [0.0] * dim
        for ind in parents:
            x, sig, _ = ind
            for i in range(dim):
                mean_x[i] += x[i]
                mean_sig[i] += sig[i]
        inv_mu = 1.0 / mu
        for i in range(dim):
            mean_x[i] *= inv_mu
            mean_sig[i] *= inv_mu

        # Generate offspring
        offspring = []
        for _ in range(lam):
            if time_left() <= 0:
                return best

            # occasional "directed" offspring near best to exploit
            if random.random() < 0.15:
                base_x = best_x
                base_sig = mean_sig
            else:
                base_x = mean_x
                base_sig = mean_sig

            # mutate sigmas (log-normal)
            common = random.gauss(0.0, 1.0)
            sig_new = [0.0] * dim
            for i in range(dim):
                s = base_sig[i] * math.exp(tau0 * common + tau * random.gauss(0.0, 1.0))
                if s < sigma_min:
                    s = sigma_min
                elif s > sigma_max:
                    s = sigma_max
                sig_new[i] = s

            # mutate parameters
            x_new = [0.0] * dim
            for i in range(dim):
                x_new[i] = base_x[i] + sig_new[i] * random.gauss(0.0, 1.0)

            x_new = reflect_into_bounds(x_new)
            f_new = evaluate(x_new)

            if f_new < best:
                best = f_new
                best_x = x_new[:]

            offspring.append([x_new, sig_new, f_new])

        # (mu+lambda) selection
        population = (population + offspring)
        population.sort(key=lambda ind: ind[2])
        population = population[:mu]

        # Soft restart if stagnating: inject diversity but keep elite
        if no_improve >= stagnation_limit:
            no_improve = 0
            # keep best 1-2
            elites = population[:2]
            # re-seed rest randomly around best and globally
            new_pop = elites[:]
            while len(new_pop) < mu:
                if random.random() < 0.6:
                    # local reseed around best
                    x = [best_x[i] + 0.3 * avg_span * random.gauss(0.0, 1.0) for i in range(dim)]
                    x = reflect_into_bounds(x)
                else:
                    x = rand_uniform()
                sig = [sigma0 for _ in range(dim)]
                f = evaluate(x)
                if f < best:
                    best = f
                    best_x = x[:]
                new_pop.append([x, sig, f])
            population = new_pop

    return best
