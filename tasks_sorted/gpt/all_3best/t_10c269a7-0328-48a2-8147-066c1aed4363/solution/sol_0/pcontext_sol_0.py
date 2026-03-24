import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Minimization under a strict time budget using a self-contained strategy:
    - Latin-hypercube-like stratified initialization (no numpy)
    - Best-of-population selection
    - Adaptive local search (coordinate + random-direction steps)
    - Occasional restarts to escape local minima

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # Guard against degenerate bounds
    for i in range(dim):
        if spans[i] < 0:
            # swap if user passed reversed bounds
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = highs[i] - lows[i]
        if spans[i] == 0:
            spans[i] = 1.0  # avoid division by zero; variable effectively fixed

    def clamp(x, i):
        if x < lows[i]: return lows[i]
        if x > highs[i]: return highs[i]
        return x

    def eval_params(x):
        # func expects an array-like; list is acceptable per template example
        return float(func(x))

    def rand_uniform(i):
        return lows[i] + random.random() * (highs[i] - lows[i])

    def rand_point():
        return [rand_uniform(i) for i in range(dim)]

    def copy_vec(x):
        return [v for v in x]

    # ---------- initial population (stratified per-dimension) ----------
    # Size: moderate, depends on dim; keep small for speed and for heavy func calls
    pop_size = max(12, min(60, 8 * dim))
    # Create stratified samples per dimension and then shuffle each dimension list
    strata = []
    for i in range(dim):
        k = pop_size
        # positions in [0,1) stratified + jitter
        vals = [((j + random.random()) / k) for j in range(k)]
        random.shuffle(vals)
        # map to bounds
        strata.append([lows[i] + v * (highs[i] - lows[i]) for v in vals])

    population = []
    for j in range(pop_size):
        x = [strata[i][j] for i in range(dim)]
        population.append(x)

    # Always include a purely random point too (in case bounds are weird)
    population.append(rand_point())
    pop_size = len(population)

    best_x = None
    best = float("inf")

    # Evaluate initial population
    for x in population:
        if time.time() >= deadline:
            return best
        fx = eval_params(x)
        if fx < best:
            best = fx
            best_x = copy_vec(x)

    # ---------- adaptive local search with restarts ----------
    # Step sizes are proportional to spans; decay on stagnation
    base_step = [0.25 * spans[i] for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] != 0 else 1.0) for i in range(dim)]

    # track improvement to adapt steps
    no_improve = 0
    # restart schedule: after some stagnation, jump to a new promising region
    restart_after = max(40, 10 * dim)

    # For choosing parents: keep a small elite set
    elite_size = max(3, min(10, pop_size // 5))

    # Recompute elites periodically
    evaluated = [(eval_params(x), x) for x in population]
    evaluated.sort(key=lambda t: t[0])
    elites = [copy_vec(evaluated[i][1]) for i in range(min(elite_size, len(evaluated)))]

    # Local search state
    step = base_step[:]

    while time.time() < deadline:
        # Periodically refresh elites with a few new candidates
        if time.time() >= deadline:
            break

        # pick a parent (biased towards best)
        if elites and random.random() < 0.85:
            parent = elites[random.randrange(len(elites))]
        else:
            parent = best_x if best_x is not None else rand_point()

        x = copy_vec(parent)

        # With some probability, try a random-direction move; else coordinate move
        if random.random() < 0.5:
            # random direction
            # generate a random unit-ish direction
            direction = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            norm = math.sqrt(sum(d*d for d in direction)) or 1.0
            direction = [d / norm for d in direction]
            # random step scale
            scale = random.random()
            for i in range(dim):
                x[i] = clamp(x[i] + direction[i] * step[i] * scale, i)
        else:
            # coordinate-wise perturbation on a subset of dimensions
            # perturb about ~sqrt(dim) coords on average (but at least 1)
            k = max(1, int(math.sqrt(dim)))
            for _ in range(k):
                i = random.randrange(dim)
                delta = (2.0 * random.random() - 1.0) * step[i]
                x[i] = clamp(x[i] + delta, i)

        if time.time() >= deadline:
            break
        fx = eval_params(x)

        if fx < best:
            best = fx
            best_x = copy_vec(x)
            no_improve = 0

            # If we improved, slightly expand step (but cap to spans)
            for i in range(dim):
                step[i] = min(spans[i], step[i] * 1.05 + 1e-18)
        else:
            no_improve += 1
            # No improvement => shrink steps slowly
            if no_improve % (5 + dim // 2) == 0:
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.85)

        # Update elites occasionally by injecting best_x and some randoms
        if no_improve % max(15, 3 * dim) == 0:
            # Build a small candidate set: current elites + best + some random points
            cand = []
            for e in elites:
                cand.append((eval_params(e), e))
                if time.time() >= deadline:
                    return best
            if best_x is not None:
                cand.append((best, copy_vec(best_x)))

            # Add a few exploration points
            extra = max(2, elite_size)
            for _ in range(extra):
                if time.time() >= deadline:
                    return best
                r = rand_point()
                fr = eval_params(r)
                cand.append((fr, r))
                if fr < best:
                    best = fr
                    best_x = copy_vec(r)
                    no_improve = 0

            cand.sort(key=lambda t: t[0])
            elites = [copy_vec(cand[i][1]) for i in range(min(elite_size, len(cand)))]

        # Restart if stagnant: jump near best or to a new random region
        if no_improve >= restart_after:
            no_improve = 0
            # reset step sizes (a bit smaller than initial)
            step = [0.15 * spans[i] for i in range(dim)]

            if best_x is not None and random.random() < 0.7:
                # restart near best
                x = copy_vec(best_x)
                for i in range(dim):
                    # gaussian-like via sum of uniforms
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random()) / 6.0
                    g = (g - 0.5) * 2.0  # approx in [-1,1]
                    x[i] = clamp(x[i] + g * 0.25 * spans[i], i)
            else:
                x = rand_point()

            if time.time() >= deadline:
                break
            fx = eval_params(x)
            if fx < best:
                best = fx
                best_x = copy_vec(x)

    return best
