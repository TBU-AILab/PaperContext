import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like initial sampling (stratified per dimension)
      - Local refinement with an adaptive Gaussian step (1+lambda evolution strategy flavor)
      - Occasional random restarts to escape local minima

    Args:
        func: callable(params:list[float]) -> float
        dim: int
        bounds: list of (low, high) for each dimension
        max_time: seconds (int/float)

    Returns:
        best fitness (float) found within time budget
    """
    t0 = time.time()
    eps_time = 1e-4

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # Ensure list/sequence is passed (template says np.array, but keep it generic)
        return float(func(x))

    # --- Initial design: stratified sampling per dimension (LHS-like) ---
    # Choose number of initial samples based on dimension but keep it modest for speed.
    n_init = max(10, min(60, 10 * dim))
    # Create strata per dimension: permute bins so each bin used once per dimension.
    perms = []
    for d in range(dim):
        p = list(range(n_init))
        random.shuffle(p)
        perms.append(p)

    best_x = None
    best = float("inf")

    # Evaluate initial samples
    for k in range(n_init):
        if time.time() - t0 >= max_time - eps_time:
            return best
        x = []
        for d in range(dim):
            # pick within stratum [i/n, (i+1)/n)
            i = perms[d][k]
            u = (i + random.random()) / n_init
            x.append(lows[d] + u * spans[d])
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    # If somehow no evaluation happened (very tiny max_time), return best
    if best_x is None:
        return best

    # --- Local search with adaptive step size and restarts ---
    # Step sizes relative to bounds; start moderate.
    sigma = [0.15 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    # Evolution strategy parameters
    lam = max(8, 4 * dim)   # offspring per iteration
    success_target = 0.2    # desired success rate
    adapt = 1.5             # adaptation factor
    # Restart control
    no_improve_limit = 40   # iterations without improvement before restart
    no_improve = 0

    # Also keep a "center" which we refine (start from best_x)
    center = best_x[:]
    center_f = best

    while True:
        if time.time() - t0 >= max_time - eps_time:
            return best

        # Generate offspring around center
        successes = 0
        best_off_x = None
        best_off_f = float("inf")

        for _ in range(lam):
            if time.time() - t0 >= max_time - eps_time:
                return best

            child = []
            for d in range(dim):
                # Gaussian perturbation
                step = random.gauss(0.0, sigma[d])
                val = center[d] + step
                val = clamp(val, lows[d], highs[d])
                child.append(val)

            f = eval_f(child)

            # Track best overall
            if f < best:
                best, best_x = f, child[:]

            # Track best in this generation
            if f < best_off_f:
                best_off_f = f
                best_off_x = child

            if f < center_f:
                successes += 1

        # Accept the best offspring if it improves the center
        if best_off_f < center_f:
            center, center_f = best_off_x, best_off_f
            no_improve = 0
        else:
            no_improve += 1

        # Adapt sigma based on success rate (simple 1/5th success rule style)
        sr = successes / float(lam)
        if sr > success_target:
            # increase step sizes a bit
            for d in range(dim):
                sigma[d] *= adapt
        else:
            # decrease step sizes a bit
            for d in range(dim):
                sigma[d] /= adapt

        # Keep sigma within reasonable limits
        for d in range(dim):
            # lower bound avoids stagnation; upper bound avoids jumping wildly
            min_s = 1e-12 * (spans[d] if spans[d] > 0 else 1.0)
            max_s = 0.5 * (spans[d] if spans[d] > 0 else 1.0)
            if sigma[d] < min_s: sigma[d] = min_s
            if sigma[d] > max_s: sigma[d] = max_s

        # Random restart if stuck: jump to a new area, but keep best found
        if no_improve >= no_improve_limit:
            # 50% restart near best, 50% completely random
            if random.random() < 0.5:
                center = best_x[:]
                center_f = best
                # broaden a bit after restart
                for d in range(dim):
                    sigma[d] = max(sigma[d], 0.2 * spans[d] if spans[d] > 0 else 1.0)
            else:
                center = rand_point()
                center_f = eval_f(center)
                if center_f < best:
                    best, best_x = center_f, center[:]
                for d in range(dim):
                    sigma[d] = 0.2 * spans[d] if spans[d] > 0 else 1.0
            no_improve = 0
