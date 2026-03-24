import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a hybrid of:
      - Latin-hypercube-like randomized initialization (stratified per-dimension)
      - Local pattern search (coordinate search with adaptive step size)
      - Random restarts

    Arguments:
        func: callable(list[float] or similar) -> float
        dim: int
        bounds: list of (low, high) tuples, length == dim
        max_time: seconds (int/float)

    Returns:
        best: best (minimum) fitness value found within time limit
    """

    # ---- helpers ----
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def sample_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func in the prompt expects an "array-like"; we pass a plain list
        return float(func(x))

    # timekeeping
    t0 = time.time()
    deadline = t0 + float(max_time)

    # Basic sanity
    if dim <= 0:
        return float('inf')
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    # Range per dimension
    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # If any span is invalid, just evaluate the single feasible point
    for i in range(dim):
        if not (bounds[i][1] >= bounds[i][0]):
            raise ValueError("Each bound must be (low, high) with high >= low")

    # ---- Initialization: stratified sampling (LHS-style) ----
    # Choose number of initial points based on dimension and time budget (lightweight)
    # Keep it modest to avoid blowing the time budget on expensive functions.
    init_n = max(5, min(40, 10 + 2 * dim))

    # Build per-dimension strata permutations
    perms = []
    for d in range(dim):
        perm = list(range(init_n))
        random.shuffle(perm)
        perms.append(perm)

    best_x = None
    best = float('inf')

    # Evaluate initial points
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for d in range(dim):
            lo, hi = bounds[d]
            if span[d] == 0.0:
                x.append(lo)
                continue
            # sample within stratum
            a = perms[d][k]
            u = (a + random.random()) / init_n
            x.append(lo + u * span[d])
        f = eval_f(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        # Fallback
        x = sample_uniform()
        return eval_f(x)

    # ---- Local search: adaptive coordinate/pattern search with restarts ----
    # Initial step sizes as a fraction of span
    step = [0.25 * s if s > 0 else 0.0 for s in span]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # Control parameters
    shrink = 0.5         # step shrink factor when stuck
    expand = 1.2         # step expand factor after improvement
    restart_patience = max(10, 5 * dim)  # iterations without improvement before restart

    no_improve = 0

    # Main loop
    while time.time() < deadline:
        improved = False

        # Coordinate search around current best_x
        # Randomize coordinate order each iteration
        coords = list(range(dim))
        random.shuffle(coords)

        x0 = best_x
        f0 = best

        for d in coords:
            if time.time() >= deadline:
                return best

            if span[d] == 0.0 or step[d] == 0.0:
                continue

            lo, hi = bounds[d]
            sd = step[d]

            # Try positive and negative directions
            for direction in (1.0, -1.0):
                if time.time() >= deadline:
                    return best

                x = list(x0)
                x[d] = clamp(x[d] + direction * sd, lo, hi)

                # If clamping caused no move, skip
                if x[d] == x0[d]:
                    continue

                f = eval_f(x)
                if f < best:
                    best = f
                    best_x = x
                    improved = True
                    x0 = best_x
                    f0 = best
                    # Slightly expand this dimension's step after success
                    step[d] = min(step[d] * expand, span[d])
                    break  # move to next coordinate (greedy)
            # end direction loop
        # end coordinate loop

        if improved:
            no_improve = 0
            continue

        no_improve += 1

        # If no improvement, shrink steps
        for d in range(dim):
            if span[d] > 0:
                step[d] *= shrink
                if step[d] < min_step[d]:
                    step[d] = min_step[d]

        # Restart if stuck for too long or if all steps are tiny
        tiny = True
        for d in range(dim):
            if span[d] > 0 and step[d] > (1e-6 * span[d]):
                tiny = False
                break

        if no_improve >= restart_patience or tiny:
            if time.time() >= deadline:
                return best

            # Random restart near current best, with probability; otherwise global random
            if random.random() < 0.7:
                # local perturbation around best
                x = list(best_x)
                for d in range(dim):
                    lo, hi = bounds[d]
                    if span[d] == 0.0:
                        x[d] = lo
                    else:
                        # perturb scale based on span
                        scale = 0.1 * span[d]
                        x[d] = clamp(x[d] + random.uniform(-scale, scale), lo, hi)
            else:
                x = sample_uniform()

            f = eval_f(x)
            if f < best:
                best = f
                best_x = x

            # Reset steps for exploration after restart
            step = [0.25 * s if s > 0 else 0.0 for s in span]
            no_improve = 0

    return best
