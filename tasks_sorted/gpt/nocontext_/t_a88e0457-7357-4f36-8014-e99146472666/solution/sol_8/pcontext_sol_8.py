import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a hybrid of:
      - Latin-hypercube-like diversified initialization
      - Adaptive local search (coordinate/axis steps)
      - Occasional random restarts
      - Shrinking step size when progress stalls

    Returns:
      best (float): best objective value found within max_time seconds
    """
    t0 = time.time()

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_point(x):
        # func expects an array-like; keep it plain Python list to avoid numpy dependency
        return float(func(x))

    # --- initialization: diversified sampling ---
    # build a "stratified" set per dimension then combine randomly (cheap LHS-like)
    init_n = max(10, 4 * dim)
    strata = []
    for i in range(dim):
        s = []
        for k in range(init_n):
            a = k / init_n
            b = (k + 1) / init_n
            u = a + (b - a) * random.random()
            s.append(lows[i] + u * spans[i])
        random.shuffle(s)
        strata.append(s)

    best = float("inf")
    best_x = None

    # evaluate initial candidates
    for k in range(init_n):
        if time.time() - t0 >= max_time:
            return best
        x = [strata[i][k] for i in range(dim)]
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, x

    # --- adaptive local search with restarts ---
    # step size starts as a fraction of range and shrinks on stagnation
    step = [0.15 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # control parameters
    patience = 12 + 4 * dim          # attempts before shrinking steps
    shrink = 0.5                     # shrink factor
    expand = 1.05                    # mild expansion when improving
    restart_prob = 0.10             # probability of random restart per outer loop
    no_improve = 0

    # current point starts at best
    x = best_x[:] if best_x is not None else rand_point()
    fx = best

    while time.time() - t0 < max_time:
        # occasional restart to escape local minima
        if random.random() < restart_prob and best_x is not None:
            x = rand_point()
            fx = eval_point(x)

        improved = False

        # try a randomized order of coordinates
        order = list(range(dim))
        random.shuffle(order)

        for i in order:
            if time.time() - t0 >= max_time:
                return best

            si = step[i]
            if si <= min_step[i]:
                continue

            # try +step and -step along coordinate i
            # also include a small random "dither" to reduce deterministic traps
            dither = (random.random() - 0.5) * 0.1 * si

            # candidate +
            xp = x[:]
            xp[i] = clamp(xp[i] + si + dither, i)
            fp = eval_point(xp)

            # candidate -
            xm = x[:]
            xm[i] = clamp(xm[i] - si + dither, i)
            fm = eval_point(xm)

            # pick best move if it improves current
            if fp < fx or fm < fx:
                if fp <= fm:
                    x, fx = xp, fp
                else:
                    x, fx = xm, fm
                improved = True

                # update global best
                if fx < best:
                    best, best_x = fx, x[:]

                # slight step increase on success (kept bounded)
                step[i] = min(step[i] * expand, spans[i] if spans[i] > 0 else step[i] * expand)

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        # shrink all steps when stuck
        if no_improve >= patience:
            for i in range(dim):
                step[i] *= shrink
            no_improve = 0

        # if steps are tiny across all dims, restart with fresh steps around best
        if all(step[i] <= min_step[i] for i in range(dim)):
            # reinitialize around the best point found so far (or random if none)
            base = best_x[:] if best_x is not None else rand_point()
            x = base[:]
            fx = eval_point(x)
            # reset steps
            step = [0.15 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]

    return best
