import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like stratified initialization
      - (1+lambda) Evolution Strategy with self-adapting step size
      - Occasional local coordinate refinement
      - Lightweight random restarts if stagnating

    Returns: best (float) = best objective value found within max_time seconds
    """
    # -------- helpers --------
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects "array-like"; list is acceptable in the template example.
        return float(func(x))

    def lhs_init(n):
        # Simple stratified sampling per dimension (LHS-style without numpy)
        # Produces n points in [0,1]^dim then scales to bounds.
        strata = []
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            strata.append(perm)
        points = []
        for i in range(n):
            u = []
            for d in range(dim):
                # sample within stratum
                a = (strata[d][i] + random.random()) / n
                lo, hi = bounds[d]
                u.append(lo + a * (hi - lo))
            points.append(u)
        return points

    def mutate_gaussian(x, sigmas):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
                continue
            y[i] = clip(y[i] + random.gauss(0.0, sigmas[i]), lo, hi)
        return y

    def coordinate_refine(x, fx, base_steps):
        # A small coordinate search around x using +/- step per dim
        # Returns possibly improved (x, fx)
        improved = True
        curx = x[:]
        curf = fx
        steps = base_steps[:]
        # Limit work: a couple passes
        for _ in range(2):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if steps[i] <= 0:
                    continue
                lo, hi = bounds[i]
                xi = curx[i]
                # try +step
                xp = curx[:]
                xp[i] = clip(xi + steps[i], lo, hi)
                fp = eval_f(xp)
                if fp < curf:
                    curx, curf = xp, fp
                    improved = True
                    continue
                # try -step
                xm = curx[:]
                xm[i] = clip(xi - steps[i], lo, hi)
                fm = eval_f(xm)
                if fm < curf:
                    curx, curf = xm, fm
                    improved = True
            # shrink steps if no progress in a pass
            if not improved:
                for i in range(dim):
                    steps[i] *= 0.5
        return curx, curf

    # -------- main --------
    start = time.time()
    deadline = start + float(max_time)

    # Handle degenerate bounds
    widths = [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]
    # Initial per-dimension step sizes as a fraction of range
    init_sigmas = [w * 0.15 if w > 0 else 0.0 for w in widths]

    # Population size heuristic
    lam = max(8, 4 + int(3 * math.log(dim + 1)))
    # Initial design size
    n0 = max(lam, 12)

    best = float("inf")
    best_x = None

    # Initial sampling (LHS-like) + a few randoms
    init_pts = lhs_init(n0)
    for _ in range(4):
        init_pts.append(rand_uniform_vec())

    for x in init_pts:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    # Evolution strategy state (current parent)
    x = best_x[:] if best_x is not None else rand_uniform_vec()
    fx = best
    sigmas = init_sigmas[:]

    # Success-based step-size control parameters
    target_success = 0.2
    c_up = 1.25
    c_down = 0.82

    # Stagnation / restart control
    no_improve = 0
    restart_after = 60  # iterations without improvement (heuristic)
    refine_every = 25

    it = 0
    while time.time() < deadline:
        it += 1

        # Generate lambda offspring and pick best
        best_child = None
        best_fchild = float("inf")
        success = False

        for _ in range(lam):
            if time.time() >= deadline:
                return best
            child = mutate_gaussian(x, sigmas)
            fchild = eval_f(child)
            if fchild < best_fchild:
                best_fchild = fchild
                best_child = child
            if fchild < fx:
                success = True

        # (1+lambda) selection
        if best_fchild < fx:
            x, fx = best_child, best_fchild
            no_improve = 0
        else:
            no_improve += 1

        # Update global best
        if fx < best:
            best, best_x = fx, x[:]

        # Step-size adaptation (simple success rule on the batch)
        # If at least one offspring improved parent: expand a bit, else shrink.
        if success:
            for i in range(dim):
                sigmas[i] *= c_up
        else:
            for i in range(dim):
                sigmas[i] *= c_down

        # Keep sigmas within sensible limits
        for i in range(dim):
            w = widths[i]
            if w <= 0:
                sigmas[i] = 0.0
            else:
                # Lower bound avoids freezing; upper bound avoids jumping wildly
                sigmas[i] = clip(sigmas[i], w * 1e-6, w * 0.5)

        # Occasional coordinate refinement around current best
        if (it % refine_every) == 0 and time.time() < deadline and best_x is not None:
            base_steps = [max(widths[i] * 0.05, sigmas[i]) for i in range(dim)]
            rx, rf = coordinate_refine(best_x, best, base_steps)
            if rf < best:
                best, best_x = rf, rx[:]
                x, fx = best_x[:], best
                no_improve = 0

        # Restart if stagnating: sample a new promising point and reset sigmas
        if no_improve >= restart_after and time.time() < deadline:
            no_improve = 0
            # Biased restart: mix random point with best point to keep exploitation
            r = rand_uniform_vec()
            mix = random.random() * 0.7  # weight toward best
            x = [clip(mix * best_x[i] + (1.0 - mix) * r[i], bounds[i][0], bounds[i][1])
                 for i in range(dim)]
            fx = eval_f(x)
            if fx < best:
                best, best_x = fx, x[:]
            # reset sigmas moderately
            sigmas = [w * 0.15 if w > 0 else 0.0 for w in widths]

    return best
