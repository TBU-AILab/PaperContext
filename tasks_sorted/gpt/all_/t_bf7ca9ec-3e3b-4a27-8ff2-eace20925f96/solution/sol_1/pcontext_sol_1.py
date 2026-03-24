import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Strategy:
      - Multi-start local search with:
          * coordinate/pattern search (fast exploitation, respects bounds)
          * occasional random/global proposals (exploration)
          * adaptive step size per run + restart on stagnation
      - Keeps the best fitness found (returns best fitness only, per template)

    Works well when evaluations are moderately expensive and gradients unavailable.
    Uses only Python standard library.
    """
    t0 = time.time()

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # avoid zero spans causing issues
    spans = [s if s > 0 else 1.0 for s in spans]

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; we provide list
        return func(x)

    # --- Global best ---
    best = float("inf")
    best_x = None

    # --- Initial "center" ---
    x = rand_point()
    fx = evaluate(x)
    best, best_x = fx, x[:]

    # Budget control heuristics
    # restart if no improvement for this many seconds
    restart_patience = max(0.08 * max_time, 0.6)
    last_best_time = time.time()

    # Step sizes: start relative to span, then shrink as we converge
    base_step = 0.25  # fraction of span
    min_step_frac = 1e-9
    max_step_frac = 0.5

    # coordinate ordering randomization interval
    permute_every = 8

    # exploration probabilities
    p_global_jump = 0.03     # random point jump
    p_heavy_jump = 0.05      # heavy-tailed perturbation around current best
    p_try_best = 0.06        # sometimes continue from global best

    # Make an initial step fraction based on dimension
    # Slightly smaller in higher dimensions
    step_frac = max(0.08, base_step / (1.0 + 0.15 * math.log(1 + dim)))

    # per-dimension step sizes
    steps = [step_frac * spans[i] for i in range(dim)]
    steps_min = [min_step_frac * spans[i] for i in range(dim)]

    # Pattern search state
    improve_streak = 0
    no_improve_iters = 0
    iters = 0

    while True:
        if time.time() - t0 >= max_time:
            return best

        iters += 1

        # --- Restarts / exploration ---
        if random.random() < p_global_jump:
            x = rand_point()
            fx = evaluate(x)
        elif random.random() < p_try_best and best_x is not None:
            # Resume from best point found so far (intensification)
            x = best_x[:]
            fx = best
        elif random.random() < p_heavy_jump:
            # Heavy-tailed perturbation from current point
            # (Cauchy-like via tan(pi*(u-0.5)))
            y = x[:]
            for i in range(dim):
                u = random.random()
                c = math.tan(math.pi * (u - 0.5))  # heavy tail
                y[i] += 0.15 * spans[i] * c
            clamp_inplace(y)
            fy = evaluate(y)
            if fy <= fx:
                x, fx = y, fy

        # Update global best
        if fx < best:
            best, best_x = fx, x[:]
            last_best_time = time.time()

        # Restart on stagnation (time-based)
        if time.time() - last_best_time >= restart_patience:
            x = rand_point()
            fx = evaluate(x)
            # reset step sizes moderately
            step_frac = max(0.06, base_step / (1.0 + 0.15 * math.log(1 + dim)))
            steps = [max(steps_min[i], min(max_step_frac * spans[i], step_frac * spans[i])) for i in range(dim)]
            improve_streak = 0
            no_improve_iters = 0
            last_best_time = time.time()
            # don't "continue" without doing local work

        # --- Local coordinate/pattern search with adaptive steps ---
        # randomize coordinate order sometimes
        coords = list(range(dim))
        if iters % permute_every == 0:
            random.shuffle(coords)

        improved_any = False

        for i in coords:
            if time.time() - t0 >= max_time:
                return best

            si = steps[i]
            if si < steps_min[i]:
                continue

            xi = x[i]

            # Try +step and -step (best-of-two)
            cand_best = None
            f_cand_best = None

            # + step
            xp = x[:]
            xp[i] = xi + si
            clamp_inplace(xp)
            fp = evaluate(xp)

            cand_best = xp
            f_cand_best = fp

            # - step
            xm = x[:]
            xm[i] = xi - si
            clamp_inplace(xm)
            fm = evaluate(xm)

            if fm < f_cand_best:
                cand_best = xm
                f_cand_best = fm

            # accept if improvement
            if f_cand_best <= fx:
                x, fx = cand_best, f_cand_best
                improved_any = True

                if fx < best:
                    best, best_x = fx, x[:]
                    last_best_time = time.time()

                # mild step increase on success in this coordinate
                steps[i] = min(max_step_frac * spans[i], steps[i] * 1.15)
            else:
                # shrink step on failure
                steps[i] = max(steps_min[i], steps[i] * 0.6)

        # Global step adaptation based on whether we improved in this sweep
        if improved_any:
            improve_streak += 1
            no_improve_iters = 0
            # if consistently improving, slightly expand all steps (up to cap)
            if improve_streak >= 3:
                for i in range(dim):
                    steps[i] = min(max_step_frac * spans[i], steps[i] * 1.05)
                improve_streak = 0
        else:
            no_improve_iters += 1
            improve_streak = 0
            # if repeatedly failing, shrink all steps
            if no_improve_iters >= 2:
                for i in range(dim):
                    steps[i] = max(steps_min[i], steps[i] * 0.75)
                no_improve_iters = 0
