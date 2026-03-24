import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded global optimization (minimization) using a hybrid of:
      - low-discrepancy sampling (Halton sequence) for broad coverage
      - local coordinate search (pattern/compass search) for refinement
      - step-size adaptation and occasional random restarts

    Returns:
        best (float): fitness of best found solution within max_time seconds
    """

    # ---------------------------
    # Helpers (no external libs)
    # ---------------------------
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def vec_clamp(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def halton_value(index, base):
        # Radical inverse in given base
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            for p in primes:
                if p * p > x:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(dim)

    def halton_vec(k):
        # Map Halton point in [0,1]^dim to bounds
        v = []
        for i in range(dim):
            u = halton_value(k, primes[i])
            lo, hi = bounds[i]
            v.append(lo + u * (hi - lo))
        return v

    def eval_f(x):
        # func expects an array-like; keep it as list to avoid dependencies
        try:
            y = func(x)
        except TypeError:
            # Some funcs might expect positional args; fall back
            y = func(*x)
        # Ensure float-like
        try:
            return float(y)
        except Exception:
            # If func returns something unexpected, treat as bad
            return float("inf")

    # ---------------------------
    # Initialization
    # ---------------------------
    t0 = time.time()
    deadline = t0 + max_time

    # Step size based on bounds span
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    max_span = max(spans) if spans else 1.0
    min_span = min(spans) if spans else 1.0

    # Start with a moderate step relative to scale
    step0 = 0.15 * max_span
    step_min = 1e-12 * (max_span if max_span > 0 else 1.0)
    step_max = 0.5 * max_span

    # Best-so-far
    best = float("inf")
    best_x = None

    # Use a mix of Halton samples + random samples initially
    # Keep it light; we refine via local search.
    halton_index = 1

    # A small evaluation budget per "local phase" (time-driven, not count-driven)
    # We'll just check time frequently.
    # Restart controls
    no_improve_restarts = 0
    max_no_improve_restarts = 50  # but time typically ends earlier

    # ---------------------------
    # Main loop: global sampling + local search
    # ---------------------------
    while time.time() < deadline:
        # --- Global proposal (choose between Halton and random) ---
        if random.random() < 0.7:
            x = halton_vec(halton_index)
            halton_index += 1
        else:
            x = rand_vec()

        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x[:]

        # --- Local refinement around current best (or around x if no best yet) ---
        if best_x is None:
            center = x
            f_center = fx
        else:
            center = best_x[:]
            f_center = best

        # Initialize local step; adapt based on problem scale
        step = clamp(step0, step_min, step_max)

        # Randomize dimension order for coordinate search
        dims = list(range(dim))
        random.shuffle(dims)

        improved_any = False

        # Pattern/compass search: reduce step until no progress or time ends
        while step >= step_min and time.time() < deadline:
            improved = False

            # Try coordinate directions
            for j in dims:
                if time.time() >= deadline:
                    break

                for sgn in (-1.0, 1.0):
                    cand = center[:]
                    cand[j] = cand[j] + sgn * step
                    # Clamp to bounds
                    lo, hi = bounds[j]
                    if cand[j] < lo:
                        cand[j] = lo
                    elif cand[j] > hi:
                        cand[j] = hi

                    # If clamping made no change, still may be worth evaluating once,
                    # but avoid duplicates if it doesn't move.
                    if cand[j] == center[j]:
                        continue

                    fc = eval_f(cand)
                    if fc < f_center:
                        center = cand
                        f_center = fc
                        improved = True
                        improved_any = True

                        # Update global best
                        if fc < best:
                            best = fc
                            best_x = cand[:]

                        # Small acceleration: keep moving in same direction (pattern move)
                        # as long as it improves and within bounds.
                        # Limit number of pattern steps to keep time predictable.
                        for _ in range(3):
                            if time.time() >= deadline:
                                break
                            cand2 = center[:]
                            cand2[j] = cand2[j] + sgn * step
                            lo, hi = bounds[j]
                            if cand2[j] < lo:
                                cand2[j] = lo
                            elif cand2[j] > hi:
                                cand2[j] = hi
                            if cand2[j] == center[j]:
                                break
                            fc2 = eval_f(cand2)
                            if fc2 < f_center:
                                center = cand2
                                f_center = fc2
                                if fc2 < best:
                                    best = fc2
                                    best_x = cand2[:]
                            else:
                                break

            if improved:
                # Mildly increase step to speed traversal when making progress
                step = min(step * 1.2, step_max)
                random.shuffle(dims)
            else:
                # No improvement: reduce step
                step *= 0.5

        # --- Restart logic: if local phase didn't help, perturb / restart ---
        if improved_any:
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
            # Occasionally jump near the best with shrinking noise, else random restart
            if best_x is not None and random.random() < 0.7:
                # Perturbation scale tied to spans and restart count
                scale = 0.05 / (1.0 + 0.1 * no_improve_restarts)
                x = [
                    clamp(best_x[i] + random.uniform(-1.0, 1.0) * spans[i] * scale,
                          bounds[i][0], bounds[i][1])
                    for i in range(dim)
                ]
                fx = eval_f(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
            else:
                x = rand_vec()
                fx = eval_f(x)
                if fx < best:
                    best = fx
                    best_x = x[:]

            if no_improve_restarts > max_no_improve_restarts:
                # Reset counter to avoid spending too long in stale mode
                no_improve_restarts = 0

    return best
