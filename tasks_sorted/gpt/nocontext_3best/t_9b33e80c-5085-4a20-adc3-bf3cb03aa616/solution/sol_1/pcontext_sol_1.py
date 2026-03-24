import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Hybrid strategy:
      1) Low-discrepancy-ish exploration via a simple Halton sequence (better coverage than pure random/LHS with same N)
      2) Multi-start local refinement using an adaptive pattern search (coordinate search with per-dimension steps)
      3) Occasional short "tunnel" moves (heavy-tailed perturbations) to escape local minima
      4) Time-aware scheduling (keeps exploring until late, then refines best)

    Returns:
        best (float): best objective value found within time limit
    """

    # --------------------- helpers ---------------------
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def eval_f(x):
        return float(func(x))

    def spans_from_bounds():
        sp = []
        for lo, hi in bounds:
            if hi < lo:
                raise ValueError("Each bound must be (low, high) with high >= low")
            sp.append(hi - lo)
        return sp

    # first primes for Halton
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton_value(index, base):
        # index >= 1
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, bases):
        # k >= 1
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            if spans[d] == 0.0:
                x[d] = lo
            else:
                u = halton_value(k, bases[d])
                x[d] = lo + u * spans[d]
        return x

    def uniform_point():
        x = []
        for d in range(dim):
            lo, hi = bounds[d]
            if spans[d] == 0.0:
                x.append(lo)
            else:
                x.append(random.uniform(lo, hi))
        return x

    def cauchy_step(scale):
        # heavy-tailed step without external libs; tan(pi*(u-0.5)) is Cauchy
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # Local pattern search around x, starting with given step vector.
    def local_refine(x, fx, step, deadline, max_iters):
        best_x = list(x)
        best_f = fx

        # parameters tuned for robustness
        shrink = 0.5
        expand = 1.25
        stall_limit = max(8, 3 * dim)

        stall = 0
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            improved = False

            # randomized coordinate order
            coords = list(range(dim))
            random.shuffle(coords)

            for d in coords:
                if time.time() >= deadline:
                    return best_x, best_f, step
                if spans[d] == 0.0 or step[d] <= 0.0:
                    continue

                lo, hi = bounds[d]
                sd = step[d]

                # try best direction first using a small look
                # (sometimes avoids evaluating both directions)
                # pick direction based on a random coin to avoid bias
                dirs = (1.0, -1.0) if random.random() < 0.5 else (-1.0, 1.0)

                for direction in dirs:
                    if time.time() >= deadline:
                        return best_x, best_f, step
                    cand = list(best_x)
                    cand[d] = clamp(cand[d] + direction * sd, lo, hi)
                    if cand[d] == best_x[d]:
                        continue
                    f = eval_f(cand)
                    if f < best_f:
                        best_x = cand
                        best_f = f
                        improved = True
                        # expand step for this coordinate after success
                        step[d] = min(step[d] * expand, spans[d])
                        break

            if improved:
                stall = 0
                continue

            stall += 1

            # shrink all steps if no improvement
            for d in range(dim):
                if spans[d] > 0.0:
                    step[d] *= shrink
                    # keep a minimal meaningful step relative to span
                    min_sd = 1e-12 * (spans[d] if spans[d] > 0 else 1.0)
                    if step[d] < min_sd:
                        step[d] = min_sd

            if stall >= stall_limit:
                break

        return best_x, best_f, step

    # --------------------- main ---------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    spans = spans_from_bounds()

    # If everything is fixed, evaluate once.
    if all(s == 0.0 for s in spans):
        x = [bounds[d][0] for d in range(dim)]
        return eval_f(x)

    # Seed randomness (deterministic seed not required; time-based is fine)
    random.seed()

    bases = first_primes(dim)

    best = float("inf")
    best_x = None

    # Budget-aware initial sampling:
    # Use Halton for coverage + a few uniform samples for randomness.
    # Keep it modest (func may be expensive).
    init_halton = max(16, min(120, 20 + 6 * dim))
    init_uniform = max(4, min(40, 6 + 2 * dim))

    k = 1
    while k <= init_halton and time.time() < deadline:
        x = halton_point(k, bases)
        f = eval_f(x)
        if f < best:
            best, best_x = f, x
        k += 1

    for _ in range(init_uniform):
        if time.time() >= deadline:
            return best
        x = uniform_point()
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    if best_x is None:
        x = uniform_point()
        return eval_f(x)

    # Main loop: alternate exploration and refinement.
    # Exploration proposes candidates (Halton + tunnel moves), refinement does local search.
    # As we approach the deadline, reduce exploration and spend more on refining the best.
    iter_id = 0
    while time.time() < deadline:
        iter_id += 1
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        # schedule: exploration weight decreases over time
        elapsed = time.time() - t0
        frac = elapsed / max(1e-9, float(max_time))
        explore_prob = 0.65 * (1.0 - frac) + 0.10  # from ~0.75 to ~0.10

        # propose a start point
        r = random.random()
        if r < explore_prob:
            # mix halton and uniform
            if random.random() < 0.75:
                # continue halton stream
                x0 = halton_point(k, bases)
                k += 1
            else:
                x0 = uniform_point()
        else:
            # intensify around current best (with occasional heavy-tailed tunnel)
            x0 = list(best_x)
            for d in range(dim):
                lo, hi = bounds[d]
                if spans[d] == 0.0:
                    x0[d] = lo
                else:
                    # mostly Gaussian-ish small step + rare Cauchy tunnel
                    if random.random() < 0.12:
                        delta = cauchy_step(0.05 * spans[d])
                    else:
                        # triangular perturbation concentrates near 0 without importing numpy
                        delta = (random.random() - random.random()) * (0.08 * spans[d])
                    x0[d] = clamp(x0[d] + delta, lo, hi)

        f0 = eval_f(x0)
        if f0 < best:
            best, best_x = f0, x0

        # local refinement from x0 (or from best if x0 is poor)
        start_x = x0 if f0 <= best * 1.10 else best_x
        start_f = f0 if start_x is x0 else best

        # initial steps: relative to spans, but smaller later in time
        base_scale = 0.22 * (1.0 - 0.7 * frac)  # shrinks over time
        step = [base_scale * s if s > 0 else 0.0 for s in spans]

        # refinement iterations based on remaining time and dimension
        # keep short to allow multiple restarts
        max_refine_iters = max(10, min(80, 8 * dim))
        # if close to deadline, do a shorter refine
        if remaining < 0.15 * float(max_time):
            max_refine_iters = max(6, min(40, 5 * dim))

        rx, rf, _ = local_refine(start_x, start_f, step, deadline, max_refine_iters)
        if rf < best:
            best, best_x = rf, rx

    return best
