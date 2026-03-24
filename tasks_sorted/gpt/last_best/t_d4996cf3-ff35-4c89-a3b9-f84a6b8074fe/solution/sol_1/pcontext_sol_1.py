import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (derivative-free) using:
      1) Quasi-random (Halton) global sampling to cover space better than stratified random
      2) Keep an elite set of best points found so far
      3) Multi-start local refinement on elites via coordinate pattern search
         with adaptive step sizes and occasional heavy-tailed kicks
      4) Periodic re-injection of global samples to avoid premature convergence

    Returns:
      best (float): best (minimum) fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    nonzero = [spans[i] > 0.0 for i in range(dim)]

    # ----------------- helpers -----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        return float(func(x))

    # --- Halton sequence for low-discrepancy sampling (no external libs) ---
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            isprime = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    isprime = False
                    break
            if isprime:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))

    def van_der_corput(n, base):
        vdc = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    halton_index = 1

    def halton_point():
        nonlocal halton_index
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(halton_index, primes[i])
            # Cranley-Patterson style random shift helps avoid alignment
            u = (u + random.random()) % 1.0
            x[i] = lows[i] + u * spans[i]
        halton_index += 1
        return x

    # --- elite management ---
    # Keep a small list of best points (fitness, x)
    elite_k = max(4, min(12, 2 + dim // 2))
    elites = []

    def add_elite(fx, x):
        nonlocal elites
        # Insert sorted by fitness, keep unique-ish
        # (avoid filling elites with near-identical points)
        for (f2, x2) in elites:
            if abs(fx - f2) < 1e-12:
                # if same fitness, keep the one with some diversity
                pass
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # --- init ---
    # Start with a few random points plus Halton points
    x_best = [lows[i] + random.random() * spans[i] if nonzero[i] else lows[i] for i in range(dim)]
    f_best = eval_f(x_best)
    add_elite(f_best, x_best)

    # Evaluate a batch of global points (time-aware)
    # Aim for enough coverage early; scale modestly with dim
    target_global = max(50, min(600, 60 + 30 * dim))
    # If max_time very small, reduce work
    if max_time < 0.2:
        target_global = max(10, min(target_global, 40))

    for _ in range(target_global):
        if time.time() >= deadline:
            return f_best
        x = halton_point()
        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]
        add_elite(fx, x)

    # ----------------- local search (pattern search) -----------------
    # Coordinate pattern search around a seed point:
    # try +/- step along each coordinate; adapt step per coordinate.
    def refine_from(seed_x, seed_f, time_limit):
        x = seed_x[:]
        fx = seed_f

        # Per-coordinate step: start moderate, shrink on failure
        step = [0.25 * spans[i] if nonzero[i] else 0.0 for i in range(dim)]
        # Minimum step: relative to range
        min_step = [1e-12 * (spans[i] if nonzero[i] else 1.0) for i in range(dim)]

        # Order dimensions by span (bigger spans first)
        order = list(range(dim))
        order.sort(key=lambda i: spans[i], reverse=True)

        no_gain_sweeps = 0
        while time.time() < time_limit:
            improved = False

            # sweep coordinates
            for i in order:
                if not nonzero[i]:
                    continue

                si = step[i]
                if si <= min_step[i]:
                    continue

                # try both directions
                best_local_f = fx
                best_local_x = None

                # Occasionally do a small heavy-tailed kick on this coordinate
                # to jump across flat regions / local traps
                if random.random() < 0.03:
                    u = random.random()
                    kick = math.tan(math.pi * (u - 0.5)) * 0.02 * spans[i]
                    cand = x[:]
                    cand[i] += kick
                    clip_inplace(cand)
                    fc = eval_f(cand)
                    if fc < best_local_f:
                        best_local_f = fc
                        best_local_x = cand

                for direction in (-1.0, 1.0):
                    cand = x[:]
                    cand[i] += direction * si
                    if cand[i] < lows[i]:
                        cand[i] = lows[i]
                    elif cand[i] > highs[i]:
                        cand[i] = highs[i]
                    fc = eval_f(cand)
                    if fc < best_local_f:
                        best_local_f = fc
                        best_local_x = cand

                if best_local_x is not None:
                    x, fx = best_local_x, best_local_f
                    improved = True
                    # If move worked, very gently increase step for that coordinate
                    step[i] = min(0.5 * spans[i], step[i] * 1.15 + 1e-18)
                    add_elite(fx, x)
                else:
                    # If it didn't work, shrink step
                    step[i] = max(min_step[i], step[i] * 0.5)

                if time.time() >= time_limit:
                    break

            if improved:
                no_gain_sweeps = 0
            else:
                no_gain_sweeps += 1
                # If no progress, shrink all a bit
                for i in range(dim):
                    if nonzero[i]:
                        step[i] = max(min_step[i], step[i] * 0.7)

            # Terminate if we're effectively stuck
            if no_gain_sweeps >= 6:
                break

        return fx, x

    # Allocate time: cycle between refining elites and injecting more global points
    # so we don't over-exploit early.
    cycle = 0
    while time.time() < deadline:
        cycle += 1
        now = time.time()
        remaining = deadline - now
        if remaining <= 0:
            break

        # Periodically add a few new global samples (diversification)
        inject = 10 if dim <= 10 else 6
        inject = min(inject, 30)
        for _ in range(inject):
            if time.time() >= deadline:
                return f_best
            x = halton_point()
            fx = eval_f(x)
            if fx < f_best:
                f_best, x_best = fx, x[:]
            add_elite(fx, x)

        # Refine a couple of best elites (intensification)
        elites.sort(key=lambda t: t[0])
        refine_count = 2 if dim <= 20 else 1
        refine_count = min(refine_count, len(elites))

        # Each refinement gets a slice of remaining time
        # but keep it small to cycle frequently.
        slice_time = min(0.12 * remaining, 0.25)  # seconds
        if slice_time <= 0.0:
            break

        for j in range(refine_count):
            if time.time() >= deadline:
                break
            fx0, x0 = elites[j]
            # Give each refinement an equal portion of the slice
            per = slice_time / max(1, refine_count)
            f1, x1 = refine_from(x0, fx0, min(deadline, time.time() + per))
            if f1 < f_best:
                f_best, x_best = f1, x1[:]
            add_elite(f1, x1)

        # Opportunistic restart: refine around current best with random small perturbation
        if time.time() < deadline and (cycle % 3 == 0):
            xb = x_best[:]
            for i in range(dim):
                if nonzero[i]:
                    xb[i] += random.gauss(0.0, 0.02 * spans[i])
            clip_inplace(xb)
            fb = eval_f(xb)
            if fb < f_best:
                f_best, x_best = fb, xb[:]
            add_elite(fb, xb)

    return f_best
