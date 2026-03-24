import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (self-contained, no external libs).

    Improvements over provided versions:
      - Maintains an elite set and updates a sampling distribution (CEM-like) for strong global guidance
      - Uses mirrored/antithetic sampling for variance reduction
      - Interleaves local refinement via adaptive pattern search around the best
      - Adds occasional heavy-tailed (Cauchy-like) jumps to escape basins
      - Budget-aware: dynamically sizes batches based on time and dim
    Returns:
      best (float): best (minimum) objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        return float(func([]))

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # Handle degenerate spans
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def evalf(x):
        return float(func(x))

    # --- helpers: normal + heavy-tailed noise (no libs) ---
    def randn():
        # approx N(0,1): sum of 12 uniforms - 6
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy_like():
        # tan(pi*(u-0.5)) gives Cauchy; clamp to avoid huge infinities
        u = random.random()
        # avoid exactly 0 or 1
        if u <= 1e-12:
            u = 1e-12
        elif u >= 1.0 - 1e-12:
            u = 1.0 - 1e-12
        v = math.tan(math.pi * (u - 0.5))
        # clamp extreme tails (still heavy-tailed)
        if v > 50.0:
            v = 50.0
        elif v < -50.0:
            v = -50.0
        return v

    # --- initialize mean at center, std to cover space ---
    mean = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    sigma = [0.35 * (highs[i] - lows[i]) for i in range(dim)]
    for i in range(dim):
        if sigma[i] <= 0.0:
            sigma[i] = 1.0

    # evaluate center and a few random points quickly
    best_x = mean[:]
    clip_inplace(best_x)
    best = evalf(best_x)

    # quick diverse seeding (small LHS-ish + opposition)
    init_n = max(10, 6 * dim)
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]
        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        clip_inplace(xo)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    # local search step (pattern search) around best
    loc_step = [0.20 * (highs[i] - lows[i]) for i in range(dim)]
    loc_min_step = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    for i in range(dim):
        if loc_step[i] <= 0.0:
            loc_step[i] = 1.0

    # CEM-like parameters
    # batch size: moderate; bigger for small dim and more time
    base_batch = 8 + 4 * dim
    elite_frac = 0.2
    if elite_frac * base_batch < 3:
        elite_frac = 3.0 / base_batch
    # learning rates
    alpha_mean = 0.35
    alpha_sigma = 0.25
    # sigma floors/ceilings
    sig_floor = [1e-12 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    sig_ceil = [0.75 * (highs[i] - lows[i]) if highs[i] != lows[i] else 1.0 for i in range(dim)]

    # stagnation control
    no_improve = 0

    # Maintain a tiny elite archive to bias updates robustly
    archive = []  # list of (f, x)
    archive_cap = max(10, 3 * dim)

    def add_archive(f, x):
        nonlocal archive
        archive.append((f, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > archive_cap:
            archive = archive[:archive_cap]

    add_archive(best, best_x)

    # time-aware loop
    while True:
        if time.time() >= deadline:
            return best

        # Adjust batch by time left (avoid being caught mid-batch)
        tl = deadline - time.time()
        # if very low time, do a couple of quick local tweaks and return
        if tl < 0.01:
            return best

        batch = base_batch
        if tl < 0.10:
            batch = max(6, 2 * dim)
        elif tl > 1.0 and dim <= 12:
            batch = int(base_batch * 1.5)

        # Create population via Gaussian around mean, with antithetic pairing
        pop = []
        half = (batch + 1) // 2

        # Occasionally mix mean between current model mean and best_x (helps when model drifts)
        if random.random() < 0.15:
            mean = [(0.75 * mean[i] + 0.25 * best_x[i]) for i in range(dim)]

        for _ in range(half):
            if time.time() >= deadline:
                return best

            z = [randn() for _ in range(dim)]

            # With small probability use heavy-tailed jump (escape)
            use_cauchy = (random.random() < 0.10)

            x1 = [0.0] * dim
            x2 = [0.0] * dim
            for i in range(dim):
                if use_cauchy:
                    step = sigma[i] * cauchy_like()
                else:
                    step = sigma[i] * z[i]
                x1[i] = mean[i] + step
                x2[i] = mean[i] - step  # antithetic
            clip_inplace(x1)
            clip_inplace(x2)

            f1 = evalf(x1)
            pop.append((f1, x1))
            if f1 < best:
                best, best_x = f1, x1[:]
                add_archive(best, best_x)
                no_improve = 0

            if len(pop) < batch:
                f2 = evalf(x2)
                pop.append((f2, x2))
                if f2 < best:
                    best, best_x = f2, x2[:]
                    add_archive(best, best_x)
                    no_improve = 0

        pop.sort(key=lambda t: t[0])

        # Also include some archive points as "virtual elites" (robust update)
        # (Keeps memory of good areas even if one batch is unlucky)
        elites_n = max(3, int(elite_frac * len(pop)))
        elites = pop[:elites_n]

        # If archive has better than current elites, mix them in
        if archive:
            take = min(len(archive), max(1, elites_n // 2))
            # archive already sorted
            elites = elites[:max(1, elites_n - take)] + archive[:take]
            elites.sort(key=lambda t: t[0])
            elites = elites[:elites_n]

        # Update mean/sigma using weighted elites (rank weights)
        weights = []
        wsum = 0.0
        for r in range(len(elites)):
            # log rank weights (CEM/ES style), positive
            w = math.log(len(elites) + 1.0) - math.log(r + 1.0)
            weights.append(w)
            wsum += w
        if wsum <= 0.0:
            wsum = 1.0

        new_mean = [0.0] * dim
        for (w, (f, x)) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                new_mean[i] += ww * x[i]

        # robust sigma from elite deviations around new_mean
        new_sigma = [0.0] * dim
        for (w, (f, x)) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                d = x[i] - new_mean[i]
                new_sigma[i] += ww * (d * d)
        for i in range(dim):
            new_sigma[i] = math.sqrt(max(new_sigma[i], sig_floor[i] * sig_floor[i]))

        # smooth updates
        for i in range(dim):
            mean[i] = (1.0 - alpha_mean) * mean[i] + alpha_mean * new_mean[i]
            sigma[i] = (1.0 - alpha_sigma) * sigma[i] + alpha_sigma * new_sigma[i]
            # clamp sigma
            if sigma[i] < sig_floor[i]:
                sigma[i] = sig_floor[i]
            elif sigma[i] > sig_ceil[i]:
                sigma[i] = sig_ceil[i]

        # --- local refinement around current best (pattern search) ---
        # Do a few coordinate tries; cheap and effective after distribution narrows.
        # Frequency increases with stagnation.
        local_tries = 2 if no_improve < 15 else 5
        for _ in range(local_tries):
            if time.time() >= deadline:
                return best
            i = random.randrange(dim)
            h = loc_step[i]
            if h <= loc_min_step[i]:
                continue

            base = best_x[:]
            # try +h and -h
            cand1 = base[:]
            cand1[i] += h
            clip_inplace(cand1)
            f1 = evalf(cand1)

            cand2 = base[:]
            cand2[i] -= h
            clip_inplace(cand2)
            f2 = evalf(cand2)

            if f1 < best or f2 < best:
                if f1 <= f2:
                    best, best_x = f1, cand1[:]
                else:
                    best, best_x = f2, cand2[:]
                add_archive(best, best_x)
                no_improve = 0
                # slightly enlarge local step on success
                loc_step[i] = min(loc_step[i] * 1.15, highs[i] - lows[i] if highs[i] != lows[i] else loc_step[i])
            else:
                # shrink that coordinate step on failure
                loc_step[i] = max(loc_step[i] * 0.6, loc_min_step[i])

        # stagnation / restart controls
        prev_best = pop[0][0] if pop else float("inf")
        if prev_best < best:
            # should rarely happen since best tracks; keep consistent
            best = prev_best
            best_x = pop[0][1][:]
            add_archive(best, best_x)
            no_improve = 0
        else:
            no_improve += 1

        # If stuck: broaden distribution or restart mean near best / random
        if no_improve % 20 == 0:
            # broaden a bit to escape
            for i in range(dim):
                sigma[i] = min(sig_ceil[i], sigma[i] * 1.35)

        if no_improve % 60 == 0:
            # partial restart
            if random.random() < 0.7:
                mean = best_x[:]
            else:
                mean = [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]
            # reset sigmas moderately
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.25 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0))
            no_improve = 0

    # unreachable
    # return best
