import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Hybrid strategy (fast + robust):
      1) Quasi-random exploration (Halton sequence) + opposition points
      2) Maintain a small elite pool
      3) Local refinement using a lightweight Nelder–Mead simplex (best for smooth-ish landscapes)
      4) When NM stalls, switch to adaptive (1+1)-ES with success rule + occasional heavy-tail jumps
      5) Restarts from elites and global Halton samples

    Returns:
      best (float): best fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        # Degenerate: call func with empty vector if possible
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    fixed = [spans[i] == 0.0 for i in range(dim)]

    # ---------- helpers ----------
    def now():
        return time.time()

    def clamp(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def evalf(x):
        return float(func(x))

    # Cauchy heavy-tail
    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- Halton sequence (better than plain random for coverage) ----------
    def first_primes(k):
        primes = []
        c = 2
        while len(primes) < k:
            is_p = True
            r = int(c ** 0.5)
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(c)
            c += 1
        return primes

    bases = first_primes(max(1, dim))

    def van_der_corput(n, base):
        vdc = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton(idx):
        # idx >= 0
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                u = van_der_corput(idx + 1, bases[i])  # avoid 0-vector
                x[i] = lows[i] + u * spans[i]
        return x

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                y[i] = lows[i]
            else:
                y[i] = lows[i] + highs[i] - x[i]
        return y

    # ---------- elite pool ----------
    elite_k = max(5, min(14, 4 + dim // 2))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites.pop()

    best = float("inf")
    best_x = None

    # ---------- initial exploration ----------
    # Mix Halton, pure random, and opposition points.
    init_n = max(60, min(700, 50 + 35 * dim))
    idx = 0
    for j in range(init_n):
        if now() >= deadline:
            return best
        if j % 7 == 0:
            x = [lows[i] + random.random() * spans[i] if not fixed[i] else lows[i] for i in range(dim)]
        else:
            x = halton(idx)
            idx += 1

        x = clamp(x)
        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]
        push_elite(f, x)

        # opposition evaluation often helps on bounded domains
        if now() >= deadline:
            return best
        xo = opposition(x)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]
        push_elite(fo, xo)

    if best_x is None:
        x = halton(0)
        best_x = x
        best = evalf(x)
        push_elite(best, x)

    # ---------- Nelder–Mead (bounded) ----------
    # Works well as a local intensifier; we keep it lightweight & restartable.
    def nm_refine(start_x, start_step_scale, time_slice_end):
        """
        Run a bounded Nelder–Mead from start_x until time_slice_end.
        Returns (best_f, best_x, did_progress)
        """
        nonlocal best, best_x

        # Build initial simplex of size dim+1
        x0 = clamp(start_x)
        f0 = evalf(x0)

        # Per-dim initial steps
        steps = []
        for i in range(dim):
            if fixed[i]:
                steps.append(0.0)
            else:
                # scale with span; ensure nonzero
                s = spans[i] * start_step_scale
                if s <= 0.0:
                    s = 1e-6
                steps.append(s)

        simplex = [(f0, x0)]
        for i in range(dim):
            if now() >= time_slice_end:
                break
            xi = x0[:]
            if not fixed[i]:
                xi[i] = xi[i] + steps[i]
            xi = clamp(xi)
            fi = evalf(xi)
            simplex.append((fi, xi))
        # If we couldn't build full simplex in time, return current best
        simplex.sort(key=lambda t: t[0])

        if simplex[0][0] < best:
            best, best_x = simplex[0][0], simplex[0][1][:]
            push_elite(best, best_x)

        # NM parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

        # track progress
        start_best = simplex[0][0]
        iters = 0

        while now() < time_slice_end and len(simplex) >= 2:
            iters += 1
            simplex.sort(key=lambda t: t[0])
            f_best, x_best = simplex[0]
            f_worst, x_worst = simplex[-1]
            f_second_worst, _ = simplex[-2]

            # centroid of all but worst
            centroid = [0.0] * dim
            m = len(simplex) - 1
            for (_, x) in simplex[:-1]:
                for i in range(dim):
                    centroid[i] += x[i]
            for i in range(dim):
                centroid[i] /= float(m)

            # reflection
            xr = [centroid[i] + alpha * (centroid[i] - x_worst[i]) for i in range(dim)]
            xr = clamp(xr)
            fr = evalf(xr)

            if fr < f_best:
                # expansion
                xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(dim)]
                xe = clamp(xe)
                fe = evalf(xe)
                if fe < fr:
                    simplex[-1] = (fe, xe)
                else:
                    simplex[-1] = (fr, xr)
            elif fr < f_second_worst:
                simplex[-1] = (fr, xr)
            else:
                # contraction
                if fr < f_worst:
                    # outside contraction
                    xc = [centroid[i] + rho * (xr[i] - centroid[i]) for i in range(dim)]
                else:
                    # inside contraction
                    xc = [centroid[i] - rho * (centroid[i] - x_worst[i]) for i in range(dim)]
                xc = clamp(xc)
                fc = evalf(xc)

                if fc < f_worst:
                    simplex[-1] = (fc, xc)
                else:
                    # shrink towards best
                    new_simplex = [simplex[0]]
                    for (fi, xi) in simplex[1:]:
                        xs = [x_best[i] + sigma * (xi[i] - x_best[i]) for i in range(dim)]
                        xs = clamp(xs)
                        fs = evalf(xs)
                        new_simplex.append((fs, xs))
                    simplex = new_simplex

            # global best update
            simplex.sort(key=lambda t: t[0])
            if simplex[0][0] < best:
                best, best_x = simplex[0][0], simplex[0][1][:]
                push_elite(best, best_x)

            # stop if simplex collapsed enough (very small spread in coords)
            if iters % (10 + dim) == 0:
                spread = 0.0
                xb = simplex[0][1]
                for (_, x) in simplex[1:]:
                    for i in range(dim):
                        d = abs(x[i] - xb[i])
                        if d > spread:
                            spread = d
                if spread < 1e-12 * (max(spans) + 1.0):
                    break

        did_progress = (simplex[0][0] + 0.0) < start_best
        return simplex[0][0], simplex[0][1][:], did_progress

    # ---------- ES fallback (global+local) ----------
    # Adaptive step sizes + elite restarts; runs until deadline.
    sigma0 = [0.18 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
    span_max = max(spans) if spans else 0.0
    sigma_floor = 1e-12 + 1e-10 * (span_max + 1.0)

    x_cur = best_x[:]
    f_cur = best

    window = max(24, 12 + 2 * dim)
    succ = 0
    trials = 0
    no_improve = 0
    restart_after = 90 + 18 * dim

    sigma = sigma0[:]

    # Alternate between NM time slices and ES, choosing from elites.
    while now() < deadline:
        # --- NM slice ---
        # Use a short slice early for exploitation, longer later when near good basins.
        remaining = deadline - now()
        if remaining <= 0:
            break

        # pick seed: mostly best elite, sometimes random elite, sometimes current
        r = random.random()
        if elites and r < 0.55:
            seed = elites[0][1][:]
        elif elites and r < 0.85:
            seed = random.choice(elites)[1][:]
        else:
            seed = x_cur[:]

        # step scale shrinks over time
        frac = (now() - t0) / max_time if max_time > 0 else 1.0
        step_scale = 0.12 * (1.0 - 0.85 * frac)  # from ~0.12 down to ~0.018

        slice_len = min(remaining, 0.12 + 0.01 * dim)  # small bounded slice
        slice_end = now() + slice_len
        _, nm_x, nm_prog = nm_refine(seed, step_scale, slice_end)
        if nm_prog:
            x_cur = nm_x[:]
            f_cur = evalf(x_cur)
            if f_cur < best:
                best, best_x = f_cur, x_cur[:]
                push_elite(best, best_x)
            no_improve = 0

        if now() >= deadline:
            break

        # --- ES steps until next potential NM slice or time end ---
        # do a small batch of ES mutations
        es_batch = 30 + 5 * dim
        for _ in range(es_batch):
            if now() >= deadline:
                break

            trials += 1

            # choose base
            if elites and random.random() < 0.12:
                x_base = random.choice(elites)[1][:]
            else:
                x_base = x_cur

            rr = random.random()
            x_new = x_base[:]
            if rr < 0.78:
                # gaussian-ish
                for i in range(dim):
                    if fixed[i]:
                        continue
                    s = max(sigma[i], sigma_floor)
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random() - 3.0) / 3.0
                    x_new[i] += g * s
            elif rr < 0.94:
                # heavy tail
                for i in range(dim):
                    if fixed[i]:
                        continue
                    s = max(sigma[i], sigma_floor)
                    x_new[i] += cauchy() * 0.30 * s
            else:
                # global Halton
                x_new = halton(idx)
                idx += 1

            x_new = clamp(x_new)
            f_new = evalf(x_new)

            if f_new <= f_cur:
                x_cur, f_cur = x_new, f_new
                succ += 1

            if f_new < best:
                best, best_x = f_new, x_new[:]
                push_elite(best, best_x)
                no_improve = 0
            else:
                no_improve += 1

            # 1/5-ish adaptation
            if trials >= window:
                rate = succ / float(trials)
                if rate > 0.22:
                    factor = 1.30
                elif rate < 0.18:
                    factor = 0.72
                else:
                    factor = 1.0

                if factor != 1.0:
                    for i in range(dim):
                        if fixed[i]:
                            continue
                        sigma[i] *= factor
                        if sigma[i] > 0.85 * spans[i]:
                            sigma[i] = 0.85 * spans[i]
                        if sigma[i] < sigma_floor:
                            sigma[i] = sigma_floor

                succ = 0
                trials = 0

            # stagnation restart
            if no_improve >= restart_after:
                # restart from best or random elite, or global
                if elites and random.random() < 0.80:
                    x_cur = (elites[0][1] if random.random() < 0.65 else random.choice(elites)[1])[:]
                else:
                    x_cur = halton(idx)
                    idx += 1

                # jitter + reset sigma (smaller later)
                frac = (now() - t0) / max_time if max_time > 0 else 1.0
                shrink = 0.55 + 0.45 * (1.0 - frac)
                sigma = [max(sigma_floor, sigma0[i] * shrink) for i in range(dim)]
                for i in range(dim):
                    if fixed[i]:
                        continue
                    x_cur[i] += (random.random() * 2.0 - 1.0) * 0.08 * spans[i]
                x_cur = clamp(x_cur)
                f_cur = evalf(x_cur)
                if f_cur < best:
                    best, best_x = f_cur, x_cur[:]
                    push_elite(best, best_x)
                no_improve = 0

    return best
