import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Hybrid time-bounded derivative-free minimizer (stdlib only).

    Improvements vs the provided winners:
      - Keeps the strong global guidance (CEM/ES-style) but adds *true* low-discrepancy seeding
        (Halton) instead of LHS-ish permutations.
      - Adds an explicit *coordinate-wise line search* (very cheap, very effective on many
        benchmark-style problems) with adaptive step sizes.
      - Adds a *multi-temperature sampler*: mostly Gaussian (exploitation) with occasional
        heavy-tailed/global samples (exploration), time-scheduled.
      - Uses a small archive + "best-so-far recentering" restarts.
      - More careful time guarding (never commits to long inner loops near deadline).

    Returns:
        best (float): best (minimum) objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    # ------------ helpers ------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def evalf(x):
        return float(func(x))

    def randn():
        # approx N(0,1): sum of 12 uniforms - 6
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy():
        u = random.random()
        # avoid infinities
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        # clamp to keep numerically sane but still heavy-tailed
        if v > 80.0:
            v = 80.0
        elif v < -80.0:
            v = -80.0
        return v

    def time_left():
        return deadline - time.time()

    # Halton sequence for better coverage than pseudo-LHS
    def _first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def _halton_index(k, base):
        # radical inverse in given base, k>=1
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    primes = _first_primes(dim)

    def halton_point(k):
        # k starts at 1
        x = [0.0] * dim
        for i in range(dim):
            u = _halton_index(k, primes[i])
            x[i] = lows[i] + u * (highs[i] - lows[i])
        return x

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ------------ initialization / seeding ------------
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    clip_inplace(center)
    best_x = center[:]
    best = evalf(best_x)

    # Seeding budget: good coverage but not too expensive
    # (Halton is deterministic given k; we still add small jitter for robustness)
    init_n = max(18, 8 * dim)
    if max_time < 0.25:
        init_n = max(6, 2 * dim)
    elif max_time > 2.0:
        init_n = max(init_n, 26 + 10 * dim)

    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best

        x = halton_point(k)
        # tiny jitter to avoid pathological alignments, scaled by bounds
        for i in range(dim):
            x[i] += (random.random() - 0.5) * 0.002 * spans[i]
        clip_inplace(x)

        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]

        xo = opposite(x)
        clip_inplace(xo)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    # ------------ search state (diagonal ES/CEM-like) ------------
    mean = best_x[:]
    sigma = [0.35 * (highs[i] - lows[i]) for i in range(dim)]
    for i in range(dim):
        if sigma[i] <= 0.0:
            sigma[i] = 1.0

    sig_floor = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    sig_ceil = [0.85 * (highs[i] - lows[i]) if highs[i] != lows[i] else 1.0 for i in range(dim)]

    # smoothing
    alpha_m = 0.30
    alpha_s = 0.22

    # small archive
    archive = []
    archive_cap = max(14, 3 * dim)

    def add_archive(f, x):
        archive.append((f, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > archive_cap:
            del archive[archive_cap:]

    add_archive(best, best_x)

    # ------------ coordinate line search around best ------------
    # start relatively large, shrink per-coordinate when failing
    step = [0.20 * (highs[i] - lows[i]) for i in range(dim)]
    step_min = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    for i in range(dim):
        if step[i] <= 0.0:
            step[i] = 1.0

    def coord_line_search(rounds):
        """Few cheap coordinate-wise trials with 1D backtracking-like behavior."""
        nonlocal best, best_x
        for _ in range(rounds):
            if time.time() >= deadline:
                return
            i = random.randrange(dim)
            h = step[i]
            if h <= step_min[i]:
                continue

            base = best_x[:]
            # try +h and -h, take best, optionally try 2h if success
            cand_p = base[:]
            cand_p[i] += h
            clip_inplace(cand_p)
            fp = evalf(cand_p)

            cand_m = base[:]
            cand_m[i] -= h
            clip_inplace(cand_m)
            fm = evalf(cand_m)

            improved = False
            if fp < best or fm < best:
                improved = True
                if fp <= fm:
                    best, best_x = fp, cand_p[:]
                else:
                    best, best_x = fm, cand_m[:]
                add_archive(best, best_x)

                # opportunistic extrapolation in same direction
                if time.time() < deadline:
                    cand2 = best_x[:]
                    # move another h in same direction as last improvement
                    if fp <= fm:
                        cand2[i] += h
                    else:
                        cand2[i] -= h
                    clip_inplace(cand2)
                    f2 = evalf(cand2)
                    if f2 < best:
                        best, best_x = f2, cand2[:]
                        add_archive(best, best_x)

                # increase step on success (bounded)
                step[i] = min(step[i] * 1.25, highs[i] - lows[i] if highs[i] != lows[i] else step[i])
            else:
                # shrink on failure
                step[i] = max(step[i] * 0.60, step_min[i])

    # ------------ main loop ------------
    no_improve = 0
    last_best = best
    it = 0

    while True:
        if time.time() >= deadline:
            return best

        it += 1
        tl = time_left()
        if tl <= 0.0:
            return best

        # population size (time/dim aware)
        base_lam = 10 + 4 * dim
        if tl < 0.08:
            lam = max(6, 2 * dim)
        elif tl > 1.0 and dim <= 16:
            lam = int(base_lam * 1.7)
        else:
            lam = base_lam

        # exploration schedule: early more heavy/global, late more local
        # p_heavy decreases with time
        frac_left = max(0.0, min(1.0, tl / max(1e-9, max_time)))
        p_heavy = 0.06 + 0.18 * frac_left  # ~0.24 early, ~0.06 late
        p_global = 0.03 + 0.12 * frac_left # global uniform samples

        pop = []
        half = (lam + 1) // 2

        # prevent drift: nudge mean toward best
        if random.random() < 0.22:
            mean = [0.82 * mean[i] + 0.18 * best_x[i] for i in range(dim)]

        for _ in range(half):
            if time.time() >= deadline:
                return best

            if random.random() < p_global:
                x1 = [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]
                x2 = opposite(x1)
                clip_inplace(x2)
            else:
                heavy = (random.random() < p_heavy)
                z = [randn() for _ in range(dim)]
                x1 = [0.0] * dim
                x2 = [0.0] * dim
                for i in range(dim):
                    stepv = sigma[i] * (cauchy() if heavy else z[i])
                    x1[i] = mean[i] + stepv
                    x2[i] = mean[i] - stepv
                clip_inplace(x1)
                clip_inplace(x2)

            f1 = evalf(x1)
            pop.append((f1, x1))
            if f1 < best:
                best, best_x = f1, x1[:]
                add_archive(best, best_x)
                no_improve = 0

            if len(pop) < lam:
                f2 = evalf(x2)
                pop.append((f2, x2))
                if f2 < best:
                    best, best_x = f2, x2[:]
                    add_archive(best, best_x)
                    no_improve = 0

        pop.sort(key=lambda t: t[0])

        # elites selection + archive mix-in
        mu = max(4, lam // 5)
        elites = pop[:mu]
        if archive:
            take = min(len(archive), max(1, mu // 3))
            elites = elites[:max(1, mu - take)] + archive[:take]
            elites.sort(key=lambda t: t[0])
            elites = elites[:mu]

        # log-rank weights
        weights = []
        wsum = 0.0
        for r in range(mu):
            w = math.log(mu + 1.0) - math.log(r + 1.0)
            weights.append(w)
            wsum += w
        if wsum <= 0.0:
            wsum = 1.0

        new_mean = [0.0] * dim
        for w, (f, x) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                new_mean[i] += ww * x[i]

        new_var = [0.0] * dim
        for w, (f, x) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                d = x[i] - new_mean[i]
                new_var[i] += ww * d * d

        new_sigma = [0.0] * dim
        for i in range(dim):
            s = math.sqrt(max(new_var[i], sig_floor[i] * sig_floor[i]))
            if s < sig_floor[i]:
                s = sig_floor[i]
            elif s > sig_ceil[i]:
                s = sig_ceil[i]
            new_sigma[i] = s

        # smooth update
        for i in range(dim):
            mean[i] = (1.0 - alpha_m) * mean[i] + alpha_m * new_mean[i]
            sigma[i] = (1.0 - alpha_s) * sigma[i] + alpha_s * new_sigma[i]
            if sigma[i] < sig_floor[i]:
                sigma[i] = sig_floor[i]
            elif sigma[i] > sig_ceil[i]:
                sigma[i] = sig_ceil[i]

        # local exploitation: coordinate line search
        # increase effort when stagnating, but stay time-safe
        tl = time_left()
        if tl > 0.02:
            rounds = 2 if no_improve < 10 else 6
            if tl < 0.08:
                rounds = 1
            coord_line_search(rounds)

        # stagnation + restarts
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # small sigma inflation to escape
        if no_improve % 16 == 0:
            for i in range(dim):
                sigma[i] = min(sig_ceil[i], sigma[i] * 1.30)

        # partial restart
        if no_improve % 55 == 0:
            # recenter mean at best or a good archive point
            if archive and random.random() < 0.65:
                mean = archive[0][1][:]
            elif random.random() < 0.80:
                mean = best_x[:]
            else:
                mean = [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

            # reset sigma moderately (not too large)
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.20 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0))
            no_improve = 0
