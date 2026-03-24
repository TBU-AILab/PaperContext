import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (stdlib only).

    Key upgrades over the provided best (CEM/diagonal-ES + local search):
      1) Better space-filling seeding (Halton + opposition + center) for a stronger initial best.
      2) True coordinate-descent with cheap bracket+refine (1D line search) around current best.
      3) Hybrid global model:
           - diagonal ES (CEM-like) to guide search
           - periodic "model restarts" from elite archive
           - mixed proposals: Gaussian, heavy-tailed, and occasional uniform
      4) More effective step-size control:
           - per-coordinate local steps (for line search)
           - sigma inflation on stagnation
      5) Strict time-guarding (never commits to long loops near deadline)

    Returns:
        best (float): best objective value found within max_time seconds.
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

    # ---------------- helpers ----------------
    def time_left():
        return deadline - time.time()

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
        # Cauchy via tan(pi*(u-0.5)), clamped to keep finite numerically
        u = random.random()
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        if v > 60.0:
            v = 60.0
        elif v < -60.0:
            v = -60.0
        return v

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def rand_uniform():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    # Halton for deterministic coverage
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

    # ---------------- initialization ----------------
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    clip_inplace(center)
    best_x = center[:]
    best = evalf(best_x)

    # Elite archive (memory)
    archive = []  # list of (f, x)
    archive_cap = max(18, 4 * dim)

    def add_archive(f, x):
        archive.append((f, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > archive_cap:
            del archive[archive_cap:]

    add_archive(best, best_x)

    # Space-filling seeding (time-aware)
    # Keep moderate: good coverage but leave time for exploitation.
    init_n = max(22, 10 * dim)
    if max_time < 0.25:
        init_n = max(8, 3 * dim)
    elif max_time > 2.0:
        init_n = max(init_n, 30 + 12 * dim)

    # Halton + jitter + opposition + a few pure randoms
    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best
        x = halton_point(k)
        # small jitter to break determinism / alignments
        for i in range(dim):
            x[i] += (random.random() - 0.5) * 0.003 * spans[i]
        clip_inplace(x)

        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]
            add_archive(best, best_x)

        xo = opposite(x)
        clip_inplace(xo)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]
            add_archive(best, best_x)

        # sprinkle a few uniforms early
        if k <= max(3, dim // 3):
            xr = rand_uniform()
            fr = evalf(xr)
            if fr < best:
                best, best_x = fr, xr[:]
                add_archive(best, best_x)

    # ---------------- global model (diagonal ES/CEM-like) ----------------
    mean = best_x[:]
    sigma = [0.33 * (highs[i] - lows[i]) for i in range(dim)]
    for i in range(dim):
        if sigma[i] <= 0.0:
            sigma[i] = 1.0

    sig_floor = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    sig_ceil = [0.90 * (highs[i] - lows[i]) if highs[i] != lows[i] else 1.0 for i in range(dim)]

    alpha_m = 0.28
    alpha_s = 0.20

    # ---------------- local search: coordinate line search ----------------
    # per-coordinate step sizes (used to propose brackets)
    lstep = [0.20 * (highs[i] - lows[i]) for i in range(dim)]
    lstep_min = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    for i in range(dim):
        if lstep[i] <= 0.0:
            lstep[i] = 1.0

    def try_update(xcand, fcand):
        nonlocal best, best_x
        if fcand < best:
            best, best_x = fcand, xcand[:]
            add_archive(best, best_x)
            return True
        return False

    def coord_line_search(budget_evals):
        """
        Very cheap 1D search along a coordinate around best_x.
        Uses a small bracket and then a couple of refinements.
        """
        nonlocal best_x, best
        if budget_evals <= 0:
            return

        evals = 0
        while evals < budget_evals and time.time() < deadline:
            i = random.randrange(dim)
            h = lstep[i]
            if h <= lstep_min[i]:
                continue

            base = best_x[:]
            f0 = best

            # Bracket: try +/-h; keep best direction if any improvement
            cand_p = base[:]
            cand_p[i] = min(highs[i], cand_p[i] + h)
            fp = evalf(cand_p); evals += 1
            if try_update(cand_p, fp):
                # expand once in same direction if time/budget
                if evals < budget_evals and time.time() < deadline:
                    cand2 = best_x[:]
                    cand2[i] = min(highs[i], cand2[i] + h)
                    f2 = evalf(cand2); evals += 1
                    try_update(cand2, f2)
                lstep[i] = min(lstep[i] * 1.25, highs[i] - lows[i] if highs[i] != lows[i] else lstep[i])
                continue

            if evals >= budget_evals or time.time() >= deadline:
                break

            cand_m = base[:]
            cand_m[i] = max(lows[i], cand_m[i] - h)
            fm = evalf(cand_m); evals += 1
            if try_update(cand_m, fm):
                if evals < budget_evals and time.time() < deadline:
                    cand2 = best_x[:]
                    cand2[i] = max(lows[i], cand2[i] - h)
                    f2 = evalf(cand2); evals += 1
                    try_update(cand2, f2)
                lstep[i] = min(lstep[i] * 1.25, highs[i] - lows[i] if highs[i] != lows[i] else lstep[i])
                continue

            # No improvement: small refine around base using smaller step (backtracking)
            # (One extra attempt if budget allows)
            if evals < budget_evals and time.time() < deadline:
                hh = h * 0.35
                cand = base[:]
                cand[i] = min(highs[i], cand[i] + (hh if random.random() < 0.5 else -hh))
                fc = evalf(cand); evals += 1
                if try_update(cand, fc):
                    lstep[i] = min(lstep[i] * 1.10, highs[i] - lows[i] if highs[i] != lows[i] else lstep[i])
                else:
                    lstep[i] = max(lstep[i] * 0.60, lstep_min[i])
            else:
                lstep[i] = max(lstep[i] * 0.60, lstep_min[i])

            # keep model mean softly synced with best (helps exploitation)
            if best < f0:
                for j in range(dim):
                    mean[j] = 0.85 * mean[j] + 0.15 * best_x[j]

    # ---------------- main loop ----------------
    no_improve = 0
    last_best = best

    while True:
        if time.time() >= deadline:
            return best

        tl = time_left()
        if tl <= 0.0:
            return best

        # Population size: moderate; reduce near deadline
        base_lam = 10 + 4 * dim
        if tl < 0.08:
            lam = max(6, 2 * dim)
        elif tl > 1.0 and dim <= 18:
            lam = int(base_lam * 1.8)
        else:
            lam = base_lam

        # Exploration schedule: more global/heavy early, more Gaussian late
        frac_left = max(0.0, min(1.0, tl / max(1e-9, max_time)))
        p_uniform = 0.02 + 0.10 * frac_left
        p_heavy = 0.05 + 0.18 * frac_left

        # Generate population with antithetic pairs
        pop = []
        half = (lam + 1) // 2

        # occasionally re-center mean to best or best archive
        if archive and random.random() < 0.10:
            mean = archive[0][1][:]
        elif random.random() < 0.12:
            mean = best_x[:]

        for _ in range(half):
            if time.time() >= deadline:
                return best

            if random.random() < p_uniform:
                x1 = rand_uniform()
                x2 = opposite(x1)
                clip_inplace(x2)
            else:
                heavy = (random.random() < p_heavy)
                z = [randn() for _ in range(dim)]
                x1 = [0.0] * dim
                x2 = [0.0] * dim
                for i in range(dim):
                    step = sigma[i] * (cauchy() if heavy else z[i])
                    x1[i] = mean[i] + step
                    x2[i] = mean[i] - step
                clip_inplace(x1)
                clip_inplace(x2)

            f1 = evalf(x1)
            pop.append((f1, x1))
            if f1 < best:
                best, best_x = f1, x1[:]
                add_archive(best, best_x)

            if len(pop) < lam:
                f2 = evalf(x2)
                pop.append((f2, x2))
                if f2 < best:
                    best, best_x = f2, x2[:]
                    add_archive(best, best_x)

        pop.sort(key=lambda t: t[0])

        # Elite recombination + archive mix-in
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

        # Smooth update
        for i in range(dim):
            mean[i] = (1.0 - alpha_m) * mean[i] + alpha_m * new_mean[i]
            sigma[i] = (1.0 - alpha_s) * sigma[i] + alpha_s * new_sigma[i]
            if sigma[i] < sig_floor[i]:
                sigma[i] = sig_floor[i]
            elif sigma[i] > sig_ceil[i]:
                sigma[i] = sig_ceil[i]

        # Local exploitation: allocate a small evaluation budget
        # More when stagnating, but keep time-safe
        tl = time_left()
        if tl > 0.02:
            if best < last_best - 1e-15:
                no_improve = 0
                last_best = best
            else:
                no_improve += 1

            local_budget = 2 if no_improve < 10 else 6
            if tl < 0.08:
                local_budget = 1
            coord_line_search(local_budget)

        # Stagnation handling: inflate sigma, and occasional partial restart
        if no_improve > 0 and (no_improve % 14 == 0):
            for i in range(dim):
                sigma[i] = min(sig_ceil[i], sigma[i] * 1.30)

        if no_improve > 0 and (no_improve % 50 == 0):
            # restart mean near best/archive, and widen sigma moderately
            if archive and random.random() < 0.70:
                mean = archive[0][1][:]
            elif random.random() < 0.85:
                mean = best_x[:]
            else:
                mean = rand_uniform()
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.22 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0))
            no_improve = 0
