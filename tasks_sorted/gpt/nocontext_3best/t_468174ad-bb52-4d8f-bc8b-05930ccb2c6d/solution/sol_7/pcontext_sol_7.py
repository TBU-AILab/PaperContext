import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (stdlib only).

    Main improvements vs. the provided best:
      - Strong initialization: Halton (low-discrepancy) + opposition + a few randoms
      - Hybrid optimizer:
          (1) Diagonal ES/CEM-like global search with antithetic sampling + elite archive
          (2) Strong local refinement: coordinate line-search (multi-step) + occasional random-direction steps
      - Better stagnation handling: adaptive sigma inflation + controlled restarts
      - Time-safe evaluation: avoids starting expensive blocks near deadline

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

    # ---------- helpers ----------
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
        # approx N(0,1)
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy():
        u = random.random()
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        # clamp tails (still heavy)
        if v > 60.0:
            v = 60.0
        elif v < -60.0:
            v = -60.0
        return v

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def rand_uniform():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    # ----- Halton sequence for better seeding -----
    def first_primes(n):
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

    def halton_index(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    primes = first_primes(dim)

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_index(k, primes[i])
            x[i] = lows[i] + u * (highs[i] - lows[i])
        return x

    # ---------- initialization ----------
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    clip_inplace(center)
    best_x = center[:]
    best = evalf(best_x)

    archive = []  # (f, x)
    archive_cap = max(24, 5 * dim)

    def add_archive(f, x):
        archive.append((f, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > archive_cap:
            del archive[archive_cap:]

    add_archive(best, best_x)

    # time-aware seeding
    init_n = max(24, 12 * dim)
    if max_time < 0.25:
        init_n = max(8, 3 * dim)
    elif max_time > 2.0:
        init_n = max(init_n, 36 + 14 * dim)

    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best

        x = halton_point(k)
        # small jitter
        for i in range(dim):
            x[i] += (random.random() - 0.5) * 0.004 * spans[i]
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

        # a few pure randoms early
        if k <= max(4, dim // 2):
            xr = rand_uniform()
            fr = evalf(xr)
            if fr < best:
                best, best_x = fr, xr[:]
                add_archive(best, best_x)

    # ---------- global ES/CEM-like model ----------
    mean = best_x[:]
    sigma = [0.35 * (highs[i] - lows[i]) for i in range(dim)]
    for i in range(dim):
        if sigma[i] <= 0.0:
            sigma[i] = 1.0

    sig_floor = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    sig_ceil = [0.95 * (highs[i] - lows[i]) if highs[i] != lows[i] else 1.0 for i in range(dim)]

    alpha_m = 0.25
    alpha_s = 0.18

    # ---------- local search state ----------
    # coordinate step for 1D local line search
    lstep = [0.22 * (highs[i] - lows[i]) for i in range(dim)]
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

    def coord_line_search(evals_budget):
        """
        Stronger-than-pattern-search 1D coordinate line search:
          - try +/-h
          - if improvement, attempt another step in same direction (greedy)
          - if no improvement, try a smaller step
        """
        nonlocal mean
        used = 0
        while used < evals_budget and time.time() < deadline:
            i = random.randrange(dim)
            h = lstep[i]
            if h <= lstep_min[i]:
                continue

            base = best_x[:]
            f0 = best

            # randomize direction order
            if random.random() < 0.5:
                dirs = (1.0, -1.0)
            else:
                dirs = (-1.0, 1.0)

            improved = False
            best_dir = 0.0

            # try +/-h
            for sgn in dirs:
                cand = base[:]
                cand[i] = max(lows[i], min(highs[i], cand[i] + sgn * h))
                fc = evalf(cand); used += 1
                if try_update(cand, fc):
                    improved = True
                    best_dir = sgn
                    break
                if used >= evals_budget or time.time() >= deadline:
                    break

            if time.time() >= deadline or used >= evals_budget:
                break

            if improved:
                # take one more step in same direction (if possible)
                cand2 = best_x[:]
                cand2[i] = max(lows[i], min(highs[i], cand2[i] + best_dir * h))
                fc2 = evalf(cand2); used += 1
                try_update(cand2, fc2)

                lstep[i] = min(lstep[i] * 1.30, (highs[i] - lows[i]) if highs[i] != lows[i] else lstep[i])
                # softly sync mean toward best
                for j in range(dim):
                    mean[j] = 0.88 * mean[j] + 0.12 * best_x[j]
            else:
                # try smaller step (one shot)
                hh = h * 0.35
                cand = base[:]
                cand[i] = max(lows[i], min(highs[i], cand[i] + (hh if random.random() < 0.5 else -hh)))
                fc = evalf(cand); used += 1
                if try_update(cand, fc):
                    lstep[i] = min(lstep[i] * 1.10, (highs[i] - lows[i]) if highs[i] != lows[i] else lstep[i])
                else:
                    lstep[i] = max(lstep[i] * 0.60, lstep_min[i])

    def random_direction_try(evals_budget):
        used = 0
        while used < evals_budget and time.time() < deadline:
            # normalized random direction
            d = [randn() for _ in range(dim)]
            n2 = sum(v * v for v in d)
            if n2 <= 0.0:
                continue
            inv = 1.0 / math.sqrt(n2)
            d = [v * inv for v in d]

            # scale tied to sigma (trust-ish)
            avg_sig = sum(sigma) / dim
            scale = avg_sig * (0.10 + 0.60 * random.random())
            cand = [best_x[i] + scale * d[i] for i in range(dim)]
            clip_inplace(cand)
            fc = evalf(cand); used += 1
            try_update(cand, fc)

    # ---------- main loop ----------
    no_improve = 0
    last_best = best

    while True:
        if time.time() >= deadline:
            return best

        tl = time_left()
        if tl <= 0.0:
            return best

        # population size (time-aware)
        base_lam = 12 + 5 * dim
        if tl < 0.07:
            lam = max(6, 2 * dim)
        elif tl > 1.0 and dim <= 18:
            lam = int(base_lam * 1.9)
        else:
            lam = base_lam

        # exploration schedule
        frac_left = max(0.0, min(1.0, tl / max(1e-9, max_time)))
        p_uniform = 0.01 + 0.10 * frac_left
        p_heavy = 0.04 + 0.20 * frac_left

        # occasional recenter
        if archive and random.random() < 0.12:
            mean = archive[0][1][:]
        elif random.random() < 0.10:
            mean = best_x[:]

        # generate population with antithetic pairing
        pop = []
        half = (lam + 1) // 2
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

        # elites with archive mix-in
        mu = max(5, lam // 5)
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

        # recombine mean
        new_mean = [0.0] * dim
        for w, (f, x) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                new_mean[i] += ww * x[i]

        # recombine variance (diagonal)
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

        # local exploitation (time-safe)
        if best < last_best - 1e-15:
            no_improve = 0
            last_best = best
        else:
            no_improve += 1

        tl = time_left()
        if tl > 0.02:
            # more local budget when stagnating
            local_budget = 2 if no_improve < 10 else 8
            if tl < 0.07:
                local_budget = 1
            coord_line_search(local_budget)

            # occasional random-direction probe (helps in rotated valleys)
            if tl > 0.05 and random.random() < (0.15 if no_improve < 12 else 0.30):
                random_direction_try(1 if tl < 0.10 else 2)

        # stagnation handling
        if no_improve > 0 and (no_improve % 12 == 0):
            for i in range(dim):
                sigma[i] = min(sig_ceil[i], sigma[i] * 1.28)

        if no_improve > 0 and (no_improve % 45 == 0):
            # controlled restart: move mean to best or good archive; widen sigma moderately
            if archive and random.random() < 0.75:
                mean = archive[0][1][:]
            elif random.random() < 0.85:
                mean = best_x[:]
            else:
                mean = rand_uniform()
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.20 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0))
            no_improve = 0
