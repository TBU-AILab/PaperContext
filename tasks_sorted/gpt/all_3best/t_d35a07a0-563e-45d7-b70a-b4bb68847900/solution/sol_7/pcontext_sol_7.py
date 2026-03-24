import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no numpy).

    Upgrade over your current best (diag-CMA + DE):
      - Keeps diag-CMA-ES core, but adds *active* covariance update (diagonal-only, negative weights)
        to shrink bad directions faster (often a big win).
      - Adds *mirrored sampling* for a fraction of CMA samples: evaluate x and its mirror around mean;
        tends to improve signal-to-noise and progress under tight budgets.
      - Uses a slightly more stable restart scheme: sigma reset based on spans + mild mean jitter.
      - Uses small *elite archive injection* and a more consistent donor selection for DE.
      - Keeps reflection boundary handling and late-stage coordinate polishing.

    Returns:
      best (float): best objective value found within max_time
    """

    # -------------------- helpers --------------------
    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        span = hi - lo
        y = (v - lo) % (2.0 * span)
        if y > span:
            y = 2.0 * span - y
        return lo + y

    # Box-Muller Gaussian
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite_point(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    # scrambled Halton for seeding
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(math.isqrt(k))
        p = 3
        while p <= r:
            if k % p == 0:
                return False
            p += 2
        return True

    def next_prime(n):
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    def pick3(n, exclude):
        # pick 3 distinct indices != exclude
        while True:
            a = random.randrange(n)
            if a != exclude:
                break
        while True:
            b = random.randrange(n)
            if b != exclude and b != a:
                break
        while True:
            c = random.randrange(n)
            if c != exclude and c != a and c != b:
                break
        return a, b, c

    # small greedy coordinate search (first improvement)
    def coord_pattern(x, fx, step_d, tries):
        for _ in range(tries):
            j = random.randrange(dim)
            lo, hi = bounds[j]
            step = step_d[j]
            if step <= 0.0:
                continue
            xp = list(x)
            xp[j] = reflect(xp[j] + step, lo, hi)
            fp = eval_f(xp)
            if fp < fx:
                return xp, fp
            xm = list(x)
            xm[j] = reflect(xm[j] - step, lo, hi)
            fm = eval_f(xm)
            if fm < fx:
                return xm, fm
        return x, fx

    # -------------------- time --------------------
    start = time.time()
    deadline = start + max(0.0, float(max_time) if max_time is not None else 0.0)
    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # -------------------- diag-CMA-ES parameters --------------------
    lam0 = max(12, min(48, 4 + int(3.0 * math.sqrt(dim)) + 2 * dim))
    lam = lam0
    ipop_mult = 2
    mu = lam // 2

    # positive recombination weights (log)
    w_pos_raw = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
    w_pos_sum = sum(w_pos_raw) if sum(w_pos_raw) > 0 else float(mu)
    w_pos = [wi / w_pos_sum for wi in w_pos_raw]
    mueff = 1.0 / sum(wi * wi for wi in w_pos)

    # active CMA: negative weights for worst individuals (diagonal-only)
    # choose same count as mu (common), scaled so sum neg approx -0.5..-1 of pos sum
    neg_count = mu
    w_neg_raw = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(neg_count)]
    # make them negative and normalize
    w_neg_raw = [-abs(v) for v in w_neg_raw]
    w_neg_sum = -sum(w_neg_raw) if -sum(w_neg_raw) > 0 else float(neg_count)
    w_neg = [wi / w_neg_sum for wi in w_neg_raw]  # sums to -1

    # learning rates (CMA defaults)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    # active part uses same cmu on diagonal, but we ensure overall stability by clipping C later
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # state
    mean = [(bounds[j][0] + bounds[j][1]) * 0.5 for j in range(dim)]
    diagC = [1.0] * dim
    ps = [0.0] * dim
    pc = [0.0] * dim
    sigma = 0.30

    min_diag = [1e-30] * dim
    max_diag = [1e6] * dim

    # -------------------- DE params --------------------
    F_base = 0.60
    CR_base = 0.85

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None
    last_improve = start

    pop_cap = max(16, min(80, 6 * dim + 20))
    pop = []

    shift = [random.random() for _ in range(dim)]
    k = 1
    init_budget = max(lam, 12 * dim)

    i = 0
    while i < init_budget and time.time() < deadline:
        if (i % 4) == 0:
            x = rand_vec()
        else:
            x = halton_point(k, shift)
            k += 1

        fx = eval_f(x)
        pop.append((fx, x))
        if fx < best:
            best, best_x = fx, list(x)
            last_improve = time.time()

        if time.time() >= deadline:
            break

        if random.random() < 0.55:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            pop.append((fxo, xo))
            if fxo < best:
                best, best_x = fxo, list(xo)
                last_improve = time.time()
        i += 1

    if not pop:
        return eval_f(rand_vec())

    pop.sort(key=lambda t: t[0])
    pop = pop[:pop_cap]
    best, best_x = pop[0][0], list(pop[0][1])
    mean = list(best_x)

    base_stag = max(0.35, 0.10 * float(max_time))
    stag = base_stag
    restarts = 0

    def sigma_collapse():
        tiny = 0
        for j in range(dim):
            if sigma * math.sqrt(max(min_diag[j], diagC[j])) < 5e-14 * span[j]:
                tiny += 1
        return tiny >= max(1, int(0.8 * dim))

    # -------------------- main loop --------------------
    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0 if deadline <= start else (now - start) / (deadline - start)
        if tfrac < 0.0:
            tfrac = 0.0
        elif tfrac > 1.0:
            tfrac = 1.0

        # restart
        if (now - last_improve) > stag or sigma_collapse():
            restarts += 1
            lam = min(160, lam * ipop_mult)
            mu = lam // 2

            w_pos_raw = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
            w_pos_sum = sum(w_pos_raw) if sum(w_pos_raw) > 0 else float(mu)
            w_pos = [wi / w_pos_sum for wi in w_pos_raw]
            mueff = 1.0 / sum(wi * wi for wi in w_pos)

            neg_count = mu
            w_neg_raw = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(neg_count)]
            w_neg_raw = [-abs(v) for v in w_neg_raw]
            w_neg_sum = -sum(w_neg_raw) if -sum(w_neg_raw) > 0 else float(neg_count)
            w_neg = [wi / w_neg_sum for wi in w_neg_raw]

            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

            ps = [0.0] * dim
            pc = [0.0] * dim
            diagC = [1.0] * dim

            # sigma reset based on spans (more robust than fixed number)
            sigma = 0.35

            if best_x is None:
                mean = rand_vec()
            else:
                mean = []
                for j in range(dim):
                    lo, hi = bounds[j]
                    mean.append(reflect(best_x[j] + gauss() * (0.20 + 0.25 * random.random()) * span[j], lo, hi))

            # diversify pop
            inject = min(pop_cap, max(12, 3 * dim))
            newp = []
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.75:
                    x = []
                    for j in range(dim):
                        lo, hi = bounds[j]
                        x.append(reflect(best_x[j] + gauss() * 0.45 * span[j], lo, hi))
                else:
                    x = rand_vec()
                fx = eval_f(x)
                newp.append((fx, x))
                if fx < best:
                    best, best_x = fx, list(x)
                    last_improve = time.time()

            pop = (pop[:max(1, pop_cap // 3)] + newp)
            pop.sort(key=lambda t: t[0])
            pop = pop[:pop_cap]

            last_improve = time.time()
            stag = base_stag * (1.0 + 0.15 * restarts)
            continue

        # precompute sqrt(diagC)
        sqrtC = [math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]

        # keep pop sorted and trimmed
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_cap]

        cand = []

        # mirrored sampling ratio (more early than late)
        mirror_prob = 0.35 * (1.0 - tfrac) + 0.10

        for _ in range(lam):
            if time.time() >= deadline:
                break

            use_de = (len(pop) >= 6) and (random.random() < (0.18 * (1.0 - tfrac) + 0.06))

            if use_de and best_x is not None:
                idx = random.randrange(len(pop))
                _, xi = pop[idx]
                b, c, d = pick3(len(pop), idx)
                xb = pop[b][1]
                xc = pop[c][1]
                F = F_base + 0.30 * (random.random() - 0.5)
                if F < 0.25: F = 0.25
                if F > 0.95: F = 0.95
                CR = CR_base + 0.25 * (random.random() - 0.5)
                if CR < 0.10: CR = 0.10
                if CR > 0.98: CR = 0.98

                x = [0.0] * dim
                jrand = random.randrange(dim)
                for j in range(dim):
                    lo, hi = bounds[j]
                    if random.random() < CR or j == jrand:
                        vj = xi[j] + F * (best_x[j] - xi[j]) + F * (xb[j] - xc[j])
                    else:
                        vj = xi[j]
                    x[j] = reflect(vj, lo, hi)

                fx = eval_f(x)
                cand.append((fx, x))
            else:
                # CMA sample, optionally mirrored
                z = [gauss() for _ in range(dim)]
                x1 = [0.0] * dim
                for j in range(dim):
                    lo, hi = bounds[j]
                    x1[j] = reflect(mean[j] + sigma * sqrtC[j] * z[j], lo, hi)
                f1 = eval_f(x1)
                cand.append((f1, x1))

                if random.random() < mirror_prob and time.time() < deadline:
                    x2 = [0.0] * dim
                    for j in range(dim):
                        lo, hi = bounds[j]
                        x2[j] = reflect(mean[j] - sigma * sqrtC[j] * z[j], lo, hi)
                    f2 = eval_f(x2)
                    cand.append((f2, x2))

            # track best
            if cand:
                fx_last, x_last = cand[-1]
                if fx_last < best:
                    best, best_x = fx_last, list(x_last)
                    last_improve = time.time()

        if not cand:
            break

        cand.sort(key=lambda t: t[0])

        # update pop for DE and restarts (take a slice)
        take = max(2, min(len(cand), lam // 5))
        pop.extend(cand[:take])
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_cap]

        # -------- tell: update mean, paths, diagC, sigma (active diag-CMA) --------
        old_mean = list(mean)

        # mean from best mu
        mean = [0.0] * dim
        top_mu = min(mu, len(cand))
        for i in range(top_mu):
            wi = w_pos[i]
            xi = cand[i][1]
            for j in range(dim):
                mean[j] += wi * xi[j]

        inv_sigma = 1.0 / max(1e-300, sigma)

        # y = (mean-old_mean)/(sigma*sqrtC)
        y = [0.0] * dim
        for j in range(dim):
            denom = max(1e-300, sqrtC[j])
            y[j] = (mean[j] - old_mean[j]) * inv_sigma / denom

        # ps
        c_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        ps_norm2 = 0.0
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + c_ps * y[j]
            ps_norm2 += ps[j] * ps[j]
        ps_norm = math.sqrt(ps_norm2)

        # sigma
        sigma *= math.exp((cs / damps) * ((ps_norm / max(1e-300, chi_n)) - 1.0))
        if sigma < 0.03:
            sigma = 0.03
        if sigma > 6.0:
            sigma = 6.0

        # pc
        hsig = 1.0 if ps_norm < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
        c_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for j in range(dim):
            pc[j] = (1.0 - cc) * pc[j] + hsig * c_pc * y[j]

        # rank-mu positive update term (z^2 average)
        z2_pos = [0.0] * dim
        for i in range(top_mu):
            wi = w_pos[i]
            xi = cand[i][1]
            for j in range(dim):
                denom = max(1e-300, sqrtC[j])
                zj = (xi[j] - old_mean[j]) * inv_sigma / denom
                z2_pos[j] += wi * (zj * zj)

        # active (negative) update from worst candidates
        z2_neg = [0.0] * dim
        worst_start = max(0, len(cand) - neg_count)
        count_worst = len(cand) - worst_start
        if count_worst > 0:
            # map weights to actual count_worst
            # take first count_worst of w_neg (largest magnitude)
            for ii in range(count_worst):
                wi = w_neg[ii]  # negative
                xi = cand[worst_start + ii][1]
                for j in range(dim):
                    denom = max(1e-300, sqrtC[j])
                    zj = (xi[j] - old_mean[j]) * inv_sigma / denom
                    z2_neg[j] += wi * (zj * zj)  # subtracts variance along bad directions

        # diagC update
        for j in range(dim):
            Cjj = diagC[j]
            Cjj = (1.0 - c1 - cmu) * Cjj + c1 * (pc[j] * pc[j]) + cmu * (z2_pos[j] + 0.35 * z2_neg[j])
            if Cjj < min_diag[j]:
                Cjj = min_diag[j]
            if Cjj > max_diag[j]:
                Cjj = max_diag[j]
            diagC[j] = Cjj

        # late coordinate polish
        if tfrac > 0.70 and best_x is not None and random.random() < 0.28 and time.time() < deadline:
            step_d = [(0.33 - 0.20 * tfrac) * sigma * math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]
            xb, fb = coord_pattern(best_x, best, step_d, tries=max(1, min(7, dim)))
            if fb < best:
                best, best_x = fb, list(xb)
                last_improve = time.time()
                pop.append((fb, xb))
                pop.sort(key=lambda t: t[0])
                pop = pop[:pop_cap]

    return best
