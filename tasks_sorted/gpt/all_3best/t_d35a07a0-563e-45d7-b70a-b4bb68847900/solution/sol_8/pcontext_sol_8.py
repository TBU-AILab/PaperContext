import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (self-contained; no numpy).

    Improvements over the previous best (active diag-CMA + DE):
      1) Uses a *proper diagonal CMA-ES in an unconstrained (normalized) space* with a smooth
         box transform (tanh). This avoids boundary distortion from reflection/clamping and
         makes covariance learning/step-size control much more reliable.
      2) Keeps *active* diagonal covariance update (negative weights) + *mirrored* sampling.
      3) Adds *Two-Point step-size adaptation (TPA)* (cheap extra signal for sigma update)
         to stabilize progress in tight budgets.
      4) Uses *separable NES-style* mean-gradient fallback when CMA is stagnating (very cheap,
         helps on noisy / deceptive landscapes).
      5) Restarts: IPOP-ish + sigma reset + mean jitter in normalized space + archive reinjection.
      6) Late-stage: small coordinate + 2-opt like random subspace polish in normalized space.

    Returns:
      best (float): best objective value found within max_time
    """

    # -------------------- time --------------------
    start = time.time()
    deadline = start + max(0.0, float(max_time) if max_time is not None else 0.0)
    if dim <= 0:
        return float("inf")

    # -------------------- helpers --------------------
    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    # --- bound transform: y in R^d -> x in [lo,hi]^d via tanh ---
    # x = mid + halfspan * tanh(y)
    # inverse: y = atanh((x-mid)/halfspan)
    mids = []
    halfspans = []
    for j in range(dim):
        lo, hi = bounds[j]
        if hi < lo:
            lo, hi = hi, lo
        mid = 0.5 * (lo + hi)
        hs = 0.5 * (hi - lo)
        if hs <= 0.0:
            hs = 1.0
        mids.append(mid)
        halfspans.append(hs)

    def to_x(y):
        x = [0.0] * dim
        for j in range(dim):
            # tanh saturates safely
            x[j] = mids[j] + halfspans[j] * math.tanh(y[j])
        return x

    def atanh(u):
        # stable-ish atanh with clamp
        if u <= -0.999999999999:
            u = -0.999999999999
        elif u >= 0.999999999999:
            u = 0.999999999999
        return 0.5 * math.log((1.0 + u) / (1.0 - u))

    def to_y(x):
        y = [0.0] * dim
        for j in range(dim):
            u = (x[j] - mids[j]) / halfspans[j]
            y[j] = atanh(u)
        return y

    # Box-Muller Gaussian
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_x():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # scrambled Halton for seeding (same as before, but we store x then convert)
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

    def opposite_x(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    def pick3(n, exclude):
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

    # late polish: small coordinate tweaks in y-space (unconstrained, then map to x)
    def coord_polish_y(best_y, best_f, step_y, tries):
        y = list(best_y)
        f = best_f
        for _ in range(tries):
            j = random.randrange(dim)
            s = step_y[j]
            if s <= 0.0:
                continue

            yp = list(y)
            yp[j] += s
            fp = eval_f(to_x(yp))
            if fp < f:
                return yp, fp

            ym = list(y)
            ym[j] -= s
            fm = eval_f(to_x(ym))
            if fm < f:
                return ym, fm
        return y, f

    # random low-dim subspace polish (2-5 dims)
    def subspace_polish_y(best_y, best_f, step_scale, reps):
        y = list(best_y)
        f = best_f
        for _ in range(reps):
            k = 2 if dim < 5 else random.randint(2, min(5, dim))
            idxs = random.sample(range(dim), k)
            yp = list(y)
            for j in idxs:
                yp[j] += gauss() * step_scale
            fp = eval_f(to_x(yp))
            if fp < f:
                y, f = yp, fp
        return y, f

    # -------------------- initialization --------------------
    # We'll build an archive of good points in x and y
    archive_cap = max(40, min(200, 12 * dim + 40))
    archive = []  # list of (f, x, y)

    best = float("inf")
    best_x = None
    best_y = None
    last_improve = start

    # seed with Halton + random + opposition
    shift = [random.random() for _ in range(dim)]
    kseed = 1
    init_budget = max(16, 12 * dim)

    for i in range(init_budget):
        if time.time() >= deadline:
            return best
        if i % 4 == 0:
            x = rand_x()
        else:
            x = halton_point(kseed, shift)
            kseed += 1

        fx = eval_f(x)
        y = to_y(x)
        archive.append((fx, list(x), list(y)))
        if fx < best:
            best, best_x, best_y = fx, list(x), list(y)
            last_improve = time.time()

        if time.time() >= deadline:
            break
        if random.random() < 0.55:
            xo = opposite_x(x)
            fxo = eval_f(xo)
            yo = to_y(xo)
            archive.append((fxo, list(xo), list(yo)))
            if fxo < best:
                best, best_x, best_y = fxo, list(xo), list(yo)
                last_improve = time.time()

    if not archive:
        x = rand_x()
        return eval_f(x)

    archive.sort(key=lambda t: t[0])
    archive = archive[:archive_cap]
    best, best_x, best_y = archive[0][0], list(archive[0][1]), list(archive[0][2])

    # -------------------- diag-CMA-ES in y-space --------------------
    # population size
    lam0 = max(12, min(48, 4 + int(3.0 * math.sqrt(dim)) + 2 * dim))
    lam = lam0
    ipop_mult = 2
    mu = lam // 2

    def setup_weights(mu_):
        wpos_raw = [max(0.0, math.log(mu_ + 0.5) - math.log(i + 1.0)) for i in range(mu_)]
        s = sum(wpos_raw)
        if s <= 0.0:
            wpos = [1.0 / mu_] * mu_
        else:
            wpos = [v / s for v in wpos_raw]
        mueff_ = 1.0 / sum(wi * wi for wi in wpos)
        # negative weights for active update (same count)
        wneg_raw = [-(abs(math.log(mu_ + 0.5) - math.log(i + 1.0))) for i in range(mu_)]
        sn = -sum(wneg_raw)
        if sn <= 0.0:
            wneg = [-1.0 / mu_] * mu_
        else:
            wneg = [v / sn for v in wneg_raw]  # sums to -1
        return wpos, wneg, mueff_

    w_pos, w_neg, mueff = setup_weights(mu)

    # CMA coefficients
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # state in y-space
    mean = list(best_y)
    diagC = [1.0] * dim
    ps = [0.0] * dim
    pc = [0.0] * dim
    sigma = 0.9  # in y-space, this is a sensible initial global scale

    min_diag = [1e-30] * dim
    max_diag = [1e6] * dim

    # DE operator in x-space using archive as donor pool (still useful for global jumps)
    F_base = 0.60
    CR_base = 0.85

    pop_cap = max(16, min(90, 6 * dim + 24))  # donor pool size (from archive best slice)
    stag_base = max(0.35, 0.10 * float(max_time))
    stag = stag_base
    restarts = 0

    # TPA (two-point step-size adaptation) parameters
    tpa_c = 0.30

    def sigma_collapse():
        tiny = 0
        for j in range(dim):
            if sigma * math.sqrt(max(min_diag[j], diagC[j])) < 1e-12:
                tiny += 1
        return tiny >= max(1, int(0.85 * dim))

    # NES fallback (separable) step for mean when stuck
    def nes_mean_step():
        # Use a few antithetic samples to estimate gradient of expected fitness wrt mean.
        nonlocal mean, best, best_x, best_y, last_improve
        m = 6 if dim <= 10 else 8
        sqrtC = [math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]
        g = [0.0] * dim
        base = eval_f(to_x(mean))
        # If evaluating base improved, record it
        if base < best:
            best = base
            best_x = to_x(mean)
            best_y = list(mean)
            last_improve = time.time()

        for _ in range(m):
            z = [gauss() for _ in range(dim)]
            yp = [mean[j] + sigma * sqrtC[j] * z[j] for j in range(dim)]
            ym = [mean[j] - sigma * sqrtC[j] * z[j] for j in range(dim)]
            fp = eval_f(to_x(yp))
            fm = eval_f(to_x(ym))

            if fp < best:
                best = fp
                best_x = to_x(yp)
                best_y = list(yp)
                last_improve = time.time()
            if fm < best:
                best = fm
                best_x = to_x(ym)
                best_y = list(ym)
                last_improve = time.time()

            # gradient estimate: (fm - fp) * z
            # (if fp < fm, move toward +z)
            diff = (fm - fp)
            for j in range(dim):
                g[j] += diff * z[j]

        # normalize and apply small step
        gn = math.sqrt(sum(v * v for v in g))
        if gn > 1e-18:
            alpha = 0.12 / gn
            for j in range(dim):
                mean[j] += alpha * g[j]

    # -------------------- main loop --------------------
    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0 if deadline <= start else (now - start) / (deadline - start)
        if tfrac < 0.0:
            tfrac = 0.0
        elif tfrac > 1.0:
            tfrac = 1.0

        # build donor pool from archive
        archive.sort(key=lambda t: t[0])
        donors = archive[:min(pop_cap, len(archive))]
        if not donors:
            break

        # restart
        if (now - last_improve) > stag or sigma_collapse():
            restarts += 1
            lam = min(180, lam * ipop_mult)
            mu = lam // 2
            w_pos, w_neg, mueff = setup_weights(mu)

            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

            ps = [0.0] * dim
            pc = [0.0] * dim
            diagC = [1.0] * dim

            # sigma reset (in y-space): exploration early, smaller later
            sigma = 1.1 - 0.5 * tfrac
            if sigma < 0.25:
                sigma = 0.25

            # mean jitter around best_y in y-space
            if best_y is None:
                mean = [gauss() for _ in range(dim)]
            else:
                mean = [best_y[j] + gauss() * (0.35 + 0.25 * random.random()) for j in range(dim)]

            # archive reinjection: add a few global + near-best points
            inject = max(10, 3 * dim)
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.70:
                    y = [best_y[j] + gauss() * 0.9 for j in range(dim)]
                    x = to_x(y)
                else:
                    x = rand_x()
                    y = to_y(x)
                fx = eval_f(x)
                archive.append((fx, list(x), list(y)))
                if fx < best:
                    best, best_x, best_y = fx, list(x), list(y)
                    last_improve = time.time()

            archive.sort(key=lambda t: t[0])
            archive = archive[:archive_cap]
            stag = stag_base * (1.0 + 0.18 * restarts)
            continue

        sqrtC = [math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]

        # mirrored sampling more early than late
        mirror_prob = 0.40 * (1.0 - tfrac) + 0.08

        # TPA: evaluate two symmetric points along ps direction sometimes
        do_tpa = (random.random() < (0.25 + 0.25 * (1.0 - tfrac))) and (time.time() < deadline)

        cand = []  # list of (f, y, x, z(optional))
        # Add some DE candidates early for global movement
        de_trials = 0
        max_de = max(1, lam // 6)

        # --- ask ---
        for _ in range(lam):
            if time.time() >= deadline:
                break

            use_de = (de_trials < max_de) and (len(donors) >= 6) and (random.random() < (0.22 * (1.0 - tfrac) + 0.05))
            if use_de and best_x is not None:
                de_trials += 1
                idx = random.randrange(len(donors))
                _, xi, _ = donors[idx]
                a, b, c = pick3(len(donors), idx)
                xb = donors[b][1]
                xc = donors[c][1]

                F = F_base + 0.30 * (random.random() - 0.5)
                if F < 0.25: F = 0.25
                if F > 0.95: F = 0.95
                CR = CR_base + 0.25 * (random.random() - 0.5)
                if CR < 0.10: CR = 0.10
                if CR > 0.98: CR = 0.98

                x = [0.0] * dim
                jrand = random.randrange(dim)
                for j in range(dim):
                    if random.random() < CR or j == jrand:
                        x[j] = xi[j] + F * (best_x[j] - xi[j]) + F * (xb[j] - xc[j])
                    else:
                        x[j] = xi[j]
                    # clamp to bounds (DE is in x-space)
                    lo, hi = bounds[j]
                    if x[j] < lo: x[j] = lo
                    elif x[j] > hi: x[j] = hi

                fx = eval_f(x)
                y = to_y(x)
                cand.append((fx, y, x, None))
            else:
                z = [gauss() for _ in range(dim)]
                y1 = [mean[j] + sigma * sqrtC[j] * z[j] for j in range(dim)]
                x1 = to_x(y1)
                f1 = eval_f(x1)
                cand.append((f1, y1, x1, z))

                if random.random() < mirror_prob and time.time() < deadline:
                    y2 = [mean[j] - sigma * sqrtC[j] * z[j] for j in range(dim)]
                    x2 = to_x(y2)
                    f2 = eval_f(x2)
                    cand.append((f2, y2, x2, z))

            # track best
            fx_last, y_last, x_last, _ = cand[-1]
            if fx_last < best:
                best, best_x, best_y = fx_last, list(x_last), list(y_last)
                last_improve = time.time()

        if not cand:
            break

        # Add evaluated points to archive (keep small, biased to good)
        for fx, y, x, _ in cand:
            archive.append((fx, list(x), list(y)))
        archive.sort(key=lambda t: t[0])
        archive = archive[:archive_cap]

        cand.sort(key=lambda t: t[0])

        # --- tell (CMA updates) ---
        old_mean = list(mean)
        top_mu = min(mu, len(cand))

        # recombination on y
        mean = [0.0] * dim
        for i in range(top_mu):
            wi = w_pos[i]
            yi = cand[i][1]
            for j in range(dim):
                mean[j] += wi * yi[j]

        inv_sigma = 1.0 / max(1e-300, sigma)

        # normalized mean step
        y_step = [0.0] * dim
        for j in range(dim):
            y_step[j] = (mean[j] - old_mean[j]) * inv_sigma / max(1e-300, sqrtC[j])

        # ps update
        c_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        ps_norm2 = 0.0
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + c_ps * y_step[j]
            ps_norm2 += ps[j] * ps[j]
        ps_norm = math.sqrt(ps_norm2)

        # base sigma update (CSA)
        sigma *= math.exp((cs / damps) * ((ps_norm / max(1e-300, chi_n)) - 1.0))

        # optional TPA correction (stabilizes sigma)
        if do_tpa and best_y is not None and time.time() < deadline:
            # direction is ps in y-space (already normalized-ish); make two probes
            # scale by sigma to be meaningful
            norm_ps = math.sqrt(sum(v * v for v in ps))
            if norm_ps > 1e-12:
                diry = [ps[j] / norm_ps for j in range(dim)]
                yp = [old_mean[j] + tpa_c * sigma * sqrtC[j] * diry[j] for j in range(dim)]
                ym = [old_mean[j] - tpa_c * sigma * sqrtC[j] * diry[j] for j in range(dim)]
                fp = eval_f(to_x(yp))
                fm = eval_f(to_x(ym))
                if fp < best:
                    best, best_x, best_y = fp, to_x(yp), list(yp)
                    last_improve = time.time()
                if fm < best:
                    best, best_x, best_y = fm, to_x(ym), list(ym)
                    last_improve = time.time()
                # if plus better than minus, slightly increase sigma; else decrease
                if fp < fm:
                    sigma *= 1.05
                elif fm < fp:
                    sigma *= 0.95

        if sigma < 0.08:
            sigma = 0.08
        if sigma > 6.0:
            sigma = 6.0

        # pc update
        hsig = 1.0 if ps_norm < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
        c_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for j in range(dim):
            pc[j] = (1.0 - cc) * pc[j] + hsig * c_pc * y_step[j]

        # rank-mu positive
        z2_pos = [0.0] * dim
        for i in range(top_mu):
            wi = w_pos[i]
            yi = cand[i][1]
            for j in range(dim):
                zj = (yi[j] - old_mean[j]) * inv_sigma / max(1e-300, sqrtC[j])
                z2_pos[j] += wi * (zj * zj)

        # active negative from worst
        neg_count = top_mu
        z2_neg = [0.0] * dim
        worst_start = max(0, len(cand) - neg_count)
        count_worst = len(cand) - worst_start
        for ii in range(count_worst):
            wi = w_neg[ii]  # negative
            yi = cand[worst_start + ii][1]
            for j in range(dim):
                zj = (yi[j] - old_mean[j]) * inv_sigma / max(1e-300, sqrtC[j])
                z2_neg[j] += wi * (zj * zj)

        # diagC update
        for j in range(dim):
            Cjj = diagC[j]
            Cjj = (1.0 - c1 - cmu) * Cjj + c1 * (pc[j] * pc[j]) + cmu * (z2_pos[j] + 0.30 * z2_neg[j])
            if Cjj < min_diag[j]:
                Cjj = min_diag[j]
            elif Cjj > max_diag[j]:
                Cjj = max_diag[j]
            diagC[j] = Cjj

        # stagnation help: occasional NES mean step (cheap)
        if (now - last_improve) > (0.45 * stag) and random.random() < 0.20 and time.time() < deadline:
            nes_mean_step()

        # late polish in y-space
        if tfrac > 0.70 and best_y is not None and time.time() < deadline:
            if random.random() < 0.30:
                step_y = [(0.18 - 0.10 * (tfrac - 0.70) / 0.30) * sigma * math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]
                yb, fb = coord_polish_y(best_y, best, step_y, tries=max(1, min(8, dim)))
                if fb < best:
                    best, best_y = fb, list(yb)
                    best_x = to_x(best_y)
                    last_improve = time.time()
                    archive.append((best, list(best_x), list(best_y)))
                    archive.sort(key=lambda t: t[0])
                    archive = archive[:archive_cap]
            if random.random() < 0.20:
                yb, fb = subspace_polish_y(best_y, best, step_scale=0.25 * sigma, reps=2)
                if fb < best:
                    best, best_y = fb, list(yb)
                    best_x = to_x(best_y)
                    last_improve = time.time()
                    archive.append((best, list(best_x), list(best_y)))
                    archive.sort(key=lambda t: t[0])
                    archive = archive[:archive_cap]

    return best
