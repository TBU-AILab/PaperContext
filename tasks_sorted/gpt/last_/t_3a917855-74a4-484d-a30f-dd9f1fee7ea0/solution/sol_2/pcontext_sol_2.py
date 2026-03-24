import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect_1d(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if t <= w else (hi - (t - w))

    span = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    def ensure_bounds(x):
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = clamp(x[i], lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # Halton sequence for space-filling initialization
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
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

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(dim)

    def halton_vec(k):
        x = []
        for i in range(dim):
            u = halton(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------- incumbent ----------
    best_x = rand_vec()
    best = eval_f(best_x)

    # ---------- small elite archive ----------
    archive = []  # (f, x)
    archive_cap = 10

    def push_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])
        # prune near-duplicates (normalized L1)
        pruned = []
        for f, v in archive:
            ok = True
            for f2, v2 in pruned:
                d = 0.0
                for i in range(dim):
                    d += abs(v[i] - v2[i]) / span[i]
                if d < 2e-3 * dim:
                    ok = False
                    break
            if ok:
                pruned.append((f, v))
            if len(pruned) >= archive_cap:
                break
        archive = pruned

    push_archive(best, best_x)

    # ---------- (diagonal) CMA-ES core ----------
    # We keep only diagonal covariance (per-dim stds) -> fast, robust, no numpy.
    # Default population sizes
    lam = max(8, 4 + int(3 * math.log(dim + 1.0)))
    mu = lam // 2

    # recombination weights (log)
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    w_sum = sum(w)
    w = [wi / w_sum for wi in w]
    w2_sum = sum(wi * wi for wi in w)
    mueff = 1.0 / w2_sum

    # strategy parameters (diagonal CMA-ES)
    # constants adapted from standard CMA-ES choices
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    # diagonal update can use larger c_mu
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    # expected length of N(0,I)
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # initial mean: best among a short Halton batch (better than pure random)
    k_hal = 1
    halton_budget = max(40, 30 * dim)
    for _ in range(halton_budget):
        if time.time() >= deadline:
            return best
        x = halton_vec(k_hal)
        k_hal += 1
        f = eval_f(x)
        if f < best:
            best, best_x = f, x
            push_archive(best, best_x)

    m = best_x[:]  # mean

    # initial global step-size and per-dim stds
    # start moderately large; will adapt quickly
    sigma = 0.30
    D = [1.0 for _ in range(dim)]  # diagonal scaling factors (sqrt of diag(C))

    # evolution paths (diagonal)
    ps = [0.0 for _ in range(dim)]
    pc = [0.0 for _ in range(dim)]

    # restart / stall logic
    stall = 0
    stall_limit = 12 + 3 * dim  # time-safe
    restarts = 0
    max_restarts = 12  # safety

    # occasional cheap local coordinate polish around current best
    def coord_polish(x0, f0, step_rel):
        x = x0[:]
        f = f0
        idx = list(range(dim))
        random.shuffle(idx)
        for i in idx:
            if time.time() >= deadline:
                break
            lo, hi = bounds[i]
            step = step_rel * span[i]
            if step <= 0.0:
                continue

            orig = x[i]
            # try + then -
            xp = x[:]
            xp[i] = reflect_1d(orig + step, lo, hi)
            fp = eval_f(xp) if xp[i] != orig else float("inf")

            xm = x[:]
            xm[i] = reflect_1d(orig - step, lo, hi)
            fm = eval_f(xm) if xm[i] != orig else float("inf")

            if fp < f and fp <= fm:
                x, f = xp, fp
            elif fm < f:
                x, f = xm, fm
        return f, x

    # --------- main loop ----------
    while True:
        if time.time() >= deadline:
            return best

        # produce offspring
        # store (f, x, z) where z is the normalized step in isotropic coords
        pop = []

        # choose parent mean sometimes from archive to escape bad basin
        if archive and random.random() < 0.10:
            m = archive[random.randrange(len(archive))][1][:]

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            x = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                # x = m + sigma * D * z (then reflect into bounds)
                xi = m[i] + (sigma * D[i] * span[i]) * z[i]
                x[i] = reflect_1d(xi, lo, hi)

            fx = eval_f(x)
            pop.append((fx, x, z))

        pop.sort(key=lambda t: t[0])

        # track global best
        if pop[0][0] < best:
            best, best_x = pop[0][0], pop[0][1][:]
            push_archive(best, best_x)
            stall = 0
        else:
            stall += 1

        # recombine to new mean (using x directly; z used for path update)
        m_new = [0.0] * dim
        z_w = [0.0] * dim
        for j in range(mu):
            fj, xj, zj = pop[j]
            wj = w[j]
            for i in range(dim):
                m_new[i] += wj * xj[i]
                z_w[i] += wj * zj[i]

        # update ps (conjugate evolution path for sigma), diagonal approx
        # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * z_w
        c_sigma = math.sqrt(cs * (2.0 - cs) * mueff)
        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + c_sigma * z_w[i]

        # update sigma (CSA)
        ps_norm = math.sqrt(sum(pi * pi for pi in ps))
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        # keep sigma reasonable
        if sigma < 1e-12:
            sigma = 1e-12
        elif sigma > 0.75:
            sigma = 0.75

        # hsig for covariance/path update
        # use standard criterion
        hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0)) / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0

        # update pc
        # pc = (1-cc)pc + hsig*sqrt(cc(2-cc)mueff) * (m_new - m)/(sigma*D*span)
        c_c = math.sqrt(cc * (2.0 - cc) * mueff)
        for i in range(dim):
            denom = (sigma * D[i] * span[i])
            yi = (m_new[i] - m[i]) / denom if denom > 0 else 0.0
            pc[i] = (1.0 - cc) * pc[i] + hsig * c_c * yi

        # update diagonal covariance (D) using rank-1 and rank-mu on y_i^2
        # Here y = (x - m)/(sigma*D*span) in old coordinate
        # diagC <- (1-c1-cmu)diagC + c1*pc^2 + cmu*sum w*y^2
        # We store D = sqrt(diagC); so update diagC then sqrt.
        diagC = [D[i] * D[i] for i in range(dim)]

        # precompute rank-mu term
        rank_mu = [0.0] * dim
        for j in range(mu):
            _, xj, _ = pop[j]
            wj = w[j]
            for i in range(dim):
                denom = (sigma * D[i] * span[i])
                yi = (xj[i] - m[i]) / denom if denom > 0 else 0.0
                rank_mu[i] += wj * (yi * yi)

        a = 1.0 - c1 - cmu
        # slightly protect against negative due to numeric drift
        if a < 0.0:
            a = 0.0

        for i in range(dim):
            diagC[i] = a * diagC[i] + c1 * (pc[i] * pc[i]) + cmu * rank_mu[i]
            if diagC[i] < 1e-30:
                diagC[i] = 1e-30
            D[i] = math.sqrt(diagC[i])

        m = ensure_bounds(m_new)

        # occasional polish (cheap, helps on separable/sharp problems)
        if stall > 0 and (stall % (4 + dim // 3) == 0):
            step_rel = min(0.10, max(1e-6, 0.25 * sigma))
            f2, x2 = coord_polish(best_x, best, step_rel)
            if f2 < best:
                best, best_x = f2, x2
                push_archive(best, best_x)
                stall = 0

        # restart if stuck or sigma collapsed
        if (stall >= stall_limit or sigma < 5e-12) and restarts < max_restarts:
            restarts += 1
            stall = 0

            # restart mean from mixture: elite + random, also reset paths
            if archive and random.random() < 0.75:
                seed = archive[0][1][:]
            else:
                seed = best_x[:]
            r = rand_vec()
            mix = 0.2 if random.random() < 0.5 else 0.5
            m = [mix * seed[i] + (1.0 - mix) * r[i] for i in range(dim)]
            ensure_bounds(m)

            # reset strategy parameters
            sigma = 0.35
            D = [1.0 for _ in range(dim)]
            ps = [0.0 for _ in range(dim)]
            pc = [0.0 for _ in range(dim)]

            # small burst of Halton samples post-restart for diversification
            burst = max(10, 5 * dim)
            for _ in range(burst):
                if time.time() >= deadline:
                    return best
                x = halton_vec(k_hal)
                k_hal += 1
                f = eval_f(x)
                if f < best:
                    best, best_x = f, x
                    push_archive(best, best_x)
                    m = best_x[:]
#
#
