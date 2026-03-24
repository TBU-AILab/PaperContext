import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-6

    # ---------------- helpers ----------------
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

    # Halton for initialization / diversification
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

    # ---------------- archive ----------------
    archive = []  # (f, x)
    archive_cap = 12

    def norm_l1(a, b):
        d = 0.0
        for i in range(dim):
            d += abs(a[i] - b[i]) / span[i]
        return d / max(1, dim)

    def push_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])
        pruned = []
        for f, v in archive:
            ok = True
            for _, v2 in pruned:
                if norm_l1(v, v2) < 2e-3:
                    ok = False
                    break
            if ok:
                pruned.append((f, v))
            if len(pruned) >= archive_cap:
                break
        archive = pruned

    # ---------------- incumbent ----------------
    best_x = rand_vec()
    best = eval_f(best_x)
    push_archive(best, best_x)

    # ---------------- initialization (stronger) ----------------
    # More aggressive space-filling at the beginning, but time-safe
    k_hal = 1
    init_budget = max(60, 40 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline - eps_time:
            return best
        x = halton_vec(k_hal)
        k_hal += 1
        f = eval_f(x)
        if f < best:
            best, best_x = f, x
            push_archive(best, best_x)

    # start mean from best
    m = best_x[:]

    # ---------------- diagonal CMA-ES parameters ----------------
    lam = max(10, 4 + int(4 * math.log(dim + 1.0)))  # slightly larger than before
    mu = lam // 2

    # log-weights
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    w_sum = sum(w)
    w = [wi / w_sum for wi in w]
    w2_sum = sum(wi * wi for wi in w)
    mueff = 1.0 / w2_sum

    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # strategy state
    sigma = 0.35
    D = [1.0 for _ in range(dim)]  # sqrt(diag(C))
    ps = [0.0 for _ in range(dim)]
    pc = [0.0 for _ in range(dim)]

    # restart / stall
    stall = 0
    restarts = 0
    max_restarts = 20
    stall_limit = 10 + 4 * dim

    # ---------------- local search: (1+1)-ES + coordinate tries ----------------
    def local_polish(x0, f0, budget_evals):
        """Small budget local search around x0 with adaptive step per dimension."""
        x = x0[:]
        f = f0
        # per-dimension step sizes (relative to span)
        step = [0.05 * s for s in span]
        succ = [0 for _ in range(dim)]
        tries = [0 for _ in range(dim)]
        idx = list(range(dim))

        evals = 0
        while evals < budget_evals and time.time() < deadline - eps_time:
            random.shuffle(idx)
            improved = False

            for i in idx:
                if evals >= budget_evals or time.time() >= deadline - eps_time:
                    break
                lo, hi = bounds[i]
                orig = x[i]
                si = step[i]
                if si <= 0.0:
                    continue

                # try two directions (randomize order)
                if random.random() < 0.5:
                    dirs = (1.0, -1.0)
                else:
                    dirs = (-1.0, 1.0)

                best_dir_f = f
                best_dir_val = orig

                for d in dirs:
                    xi = reflect_1d(orig + d * si, lo, hi)
                    if xi == orig:
                        continue
                    xt = x[:]
                    xt[i] = xi
                    ft = eval_f(xt)
                    evals += 1
                    tries[i] += 1
                    if ft < best_dir_f:
                        best_dir_f = ft
                        best_dir_val = xi

                    if evals >= budget_evals or time.time() >= deadline - eps_time:
                        break

                if best_dir_f < f:
                    x[i] = best_dir_val
                    f = best_dir_f
                    succ[i] += 1
                    improved = True

                # 1/5-ish success adaptation (very cheap)
                if tries[i] >= 8:
                    rate = succ[i] / float(tries[i])
                    if rate > 0.25:
                        step[i] *= 1.4
                    elif rate < 0.10:
                        step[i] *= 0.7
                    # clamp steps
                    if step[i] > 0.35 * span[i]:
                        step[i] = 0.35 * span[i]
                    if step[i] < 1e-12 * span[i]:
                        step[i] = 1e-12 * span[i]
                    succ[i] = 0
                    tries[i] = 0

            if not improved:
                # isotropic tiny gaussian shake (keeps it from stalling on ridges)
                if time.time() >= deadline - eps_time:
                    break
                xt = x[:]
                for i in range(dim):
                    lo, hi = bounds[i]
                    xt[i] = reflect_1d(xt[i] + random.gauss(0.0, 0.15) * step[i], lo, hi)
                ft = eval_f(xt)
                evals += 1
                if ft < f:
                    x, f = xt, ft
        return f, x

    # ---------------- main loop ----------------
    while True:
        if time.time() >= deadline - eps_time:
            return best

        # Occasionally use an elite mean (helps basin-hopping)
        if archive and random.random() < 0.15:
            m = archive[random.randrange(len(archive))][1][:]

        pop = []  # (f, x, z)
        for _ in range(lam):
            if time.time() >= deadline - eps_time:
                return best
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            x = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                xi = m[i] + (sigma * D[i] * span[i]) * z[i]
                x[i] = reflect_1d(xi, lo, hi)
            fx = eval_f(x)
            pop.append((fx, x, z))

        pop.sort(key=lambda t: t[0])

        if pop[0][0] < best:
            best, best_x = pop[0][0], pop[0][1][:]
            push_archive(best, best_x)
            stall = 0
        else:
            stall += 1

        # recombination
        m_new = [0.0] * dim
        z_w = [0.0] * dim
        for j in range(mu):
            _, xj, zj = pop[j]
            wj = w[j]
            for i in range(dim):
                m_new[i] += wj * xj[i]
                z_w[i] += wj * zj[i]

        # update ps
        c_sigma = math.sqrt(cs * (2.0 - cs) * mueff)
        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + c_sigma * z_w[i]

        # CSA sigma update
        ps_norm = math.sqrt(sum(pi * pi for pi in ps))
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        if sigma < 1e-14:
            sigma = 1e-14
        elif sigma > 0.9:
            sigma = 0.9

        # hsig and pc
        denom_h = math.sqrt(max(1e-30, 1.0 - (1.0 - cs) ** (2.0)))
        hsig = 1.0 if (ps_norm / denom_h / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0
        c_c = math.sqrt(cc * (2.0 - cc) * mueff)
        for i in range(dim):
            denom = (sigma * D[i] * span[i])
            yi = (m_new[i] - m[i]) / denom if denom > 0 else 0.0
            pc[i] = (1.0 - cc) * pc[i] + hsig * c_c * yi

        # diagonal covariance update
        diagC = [D[i] * D[i] for i in range(dim)]
        rank_mu = [0.0] * dim
        for j in range(mu):
            _, xj, _ = pop[j]
            wj = w[j]
            for i in range(dim):
                denom = (sigma * D[i] * span[i])
                yi = (xj[i] - m[i]) / denom if denom > 0 else 0.0
                rank_mu[i] += wj * (yi * yi)

        a = 1.0 - c1 - cmu
        if a < 0.0:
            a = 0.0
        for i in range(dim):
            diagC[i] = a * diagC[i] + c1 * (pc[i] * pc[i]) + cmu * rank_mu[i]
            if diagC[i] < 1e-32:
                diagC[i] = 1e-32
            D[i] = math.sqrt(diagC[i])

        m = ensure_bounds(m_new)

        # ---- triggered local improvement bursts ----
        # If we stall, spend a small evaluation budget polishing best_x.
        if stall > 0 and (stall % (3 + dim // 2) == 0):
            # budget is small and time-safe; increases with dimension a bit
            polish_evals = 10 + 3 * dim
            f2, x2 = local_polish(best_x, best, polish_evals)
            if f2 < best:
                best, best_x = f2, x2
                push_archive(best, best_x)
                m = best_x[:]
                stall = 0

        # ---- restart logic ----
        # Restart not only on stall, but also if sigma collapses early.
        if (stall >= stall_limit or sigma < 5e-13) and restarts < max_restarts:
            restarts += 1
            stall = 0

            # choose seed: best or random elite, then mix with random
            if archive and random.random() < 0.8:
                seed = archive[0][1][:]
            else:
                seed = best_x[:]
            r = rand_vec()
            mix = 0.35 if random.random() < 0.5 else 0.6
            m = [mix * seed[i] + (1.0 - mix) * r[i] for i in range(dim)]
            ensure_bounds(m)

            # reset strategy state
            sigma = 0.45
            D = [1.0 for _ in range(dim)]
            ps = [0.0 for _ in range(dim)]
            pc = [0.0 for _ in range(dim)]

            # diversification burst via Halton + random
            burst = max(12, 6 * dim)
            for _ in range(burst):
                if time.time() >= deadline - eps_time:
                    return best
                if random.random() < 0.7:
                    x = halton_vec(k_hal)
                    k_hal += 1
                else:
                    x = rand_vec()
                f = eval_f(x)
                if f < best:
                    best, best_x = f, x
                    push_archive(best, best_x)
                    m = best_x[:]
