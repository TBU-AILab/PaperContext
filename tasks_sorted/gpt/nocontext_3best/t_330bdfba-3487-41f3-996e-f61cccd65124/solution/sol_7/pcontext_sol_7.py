import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only, self-contained).

    Improvements vs your current best (#1):
      1) Hybrid global+local engine:
         - Global exploration using low-discrepancy (Halton) + random
         - Main optimizer: diagonal CMA-ES-like with mirrored sampling
         - Periodic *budgeted* trust-region local search (coordinate + random directions)
      2) Better bounded handling:
         - Reflection into bounds (reduces boundary bias vs clamp)
      3) Smarter restarts:
         - Multi-armed restart portfolio (different sigmas, occasional larger population)
         - Uses an archive of elites and restarts from diverse elites
      4) Noise/robustness features:
         - Re-evaluate best occasionally (very cheap) to reduce lucky outliers
         - Safe evaluation guards against NaN/Inf/exceptions

    Returns:
        best (float): best objective value found within max_time seconds.
    """

    # ----------------------------- basics ----------------------------- #
    if dim <= 0:
        return float("inf")

    # sanitize bounds
    bnds = []
    for lo, hi in bounds:
        lo = float(lo); hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        bnds.append((lo, hi))
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    t0 = time.time()
    deadline = t0 + float(max_time)

    # dimension scales
    ranges = [hi - lo for lo, hi in bnds]
    active = [r for r in ranges if r > 0.0]
    mean_range = (sum(active) / len(active)) if active else 1.0
    if mean_range <= 0.0:
        mean_range = 1.0

    # ----------------------------- utilities ----------------------------- #
    def reflect_into(v, lo, hi):
        """Reflect v into [lo,hi]. Better than clamping for stochastic steps."""
        if hi <= lo:
            return lo
        r = hi - lo
        x = (v - lo) % (2.0 * r)
        if x > r:
            x = 2.0 * r - x
        return lo + x

    def safe_eval(x):
        xx = [reflect_into(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
        try:
            y = func(xx)
            if y is None:
                return float("inf"), xx
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf"), xx
            return y, xx
        except Exception:
            return float("inf"), xx

    def randn():
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def norm(v):
        return math.sqrt(sum(a*a for a in v))

    # ----------------------------- Halton seeding ----------------------------- #
    def first_primes(k):
        ps = []
        n = 2
        while len(ps) < k:
            is_p = True
            r = int(n ** 0.5)
            for p in ps:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(n)
            n += 1
        return ps

    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index, bases):
        return [vdc(index, b) for b in bases]

    def to_bounds(u01):
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi <= lo:
                x.append(lo)
            else:
                x.append(lo + u01[i] * (hi - lo))
        return x

    def random_point():
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi <= lo:
                x.append(lo)
            else:
                x.append(lo + random.random() * (hi - lo))
        return x

    # ----------------------------- local trust-region (polisher) ----------------------------- #
    def local_trust_region(x0, f0, t_end):
        """
        Small budget local improvement:
          - coordinate steps + a few random directions
          - adaptive step sizes
        """
        x = x0[:]
        f = f0

        # step as fraction of domain
        step = [0.06 * (ranges[i] if ranges[i] > 0 else 1.0) for i in range(dim)]
        min_step = [max(1e-12, 1e-10 * (ranges[i] if ranges[i] > 0 else 1.0)) for i in range(dim)]

        budget = 18 + 7 * dim
        evals = 0

        while evals < budget and time.time() < t_end:
            improved = False

            # coordinate search
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= t_end or evals >= budget:
                    break
                if ranges[i] <= 0 or step[i] <= min_step[i]:
                    continue

                s = step[i]
                xp = x[:]; xp[i] += s
                fp, xp = safe_eval(xp); evals += 1

                xm = x[:]; xm[i] -= s
                fm, xm = safe_eval(xm); evals += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    improved = True

            # a few random-direction probes (helps rotated valleys)
            if time.time() < t_end and evals < budget:
                k_dir = 2 if dim <= 12 else 1
                for _ in range(k_dir):
                    if time.time() >= t_end or evals >= budget:
                        break
                    d = [randn() for _j in range(dim)]
                    dn = norm(d)
                    if dn <= 1e-12:
                        continue
                    inv = 1.0 / dn
                    for j in range(dim):
                        d[j] *= inv

                    # propose
                    s = 0.0
                    # use median-ish step scale
                    if dim > 0:
                        s = step[random.randrange(dim)]
                    xr = [x[j] + s * d[j] for j in range(dim)]
                    fr, xr = safe_eval(xr); evals += 1
                    if fr < f:
                        x, f = xr, fr
                        improved = True

            # adapt steps
            if improved:
                for i in range(dim):
                    if ranges[i] > 0:
                        step[i] *= 1.22
            else:
                for i in range(dim):
                    step[i] *= 0.55
                if all(step[i] <= min_step[i] for i in range(dim) if ranges[i] > 0):
                    break

        return x, f

    # ----------------------------- elite archive ----------------------------- #
    elite_k = max(8, min(25, 6 + 2 * dim))
    elites = []  # list of (f, x)

    def elite_add(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # ----------------------------- initial seeding ----------------------------- #
    best = float("inf")
    best_x = None

    bases = first_primes(dim)
    # spend small fixed slice on seeding, but time-guarded
    seed_n = 14 + 7 * dim
    idx = 1
    for _ in range(seed_n):
        if time.time() >= deadline:
            return best
        if random.random() < 0.70:
            x0 = to_bounds(halton_point(idx, bases)); idx += 1
        else:
            x0 = random_point()
        f0, x0 = safe_eval(x0)
        elite_add(f0, x0)
        if f0 < best:
            best, best_x = f0, x0

    if best_x is None:
        x0 = random_point()
        best, best_x = safe_eval(x0)
        elite_add(best, best_x)

    # occasional best re-check (for noisy functions)
    best_confirm = best
    best_confirm_x = best_x[:]
    last_confirm_t = time.time()

    # ----------------------------- CMA-ES-like core (diagonal) ----------------------------- #
    n = dim
    base_lam = max(10, 4 + int(6 * math.log(n + 1.0)))
    if base_lam % 2 == 1:
        base_lam += 1

    restart_id = 0
    no_improve = 0
    last_best = best

    # start at best elite
    m = elites[0][1][:] if elites else best_x[:]
    sigma = 0.28 * mean_range
    diag = [1.0] * n
    p_sigma = [0.0] * n
    p_c = [0.0] * n

    # expected length of N(0,I)
    chi_n = (math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))) if n > 1 else 1.0

    while time.time() < deadline:
        # occasionally confirm best (cheap, rare)
        if time.time() - last_confirm_t > 0.25 and best_x is not None:
            last_confirm_t = time.time()
            fchk, _ = safe_eval(best_x)
            # keep the better of stored/confirmed
            if fchk < best_confirm:
                best_confirm, best_confirm_x = fchk, best_x[:]
            # if stored best looks like an outlier (worse on re-eval), don't overwrite best,
            # but allow optimizer center to follow the more reliable value sometimes
            if fchk > best + 1e-12 and random.random() < 0.15:
                m = best_confirm_x[:]

        # local polish sometimes, but strictly time-bounded
        if best_x is not None and random.random() < 0.11:
            t_end = min(deadline, time.time() + max(0.006, 0.020 + 0.002 * dim))
            xp, fp = local_trust_region(best_x, best, t_end)
            if fp < best:
                best, best_x = fp, xp
                elite_add(best, best_x)
                m = best_x[:]
                last_best = best
                no_improve = 0

        # choose population size (occasionally larger on later restarts)
        lam = base_lam * (2 if (restart_id >= 2 and restart_id % 3 == 2) else 1)
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2

        # weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w) if mu > 0 else 1.0
        w = [wi / wsum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w) if mu > 0 else 1.0

        # learning rates
        c_sigma = min(0.35, (mu_eff + 2.0) / (n + mu_eff + 5.0))
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1e-12)) - 1.0) + c_sigma
        c_c = min(0.45, (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n))
        c1 = 0.11 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(0.30, 0.18 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        if c_mu < 0.0:
            c_mu = 0.0

        # ask: mirrored sampling, with small probability of heavy-tail jump
        pop = []
        half = lam // 2
        for _ in range(half):
            if time.time() >= deadline:
                return best

            if random.random() < 0.10:
                # Cauchy-ish for occasional basin escape
                z = []
                for _j in range(n):
                    u = random.random()
                    if u <= 1e-12: u = 1e-12
                    if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
                    z.append(math.tan(math.pi * (u - 0.5)))
            else:
                z = [randn() for _j in range(n)]

            for sgn in (1.0, -1.0):
                y = [diag[j] * (sgn * z[j]) for j in range(n)]
                x = [m[j] + sigma * y[j] for j in range(n)]
                f, x = safe_eval(x)
                pop.append((f, x, y))
                if f < best:
                    best, best_x = f, x
                    elite_add(best, best_x)

        if not pop:
            break
        pop.sort(key=lambda t: t[0])

        # tell: update mean
        m_old = m[:]
        m = [0.0] * n
        for i in range(mu):
            fi, xi, yi = pop[i]
            wi = w[i]
            for j in range(n):
                m[j] += wi * xi[j]

        # y_w
        y_w = [0.0] * n
        for i in range(mu):
            wi = w[i]
            yi = pop[i][2]
            for j in range(n):
                y_w[j] += wi * yi[j]

        # p_sigma and sigma
        cs = c_sigma
        coeff = math.sqrt(cs * (2.0 - cs) * mu_eff)
        for j in range(n):
            p_sigma[j] = (1.0 - cs) * p_sigma[j] + coeff * y_w[j]
        pnorm = norm(p_sigma)
        sigma *= math.exp((cs / d_sigma) * (pnorm / (chi_n + 1e-12) - 1.0))

        # p_c
        cc = c_c
        h_sigma = 1.0 if pnorm < (1.9 + 0.35 * n) * chi_n else 0.0
        coeff_c = h_sigma * math.sqrt(cc * (2.0 - cc) * mu_eff)
        for j in range(n):
            p_c[j] = (1.0 - cc) * p_c[j] + coeff_c * y_w[j]

        # diagonal covariance update
        for j in range(n):
            var = diag[j] * diag[j]
            var *= (1.0 - c1 - c_mu)
            var += c1 * (p_c[j] * p_c[j])
            if c_mu > 0.0:
                s = 0.0
                for i in range(mu):
                    yi = pop[i][2]
                    s += w[i] * (yi[j] * yi[j])
                var += c_mu * s
            if var < 1e-30:
                var = 1e-30
            diag[j] = math.sqrt(var)

        # keep sigma in sane domain-scale limits
        if sigma < 1e-14 * mean_range:
            sigma = 1e-14 * mean_range
        if sigma > 2.5 * mean_range:
            sigma = 2.5 * mean_range

        # stagnation detection
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # restart policy
        if no_improve >= 10:
            no_improve = 0
            restart_id += 1

            # restart center: diversify across elites (bias to better but not always best)
            if elites and random.random() < 0.92:
                j = int((random.random() ** 2.2) * len(elites))
                m = elites[j][1][:]
            else:
                m = random_point()

            # small jitter at restart
            for i in range(n):
                r = ranges[i]
                if r > 0:
                    m[i] = reflect_into(m[i] + random.uniform(-0.08, 0.08) * r, bnds[i][0], bnds[i][1])

            # reset strategy paths
            diag = [1.0] * n
            p_sigma = [0.0] * n
            p_c = [0.0] * n

            # sigma schedule: wide/narrow/very wide occasionally
            base = mean_range
            if restart_id % 5 == 0:
                sigma = 0.55 * base
            elif restart_id % 2 == 1:
                sigma = 0.34 * base
            else:
                sigma = 0.18 * base

            # evaluate restart center
            f0, m = safe_eval(m)
            elite_add(f0, m)
            if f0 < best:
                best, best_x = f0, m
                last_best = best

    return best
