import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (self-contained, no external libs).

    Improvement focus vs your best (#1, diagonal CMA-ES-like):
      - Proper *bound handling* via reflection (reduces harmful clamping bias).
      - *Two-phase search*: (A) fast global seeding (Halton+random) -> pick best seed
        (B) main optimizer = improved diagonal CMA-ES + occasional *polished local search*
        (coordinate pattern search) started from best points.
      - Better restart logic: multiple independent CMA "runs" with decreasing/increasing sigma,
        plus occasional random-direction kicks.
      - Uses cheap evaluation accounting and strict time guards.

    Returns:
        best (float): best objective found within max_time seconds.
    """

    # ------------------------- basic utilities ------------------------- #
    def sanitize_bounds(b):
        out = []
        for lo, hi in b:
            lo = float(lo); hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out

    def reflect_into(v, lo, hi):
        """Reflect value into [lo,hi] (better than clamp for stochastic optimizers)."""
        if hi <= lo:
            return lo
        # reflect with period 2R
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
        return math.sqrt(sum(t*t for t in v))

    # ------------------------- Halton (global seeding) ------------------------- #
    def first_primes(k):
        primes = []
        n = 2
        while len(primes) < k:
            is_p = True
            r = int(n ** 0.5)
            for p in primes:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(n)
            n += 1
        return primes

    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index, bases):
        return [vdc(index, b) for b in bases]

    def to_bounds(u):
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi <= lo:
                x.append(lo)
            else:
                x.append(lo + u[i] * (hi - lo))
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

    # ------------------------- small local search (polisher) ------------------------- #
    def local_polish(x0, f0, t_end):
        """
        Very lightweight coordinate/pattern search used as a 'polisher' on good candidates.
        Reflection bound handling is used implicitly by safe_eval.
        """
        x = x0[:]
        f = f0
        ranges = [bnds[i][1] - bnds[i][0] for i in range(dim)]
        step = [0.08 * r for r in ranges]
        min_step = [max(1e-12, 1e-8 * (r if r > 0 else 1.0)) for r in ranges]

        # small eval budget; time is the real constraint
        eval_budget = 20 + 6 * dim
        evals = 0

        while evals < eval_budget and time.time() < t_end:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= t_end or evals >= eval_budget:
                    break
                si = step[i]
                if ranges[i] <= 0 or si <= min_step[i]:
                    continue

                # Try +/- along coordinate
                xp = x[:]
                xp[i] = xp[i] + si
                fp, xp = safe_eval(xp); evals += 1

                xm = x[:]
                xm[i] = xm[i] - si
                fm, xm = safe_eval(xm); evals += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    improved = True

            if improved:
                for i in range(dim):
                    if ranges[i] > 0:
                        step[i] *= 1.25
            else:
                for i in range(dim):
                    step[i] *= 0.55
                # stop if all small
                tiny = True
                for i in range(dim):
                    if ranges[i] > 0 and step[i] > min_step[i]:
                        tiny = False
                        break
                if tiny:
                    break

        return x, f

    # ------------------------- setup ------------------------- #
    if dim <= 0:
        return float("inf")
    bnds = sanitize_bounds(bounds)
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    start = time.time()
    deadline = start + float(max_time)

    ranges = [bnds[i][1] - bnds[i][0] for i in range(dim)]
    active = sum(1 for r in ranges if r > 0)
    mean_range = (sum(r for r in ranges if r > 0) / active) if active else 1.0

    best = float("inf")
    best_x = None

    # ------------------------- Phase A: global seeding ------------------------- #
    bases = first_primes(dim)
    seed_pool = []  # (f, x)
    # spend small time but enough to not start badly
    seed_count = 10 + 6 * dim
    idx = 1
    for _ in range(seed_count):
        if time.time() >= deadline:
            return best
        if random.random() < 0.65:
            x0 = to_bounds(halton_point(idx, bases))
            idx += 1
        else:
            x0 = random_point()
        f0, x0 = safe_eval(x0)
        seed_pool.append((f0, x0))
        if f0 < best:
            best, best_x = f0, x0

    seed_pool.sort(key=lambda t: t[0])
    # keep a few elites for restarts/polish
    elites = seed_pool[:max(3, min(10, 2 + dim))]

    # ------------------------- Phase B: improved diagonal CMA-ES-like runs ------------------------- #
    n = dim

    # population size
    lam = max(10, 4 + int(6 * math.log(n + 1.0)))
    if lam % 2 == 1:
        lam += 1
    mu = lam // 2

    # log weights
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    w_sum = sum(w) if mu > 0 else 1.0
    w = [wi / w_sum for wi in w]
    mu_eff = 1.0 / sum(wi * wi for wi in w) if mu > 0 else 1.0

    c_sigma = min(0.35, (mu_eff + 2.0) / (n + mu_eff + 5.0))
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1e-12)) - 1.0) + c_sigma
    c_c = min(0.45, (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n))
    c1 = 0.12 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(0.30, 0.18 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    if c_mu < 0.0:
        c_mu = 0.0

    if n > 1:
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    else:
        chi_n = 1.0

    # restart schedule
    restart_id = 0
    no_improve_gens = 0
    last_best = best

    # initialize state from best seed
    m = elites[0][1][:] if elites else (best_x[:] if best_x is not None else random_point())
    sigma = 0.30 * mean_range if mean_range > 0 else 1.0
    diag = [1.0] * n
    p_sigma = [0.0] * n
    p_c = [0.0] * n

    while time.time() < deadline:
        # occasional polishing on best (very small slice)
        if best_x is not None and (random.random() < 0.10):
            t_end = min(deadline, time.time() + max(0.005, 0.02 + 0.002 * dim))
            xp, fp = local_polish(best_x, best, t_end)
            if fp < best:
                best, best_x = fp, xp
                last_best = best
                no_improve_gens = 0
                m = best_x[:]

        # generate population with mirrored sampling
        pop = []
        half = lam // 2

        for _ in range(half):
            if time.time() >= deadline:
                return best
            z = [randn() for _ in range(n)]
            for sgn in (1.0, -1.0):
                y = [diag[i] * (sgn * z[i]) for i in range(n)]
                x = [m[i] + sigma * y[i] for i in range(n)]
                f, x = safe_eval(x)
                pop.append((f, x, y))
                if f < best:
                    best, best_x = f, x

        pop.sort(key=lambda t: t[0])

        # update mean
        old_m = m[:]
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

        # p_sigma
        cs = c_sigma
        coeff = math.sqrt(cs * (2.0 - cs) * mu_eff)
        for j in range(n):
            p_sigma[j] = (1.0 - cs) * p_sigma[j] + coeff * y_w[j]

        # sigma adapt
        p_norm = norm(p_sigma)
        sigma *= math.exp((cs / d_sigma) * (p_norm / (chi_n + 1e-12) - 1.0))

        # p_c
        cc = c_c
        h_sigma = 1.0 if p_norm < (1.8 + 0.35 * n) * chi_n else 0.0
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

        # mild random-direction kick if extremely narrow or stuck on bounds
        if sigma < 1e-14 * (mean_range if mean_range > 0 else 1.0):
            sigma = 0.25 * (mean_range if mean_range > 0 else 1.0)

        # stagnation / restarts
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= 12:
            no_improve_gens = 0
            restart_id += 1

            # pick a restart center from elites/best/random
            r = random.random()
            if best_x is not None and r < 0.55:
                m = best_x[:]
            elif elites and r < 0.85:
                # pick among a few best seeds
                j = int((random.random() ** 2.2) * len(elites))
                m = elites[j][1][:]
            else:
                m = random_point()

            # jitter mean to avoid exact repeats
            for i in range(n):
                rr = ranges[i]
                if rr > 0:
                    m[i] = reflect_into(m[i] + random.uniform(-0.06, 0.06) * rr, bnds[i][0], bnds[i][1])

            # reset strategy
            diag = [1.0] * n
            p_sigma = [0.0] * n
            p_c = [0.0] * n

            # restart sigma schedule: alternate larger/smaller
            base = (mean_range if mean_range > 0 else 1.0)
            if restart_id % 2 == 1:
                sigma = 0.40 * base
            else:
                sigma = 0.18 * base

            # add a quick polish to the restart center
            f0, m2 = safe_eval(m)
            if f0 < best:
                best, best_x = f0, m2
                last_best = best
            m = m2

    return best
