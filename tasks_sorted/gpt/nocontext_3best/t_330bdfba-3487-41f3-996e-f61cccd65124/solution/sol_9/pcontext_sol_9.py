import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (stdlib-only).

    What’s improved vs the best provided (#1):
      - Uses a *proper mixed global+local strategy* that is stronger under tight budgets:
          (1) fast low-discrepancy + random seeding
          (2) main optimizer: DE/current-to-best/1 with "p-best" selection (JADE-style idea)
          (3) periodic *diagonal CMA-ES-like* exploitation bursts around the best
          (4) very small coordinate/pattern polish as a final mile improver
      - Better bound handling: reflection (reduces boundary bias vs clamping)
      - Better diversity management: adaptive reinit of worst individuals + sigma/F/CR adaptation
      - Strict time guarding on every expensive block
    Returns:
        best (float): best objective value found within max_time seconds
    """

    # --------------------------- guards & bounds --------------------------- #
    if dim <= 0:
        return float("inf")
    if max_time is None:
        max_time = 0.0
    max_time = float(max_time)
    if max_time <= 0.0:
        # Try at least one evaluation if possible
        try:
            x0 = [float(bounds[i][0]) for i in range(dim)]
            y = float(func(x0))
            return y if (not math.isnan(y) and not math.isinf(y)) else float("inf")
        except Exception:
            return float("inf")

    bnds = []
    for lo, hi in bounds:
        lo = float(lo); hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        bnds.append((lo, hi))
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    ranges = [hi - lo for lo, hi in bnds]
    active = sum(1 for r in ranges if r > 0.0)
    mean_range = (sum(r for r in ranges if r > 0.0) / active) if active else 1.0
    if mean_range <= 0.0:
        mean_range = 1.0

    start = time.time()
    deadline = start + max_time

    # ------------------------------ helpers ------------------------------ #
    def reflect_into(v, lo, hi):
        if hi <= lo:
            return lo
        r = hi - lo
        x = (v - lo) % (2.0 * r)
        if x > r:
            x = 2.0 * r - x
        return lo + x

    def repair(x):
        return [reflect_into(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]

    def safe_eval(x):
        xx = repair(x)
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
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def norm(v):
        return math.sqrt(sum(t * t for t in v))

    # ------------------------- low-discrepancy seeding ------------------------- #
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

    def from_unit(u01):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            x[i] = lo if r <= 0.0 else lo + u01[i] * r
        return x

    def random_point():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            x[i] = lo if hi <= lo else (lo + random.random() * (hi - lo))
        return x

    # ------------------------- tiny local polish (pattern search) ------------------------- #
    def local_polish(x0, f0, t_end):
        x = x0[:]
        f = f0
        step = [0.06 * (ranges[i] if ranges[i] > 0 else 1.0) for i in range(dim)]
        min_step = [max(1e-12, 1e-9 * (ranges[i] if ranges[i] > 0 else 1.0)) for i in range(dim)]
        budget = 14 + 5 * dim
        used = 0

        while used < budget and time.time() < t_end:
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if used >= budget or time.time() >= t_end:
                    break
                if ranges[i] <= 0.0 or step[i] <= min_step[i]:
                    continue

                s = step[i]
                xp = x[:]; xp[i] += s
                fp, xp = safe_eval(xp); used += 1

                xm = x[:]; xm[i] -= s
                fm, xm = safe_eval(xm); used += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    improved = True

            if improved:
                for i in range(dim):
                    step[i] *= 1.22
            else:
                for i in range(dim):
                    step[i] *= 0.55
                if all((ranges[i] <= 0.0) or (step[i] <= min_step[i]) for i in range(dim)):
                    break
        return x, f

    # ------------------------- diagonal CMA-ES-like burst (exploit) ------------------------- #
    def cma_burst(center_x, sigma0, t_end, burst_steps=2):
        """
        Very small diagonal CMA-like burst around center_x.
        Time-bounded and returns updated (best, best_x, center_x, sigma0).
        """
        nonlocal best, best_x

        n = dim
        # Keep burst small so DE remains main engine
        lam = max(10, 4 + int(4 * math.log(n + 1.0)))
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w) if mu > 0 else 1.0
        w = [wi / wsum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w) if mu > 0 else 1.0

        c_sigma = min(0.35, (mu_eff + 2.0) / (n + mu_eff + 5.0))
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1e-12)) - 1.0) + c_sigma
        c_c = min(0.45, (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n))
        c1 = 0.10 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(0.25, 0.16 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        if c_mu < 0.0:
            c_mu = 0.0

        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)) if n > 1 else 1.0

        m = center_x[:]
        sigma = float(sigma0)
        if sigma <= 1e-15:
            sigma = 0.15 * mean_range
        diag = [1.0] * n
        p_sigma = [0.0] * n
        p_c = [0.0] * n

        for _ in range(max(1, burst_steps)):
            if time.time() >= t_end:
                break

            pop = []
            half = lam // 2
            for _k in range(half):
                if time.time() >= t_end:
                    break
                z = [randn() for _j in range(n)]
                for sgn in (1.0, -1.0):
                    y = [diag[j] * (sgn * z[j]) for j in range(n)]
                    x = [m[j] + sigma * y[j] for j in range(n)]
                    fx, x = safe_eval(x)
                    pop.append((fx, x, y))
                    if fx < best:
                        best, best_x = fx, x

            if len(pop) < mu:
                break
            pop.sort(key=lambda t: t[0])

            # mean update
            m_new = [0.0] * n
            for i in range(mu):
                wi = w[i]
                xi = pop[i][1]
                for j in range(n):
                    m_new[j] += wi * xi[j]

            y_w = [0.0] * n
            for i in range(mu):
                wi = w[i]
                yi = pop[i][2]
                for j in range(n):
                    y_w[j] += wi * yi[j]

            # p_sigma
            coeff = math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff)
            for j in range(n):
                p_sigma[j] = (1.0 - c_sigma) * p_sigma[j] + coeff * y_w[j]

            # sigma
            pnorm = norm(p_sigma)
            sigma *= math.exp((c_sigma / d_sigma) * (pnorm / (chi_n + 1e-12) - 1.0))

            # p_c
            h_sigma = 1.0 if pnorm < (1.9 + 0.35 * n) * chi_n else 0.0
            coeff_c = h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff)
            for j in range(n):
                p_c[j] = (1.0 - c_c) * p_c[j] + coeff_c * y_w[j]

            # diag update
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

            m = m_new

            # keep sigma sane for bounded problems
            if sigma < 1e-14 * mean_range:
                sigma = 1e-14 * mean_range
            if sigma > 1.5 * mean_range:
                sigma = 1.5 * mean_range

        return m, sigma

    # ------------------------- initialization: seed pool ------------------------- #
    best = float("inf")
    best_x = None

    bases = first_primes(dim)
    hal_idx = 1

    # Stronger seeding than #1 but still cheap
    seed_n = 12 + 6 * dim
    # If very tight time, cut seeds
    if max_time < 0.05:
        seed_n = max(6, 2 * dim + 6)

    seeds = []
    for _ in range(seed_n):
        if time.time() >= deadline:
            return best
        if random.random() < 0.65:
            x = from_unit(halton_point(hal_idx, bases))
            hal_idx += 1
        else:
            x = random_point()
        f, x = safe_eval(x)
        seeds.append((f, x))
        if f < best:
            best, best_x = f, x

    seeds.sort(key=lambda t: t[0])

    # ------------------------- main engine: DE current-to-pbest/1 ------------------------- #
    # Population size
    NP = max(18, min(70, 10 + 5 * int(math.log(dim + 3.0))))
    # Build initial population from best seeds + extra random/halton
    pop = []
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if i < len(seeds) and random.random() < 0.85:
            x = seeds[i][1][:]
            f = seeds[i][0]
        else:
            x = from_unit(halton_point(hal_idx, bases)) if random.random() < 0.55 else random_point()
            hal_idx += 1 if random.random() < 0.55 else 0
            f, x = safe_eval(x)
        pop.append([x, f])
        if f < best:
            best, best_x = f, x

    # Adaptive parameter memories (tiny JADE-style)
    Fm = 0.6
    CRm = 0.7

    def clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # Stagnation controls
    last_best = best
    no_improve = 0

    # Exploitation schedule
    cma_sigma = 0.20 * mean_range

    while time.time() < deadline:
        # Sort by fitness for p-best selection
        pop.sort(key=lambda t: t[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]

        # occasional tiny polish
        if best_x is not None and random.random() < 0.08:
            t_end = min(deadline, time.time() + (0.012 + 0.0015 * dim))
            xp, fp = local_polish(best_x, best, t_end)
            if fp < best:
                best, best_x = fp, xp

        # occasional CMA burst near best (key improvement)
        if best_x is not None and random.random() < 0.22:
            t_end = min(deadline, time.time() + (0.020 + 0.0018 * dim))
            burst_steps = 1 if dim > 30 else 2
            new_center, cma_sigma = cma_burst(best_x, cma_sigma, t_end, burst_steps=burst_steps)
            # If CMA found something, re-center best_x already updated via safe_eval in burst
            if best_x is not None:
                cma_sigma = max(1e-14 * mean_range, min(0.35 * mean_range, cma_sigma))

        # adaptive p-best fraction
        p = 0.15 + 0.25 * random.random()  # [0.15, 0.40]
        pbest_count = max(2, int(p * NP))

        # generation parameter sampling (lightweight)
        # F: perturb around Fm; CR around CRm
        successful_F = []
        successful_CR = []

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i][0], pop[i][1]

            # Sample F and CR
            # F ~ clipped normal-ish around Fm using sum of uniforms
            F = Fm + 0.15 * ((random.random() + random.random() + random.random()) - 1.5)
            F = 0.1 if F < 0.1 else (0.95 if F > 0.95 else F)

            CR = CRm + 0.20 * ((random.random() + random.random()) - 1.0)
            CR = clip01(CR)

            # pick pbest
            pbest_idx = random.randrange(pbest_count)
            xbestp = pop[pbest_idx][0]

            # pick r1, r2 distinct and not i
            r1 = r2 = i
            while r1 == i:
                r1 = random.randrange(NP)
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)
            xr1 = pop[r1][0]
            xr2 = pop[r2][0]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + F * (xbestp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            # crossover (bin)
            jrand = random.randrange(dim)
            u = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    u[j] = v[j]

            fu, u = safe_eval(u)

            if fu <= fi:
                pop[i][0] = u
                pop[i][1] = fu
                successful_F.append(F)
                successful_CR.append(CR)
                if fu < best:
                    best, best_x = fu, u

        # Update parameter memories from successes
        if successful_F:
            # Lehmer mean for F (more weight to larger useful steps)
            num = sum(f * f for f in successful_F)
            den = sum(successful_F)
            if den > 1e-12:
                Fm = 0.8 * Fm + 0.2 * (num / den)
        if successful_CR:
            CRm = 0.85 * CRm + 0.15 * (sum(successful_CR) / len(successful_CR))

        # Stagnation handling: reinit some worst individuals
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 7:
            no_improve = 0
            pop.sort(key=lambda t: t[1])
            k = max(2, NP // 4)
            for t in range(NP - k, NP):
                if time.time() >= deadline:
                    return best
                if best_x is not None and random.random() < 0.65:
                    # jitter around best
                    x = best_x[:]
                    for j in range(dim):
                        r = ranges[j]
                        if r > 0.0:
                            x[j] = x[j] + (0.08 + 0.18 * random.random()) * r * randn()
                    x = repair(x)
                else:
                    if random.random() < 0.55:
                        x = from_unit(halton_point(hal_idx, bases))
                        hal_idx += 1
                    else:
                        x = random_point()
                f, x = safe_eval(x)
                pop[t][0] = x
                pop[t][1] = f
                if f < best:
                    best, best_x = f, x

            # also slightly widen CMA sigma after a restart event
            cma_sigma = min(0.40 * mean_range, max(cma_sigma, 0.22 * mean_range))

    return best
