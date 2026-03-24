import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; stdlib only).

    Upgrade over your best (#1):
      - Uses a *bounded* parameterization: optimize in unconstrained "u" space,
        map to box via smooth sigmoid. This avoids clamp/reflection artifacts and
        makes the landscape friendlier to CMA-like adaptation near bounds.
      - Core optimizer: diagonal CMA-ES-like with mirrored sampling + rank weights.
      - Stronger *restart* scheme (IPOP-style: sigma schedule + occasional pop growth)
        and *heavy-tail* injections to escape local minima.
      - Lightweight *polisher* in u-space (coordinate pattern) only when promising.

    Returns:
        best (float): best objective found within max_time seconds.
    """

    # ------------------------- setup / bounds ------------------------- #
    if dim <= 0:
        return float("inf")

    bnds = []
    for lo, hi in bounds:
        lo = float(lo); hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        bnds.append((lo, hi))
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    start = time.time()
    deadline = start + float(max_time)

    ranges = [hi - lo for (lo, hi) in bnds]
    active = sum(1 for r in ranges if r > 0.0)
    mean_range = (sum(r for r in ranges if r > 0.0) / active) if active else 1.0
    if mean_range <= 0.0:
        mean_range = 1.0

    # ------------------------- numerics helpers ------------------------- #
    def isfinite(x):
        return not (math.isinf(x) or math.isnan(x))

    def randn():
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def norm(v):
        return math.sqrt(sum(t * t for t in v))

    # ------------------------- bound mapping: u (R^n) -> x (box) ------------------------- #
    # Stable sigmoid and logit.
    def sigmoid(t):
        # numerically stable logistic
        if t >= 0:
            e = math.exp(-t)
            return 1.0 / (1.0 + e)
        else:
            e = math.exp(t)
            return e / (1.0 + e)

    def logit(p):
        # p in (0,1)
        if p <= 1e-15:
            p = 1e-15
        elif p >= 1.0 - 1e-15:
            p = 1.0 - 1e-15
        return math.log(p / (1.0 - p))

    def u_to_x(u):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            if r <= 0.0:
                x[i] = lo
            else:
                s = sigmoid(u[i])
                x[i] = lo + r * s
        return x

    def x_to_u(x):
        u = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            if r <= 0.0:
                u[i] = 0.0
            else:
                p = (x[i] - lo) / r
                u[i] = logit(p)
        return u

    def safe_eval_u(u):
        x = u_to_x(u)
        try:
            y = func(x)
            if y is None:
                return float("inf"), x
            y = float(y)
            if not isfinite(y):
                return float("inf"), x
            return y, x
        except Exception:
            return float("inf"), x

    # ------------------------- seeding: Halton + random (in x, convert to u) ------------------------- #
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

    def to_bounds_from_unit(u01):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            if r <= 0.0:
                x[i] = lo
            else:
                x[i] = lo + r * u01[i]
        return x

    def random_x():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            if hi <= lo:
                x[i] = lo
            else:
                x[i] = lo + random.random() * (hi - lo)
        return x

    best = float("inf")
    best_x = None
    best_u = None

    bases = first_primes(dim)
    seed_count = 12 + 6 * dim
    elites = []  # list of (f, u)
    elite_k = max(4, min(12, 2 + dim))

    def elite_add(f, u):
        nonlocal elites
        elites.append((f, u))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites.pop()

    idx = 1
    for _ in range(seed_count):
        if time.time() >= deadline:
            return best
        if random.random() < 0.65:
            x0 = to_bounds_from_unit(halton_point(idx, bases))
            idx += 1
        else:
            x0 = random_x()

        u0 = x_to_u(x0)
        f0, _ = safe_eval_u(u0)
        elite_add(f0, u0)

        if f0 < best:
            best = f0
            best_x = u_to_x(u0)
            best_u = u0[:]

    if best_u is None:
        u0 = [0.0] * dim
        f0, _ = safe_eval_u(u0)
        best = f0
        best_x = u_to_x(u0)
        best_u = u0[:]
        elite_add(f0, u0)

    # ------------------------- local polisher in u-space ------------------------- #
    def local_polish_u(u0, f0, t_end):
        u = u0[:]
        f = f0

        # step in u-space (dimensionless); adapt
        step = [0.6] * dim
        min_step = [1e-4] * dim

        budget = 18 + 5 * dim
        evals = 0

        while evals < budget and time.time() < t_end:
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if time.time() >= t_end or evals >= budget:
                    break
                if step[i] <= min_step[i]:
                    continue

                s = step[i]

                up = u[:]
                up[i] += s
                fp, _ = safe_eval_u(up); evals += 1

                um = u[:]
                um[i] -= s
                fm, _ = safe_eval_u(um); evals += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        u, f = up, fp
                    else:
                        u, f = um, fm
                    improved = True

            if improved:
                for i in range(dim):
                    step[i] *= 1.25
            else:
                for i in range(dim):
                    step[i] *= 0.55
                if all(step[i] <= min_step[i] for i in range(dim)):
                    break

        return u, f

    # ------------------------- diagonal CMA-ES-like in u-space ------------------------- #
    n = dim

    # base population size (will be increased on some restarts)
    base_lam = max(10, 4 + int(6 * math.log(n + 1.0)))
    if base_lam % 2 == 1:
        base_lam += 1

    # restart control
    restart_id = 0
    no_improve_gens = 0
    last_best = best

    # initial mean from best elite
    elites.sort(key=lambda t: t[0])
    m = elites[0][1][:] if elites else best_u[:]

    # start sigma in u-space: moderate; independent of original ranges due to sigmoid map
    sigma = 1.2

    diag = [1.0] * n
    p_sigma = [0.0] * n
    p_c = [0.0] * n

    while time.time() < deadline:
        # occasional polish on current best
        if best_u is not None and random.random() < 0.10:
            t_end = min(deadline, time.time() + max(0.006, 0.018 + 0.002 * dim))
            up, fp = local_polish_u(best_u, best, t_end)
            if fp < best:
                best = fp
                best_u = up[:]
                best_x = u_to_x(best_u)
                last_best = best
                no_improve_gens = 0
                m = best_u[:]

        # IPOP-ish: sometimes grow population on restart
        lam = base_lam * (2 if (restart_id >= 2 and restart_id % 3 == 2) else 1)
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2

        # weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w) if mu > 0 else 1.0
        w = [wi / wsum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w) if mu > 0 else 1.0

        c_sigma = min(0.35, (mu_eff + 2.0) / (n + mu_eff + 5.0))
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1e-12)) - 1.0) + c_sigma
        c_c = min(0.45, (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n))
        c1 = 0.12 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(0.30, 0.18 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        if c_mu < 0.0:
            c_mu = 0.0

        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)) if n > 1 else 1.0

        # generate population (mirrored)
        pop = []
        half = lam // 2
        for _ in range(half):
            if time.time() >= deadline:
                return best

            # mixture: mostly Gaussian, sometimes heavy-tail (Cauchy-like) to jump basins
            if random.random() < 0.12:
                # Cauchy via tan(pi*(u-0.5))
                z = []
                for _j in range(n):
                    u = random.random()
                    # avoid exact 0/1
                    if u <= 1e-12: u = 1e-12
                    if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
                    z.append(math.tan(math.pi * (u - 0.5)))
            else:
                z = [randn() for _j in range(n)]

            for sgn in (1.0, -1.0):
                y = [diag[j] * (sgn * z[j]) for j in range(n)]
                u = [m[j] + sigma * y[j] for j in range(n)]
                f, _x = safe_eval_u(u)
                pop.append((f, u, y))
                if f < best:
                    best = f
                    best_u = u[:]
                    best_x = _x

        pop.sort(key=lambda t: t[0])

        # update mean
        m_old = m[:]
        m = [0.0] * n
        for i in range(mu):
            fi, ui, yi = pop[i]
            wi = w[i]
            for j in range(n):
                m[j] += wi * ui[j]

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

        # keep sigma in a sane range in u-space (avoid numerical blow-ups/collapse)
        if sigma < 1e-6:
            sigma = 1e-6
        elif sigma > 50.0:
            sigma = 50.0

        # update elite set (for restarts)
        # add a couple of best from this gen
        for i in range(min(3, len(pop))):
            elite_add(pop[i][0], pop[i][1])

        # stagnation / restart
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= 10:
            no_improve_gens = 0
            restart_id += 1

            # choose restart center: best_u or one of elites or new random in x->u
            r = random.random()
            if best_u is not None and r < 0.55:
                m = best_u[:]
            elif elites and r < 0.90:
                j = int((random.random() ** 2.0) * len(elites))
                m = elites[j][1][:]
            else:
                # random x then to u (ensures inside box)
                m = x_to_u(random_x())

            # strategy reset
            diag = [1.0] * n
            p_sigma = [0.0] * n
            p_c = [0.0] * n

            # sigma schedule: alternate wide/narrow; occasionally very wide
            if restart_id % 5 == 0:
                sigma = 3.0
            elif restart_id % 2 == 1:
                sigma = 1.8
            else:
                sigma = 0.9

            # small jitter in u-space
            for j in range(n):
                m[j] += 0.15 * randn()

    return best
