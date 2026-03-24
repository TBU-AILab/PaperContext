import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Main upgrades vs your previous versions:
    - Uses a "CMA-ES (sep)" core (diagonal covariance) => much cheaper than full CMA,
      so it can evaluate more candidates within the same max_time.
    - Heavy-tailed sampling (Student-t like via Cauchy mixture) for occasional long jumps.
    - Strict, robust bound handling via reflection (never sticks to borders like hard clip).
    - Adaptive restarts (IPOP-like population growth) when stalled.
    - Tiny, budget-aware local refinement around the best (coordinate + random direction).

    Returns:
        best (float): best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]

    # ----- helpers -----
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_uniform_vec():
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
            else:
                x[i] = lo[i] + random.random() * span[i]
        return x

    def reflect_to_bounds(x):
        # modulo reflection into [lo, hi] per coordinate
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
                continue
            a, b = lo[i], hi[i]
            w = b - a
            if w <= 0.0:
                x[i] = a
                continue
            y = x[i] - a
            y = y % (2.0 * w)
            if y > w:
                y = 2.0 * w - y
            x[i] = a + y

    def norm2(v):
        return sum(vi * vi for vi in v)

    def vec_norm(v):
        return math.sqrt(max(0.0, norm2(v)))

    def rand_unit_vec_active(adim):
        v = [random.gauss(0.0, 1.0) for _ in range(adim)]
        n = vec_norm(v)
        if n <= 0.0:
            j = random.randrange(adim)
            v = [0.0] * adim
            v[j] = 1.0
            return v
        inv = 1.0 / n
        return [inv * a for a in v]

    # Active dims list
    active_idx = [i for i in range(dim) if not fixed[i]]
    adim = len(active_idx)
    if adim == 0:
        x0 = [lo[i] for i in range(dim)]
        return safe_eval(x0)

    def pack_active(x_full):
        return [x_full[i] for i in active_idx]

    def unpack_active(x_act, template_full):
        x = template_full[:]
        for k, i in enumerate(active_idx):
            x[i] = x_act[k]
        return x

    # ----- initial seeding -----
    best = float("inf")
    best_x = rand_uniform_vec()

    # Do a small LHS-like seeding for coverage (cheap, no numpy)
    seed_n = min(24 + 4 * adim, 120)
    # strata per active dim
    strata = []
    for k, i in enumerate(active_idx):
        m = seed_n
        perm = list(range(m))
        random.shuffle(perm)
        vals = []
        for j in range(m):
            u = (perm[j] + random.random()) / m
            vals.append(lo[i] + u * span[i])
        strata.append(vals)

    for j in range(seed_n):
        if time.time() >= deadline:
            return best
        x = best_x[:]  # template
        for kk, ii in enumerate(active_idx):
            x[ii] = strata[kk][j]
        # add opposite half the time
        if (j & 1) == 1:
            for ii in active_idx:
                x[ii] = lo[ii] + (hi[ii] - x[ii])
        reflect_to_bounds(x)
        f = safe_eval(x)
        if f < best:
            best = f
            best_x = x[:]

    # ----- Sep-CMA-ES setup -----
    # mean in full space, but we update only active components
    m_full = best_x[:]
    m = pack_active(m_full)

    avg_span = sum(span[i] for i in active_idx) / max(1, adim)
    sigma = 0.30 * avg_span if avg_span > 0.0 else 1.0

    # diagonal covariance (variances per coordinate)
    Cdiag = [1.0] * adim  # starts as identity

    # CMA params based on adim
    lam0 = max(10, min(40, 4 + int(3 * math.log(adim + 1.0)) + 2 * adim))
    lam = lam0
    mu = lam // 2

    def make_weights(mu):
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        w = [wi / s for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)
        return w, mueff

    weights, mueff = make_weights(mu)

    cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
    cs = (mueff + 2.0) / (adim + mueff + 5.0)
    c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs

    ps = [0.0] * adim
    pc = [0.0] * adim
    gen = 0

    chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

    # Local search steps (full dims) around best
    ls_step = [0.06 * s for s in span]
    ls_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # Stagnation / restarts
    last_best = best
    stall = 0
    restarts = 0

    def sample_heavy():
        # 92% Gaussian, 8% Cauchy-like (heavy tail) to escape local minima
        if random.random() < 0.08:
            u = random.random()
            return math.tan(math.pi * (u - 0.5))  # Cauchy(0,1)
        return random.gauss(0.0, 1.0)

    while time.time() < deadline:
        gen += 1

        # Ensure parameters consistent if lam changes
        mu = lam // 2
        weights, mueff = make_weights(mu)

        # update some params that depend on mueff (cheap; keeps behavior stable across restarts)
        cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
        cs = (mueff + 2.0) / (adim + mueff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs

        # sample population
        pop = []
        # small global injection when stalled
        mix_global = 0.10 if stall > 10 else 0.03

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            if random.random() < mix_global:
                x_full = rand_uniform_vec()
                f = safe_eval(x_full)
                pop.append((f, pack_active(x_full), None))
                continue

            z = [sample_heavy() for _ in range(adim)]
            # y = sqrt(Cdiag) * z
            y = [math.sqrt(max(1e-30, Cdiag[k])) * z[k] for k in range(adim)]
            x_act = [m[k] + sigma * y[k] for k in range(adim)]
            x_full = unpack_active(x_act, m_full)
            reflect_to_bounds(x_full)
            f = safe_eval(x_full)
            pop.append((f, pack_active(x_full), y))  # store y (already scaled by sqrt(C))

        pop.sort(key=lambda t: t[0])

        if pop[0][0] < best:
            best = pop[0][0]
            best_x = unpack_active(pop[0][1], best_x)
            reflect_to_bounds(best_x)

        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        # recombination: new mean
        old_m = m[:]
        new_m = [0.0] * adim
        for i in range(mu):
            xi = pop[i][1]
            wi = weights[i]
            for k in range(adim):
                new_m[k] += wi * xi[k]
        m = new_m
        m_full = unpack_active(m, m_full)
        reflect_to_bounds(m_full)
        m = pack_active(m_full)

        # y_w = (m_new - m_old)/sigma
        invsigma = 1.0 / max(1e-30, sigma)
        y_w = [(m[k] - old_m[k]) * invsigma for k in range(adim)]

        # invsqrt(C) * y_w for diagonal is y_w / sqrt(Cdiag)
        invsqrtCy = [y_w[k] / math.sqrt(max(1e-30, Cdiag[k])) for k in range(adim)]

        # update ps
        coeff_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        for k in range(adim):
            ps[k] = (1.0 - cs) * ps[k] + coeff_ps * invsqrtCy[k]

        # hsig
        ps_norm = vec_norm(ps)
        left = ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen))
        hsig = 1.0 if left < (1.4 + 2.0 / (adim + 1.0)) * chiN else 0.0

        # update pc
        coeff_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for k in range(adim):
            pc[k] = (1.0 - cc) * pc[k] + hsig * coeff_pc * y_w[k]

        # update Cdiag (sep-CMA)
        # C = (1-c1-cmu)C + c1*pc^2 + cmu*sum(w_i * y_i^2) (+ small correction when hsig=0)
        factor = 1.0 - c1 - cmu
        if factor < 0.0:
            factor = 0.0

        # rank-mu term: use y_i = (x_i - old_m)/sigma in coordinates of sqrt(C) already?
        # We stored y as sqrt(C)*z; to get (x-old_m)/sigma in coordinate space: (xi-old_m)/sigma directly:
        # use xi from pop. That's simplest and stable.
        y_mu2 = [0.0] * adim
        for i in range(mu):
            xi = pop[i][1]
            wi = weights[i]
            for k in range(adim):
                yik = (xi[k] - old_m[k]) * invsigma
                y_mu2[k] += wi * (yik * yik)

        for k in range(adim):
            Ck = Cdiag[k]
            Ck *= factor
            Ck += c1 * (pc[k] * pc[k])
            if hsig == 0.0:
                Ck += c1 * cc * (2.0 - cc) * Cdiag[k] * 0.2  # mild correction
            Ck += cmu * y_mu2[k]
            # keep sane
            if Ck < 1e-30:
                Ck = 1e-30
            elif Ck > 1e30:
                Ck = 1e30
            Cdiag[k] = Ck

        # sigma update
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        # clamp sigma relative to spans
        base = avg_span if avg_span > 0.0 else 1.0
        if sigma < 1e-16 * base:
            sigma = 1e-16 * base
        if sigma > 2.5 * base + 1e-12:
            sigma = 2.5 * base + 1e-12

        # ----- tiny local refinement when stalled -----
        if stall > max(8, 2 * adim) and time.time() + 0.002 < deadline:
            trials = min(10 + dim, 26)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                if random.random() < 0.75:
                    d = random.randrange(dim)
                    if fixed[d]:
                        continue
                    sd = max(ls_min[d], ls_step[d])
                    x[d] += sd if random.random() < 0.5 else -sd
                else:
                    dir_act = rand_unit_vec_active(adim)
                    # scale by average active ls_step
                    ssum = 0.0
                    for ii in active_idx:
                        ssum += max(ls_min[ii], ls_step[ii])
                    scale = (ssum / adim) * (0.4 + 0.6 * random.random())
                    for kk, ii in enumerate(active_idx):
                        x[ii] += scale * dir_act[kk]
                reflect_to_bounds(x)
                fx = safe_eval(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    # pull mean to best
                    m_full = best_x[:]
                    m = pack_active(m_full)
                    stall = 0
                    break
                else:
                    for d in range(dim):
                        if not fixed[d] and ls_step[d] > ls_min[d]:
                            ls_step[d] *= 0.996

        # ----- restart if heavily stalled -----
        if stall > max(22, 5 * adim) and time.time() + 0.01 < deadline:
            stall = 0
            restarts += 1

            # IPOP-like: grow population modestly to increase exploration
            lam = min(120, int(lam0 * (1.0 + 0.35 * restarts)))
            if lam < 10:
                lam = 10

            # reset paths, loosen covariance, increase sigma
            ps = [0.0] * adim
            pc = [0.0] * adim
            Cdiag = [1.0] * adim

            m_full = best_x[:]
            m = pack_active(m_full)

            sigma = max(sigma, 0.45 * base)

            # also reset local steps a bit
            for d in range(dim):
                if not fixed[d]:
                    ls_step[d] = max(ls_step[d], 0.05 * span[d])

    return best
