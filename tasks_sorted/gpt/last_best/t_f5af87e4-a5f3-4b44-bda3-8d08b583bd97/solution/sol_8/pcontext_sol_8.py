import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only time-bounded minimizer (continuous, bounded, black-box).

    Upgrade vs previous diagonal-CMA:
      - Hybrid: (1) quasi-opposition + LHS-like seeding, (2) Diagonal CMA-ES core,
                (3) periodic local refinement (stochastic coordinate search),
                (4) adaptive restarts with mixed exploration/exploitation.
      - Faster N(0,1) generator (Box-Muller with cache).
      - Better bound handling (reflect).
      - More aggressive early exploration + late-stage intensification.

    Returns:
      best (float): best objective value found within max_time
    """
    t0 = time.perf_counter()
    deadline = t0 + float(max_time)

    # -------- guards --------
    if dim <= 0:
        try:
            v = float(func([]))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def reflect_inplace(x):
        # reflect each coordinate into [lo,hi]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            r = hi - lo
            p = 2.0 * r
            y = (v - lo) % p
            if y < 0.0:
                y += p
            if y <= r:
                x[i] = lo + y
            else:
                x[i] = hi - (y - r)

    def eval_f(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        return v if is_finite(v) else float("inf")

    # -------- fast randn (Box-Muller) with cache --------
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        # protect log(0)
        u1 = random.random()
        if u1 <= 1e-300:
            u1 = 1e-300
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        _spare = r * math.sin(theta)
        _has_spare = True
        return z0

    # -------- initialize: center + random + quasi-opposition + LHS-ish --------
    # Start from center
    x_best = [0.5 * (lows[i] + highs[i]) for i in range(dim)]
    reflect_inplace(x_best)
    f_best = eval_f(x_best)

    # LHS-like: stratify per dimension by permuting bins
    # Keep small to preserve time for actual search.
    seed_budget = 10 + 2 * dim
    seed_budget = min(seed_budget, 60)

    # create bin permutations
    bins = max(4, int(math.sqrt(seed_budget)))
    perms = []
    for i in range(dim):
        p = list(range(bins))
        random.shuffle(p)
        perms.append(p)

    def make_lhs_point(k):
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] == 0.0:
                x[i] = lows[i]
                continue
            b = perms[i][k % bins]
            # sample within bin with jitter
            u = (b + random.random()) / float(bins)
            x[i] = lows[i] + u * spans[i]
        return x

    # evaluate seeds (also quasi-opposition)
    for k in range(seed_budget):
        if time.perf_counter() >= deadline:
            return float(f_best)

        if k < bins:
            x = make_lhs_point(k)
        else:
            x = [lows[i] + random.random() * spans[i] if spans[i] != 0.0 else lows[i] for i in range(dim)]
        reflect_inplace(x)
        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, list(x)

        # quasi-opposite point (helps when optimum near boundary)
        if time.perf_counter() >= deadline:
            break
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        # slight randomization towards the opposite (quasi)
        for i in range(dim):
            if spans[i] != 0.0:
                alpha = 0.85 + 0.3 * random.random()  # [0.85, 1.15]
                xo[i] = x[i] + alpha * (xo[i] - x[i])
        reflect_inplace(xo)
        fxo = eval_f(xo)
        if fxo < f_best:
            f_best, x_best = fxo, list(xo)

    # -------- local refinement: stochastic coordinate / pattern search --------
    def local_refine(x0, f0, time_limit, initial_step_frac=0.08):
        """
        Simple bound-aware stochastic coordinate search with step halving.
        Uses greedy acceptance; cheap and robust for final polishing.
        """
        t_end = min(deadline, time.perf_counter() + time_limit)
        x = list(x0)
        fx = float(f0)

        # step per dimension proportional to span
        steps = [initial_step_frac * safe_spans[i] for i in range(dim)]
        min_step = 1e-12 * (sum(safe_spans) / dim)

        # shuffle coordinates occasionally
        coords = list(range(dim))
        it = 0
        while time.perf_counter() < t_end:
            it += 1
            if it % (2 * dim + 1) == 0:
                random.shuffle(coords)

            improved = False
            for i in coords:
                if time.perf_counter() >= t_end:
                    break
                if steps[i] < min_step:
                    continue

                # try + and -
                for sgn in (1.0, -1.0):
                    if time.perf_counter() >= t_end:
                        break
                    xn = list(x)
                    xn[i] = xn[i] + sgn * steps[i]
                    reflect_inplace(xn)
                    fn = eval_f(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        break  # move to next coord (first-improvement)
            if not improved:
                # reduce steps
                for i in range(dim):
                    steps[i] *= 0.5
                if max(steps) < min_step:
                    break
        return x, fx

    # -------- diagonal CMA-ES with restarts + periodic local refine --------
    base_lambda = max(10, 4 + int(3 * math.log(dim + 1.0)))
    restart = 0
    last_improve_t = time.perf_counter()
    stall_time = max(0.18, 0.22 * float(max_time))
    stall_gens = 35 + 8 * dim

    # schedule local refinements: do a short refine a few times
    refine_count = 0
    next_refine_at = t0 + 0.55 * float(max_time)  # start mid-way

    while time.perf_counter() < deadline:
        # mixed restart sizes: grow population but also occasionally reset smaller for exploitation
        if restart == 0:
            lam = base_lambda
        else:
            if restart % 3 == 0:
                lam = base_lambda  # exploitation restart
            else:
                lam = base_lambda * (2 ** min(6, restart - 1))  # exploration growth, capped

        mu = max(2, lam // 2)

        # recombination weights (log)
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]
        w2_sum = sum(wi * wi for wi in w)
        mu_eff = 1.0 / max(1e-12, w2_sum)

        # CMA parameters (diagonal)
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)

        c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))

        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # mean starts at best, with restart-dependent jitter (more on exploration restarts)
        m = list(x_best)
        jitter_frac = 0.02 if (restart % 3 == 0) else (0.06 + 0.02 * min(4, restart))
        for i in range(dim):
            if spans[i] != 0.0:
                m[i] += jitter_frac * safe_spans[i] * randn()
        reflect_inplace(m)

        # step size: larger on exploration restarts, smaller on exploitation restarts
        if restart % 3 == 0:
            sigma = 0.12
        else:
            sigma = 0.28 / (1.0 + 0.15 * restart)

        sigma_min = 1e-12
        sigma_max = 2.0

        D = [1.0] * dim
        p_sigma = [0.0] * dim
        p_c = [0.0] * dim

        gen = 0
        gens_no_improve = 0

        half = lam // 2

        while time.perf_counter() < deadline:
            gen += 1

            # periodic local refinement when time is ripe and we have a decent incumbent
            now = time.perf_counter()
            if refine_count < 3 and now >= next_refine_at and (deadline - now) > 0.05:
                # short local polish; scale time by remaining time
                remain = deadline - now
                tl = min(0.12 * float(max_time), 0.35 * remain)
                xr, fr = local_refine(x_best, f_best, time_limit=tl, initial_step_frac=0.06)
                refine_count += 1
                next_refine_at = now + 0.25 * float(max_time)  # next later
                if fr < f_best:
                    f_best, x_best = fr, xr
                    last_improve_t = time.perf_counter()
                # also recenter mean to polished best
                m = list(x_best)

            cand = []

            # mirrored sampling
            for _ in range(half):
                if time.perf_counter() >= deadline:
                    break
                z = [randn() for _ in range(dim)]
                x1 = [m[i] + sigma * D[i] * z[i] * safe_spans[i] for i in range(dim)]
                x2 = [m[i] - sigma * D[i] * z[i] * safe_spans[i] for i in range(dim)]
                reflect_inplace(x1)
                reflect_inplace(x2)
                f1 = eval_f(x1)
                f2 = eval_f(x2)
                cand.append((f1, x1, z))
                cand.append((f2, x2, [-zi for zi in z]))

            while len(cand) < lam and time.perf_counter() < deadline:
                z = [randn() for _ in range(dim)]
                x = [m[i] + sigma * D[i] * z[i] * safe_spans[i] for i in range(dim)]
                reflect_inplace(x)
                fx = eval_f(x)
                cand.append((fx, x, z))

            if not cand:
                break

            cand.sort(key=lambda t: t[0])

            # update best
            if cand[0][0] < f_best:
                f_best = cand[0][0]
                x_best = list(cand[0][1])
                last_improve_t = time.perf_counter()
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            # restart on stagnation
            if (time.perf_counter() - last_improve_t) > stall_time or gens_no_improve > stall_gens:
                break

            # recombination
            m_old = m
            z_w = [0.0] * dim
            m = [0.0] * dim
            for j in range(mu):
                wj = w[j]
                xj = cand[j][1]
                zj = cand[j][2]
                for i in range(dim):
                    m[i] += wj * xj[i]
                    z_w[i] += wj * zj[i]
            reflect_inplace(m)

            # p_sigma update
            fac_ps = math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff)
            for i in range(dim):
                p_sigma[i] = (1.0 - c_sigma) * p_sigma[i] + fac_ps * z_w[i]

            # sigma (CSA)
            norm_p = math.sqrt(sum(v * v for v in p_sigma))
            sigma *= math.exp((c_sigma / d_sigma) * (norm_p / chi_n - 1.0))
            if sigma < sigma_min: sigma = sigma_min
            if sigma > sigma_max: sigma = sigma_max

            # hsig
            denom = math.sqrt(max(1e-300, 1.0 - (1.0 - c_sigma) ** (2.0 * gen)))
            hsig = 1.0 if (norm_p / denom) < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0

            # p_c update
            fac_pc = hsig * math.sqrt(c_c * (2.0 - c_c) * mu_eff)
            for i in range(dim):
                denom_i = sigma * D[i] * safe_spans[i]
                y_i = 0.0 if denom_i == 0.0 else (m[i] - m_old[i]) / denom_i
                p_c[i] = (1.0 - c_c) * p_c[i] + fac_pc * y_i

            # D update (diagonal covariance)
            for i in range(dim):
                rank1 = p_c[i] * p_c[i]
                rankmu = 0.0
                for j in range(mu):
                    zj = cand[j][2]
                    rankmu += w[j] * (zj[i] * zj[i])
                v = (1.0 - c1 - c_mu) * (D[i] * D[i]) + c1 * rank1 + c_mu * rankmu
                if v < 1e-30:
                    v = 1e-30
                D[i] = math.sqrt(v)

            # regularize anisotropy
            if gen % 12 == 0:
                dmin = min(D)
                dmax = max(D)
                if dmin > 0.0:
                    r = dmax / dmin
                    if r > 1e5:
                        # compress extremes (soft)
                        target = r ** 0.2
                        for i in range(dim):
                            D[i] = max(dmin, min(dmax, D[i] / target))

        restart += 1
        if restart > 10:
            restart = 10  # cap

    return float(f_best)
