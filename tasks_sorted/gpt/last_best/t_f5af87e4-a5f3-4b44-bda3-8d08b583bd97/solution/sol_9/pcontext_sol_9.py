import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only time-bounded minimizer (continuous, bounded, black-box).

    Upgrade over previous diagonal CMA variants:
      - Hybrid global-local:
          (A) Fast space-filling + quasi-opposition seeding
          (B) Diagonal CMA-ES core with mirrored sampling
          (C) Periodic trust-region local search (coordinate + occasional random subspace)
      - Better normal RNG (Box-Muller with cache) for higher-quality steps
      - Active-ish diagonal adaptation (mild negative update pressure using worst points)
      - Smarter restarts: mixture of IPOP exploration + small-pop exploitation restarts
      - Robust bound handling via reflection

    Returns:
      best (float): best objective value found within max_time
    """
    t0 = time.perf_counter()
    deadline = t0 + float(max_time)

    # ---------- guards ----------
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
        # reflect each coordinate into [lo, hi]
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

    # ---------- fast randn (Box-Muller) with cache ----------
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        if u1 <= 1e-300:
            u1 = 1e-300
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        _spare = r * math.sin(th)
        _has_spare = True
        return z0

    # ---------- initialization: center + space-filling + quasi-opposition ----------
    x_best = [0.5 * (lows[i] + highs[i]) for i in range(dim)]
    reflect_inplace(x_best)
    f_best = eval_f(x_best)

    # Budget for seeds; keep modest to save time for optimization
    seed_budget = 12 + 3 * dim
    if seed_budget > 80:
        seed_budget = 80

    # LHS-ish stratified points (per-dimension bin permutations)
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
            u = (b + random.random()) / float(bins)
            x[i] = lows[i] + u * spans[i]
        return x

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

        # quasi-opposition (often helps in bounded domains)
        if time.perf_counter() >= deadline:
            break
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        for i in range(dim):
            if spans[i] != 0.0:
                a = 0.8 + 0.4 * random.random()  # [0.8, 1.2]
                xo[i] = x[i] + a * (xo[i] - x[i])
        reflect_inplace(xo)
        fxo = eval_f(xo)
        if fxo < f_best:
            f_best, x_best = fxo, list(xo)

    # ---------- local search (trust-region coordinate / random-subspace) ----------
    def local_search(x0, f0, time_limit, step_frac=0.08):
        """
        Greedy local improvement with shrinking trust region.
        Combines coordinate steps with occasional random subspace moves.
        """
        t_end = min(deadline, time.perf_counter() + max(0.0, time_limit))
        x = list(x0)
        fx = float(f0)

        # per-dimension step
        steps = [step_frac * safe_spans[i] for i in range(dim)]
        min_step = 1e-12 * (sum(safe_spans) / float(dim))

        coords = list(range(dim))
        it = 0
        while time.perf_counter() < t_end:
            it += 1
            improved = False

            # occasional random subspace move (helps on rotated valleys)
            if dim >= 3 and (it % (3 * dim + 7) == 0):
                k = 1 + int(random.random() * min(dim, 6))
                idx = random.sample(coords, k)
                xn = list(x)
                for j in idx:
                    if steps[j] >= min_step and spans[j] != 0.0:
                        xn[j] += (2.0 * random.random() - 1.0) * steps[j]
                reflect_inplace(xn)
                fn = eval_f(xn)
                if fn < fx:
                    x, fx = xn, fn
                    improved = True

            if not improved:
                if it % (2 * dim + 1) == 0:
                    random.shuffle(coords)

                for i in coords:
                    if time.perf_counter() >= t_end:
                        break
                    si = steps[i]
                    if si < min_step:
                        continue

                    # try + and -
                    xn = list(x)
                    xn[i] = x[i] + si
                    reflect_inplace(xn)
                    fn = eval_f(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        continue

                    xn = list(x)
                    xn[i] = x[i] - si
                    reflect_inplace(xn)
                    fn = eval_f(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True

            if not improved:
                # shrink trust region
                for i in range(dim):
                    steps[i] *= 0.5
                if max(steps) < min_step:
                    break

        return x, fx

    # ---------- diagonal CMA-ES with restarts ----------
    base_lambda = max(10, 4 + int(3 * math.log(dim + 1.0)))
    restart = 0

    last_improve_t = time.perf_counter()
    stall_time = max(0.18, 0.22 * float(max_time))
    stall_gens = 35 + 8 * dim

    # schedule a few local searches (earlier than before)
    refine_done = 0
    next_refine_at = t0 + 0.35 * float(max_time)

    while time.perf_counter() < deadline:
        # restart scheme: alternate exploitation and exploration
        if restart == 0:
            lam = base_lambda
            restart_mode = "exploit"
        else:
            if restart % 3 == 0:
                lam = base_lambda
                restart_mode = "exploit"
            else:
                lam = base_lambda * (2 ** min(6, restart - 1))
                restart_mode = "explore"

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

        # initialize mean at best with jitter
        m = list(x_best)
        if restart_mode == "exploit":
            jitter_frac = 0.012
        else:
            jitter_frac = 0.05 + 0.02 * min(4, restart)
        for i in range(dim):
            if spans[i] != 0.0:
                m[i] += jitter_frac * safe_spans[i] * randn()
        reflect_inplace(m)

        # step size
        sigma = 0.10 if restart_mode == "exploit" else 0.25 / (1.0 + 0.12 * restart)
        sigma_min = 1e-12
        sigma_max = 2.0

        # diagonal covariance factors
        D = [1.0] * dim
        p_sigma = [0.0] * dim
        p_c = [0.0] * dim

        gen = 0
        gens_no_improve = 0
        half = lam // 2

        # active diagonal update strength (very mild; keep stable)
        # uses worst points to push variance down in bad directions
        active_gamma = 0.12  # small to avoid instability

        while time.perf_counter() < deadline:
            gen += 1
            now = time.perf_counter()

            # periodic local refine (more frequent early-mid)
            if refine_done < 4 and now >= next_refine_at and (deadline - now) > 0.06:
                remain = deadline - now
                tl = min(0.10 * float(max_time), 0.30 * remain)
                xr, fr = local_search(x_best, f_best, time_limit=tl, step_frac=0.07 if restart_mode == "exploit" else 0.09)
                refine_done += 1
                next_refine_at = now + 0.20 * float(max_time)
                if fr < f_best:
                    f_best, x_best = fr, xr
                    last_improve_t = time.perf_counter()
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

            # best update
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

            # sigma update (CSA)
            norm_p = math.sqrt(sum(v * v for v in p_sigma))
            sigma *= math.exp((c_sigma / d_sigma) * (norm_p / chi_n - 1.0))
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            # hsig
            denom = math.sqrt(max(1e-300, 1.0 - (1.0 - c_sigma) ** (2.0 * gen)))
            hsig = 1.0 if (norm_p / denom) < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0

            # p_c update
            fac_pc = hsig * math.sqrt(c_c * (2.0 - c_c) * mu_eff)
            for i in range(dim):
                denom_i = sigma * D[i] * safe_spans[i]
                y_i = 0.0 if denom_i == 0.0 else (m[i] - m_old[i]) / denom_i
                p_c[i] = (1.0 - c_c) * p_c[i] + fac_pc * y_i

            # diagonal covariance update with mild active component
            # best mu contribute positively; worst mu contribute negatively (variance reduction)
            for i in range(dim):
                rank1 = p_c[i] * p_c[i]

                rankmu_pos = 0.0
                for j in range(mu):
                    zj = cand[j][2]
                    rankmu_pos += w[j] * (zj[i] * zj[i])

                # active: use worst mu points to reduce variance in directions that generate poor samples
                rankmu_neg = 0.0
                for j in range(mu):
                    zj = cand[-(j + 1)][2]
                    rankmu_neg += w[j] * (zj[i] * zj[i])

                v = (1.0 - c1 - c_mu) * (D[i] * D[i]) + c1 * rank1 + c_mu * rankmu_pos
                v -= active_gamma * c_mu * (rankmu_neg - 1.0)  # centered so isotropic doesn't collapse

                if v < 1e-30:
                    v = 1e-30
                D[i] = math.sqrt(v)

            # regularize anisotropy occasionally
            if gen % 12 == 0:
                dmin = min(D)
                dmax = max(D)
                if dmin > 0.0:
                    r = dmax / dmin
                    if r > 3e5:
                        g = r ** 0.22
                        for i in range(dim):
                            D[i] = max(dmin, min(dmax, D[i] / g))

        restart += 1
        if restart > 10:
            restart = 10

    return float(f_best)
