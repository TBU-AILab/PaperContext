import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only time-bounded minimizer.

    Improvements over the provided DE:
      - Switch to CMA-ES style search (very strong on continuous black-box problems)
      - Diagonal covariance adaptation (fast, robust, cheap in stdlib)
      - Mirrored sampling + elitist update
      - Automatic restarts with increasing population (IPOP-like) on stagnation
      - Keeps best-ever solution across restarts
      - Reflection bound handling

    Returns:
      best (float): best objective value found within max_time
    """
    t0 = time.perf_counter()
    deadline = t0 + float(max_time)

    # ----------- basic guards -----------
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

    def is_finite(x):
        return not (math.isnan(x) or math.isinf(x))

    def reflect_inplace(x):
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

    # ~N(0,1) via 12 uniforms - 6
    def randn():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def clip01(a):
        if a < 0.0: return 0.0
        if a > 1.0: return 1.0
        return a

    # ----------- initialization (best-known) -----------
    x_best = [0.5 * (lows[i] + highs[i]) for i in range(dim)]
    reflect_inplace(x_best)
    f_best = eval_f(x_best)

    # small extra seeds to avoid pathological centers
    for _ in range(min(6, dim + 2)):
        if time.perf_counter() >= deadline:
            return float(f_best)
        x = [lows[i] + random.random() * spans[i] for i in range(dim)]
        reflect_inplace(x)
        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x

    # ----------- CMA-ES (diagonal) with restarts -----------
    # restart parameters
    base_lambda = max(8, 4 + int(3 * math.log(dim + 1.0)))
    restart = 0
    best_at_restart = f_best
    last_improve_t = time.perf_counter()

    # stagnation thresholds
    stall_time = max(0.15, 0.18 * float(max_time))   # time without improvement
    stall_gens = 40 + 10 * dim                       # or generations without improvement

    while time.perf_counter() < deadline:
        # population size increases on restart (IPOP-ish)
        lam = base_lambda * (2 ** restart)
        mu = max(2, lam // 2)

        # recombination weights (log)
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]
        w2_sum = sum(wi * wi for wi in w)
        mu_eff = 1.0 / max(1e-12, w2_sum)

        # learning rates (diagonal CMA-ish)
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)

        # diagonal covariance adaptation rates
        c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))

        # expected norm of N(0,I)
        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # initial mean from current global best (good exploitation)
        m = list(x_best)

        # initial step size proportional to bounds (restart-dependent)
        sigma0 = 0.22 / (1.0 + 0.35 * restart)
        sigma = sigma0

        # diagonal covariance as vector of std dev multipliers
        D = [1.0] * dim

        # evolution paths
        p_sigma = [0.0] * dim
        p_c = [0.0] * dim

        gen = 0
        gens_no_improve = 0

        # for mirrored sampling
        half = lam // 2

        # hard floor/ceiling on sigma (relative to span)
        # use average span scale for sigma control
        avg_span = sum(safe_spans) / float(dim)
        sigma_min = 1e-12
        sigma_max = 1.5

        while time.perf_counter() < deadline:
            gen += 1

            # sample population
            cand = []
            # mirrored pairs improve stability
            for k in range(half):
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

            # if odd lambda, add one extra
            while len(cand) < lam and time.perf_counter() < deadline:
                z = [randn() for _ in range(dim)]
                x = [m[i] + sigma * D[i] * z[i] * safe_spans[i] for i in range(dim)]
                reflect_inplace(x)
                fx = eval_f(x)
                cand.append((fx, x, z))

            if not cand:
                break

            cand.sort(key=lambda t: t[0])

            # global best update
            if cand[0][0] < f_best:
                f_best = cand[0][0]
                x_best = list(cand[0][1])
                last_improve_t = time.perf_counter()
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            # stop criteria for this restart (stagnation)
            now = time.perf_counter()
            if (now - last_improve_t) > stall_time or gens_no_improve > stall_gens:
                break

            # compute new mean (in x-space) and z-mean (in isotropic space)
            m_old = m
            z_w = [0.0] * dim
            m = [0.0] * dim
            for j in range(mu):
                fj, xj, zj = cand[j]
                wj = w[j]
                for i in range(dim):
                    m[i] += wj * xj[i]
                    z_w[i] += wj * zj[i]
            reflect_inplace(m)

            # update p_sigma (diagonal approx; use D^-1 * (m-m_old)/(sigma*span))
            for i in range(dim):
                # y in isotropic coordinates approximated by z_w
                p_sigma[i] = (1.0 - c_sigma) * p_sigma[i] + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_w[i]

            # sigma update (CSA)
            norm_p = math.sqrt(sum(v * v for v in p_sigma))
            sigma *= math.exp((c_sigma / d_sigma) * (norm_p / chi_n - 1.0))
            if sigma < sigma_min: sigma = sigma_min
            if sigma > sigma_max: sigma = sigma_max

            # heuristic hsig
            hsig = 1.0 if (norm_p / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * gen))) < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0

            # update p_c (in x-space normalized by sigma*D*span)
            for i in range(dim):
                # direction in x-space: (m - m_old) / (sigma * D * span)
                denom = (sigma * D[i] * safe_spans[i])
                y_i = 0.0 if denom == 0.0 else (m[i] - m_old[i]) / denom
                p_c[i] = (1.0 - c_c) * p_c[i] + hsig * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_i

            # update diagonal covariance (D^2) using rank-1 and rank-mu in y-space
            # y vectors in normalized coordinates: (x - m_old)/(sigma*D*span) approx via z
            # Use z from sampled (already isotropic), with a small stabilization from p_c.
            for i in range(dim):
                # rank-1 component
                rank1 = p_c[i] * p_c[i]
                # rank-mu component (weighted)
                rankmu = 0.0
                for j in range(mu):
                    zj = cand[j][2]
                    rankmu += w[j] * (zj[i] * zj[i])
                # update variance multiplier
                # D[i]^2 <- (1 - c1 - c_mu)*D[i]^2 + c1*rank1 + c_mu*rankmu
                v = (1.0 - c1 - c_mu) * (D[i] * D[i]) + c1 * rank1 + c_mu * rankmu
                if v < 1e-30:
                    v = 1e-30
                D[i] = math.sqrt(v)

            # mild regularization to prevent runaway anisotropy
            # (keeps D within a reasonable ratio)
            if gen % 15 == 0:
                dmin = min(D)
                dmax = max(D)
                if dmin > 0.0:
                    ratio = dmax / dmin
                    if ratio > 1e6:
                        # compress extremes
                        g = ratio ** 0.25
                        for i in range(dim):
                            D[i] = max(dmin, min(dmax, D[i] / g))

        # restart logic: if no meaningful improvement, increase restart level
        if f_best < best_at_restart:
            best_at_restart = f_best
            # keep restart level if improving; but still allow another cycle if time permits
        else:
            restart += 1
            best_at_restart = f_best

        # if almost out of time, return
        if time.perf_counter() >= deadline:
            break

        # small safeguard: if restarts get too huge, cap
        if restart > 8:
            restart = 8

        # re-center next restart around best with small jitter (helps escape)
        if time.perf_counter() < deadline:
            jitter = 0.02 / (1.0 + restart)
            m2 = list(x_best)
            for i in range(dim):
                if spans[i] != 0.0:
                    m2[i] += jitter * safe_spans[i] * randn()
            reflect_inplace(m2)
            fm2 = eval_f(m2)
            if fm2 < f_best:
                f_best, x_best = fm2, m2
                last_improve_t = time.perf_counter()

    return float(f_best)
