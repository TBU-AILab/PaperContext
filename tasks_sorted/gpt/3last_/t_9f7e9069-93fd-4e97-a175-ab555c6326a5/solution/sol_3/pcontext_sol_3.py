import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements vs previous versions:
    - Uses CMA-ES style sampling (full covariance, rank-1 + rank-mu updates)
      for strong performance on smooth/ill-conditioned problems.
    - Automatic bound handling via smooth "reflection" (keeps samples feasible
      without sticking to borders too much).
    - Occasional global restart mixture when stagnating.
    - Small, cheap local pattern search around the current best.

    Returns:
        best (float): best fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + max_time

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]

    # ---- utilities ----
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
        # Reflect each coordinate into [lo, hi] interval.
        # Works even for large excursions by using modulo reflection.
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
            # map to [0, 2w)
            y = y % (2.0 * w)
            if y > w:
                y = 2.0 * w - y
            x[i] = a + y

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def mat_vec(M, v):
        return [dot(row, v) for row in M]

    def vec_add(a, b, s=1.0):
        return [ai + s * bi for ai, bi in zip(a, b)]

    def vec_scale(a, s):
        return [s * ai for ai in a]

    def vec_norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def outer(u, v):
        # returns matrix u v^T
        return [[ui * vj for vj in v] for ui in u]

    def mat_add_inplace(A, B, alpha=1.0):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            Bi = B[i]
            for j in range(n):
                Ai[j] += alpha * Bi[j]

    def mat_scale_inplace(A, s):
        n = len(A)
        for i in range(n):
            row = A[i]
            for j in range(n):
                row[j] *= s

    def identity(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def cholesky_psd(A):
        # Cholesky for SPD; if numerical issues, add small jitter progressively.
        n = len(A)
        jitter = 0.0
        for _ in range(6):
            L = [[0.0] * n for _ in range(n)]
            ok = True
            for i in range(n):
                for j in range(i + 1):
                    s = A[i][j]
                    if i == j and jitter != 0.0:
                        s += jitter
                    for k in range(j):
                        s -= L[i][k] * L[j][k]
                    if i == j:
                        if s <= 1e-18:
                            ok = False
                            break
                        L[i][j] = math.sqrt(s)
                    else:
                        L[i][j] = s / L[j][j]
                if not ok:
                    break
            if ok:
                return L
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
        # fallback: diagonal sqrt
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            d = A[i][i]
            if not (d > 0.0):
                d = 1e-12
            L[i][i] = math.sqrt(d)
        return L

    def gaussian_vec(n):
        return [random.gauss(0.0, 1.0) for _ in range(n)]

    # ---- handle degenerate dimensions ----
    active_idx = [i for i in range(dim) if not fixed[i]]
    adim = len(active_idx)
    if adim == 0:
        x = [lo[i] for i in range(dim)]
        return safe_eval(x)

    # We optimize in full dim but fixed coords stay fixed.
    # Initial mean: random + best of a few for better start.
    best = float("inf")
    best_x = rand_uniform_vec()
    # quick seeding
    seed_trials = min(12, 4 + 2 * dim)
    for _ in range(seed_trials):
        if time.time() >= deadline:
            return best if best < float("inf") else safe_eval(best_x)
        x = rand_uniform_vec()
        f = safe_eval(x)
        if f < best:
            best = f
            best_x = x

    m = best_x[:]  # mean starts at best observed

    # ---- CMA-ES parameters ----
    # population size lambda and mu
    lam = max(12, min(60, 4 + int(3 * math.log(adim + 1.0)) + 2 * adim))
    mu = lam // 2

    # recombination weights (log)
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # strategy parameters
    cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
    cs = (mueff + 2.0) / (adim + mueff + 5.0)
    c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs

    # initial sigma as fraction of average span
    avg_span = sum(span[i] for i in active_idx) / max(1, adim)
    sigma = 0.25 * avg_span if avg_span > 0 else 1.0

    # evolution paths and covariance
    pc = [0.0] * adim
    ps = [0.0] * adim
    C = identity(adim)

    # expected norm of N(0,I)
    chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

    # For mapping between full x and active coordinates
    def pack_active(x_full):
        return [x_full[i] for i in active_idx]

    def unpack_active(x_act, template_full):
        x = template_full[:]
        for k, i in enumerate(active_idx):
            x[i] = x_act[k]
        return x

    # local search step sizes (on full vector)
    ls_step = [0.05 * s for s in span]
    ls_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # stagnation handling
    last_best = best
    stall = 0

    # ---- main loop ----
    gen = 0
    while time.time() < deadline:
        gen += 1

        # Decompose C -> L (Cholesky), then sample: y = m_act + sigma * L * z
        m_act = pack_active(m)
        L = cholesky_psd(C)

        pop = []
        # mixture sampling: mostly around mean, sometimes global for robustness
        mix_global = 0.10 if stall > 10 else 0.04
        for _ in range(lam):
            if time.time() >= deadline:
                return best
            if random.random() < mix_global:
                x = rand_uniform_vec()
                f = safe_eval(x)
                pop.append((f, x, None, None))
                continue

            z = gaussian_vec(adim)
            y = mat_vec(L, z)
            x_act = vec_add(m_act, y, s=sigma)
            x = unpack_active(x_act, m)
            reflect_to_bounds(x)
            f = safe_eval(x)
            pop.append((f, x, z, y))

        pop.sort(key=lambda t: t[0])

        # update global best
        if pop[0][0] < best:
            best = pop[0][0]
            best_x = pop[0][1][:]
        # stall tracking
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        # recombination to new mean
        old_m_act = m_act[:]
        new_m_act = [0.0] * adim
        selected = pop[:mu]
        for i in range(mu):
            x = selected[i][1]
            x_act = pack_active(x)
            for k in range(adim):
                new_m_act[k] += weights[i] * x_act[k]
        # set mean full
        m = unpack_active(new_m_act, m)
        reflect_to_bounds(m)

        # compute y_w = (m_new - m_old)/sigma in active space
        y_w = [(new_m_act[k] - old_m_act[k]) / max(1e-30, sigma) for k in range(adim)]

        # update ps
        # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrt(C) * y_w
        # approximate invsqrt(C) * y_w by solving L * v = y_w then v is inv(L)*y_w,
        # and invsqrt(C)=inv(L^T)*inv(L), so use two triangular solves approx:
        # v = solve(L, y_w), w = solve(L^T, v)
        def solve_lower(Lm, b):
            n = len(Lm)
            x = [0.0] * n
            for i in range(n):
                s = b[i]
                for j in range(i):
                    s -= Lm[i][j] * x[j]
                d = Lm[i][i]
                x[i] = s / d if d != 0.0 else 0.0
            return x

        def solve_upper_from_lower(Lm, b):
            # solve (L^T) x = b where L is lower triangular
            n = len(Lm)
            x = [0.0] * n
            for i in range(n - 1, -1, -1):
                s = b[i]
                for j in range(i + 1, n):
                    s -= Lm[j][i] * x[j]
                d = Lm[i][i]
                x[i] = s / d if d != 0.0 else 0.0
            return x

        v = solve_lower(L, y_w)
        invsqrtCy = solve_upper_from_lower(L, v)

        coeff_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        for k in range(adim):
            ps[k] = (1.0 - cs) * ps[k] + coeff_ps * invsqrtCy[k]

        # compute hsig
        ps_norm = vec_norm(ps)
        left = ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen))
        hsig = 1.0 if left < (1.4 + 2.0 / (adim + 1.0)) * chiN else 0.0

        # update pc
        coeff_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for k in range(adim):
            pc[k] = (1.0 - cc) * pc[k] + hsig * coeff_pc * y_w[k]

        # update covariance C
        # C = (1 - c1 - cmu) * C + c1*(pc pc^T + (1-hsig)*cc*(2-cc)*C) + cmu*sum(w_i*y_i*y_i^T)
        # y_i are (x_i - old_m)/sigma in active coordinates for selected individuals
        factor = (1.0 - c1 - cmu)
        if factor < 0.0:
            factor = 0.0
        mat_scale_inplace(C, factor)

        # rank-one
        rank_one = outer(pc, pc)
        mat_add_inplace(C, rank_one, alpha=c1)

        if hsig == 0.0:
            mat_add_inplace(C, C, alpha=(c1 * cc * (2.0 - cc)))  # mild compensation

        # rank-mu
        for i in range(mu):
            x = selected[i][1]
            x_act = pack_active(x)
            yi = [(x_act[k] - old_m_act[k]) / max(1e-30, sigma) for k in range(adim)]
            mat_add_inplace(C, outer(yi, yi), alpha=cmu * weights[i])

        # step-size control
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        # safety bounds for sigma
        sigma = max(1e-15 * (avg_span if avg_span > 0 else 1.0), min(sigma, 2.0 * (avg_span if avg_span > 0 else 1.0) + 1e-9))

        # ---- cheap local search around best (very small budget) ----
        if stall > max(8, 2 * adim) and time.time() + 0.002 < deadline:
            # coordinate pattern search
            trials = min(10 + dim, 30)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                if random.random() < 0.75:
                    d = random.randrange(dim)
                    if fixed[d]:
                        continue
                    stepd = max(ls_min[d], ls_step[d])
                    x[d] += stepd if random.random() < 0.5 else -stepd
                else:
                    # random direction small move
                    # build direction on active dims only
                    z = gaussian_vec(adim)
                    nz = vec_norm(z)
                    if nz <= 0.0:
                        continue
                    for k, i in enumerate(active_idx):
                        stepd = max(ls_min[i], ls_step[i])
                        x[i] += (z[k] / nz) * stepd
                reflect_to_bounds(x)
                fx = safe_eval(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    m = x[:]  # pull mean to improved best
                    stall = 0
                    break
                else:
                    # gentle decay
                    for d in range(dim):
                        if not fixed[d] and ls_step[d] > ls_min[d]:
                            ls_step[d] *= 0.995

        # ---- restart / diversification when heavily stalled ----
        if stall > max(25, 5 * adim) and time.time() + 0.01 < deadline:
            stall = 0
            # reset covariance and enlarge sigma a bit; keep mean at best
            m = best_x[:]
            C = identity(adim)
            sigma = max(sigma, 0.35 * avg_span if avg_span > 0 else 1.0)
            ps = [0.0] * adim
            pc = [0.0] * adim
            # also reset local step a bit
            for d in range(dim):
                if not fixed[d]:
                    ls_step[d] = max(ls_step[d], 0.04 * span[d])

    return best
