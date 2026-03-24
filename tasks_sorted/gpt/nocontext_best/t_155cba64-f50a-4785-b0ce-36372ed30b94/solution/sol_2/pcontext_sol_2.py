import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Hybrid:
      - Fast space-filling init (Halton) + random points
      - CMA-ES (full covariance, rank-1/rank-mu updates)
      - Mirrored sampling near bounds
      - Occasional small coordinate/pattern local search
      - IPOP-style restarts on stagnation with increased population

    func: callable(list[float]) -> float
    returns: best (minimum fitness found within max_time seconds)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ----------------- helpers -----------------
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def project(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_x(x):
        xx = project(x)
        v = func(xx)
        return float(v), xx

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Halton (bases = primes)
    def first_primes(n):
        primes = []
        v = 2
        while len(primes) < n:
            ok = True
            r = int(v ** 0.5)
            for p in primes:
                if p > r:
                    break
                if v % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(v)
            v += 1
        return primes

    def halton_index(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(max(1, dim))

    def halton_point(k):
        x = []
        for j in range(dim):
            u = halton_index(k, primes[j])
            lo, hi = bounds[j]
            x.append(lo + (hi - lo) * u)
        return x

    # linear algebra (small-dim friendly; O(n^3) per update, OK for typical dims)
    def mat_vec(A, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Ai = A[i]
            for j in range(n):
                s += Ai[j] * v[j]
            out[i] = s
        return out

    def outer(u, v):
        n = len(u)
        M = [[0.0] * n for _ in range(n)]
        for i in range(n):
            ui = u[i]
            Mi = M[i]
            for j in range(n):
                Mi[j] = ui * v[j]
        return M

    def mat_add_inplace(A, B, alpha=1.0):
        n = len(A)
        for i in range(n):
            Ai, Bi = A[i], B[i]
            for j in range(n):
                Ai[j] += alpha * Bi[j]

    def mat_scale_inplace(A, s):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            for j in range(n):
                Ai[j] *= s

    def identity(n):
        I = [[0.0] * n for _ in range(n)]
        for i in range(n):
            I[i][i] = 1.0
        return I

    # Cholesky decomposition (SPD) and sampling transform
    def cholesky(A):
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 1e-18:
                        s = 1e-18
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / (L[j][j] if L[j][j] != 0.0 else 1e-12)
        return L

    def lower_mat_vec(L, z):
        n = len(z)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Li = L[i]
            for k in range(i + 1):
                s += Li[k] * z[k]
            out[i] = s
        return out

    # mirrored sampling for bounds: if out-of-range, reflect into interval
    def reflect_into_bounds(x):
        xx = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                xx[i] = lo
                continue
            w = hi - lo
            v = xx[i]
            # reflect repeatedly; handles far out-of-range values
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
            # numerical nudge
            if v < lo: v = lo
            if v > hi: v = hi
            xx[i] = v
        return xx

    # small local search around incumbent
    def local_search(best_x, best_v, budget):
        x = best_x[:]
        v = best_v
        ranges = [(bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 1.0 for i in range(dim)]
        step = [0.05 * r for r in ranges]
        for _ in range(budget):
            if time.time() >= deadline:
                break
            j = random.randrange(dim)
            s = step[j]
            if s <= 0.0:
                continue
            improved = False
            # try +/- and a small pattern move
            for d in (1.0, -1.0):
                if time.time() >= deadline:
                    break
                cand = x[:]
                cand[j] = cand[j] + d * s
                cand = reflect_into_bounds(cand)
                vv, cc = eval_x(cand)
                if vv < v:
                    x, v = cc, vv
                    step[j] = min(step[j] * 1.3, 0.5 * ranges[j])
                    improved = True
                    break
            if not improved:
                step[j] *= 0.6
        return v, x

    # ----------------- init -----------------
    if dim <= 0:
        return float(func([]))

    ranges = [(bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 1.0 for i in range(dim)]
    avg_range = sum(ranges) / float(dim)
    scale = avg_range if avg_range > 0 else 1.0

    best = float("inf")
    best_x = None

    # Halton + random initialization
    n_init = max(24, min(200, 16 * dim))
    k = 11
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        v, x = eval_x(halton_point(k))
        k += 1
        if v < best:
            best, best_x = v, x

    for _ in range(max(8, 2 * dim)):
        if time.time() >= deadline:
            return best
        v, x = eval_x(rand_point())
        if v < best:
            best, best_x = v, x

    if best_x is None:
        best, best_x = eval_x(rand_point())

    # ----------------- CMA-ES core -----------------
    def cma_run(x_start, f_start, lam0, sigma0):
        nonlocal best, best_x

        n = dim
        lam = int(lam0)
        mu = max(2, lam // 2)

        # log weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # parameters
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        m = x_start[:]
        sigma = max(1e-12 * scale, float(sigma0))

        C = identity(n)
        pc = [0.0] * n
        ps = [0.0] * n

        # caches
        L = cholesky(C)
        evals_since_decomp = 0

        last_best = f_start
        stall = 0
        gen = 0

        while time.time() < deadline:
            gen += 1

            # periodic decomposition refresh (or when sigma changes a lot)
            if evals_since_decomp >= max(1, 2 * lam):
                L = cholesky(C)
                evals_since_decomp = 0

            # sample offspring
            pop = []
            for _ in range(lam):
                if time.time() >= deadline:
                    return
                z = [random.gauss(0.0, 1.0) for _ in range(n)]
                y = lower_mat_vec(L, z)          # ~ N(0, C)
                x = [m[i] + sigma * y[i] for i in range(n)]
                x = reflect_into_bounds(x)
                f, xx = eval_x(x)
                pop.append((f, xx, y))
                if f < best:
                    best, best_x = f, xx

            pop.sort(key=lambda t: t[0])
            elites = pop[:mu]

            # recombination
            m_old = m
            m = [0.0] * n
            y_w = [0.0] * n
            for i in range(mu):
                wi = w[i]
                xi = elites[i][1]
                yi = elites[i][2]
                for j in range(n):
                    m[j] += wi * xi[j]
                    y_w[j] += wi * yi[j]

            # update evolution path ps (using invsqrtC ~ solve(L, .) approximately)
            # approximate invsqrtC*y_w by solving L * u = y_w then use u
            # forward solve for lower-triangular L
            u = [0.0] * n
            for i in range(n):
                s = y_w[i]
                Li = L[i]
                for k in range(i):
                    s -= Li[k] * u[k]
                denom = Li[i] if Li[i] != 0.0 else 1e-12
                u[i] = s / denom

            for i in range(n):
                ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * u[i]

            ps_norm = math.sqrt(sum(vv * vv for vv in ps))
            # hsig
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chi_n) < (1.4 + 2.0 / (n + 1.0)) else 0.0

            # update pc
            for i in range(n):
                pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

            # covariance update
            # C = (1-c1-cmu)*C + c1*(pc pc^T + (1-hsig)*cc*(2-cc)*C) + cmu*sum(w_i * y_i y_i^T)
            # scale existing C
            mat_scale_inplace(C, (1.0 - c1 - cmu))

            # rank-1
            mat_add_inplace(C, outer(pc, pc), alpha=c1)

            if hsig < 0.5:
                mat_add_inplace(C, C, alpha=(c1 * cc * (2.0 - cc)))  # mild correction

            # rank-mu
            for i in range(mu):
                wi = w[i]
                yi = elites[i][2]
                mat_add_inplace(C, outer(yi, yi), alpha=cmu * wi)

            # step-size update
            sigma *= math.exp((cs / damps) * (ps_norm / chi_n - 1.0))
            sigma = max(1e-12 * scale, min(sigma, 0.8 * scale + 1e-12))

            evals_since_decomp += lam

            # stall / occasional local improvement
            if best < last_best - 1e-12 * (1.0 + abs(last_best)):
                last_best = best
                stall = 0
            else:
                stall += 1

            if stall > 10 and time.time() < deadline:
                # quick local search around incumbent
                vv, xx = local_search(best_x, best, budget=25 + dim)
                if vv < best:
                    best, best_x = vv, xx
                    m = best_x[:]
                stall = 0

            # if covariance drifts badly (numerical), reset softly
            if gen % 30 == 0:
                # ensure diagonal not tiny/huge
                for i in range(n):
                    if C[i][i] < 1e-18:
                        C[i][i] = 1e-18
                    if C[i][i] > 1e18:
                        C[i][i] = 1e18

    # ----------------- Restarts (IPOP-like) -----------------
    # initial sigma based on scale
    sigma0 = 0.22 * scale
    lam0 = max(12, 6 * dim)

    # first run from best init
    cma_run(best_x, best, lam0, sigma0)

    # restarts until time runs out
    restart = 0
    while time.time() < deadline:
        restart += 1
        # increase population, reset covariance, start around best with jitter + some random chance
        lam_r = int(lam0 * (2 ** min(6, restart)))  # cap growth
        sigma_r = max(0.05 * scale, sigma0 * (0.7 ** min(10, restart)))

        if time.time() >= deadline:
            break

        if random.random() < 0.7 and best_x is not None:
            x0 = best_x[:]
            # jitter start
            for j in range(dim):
                x0[j] = clip(x0[j] + random.gauss(0.0, 1.0) * 0.1 * ranges[j], bounds[j][0], bounds[j][1])
        else:
            x0 = rand_point()

        f0, x0 = eval_x(x0)
        if f0 < best:
            best, best_x = f0, x0

        cma_run(x0, f0, lam_r, sigma_r)

    return best
