import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (self-contained, no external libs).

    Main improvements vs your last attempt:
      - Uses a compact CMA-ES style optimizer in normalized [0,1]^dim (very strong general-purpose)
      - Multiple restarts with IPOP-like population growth + occasional random injections
      - Two-stage: quick global sampling -> CMA restarts seeded from best samples
      - Robust bounded handling via reflection in normalized space
      - Minimal overhead per evaluation (critical for short max_time)

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        return float("inf")

    # ---------------- bounds helpers ----------------
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    inv_span = [1.0 / s if s > 0.0 else 0.0 for s in span]

    def reflect01(u):
        # reflect u into [0,1]
        if u < 0.0 or u > 1.0:
            u = u % 2.0
            if u > 1.0:
                u = 2.0 - u
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        return u

    def to_x(u):
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] + u[i] * (hi[i] - lo[i])
        return x

    # ---------------- evaluation (tiny cache) ----------------
    # Coarse cache in normalized space to avoid duplicate evaluations from reflection.
    cache = {}
    def key_u(u):
        # 1e-6 resolution in [0,1] -> 1e6 buckets; ok memory for typical eval budgets
        # (still bounded by actual number of evals)
        return tuple(int(reflect01(u[i]) * 1000000.0) for i in range(dim))

    def evaluate_u(u):
        u2 = [reflect01(ui) for ui in u]
        k = key_u(u2)
        if k in cache:
            return cache[k], u2
        x = to_x(u2)
        try:
            v = func(x)
            if v is None:
                v = float("inf")
            v = float(v)
            if not math.isfinite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v, u2

    # ---------------- utilities ----------------
    def rand_u():
        return [random.random() for _ in range(dim)]

    def l1_dist_u(a, b):
        s = 0.0
        for i in range(dim):
            s += abs(a[i] - b[i])
        return s

    # Low-discrepancy-ish init: scrambled Halton (fast) + LHS
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(dim)
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def halton_scrambled(index, base):
        f = 1.0
        r = 0.0
        i = index
        perm = digit_perm[base]
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_u(k):
        return [halton_scrambled(k, primes[i]) for i in range(dim)]

    def lhs_u(n):
        per_dim = []
        for i in range(dim):
            arr = [(k + random.random()) / n for k in range(n)]
            random.shuffle(arr)
            per_dim.append(arr)
        pts = []
        for k in range(n):
            pts.append([per_dim[i][k] for i in range(dim)])
        return pts

    # ---------------- compact CMA-ES core ----------------
    # Implements rank-μ update with diagonal covariance (sep-CMA) by default,
    # with a light full-cov option when dim is small.
    def cma_run(seed_u, seed_f, sigma0, lam, max_iter, full_cov):
        # Basic CMA-ES parameters
        n = dim
        mu = lam // 2
        weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        mueff = 1.0 / sum(w * w for w in weights)

        # strategy parameters
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        # state
        m = seed_u[:]  # mean in [0,1]
        best_f = seed_f
        best_u = seed_u[:]

        # covariance representation
        if full_cov and n <= 32:
            # Full covariance matrix C and its "sqrt" B*D (we'll maintain C and do Cholesky periodically)
            C = [[0.0] * n for _ in range(n)]
            for i in range(n):
                C[i][i] = 1.0
            # evolution paths
            pc = [0.0] * n
            ps = [0.0] * n
            sigma = sigma0

            # Cholesky for sampling (recomputed every few iterations)
            L = [[0.0] * n for _ in range(n)]
            chol_ok = False

            def cholesky(A):
                Lloc = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1):
                        s = A[i][j]
                        for k in range(j):
                            s -= Lloc[i][k] * Lloc[j][k]
                        if i == j:
                            if s <= 1e-18:
                                return None
                            Lloc[i][j] = math.sqrt(s)
                        else:
                            Lloc[i][j] = s / (Lloc[j][j] + 1e-300)
                return Lloc

            def mvn_sample():
                # z ~ N(0,I), y = L z
                z = [random.gauss(0.0, 1.0) for _ in range(n)]
                y = [0.0] * n
                for i in range(n):
                    s = 0.0
                    Li = L[i]
                    for k in range(i + 1):
                        s += Li[k] * z[k]
                    y[i] = s
                return y

            # expected norm of N(0,I)
            chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

            it = 0
            while it < max_iter and time.time() < deadline:
                it += 1
                if (not chol_ok) or (it % 6 == 1):
                    Lnew = cholesky(C)
                    if Lnew is None:
                        # fallback: reset covariance
                        C = [[0.0] * n for _ in range(n)]
                        for i in range(n):
                            C[i][i] = 1.0
                        Lnew = cholesky(C)
                    L = Lnew
                    chol_ok = True

                # sample population
                pop = []
                for _ in range(lam):
                    if time.time() >= deadline:
                        break
                    y = mvn_sample()
                    u = [m[i] + sigma * y[i] for i in range(n)]
                    f, ufix = evaluate_u(u)
                    pop.append((f, ufix, y))
                    if f < best_f:
                        best_f = f
                        best_u = ufix[:]
                if not pop:
                    break
                pop.sort(key=lambda t: t[0])

                # recombination
                old_m = m[:]
                m = [0.0] * n
                yw = [0.0] * n
                for i in range(mu):
                    f, u, y = pop[i]
                    for j in range(n):
                        m[j] += weights[i] * u[j]
                        yw[j] += weights[i] * y[j]

                # update paths (approx invsqrt(C) * (m-oldm)/sigma using L solve)
                # solve L v = (m-oldm)/sigma then solve L^T w = v -> w = invsqrt(C)*...
                dx = [(m[i] - old_m[i]) / (sigma + 1e-300) for i in range(n)]
                # forward solve
                v = [0.0] * n
                for i in range(n):
                    s = dx[i]
                    for k in range(i):
                        s -= L[i][k] * v[k]
                    v[i] = s / (L[i][i] + 1e-300)
                # backward solve
                invsqrt_dx = [0.0] * n
                for i in range(n - 1, -1, -1):
                    s = v[i]
                    for k in range(i + 1, n):
                        s -= L[k][i] * invsqrt_dx[k]
                    invsqrt_dx[i] = s / (L[i][i] + 1e-300)

                for i in range(n):
                    ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * invsqrt_dx[i]
                ps_norm = math.sqrt(sum(p * p for p in ps))
                hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * it)) / (chi_n + 1e-300)) < (1.4 + 2.0 / (n + 1.0)) else 0.0

                for i in range(n):
                    pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (m[i] - old_m[i]) / (sigma + 1e-300)

                # covariance update: C = (1-c1-cmu)C + c1*(pc pc^T + (1-hsig)cc(2-cc)C) + cmu*sum(w_i y_i y_i^T)
                # we have y stored for best individuals: y = (u - old_m)/sigma approx; use stored y in pop.
                alpha = (1.0 - c1 - cmu)
                if alpha < 0.0:
                    alpha = 0.0
                # scale existing C
                for i in range(n):
                    Ci = C[i]
                    for j in range(i + 1):
                        Ci[j] *= alpha

                # rank-one
                rank1 = c1
                if hsig == 0.0:
                    rank1 *= (1.0 - cc * (2.0 - cc))
                for i in range(n):
                    for j in range(i + 1):
                        C[i][j] += rank1 * pc[i] * pc[j]

                # rank-μ
                for k in range(mu):
                    _, _, yk = pop[k]
                    wk = cmu * weights[k]
                    for i in range(n):
                        for j in range(i + 1):
                            C[i][j] += wk * yk[i] * yk[j]

                # symmetrize upper triangle implicit; ensure diagonals positive
                for i in range(n):
                    if C[i][i] <= 1e-18:
                        C[i][i] = 1e-18

                # step-size control
                sigma *= math.exp((cs / damps) * (ps_norm / (chi_n + 1e-300) - 1.0))
                if sigma < 1e-10:
                    sigma = 1e-10
                if sigma > 0.9:
                    sigma = 0.9

                # stop if sigma tiny and no improvement likely
                if sigma < 1e-7 and it > 10:
                    break

            return best_u, best_f

        else:
            # sep-CMA (diagonal covariance) - fast for high dim
            diagC = [1.0] * n
            pc = [0.0] * n
            ps = [0.0] * n
            sigma = sigma0
            chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

            it = 0
            while it < max_iter and time.time() < deadline:
                it += 1

                pop = []
                for _ in range(lam):
                    if time.time() >= deadline:
                        break
                    z = [random.gauss(0.0, 1.0) for _ in range(n)]
                    y = [z[i] * math.sqrt(diagC[i]) for i in range(n)]
                    u = [m[i] + sigma * y[i] for i in range(n)]
                    f, ufix = evaluate_u(u)
                    pop.append((f, ufix, z, y))
                    if f < best_f:
                        best_f = f
                        best_u = ufix[:]
                if not pop:
                    break
                pop.sort(key=lambda t: t[0])

                old_m = m[:]
                m = [0.0] * n
                zw = [0.0] * n
                for i in range(mu):
                    f, u, z, y = pop[i]
                    for j in range(n):
                        m[j] += weights[i] * u[j]
                        zw[j] += weights[i] * z[j]

                # update ps using invsqrtC * (m-oldm)/sigma ; in diag case invsqrtC is 1/sqrt(diagC)
                for i in range(n):
                    dx = (m[i] - old_m[i]) / (sigma + 1e-300)
                    invsqrt = 1.0 / math.sqrt(diagC[i] + 1e-300)
                    ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * (dx * invsqrt)

                ps_norm = math.sqrt(sum(p * p for p in ps))
                hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * it)) / (chi_n + 1e-300)) < (1.4 + 2.0 / (n + 1.0)) else 0.0

                for i in range(n):
                    pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (m[i] - old_m[i]) / (sigma + 1e-300)

                # update diagonal covariance
                for i in range(n):
                    # rank-one part
                    rank1 = (pc[i] * pc[i])
                    # rank-mu part: sum w * (y_i)^2, where y_i = sqrt(diagC)*z_i
                    s = 0.0
                    for k in range(mu):
                        _, _, z, y = pop[k]
                        s += weights[k] * (y[i] * y[i])
                    diagC[i] = (1.0 - c1 - cmu) * diagC[i] + c1 * rank1 + cmu * s
                    if diagC[i] < 1e-18:
                        diagC[i] = 1e-18
                    if diagC[i] > 1e6:
                        diagC[i] = 1e6

                # step-size
                sigma *= math.exp((cs / damps) * (ps_norm / (chi_n + 1e-300) - 1.0))
                if sigma < 1e-10:
                    sigma = 1e-10
                if sigma > 0.9:
                    sigma = 0.9

                # if converged
                if sigma < 1e-7 and it > 10:
                    break

            return best_u, best_f

    # ---------------- initial global sampling ----------------
    best = float("inf")
    best_u = None

    # Budget init samples based on dim; keep modest to leave time for CMA.
    n_hal = max(20, min(140, 10 * dim))
    n_lhs = max(10, min(80, 5 * dim))
    hal_start = 1 + random.randrange(512)

    samples = []
    for k in range(n_hal):
        if time.time() >= deadline:
            return best
        u = halton_u(hal_start + k)
        f, u = evaluate_u(u)
        samples.append((f, u))
        if f < best:
            best = f
            best_u = u[:]

    for u in lhs_u(n_lhs):
        if time.time() >= deadline:
            return best
        f, u = evaluate_u(u)
        samples.append((f, u))
        if f < best:
            best = f
            best_u = u[:]

    # Add a few pure random points
    for _ in range(max(6, min(30, 2 * dim))):
        if time.time() >= deadline:
            return best
        f, u = evaluate_u(rand_u())
        samples.append((f, u))
        if f < best:
            best = f
            best_u = u[:]

    if best_u is None:
        f, u = evaluate_u(rand_u())
        return f

    samples.sort(key=lambda t: t[0])

    # ---------------- restart schedule ----------------
    # IPOP-like: increase lambda after each restart. Use best few samples as seeds.
    restart = 0
    seed_pool = [u for _, u in samples[:max(3, min(12, 2 * dim))]]

    while time.time() < deadline:
        # Choose seed: mostly best, sometimes diverse among seed_pool
        if seed_pool and random.random() < 0.7:
            seed_u = seed_pool[0]
        else:
            # pick farthest-from-best among a few to promote diversity
            base = seed_pool[0] if seed_pool else best_u
            cand = seed_pool[random.randrange(len(seed_pool))] if seed_pool else rand_u()
            best_cand = cand
            best_d = l1_dist_u(cand, base)
            for _ in range(4):
                c = seed_pool[random.randrange(len(seed_pool))] if seed_pool else rand_u()
                d = l1_dist_u(c, base)
                if d > best_d:
                    best_d = d
                    best_cand = c
            seed_u = best_cand

        seed_f, seed_u = evaluate_u(seed_u)

        # population and sigma
        base_lam = 4 + int(3 * math.log(dim + 1.0))
        lam = int(base_lam * (2 ** min(6, restart // 2)))
        lam = max(8, min(lam, 64 + 8 * dim))  # cap

        # sigma: in normalized coordinates; larger on later restarts sometimes helps escaping
        sigma0 = 0.18 / (1.0 + 0.15 * restart)
        if restart % 3 == 2:
            sigma0 *= 1.8
        sigma0 = max(0.02, min(0.45, sigma0))

        # iterations: small chunks, time-bounded
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        # keep it safe; each iter uses ~lam evals
        max_iter = max(6, min(80, int(0.25 * (remaining * 200.0) / max(1.0, lam))))
        # heuristic: full covariance only for small dim and if time seems enough
        full_cov = (dim <= 24) and (remaining > 0.25 * max_time)

        u_new, f_new = cma_run(seed_u, seed_f, sigma0=sigma0, lam=lam, max_iter=max_iter, full_cov=full_cov)
        if f_new < best:
            best = f_new
            best_u = u_new[:]
            # refresh seed pool with improved best and some random near-best points
            seed_pool = [best_u[:]] + seed_pool[:]
            if len(seed_pool) > max(6, min(20, 3 * dim)):
                seed_pool = seed_pool[:max(6, min(20, 3 * dim))]

        # occasional global injection to avoid over-committing
        if time.time() < deadline and random.random() < 0.35:
            f, u = evaluate_u(halton_u(hal_start + n_hal + restart * 31 + random.randrange(256)))
            if f < best:
                best = f
                best_u = u[:]
            # add to seed pool if decent
            if seed_pool:
                # keep if within some margin
                if f <= seed_f + 0.25 * (abs(seed_f) + 1.0):
                    seed_pool.append(u)
            else:
                seed_pool = [u]

        restart += 1

    return best
