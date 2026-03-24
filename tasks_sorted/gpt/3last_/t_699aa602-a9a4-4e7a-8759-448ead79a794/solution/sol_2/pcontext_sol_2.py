import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger box-constrained, derivative-free minimizer.

    Strategy (time-budgeted):
      - Initial seeding: center + random points + a few corners
      - Main optimizer: lightweight CMA-ES style search (full covariance, rank-1 + rank-mu)
      - Restarts with increasing population (IPOP-like) and occasional random injections
      - Strict time checks; returns best fitness found

    Self-contained (no numpy).
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------- helpers ----------
    def clamp(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def corner_point(mask):
        x = [0.0] * dim
        for i in range(dim):
            x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
        return x

    def evaluate(x):
        return float(func(x))

    # standard normal via Box-Muller
    _has_spare = [False]
    _spare = [0.0]
    def randn():
        if _has_spare[0]:
            _has_spare[0] = False
            return _spare[0]
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-300)
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare[0] = z1
        _has_spare[0] = True
        return z0

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def mat_vec(M, v):
        out = [0.0] * dim
        for i in range(dim):
            row = M[i]
            s = 0.0
            for j in range(dim):
                s += row[j] * v[j]
            out[i] = s
        return out

    def outer_add(M, a, b, coef):
        # M += coef * a*b^T
        for i in range(dim):
            ai = a[i] * coef
            row = M[i]
            for j in range(dim):
                row[j] += ai * b[j]

    def identity():
        M = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            M[i][i] = 1.0
        return M

    def mat_add_scaled(A, B, alpha):
        for i in range(dim):
            Ar = A[i]
            Br = B[i]
            for j in range(dim):
                Ar[j] += alpha * Br[j]

    def symmetrize(M):
        for i in range(dim):
            for j in range(i + 1, dim):
                v = 0.5 * (M[i][j] + M[j][i])
                M[i][j] = v
                M[j][i] = v

    def cholesky_spd(A):
        # returns lower-triangular L such that A ~= L L^T
        # Adds tiny jitter on diagonal if needed.
        L = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(i + 1):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 1e-18:
                        s = 1e-18
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
        return L

    # ---------- initial seeding ----------
    best = float("inf")
    best_x = None

    # center
    if time.time() < deadline:
        x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
        f0 = evaluate(x0)
        best, best_x = f0, x0[:]

    # some corners (up to 16, but limited by dim)
    corner_tries = min(16, 1 << min(dim, 4))
    for m in range(corner_tries):
        if time.time() >= deadline:
            return best
        x = corner_point(m)
        f = evaluate(x)
        if f < best:
            best, best_x = f, x[:]

    # random seeds
    seed_count = 12 + 3 * dim
    for _ in range(seed_count):
        if time.time() >= deadline:
            return best
        x = rand_point()
        f = evaluate(x)
        if f < best:
            best, best_x = f, x[:]

    # ---------- CMA-ES core (with restarts) ----------
    # restart schedule
    restart = 0
    base_lambda = 4 + int(3 * math.log(dim + 1.0))
    base_lambda = max(base_lambda, 8)

    # global scaling for sigma based on average span
    avg_span = sum(span_safe) / float(dim)
    sigma_global0 = 0.25 * avg_span
    sigma_min = 1e-14 * avg_span
    sigma_max = 0.8 * avg_span

    while time.time() < deadline:
        # population size increases on restart (IPOP-ish)
        lam = base_lambda * (2 ** (restart // 2))
        lam = max(lam, 8)
        mu = lam // 2

        # recombination weights (log)
        weights = [0.0] * mu
        for i in range(mu):
            weights[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        w_sum = sum(weights)
        for i in range(mu):
            weights[i] /= w_sum
        w_sq_sum = sum(w * w for w in weights)
        mueff = 1.0 / w_sq_sum

        # strategy parameters
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

        # expectation of ||N(0,I)||
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # init mean: mix of best and random (diversify)
        if best_x is not None and random.random() < 0.75:
            m = best_x[:]
            # small random offset to avoid determinism
            for i in range(dim):
                m[i] += (2.0 * random.random() - 1.0) * 0.02 * span_safe[i]
            clamp(m)
        else:
            m = rand_point()

        # init covariance to diagonal scaled to box
        C = identity()
        for i in range(dim):
            C[i][i] = (span_safe[i] / avg_span) ** 2  # relative scaling
        symmetrize(C)

        # evolution paths
        pc = [0.0] * dim
        ps = [0.0] * dim

        # step size
        sigma = sigma_global0 * (0.5 + 1.5 * random.random())
        sigma = max(sigma_min, min(sigma, sigma_max))

        # bookkeeping
        no_improve_gens = 0
        last_best = best

        # decompose C occasionally
        L = cholesky_spd(C)
        decomp_period = max(1, int(0.5 * dim))
        gen = 0

        while time.time() < deadline:
            gen += 1
            if gen % decomp_period == 1:
                # refresh factorization
                symmetrize(C)
                L = cholesky_spd(C)

            # sample population
            pop = []
            arz = []  # z vectors
            arx = []  # x vectors
            for k in range(lam):
                if time.time() >= deadline:
                    return best

                z = [randn() for _ in range(dim)]
                y = mat_vec(L, z)  # ~ N(0, C)
                x = [m[i] + sigma * y[i] for i in range(dim)]
                clamp(x)
                fx = evaluate(x)

                pop.append((fx, k))
                arz.append(z)
                arx.append(x)

                if fx < best:
                    best = fx
                    best_x = x[:]

            pop.sort(key=lambda t: t[0])
            # recombination: new mean from best mu solutions
            m_old = m[:]
            m = [0.0] * dim
            zmean = [0.0] * dim
            for i in range(mu):
                _, idx = pop[i]
                wi = weights[i]
                xi = arx[idx]
                zi = arz[idx]
                for d in range(dim):
                    m[d] += wi * xi[d]
                    zmean[d] += wi * zi[d]
            clamp(m)

            # update evolution path ps
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * zmean
            c = math.sqrt(cs * (2.0 - cs) * mueff)
            for d in range(dim):
                ps[d] = (1.0 - cs) * ps[d] + c * zmean[d]

            # hsig
            ps_norm = math.sqrt(dot(ps, ps))
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0

            # update evolution path pc
            # pc = (1-cc)pc + hsig*sqrt(cc(2-cc)mueff) * (m-m_old)/sigma
            cpc = hsig * math.sqrt(cc * (2.0 - cc) * mueff)
            for d in range(dim):
                pc[d] = (1.0 - cc) * pc[d] + cpc * ((m[d] - m_old[d]) / max(sigma, 1e-300))

            # rank-one update term: pc*pc^T
            # rank-mu update term: sum wi * yi*yi^T where yi = (x_i - m_old)/sigma
            # We approximate yi using (x_i - m_old)/sigma (clamped samples distort a bit, but still works)
            # C = (1-c1-cmu)C + c1*(pc pc^T + (1-hsig)*cc(2-cc)C) + cmu * sum wi yi yi^T
            # implement:
            # scale existing C
            scale = (1.0 - c1 - cmu)
            for i in range(dim):
                row = C[i]
                for j in range(dim):
                    row[j] *= scale

            # add rank-one and correction
            rank_one = [[0.0] * dim for _ in range(dim)]
            outer_add(rank_one, pc, pc, 1.0)
            if hsig < 0.5:
                # add (1-hsig)*cc(2-cc)C_old approximately by adding to rank_one a multiple of identity-ish.
                # We don't have C_old now (already scaled), so do small stabilization instead:
                for i in range(dim):
                    rank_one[i][i] += cc * (2.0 - cc)
            mat_add_scaled(C, rank_one, c1)

            # rank-mu
            rank_mu = [[0.0] * dim for _ in range(dim)]
            for i in range(mu):
                _, idx = pop[i]
                wi = weights[i]
                yi = [(arx[idx][d] - m_old[d]) / max(sigma, 1e-300) for d in range(dim)]
                outer_add(rank_mu, yi, yi, wi)
            mat_add_scaled(C, rank_mu, cmu)
            symmetrize(C)

            # step-size control
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            # stagnation / restart logic
            if best < last_best - 1e-15 * (1.0 + abs(last_best)):
                last_best = best
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            # If sigma collapses or no progress for a while, restart
            if sigma <= sigma_min * 1.01 or no_improve_gens > (10 + 3 * dim):
                break

            # occasional random injection: helps escape overly constrained basin
            if (gen % (8 + dim)) == 0 and random.random() < 0.15 and time.time() < deadline:
                x = rand_point()
                fx = evaluate(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    # also shift mean a bit toward it
                    m = x[:]

        restart += 1

    return best
