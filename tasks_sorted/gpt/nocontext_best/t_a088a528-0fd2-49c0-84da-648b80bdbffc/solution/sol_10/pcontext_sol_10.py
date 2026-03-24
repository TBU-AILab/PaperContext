import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Changes vs previous:
      - Switch core to an *active* CMA-ES (full covariance, rank-1 + rank-mu update)
        which is much stronger on rotated / ill-conditioned problems than diagonal ES.
      - Keep box constraints via reflection.
      - Add small, budget-friendly local refinement (pattern/coordinate) near current best.
      - Strict time checks; no external libraries.

    Returns: best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        raise ValueError("dim must be positive")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            raise ValueError("Each bound must satisfy low <= high")

    # ---------- RNG: Normal (Box-Muller cached) ----------
    _have_spare = False
    _spare = 0.0

    def randn():
        nonlocal _have_spare, _spare
        if _have_spare:
            _have_spare = False
            return _spare
        u1 = 1e-12 + random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        _spare = r * math.sin(th)
        _have_spare = True
        return z0

    def reflect_1d(x, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (x - lo) % (2.0 * w)
        if y < 0.0:
            y += 2.0 * w
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def reflect_vec(x):
        y = x[:]  # copy
        for i in range(dim):
            if spans[i] <= 0.0:
                y[i] = lows[i]
            else:
                xi = y[i]
                if xi < lows[i] or xi > highs[i]:
                    y[i] = reflect_1d(xi, lows[i], highs[i])
        return y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] <= 0.0:
                x[i] = lows[i]
            else:
                x[i] = lows[i] + random.random() * spans[i]
        return x

    def evaluate(x):
        return float(func(x))

    # ---------- linear algebra helpers (lists) ----------
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def mat_vec(M, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            row = M[i]
            s = 0.0
            for j in range(n):
                s += row[j] * v[j]
            out[i] = s
        return out

    def outer(u, v):
        n = len(u)
        M = [[0.0] * n for _ in range(n)]
        for i in range(n):
            ui = u[i]
            row = M[i]
            for j in range(n):
                row[j] = ui * v[j]
        return M

    def mat_add_inplace(A, B, alpha=1.0):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            Bi = B[i]
            for j in range(n):
                Ai[j] += alpha * Bi[j]

    def mat_scale_inplace(A, alpha):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            for j in range(n):
                Ai[j] *= alpha

    def mat_copy(A):
        return [row[:] for row in A]

    # Symmetric eigen-decomposition via Jacobi rotations (no numpy).
    # Used to compute B,D such that C = B * diag(D^2) * B^T, and to sample x = m + sigma * B*(D*z)
    def jacobi_eigh_sym(A, max_sweeps=30):
        n = len(A)
        V = [[0.0] * n for _ in range(n)]
        for i in range(n):
            V[i][i] = 1.0
        M = mat_copy(A)

        def max_offdiag():
            p = 0
            q = 1
            mx = 0.0
            for i in range(n):
                Mi = M[i]
                for j in range(i + 1, n):
                    v = abs(Mi[j])
                    if v > mx:
                        mx = v
                        p, q = i, j
            return mx, p, q

        for _ in range(max_sweeps * n * n):
            mx, p, q = max_offdiag()
            if mx < 1e-14:
                break
            app = M[p][p]
            aqq = M[q][q]
            apq = M[p][q]
            if abs(apq) < 1e-20:
                continue
            tau = (aqq - app) / (2.0 * apq)
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
            if tau < 0.0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c

            # rotate M
            for k in range(n):
                if k != p and k != q:
                    mkp = M[k][p]
                    mkq = M[k][q]
                    M[k][p] = c * mkp - s * mkq
                    M[p][k] = M[k][p]
                    M[k][q] = s * mkp + c * mkq
                    M[q][k] = M[k][q]

            Mpp = c * c * app - 2.0 * s * c * apq + s * s * aqq
            Mqq = s * s * app + 2.0 * s * c * apq + c * c * aqq
            M[p][p] = Mpp
            M[q][q] = Mqq
            M[p][q] = 0.0
            M[q][p] = 0.0

            # rotate V
            for k in range(n):
                vkp = V[k][p]
                vkq = V[k][q]
                V[k][p] = c * vkp - s * vkq
                V[k][q] = s * vkp + c * vkq

        eig = [M[i][i] for i in range(n)]
        # sort descending eigenvalues to keep stable
        idx = list(range(n))
        idx.sort(key=lambda i: eig[i], reverse=True)
        eig2 = [eig[i] for i in idx]
        V2 = [[V[r][i] for i in idx] for r in range(n)]
        return eig2, V2

    # ---------- initialization: LHS-ish + opposition ----------
    best = float("inf")
    best_x = None

    def try_point(x):
        nonlocal best, best_x
        if time.time() >= deadline:
            return
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]

    center = [lows[i] + 0.5 * spans[i] for i in range(dim)]
    try_point(center)

    init_n = max(20, min(120, 10 * dim))
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] <= 0.0:
                x[i] = lows[i]
            else:
                u = (perms[i][j] + random.random()) / init_n
                x[i] = lows[i] + u * spans[i]
        try_point(x)
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        try_point(reflect_vec(xo))

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # ---------- CMA-ES parameters ----------
    n = dim
    lam = max(8, min(60, 4 + int(3 * math.log(n + 1.0)) + 4 * n // 5))
    mu = lam // 2

    # log weights
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # strategy parameters
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    # Active CMA (negative weights for worst individuals)
    neg = lam - mu
    w_neg = []
    if neg > 0:
        w2 = [math.log(lam + 0.5) - math.log(mu + i + 1.0) for i in range(neg)]
        s2 = sum(abs(x) for x in w2) or 1.0
        w_neg = [-0.25 * abs(x) / s2 for x in w2]  # modest negative pressure

    # state
    m = best_x[:]  # mean
    sigma = 0.25 * (sum(spans) / max(1, n)) if sum(spans) > 0 else 1.0
    sigma = max(1e-12, sigma)

    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = 1.0

    pc = [0.0] * n
    ps = [0.0] * n

    # decomposition
    B = [[0.0] * n for _ in range(n)]
    for i in range(n):
        B[i][i] = 1.0
    D = [1.0] * n
    invsqrtC = [[0.0] * n for _ in range(n)]
    for i in range(n):
        invsqrtC[i][i] = 1.0

    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    # local search step
    lstep = [0.08 * spans[i] if spans[i] > 0 else 0.0 for i in range(n)]
    lfloor = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(n)]

    # control
    evals_since_update = 0
    update_every = max(1, int(0.25 * n))  # update eigendecomp periodically
    stall = 0
    last_best = best

    def update_decomp():
        nonlocal B, D, invsqrtC
        # ensure symmetry (numerical)
        for i in range(n):
            for j in range(i + 1, n):
                v = 0.5 * (C[i][j] + C[j][i])
                C[i][j] = v
                C[j][i] = v
        eig, V = jacobi_eigh_sym(C, max_sweeps=25)
        # protect eigenvalues
        eig = [max(1e-30, e) for e in eig]
        D = [math.sqrt(e) for e in eig]
        B = V
        # invsqrtC = B * diag(1/D) * B^T
        invsqrtC = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += B[i][k] * (1.0 / D[k]) * B[j][k]
                invsqrtC[i][j] = s

    update_decomp()

    def sample():
        # x = m + sigma * B*(D*z)
        z = [randn() for _ in range(n)]
        Dz = [D[i] * z[i] for i in range(n)]
        BDz = [0.0] * n
        for i in range(n):
            s = 0.0
            Bi = B[i]
            for k in range(n):
                s += Bi[k] * Dz[k]
            BDz[i] = s
        x = [m[i] + sigma * BDz[i] for i in range(n)]
        return reflect_vec(x), z  # return z for recombination bookkeeping

    # ---------- main loop ----------
    while True:
        if time.time() >= deadline:
            return best

        # ask
        pop = []
        for _ in range(lam):
            if time.time() >= deadline:
                return best
            x, z = sample()
            fx = evaluate(x)
            pop.append((fx, x, z))
        pop.sort(key=lambda t: t[0])

        # track best
        if pop[0][0] < best:
            best = pop[0][0]
            best_x = pop[0][1][:]
        if best < last_best - 1e-15:
            stall = 0
            last_best = best
        else:
            stall += 1

        # recombination
        old_m = m[:]
        m = [0.0] * n
        zmean = [0.0] * n
        for i in range(mu):
            wi = w[i]
            xi = pop[i][1]
            zi = pop[i][2]
            for j in range(n):
                m[j] += wi * xi[j]
                zmean[j] += wi * zi[j]
        m = reflect_vec(m)

        # evolution paths
        # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrtC * (m-old_m)/sigma
        y = [(m[i] - old_m[i]) / max(1e-30, sigma) for i in range(n)]
        inv_y = mat_vec(invsqrtC, y)
        cfac = math.sqrt(cs * (2 - cs) * mueff)
        for i in range(n):
            ps[i] = (1 - cs) * ps[i] + cfac * inv_y[i]

        ps_norm = norm(ps)
        hsig = 1.0 if (ps_norm / math.sqrt(1 - (1 - cs) ** (2 * (1 + 1))) / chiN) < (1.4 + 2 / (n + 1)) else 0.0

        cfac_c = math.sqrt(cc * (2 - cc) * mueff)
        for i in range(n):
            pc[i] = (1 - cc) * pc[i] + hsig * cfac_c * y[i]

        # covariance update
        # C = (1-c1-cmu)*C + c1*(pc pc^T + (1-hsig)*cc*(2-cc)*C) + cmu * sum(w_i * y_i y_i^T)
        # where y_i = (x_i - old_m)/sigma
        # rank-one
        C_old = C  # alias
        a = (1 - c1 - cmu)
        if a < 0.0:
            a = 0.0
        mat_scale_inplace(C_old, a)

        # add (1-hsig) term
        if hsig < 0.5:
            mat_add_inplace(C_old, C, alpha=c1 * cc * (2 - cc))  # uses current C_old already scaled; ok modest

        mat_add_inplace(C_old, outer(pc, pc), alpha=c1)

        # rank-mu positive
        for i in range(mu):
            wi = w[i]
            xi = pop[i][1]
            yi = [(xi[j] - old_m[j]) / max(1e-30, sigma) for j in range(n)]
            mat_add_inplace(C_old, outer(yi, yi), alpha=cmu * wi)

        # active negative update from worst
        for k in range(len(w_neg)):
            wi = w_neg[k]
            xi = pop[lam - 1 - k][1]
            yi = [(xi[j] - old_m[j]) / max(1e-30, sigma) for j in range(n)]
            mat_add_inplace(C_old, outer(yi, yi), alpha=cmu * wi)

        C = C_old

        # step-size control
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        sigma = max(1e-15, min(1e5, sigma))

        evals_since_update += lam
        if evals_since_update >= update_every * lam:
            evals_since_update = 0
            update_decomp()

        # -------- local refinement near best (cheap pattern/coordinate) --------
        # only when stagnating a bit
        if stall >= max(6, n // 2) and time.time() < deadline:
            x = best_x[:]
            order = list(range(n))
            random.shuffle(order)
            improved = False
            for t in range(min(n, 10)):
                if time.time() >= deadline:
                    return best
                i = order[t]
                if spans[i] <= 0.0 or lstep[i] <= lfloor[i]:
                    continue
                base = x[i]
                for sgn in (1.0, -1.0):
                    cand = x[:]
                    cand[i] = base + sgn * lstep[i]
                    cand = reflect_vec(cand)
                    fc = evaluate(cand)
                    if fc < best:
                        best = fc
                        best_x = cand[:]
                        x = cand
                        improved = True
                        break
            if improved:
                stall = 0
                # pull mean towards refined best
                a = 0.6
                m = reflect_vec([a * best_x[i] + (1 - a) * m[i] for i in range(n)])
                for i in range(n):
                    if spans[i] > 0.0:
                        lstep[i] = min(0.25 * spans[i], lstep[i] * 1.1)
            else:
                for i in range(n):
                    if spans[i] > 0.0:
                        lstep[i] = max(lfloor[i], lstep[i] * 0.75)
                # mild restart if very stuck
                if stall > 25 + 3 * n and time.time() < deadline:
                    stall = 0
                    # restart mean: mix best + random
                    u = rand_uniform()
                    mix = 0.7
                    m = reflect_vec([mix * best_x[i] + (1 - mix) * u[i] for i in range(n)])
                    sigma = max(sigma, 0.2 * (sum(spans) / max(1, n)))
                    C = [[0.0] * n for _ in range(n)]
                    for i in range(n):
                        C[i][i] = 1.0
                    pc = [0.0] * n
                    ps = [0.0] * n
                    update_decomp()
