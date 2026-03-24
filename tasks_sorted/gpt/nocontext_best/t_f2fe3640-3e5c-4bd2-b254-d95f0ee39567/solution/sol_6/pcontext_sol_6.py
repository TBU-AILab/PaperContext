import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (stdlib-only).

    Key upgrades vs. provided code:
      1) Correct CMA-ES update geometry:
         - Uses proper whitening via B*D from eigendecomposition of C (C = B*D^2*B^T),
           instead of the diagonal surrogate for ps (which can break sigma control badly).
      2) Much cheaper/faster sampling than full Cholesky every generation:
         - Eigendecomposition is done only occasionally (like CMA-ES), sampling uses B*D*z.
      3) Better boundary handling:
         - Uses reflection in normalized [0,1] space (less bias than hard clipping).
      4) Two-phase search:
         - Fast global exploration (Halton + random) to seed, then CMA-ES with restarts.
      5) Lightweight local search:
         - Occasional coordinate-wise pattern search around incumbent.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps = 1e-14

    n = int(dim)
    lo = [bounds[i][0] for i in range(n)]
    hi = [bounds[i][1] for i in range(n)]
    span = [hi[i] - lo[i] for i in range(n)]
    for i in range(n):
        if span[i] <= 0:
            span[i] = 1.0

    def now():
        return time.time()

    def reflect01(u):
        # reflect into [0,1] to avoid boundary sticking
        # works for any real u
        if 0.0 <= u <= 1.0:
            return u
        u = u % 2.0
        if u > 1.0:
            u = 2.0 - u
        return u

    def x_from_u(u):
        return [lo[i] + u[i] * span[i] for i in range(n)]

    # --- caching with quantization in normalized space ---
    cache = {}
    q = 1e-9

    def key_of_u(u):
        return tuple(int(u[i] / q) for i in range(n))

    def eval_u(u):
        # u is expected roughly in [0,1], but we reflect to be safe
        ur = [reflect01(u[i]) for i in range(n)]
        k = key_of_u(ur)
        v = cache.get(k)
        if v is not None:
            return v, ur
        x = x_from_u(ur)
        fx = float(func(x))
        cache[k] = fx
        return fx, ur

    # -------------------- Halton sequence --------------------
    def first_primes(m):
        primes = []
        x = 2
        while len(primes) < m:
            r = int(math.isqrt(x))
            ok = True
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    def vdc(k, base):
        out = 0.0
        denom = 1.0
        while k:
            k, r = divmod(k, base)
            denom *= base
            out += r / denom
        return out

    bases = first_primes(n)
    hal_k = 1

    def halton_u(k):
        return [vdc(k, bases[i]) for i in range(n)]

    # -------------------- basic linear algebra --------------------
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def mat_vec(A, x):
        m = len(A)
        out = [0.0] * m
        for i in range(m):
            s = 0.0
            Ai = A[i]
            for j in range(len(x)):
                s += Ai[j] * x[j]
            out[i] = s
        return out

    def outer_add_sym(M, v, w, alpha):
        # M += alpha * v * w^T, but only lower triangle is stored/used
        # if v==w gives symmetric rank-1. For general, we add symmetric part.
        for i in range(n):
            Mi = M[i]
            vi = v[i]
            for j in range(i + 1):
                Mi[j] += alpha * vi * w[j]

    def symmetrize_lower_to_full(C):
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                A[i][j] = C[i][j]
                A[j][i] = C[i][j]
        return A

    # Jacobi eigenvalue algorithm for symmetric matrices (stdlib-only).
    # Returns eigenvalues and eigenvectors (columns of V).
    def jacobi_eigh(A, max_sweeps=50):
        # A is full symmetric (n x n)
        V = [[0.0] * n for _ in range(n)]
        for i in range(n):
            V[i][i] = 1.0

        def max_offdiag(A):
            p = 0
            q = 1
            mx = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    v = abs(A[i][j])
                    if v > mx:
                        mx = v
                        p, q = i, j
            return mx, p, q

        for _ in range(max_sweeps):
            mx, p, q = max_offdiag(A)
            if mx < 1e-12:
                break

            app = A[p][p]
            aqq = A[q][q]
            apq = A[p][q]

            if abs(apq) < eps:
                continue

            tau = (aqq - app) / (2.0 * apq)
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
            if tau < 0.0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c

            # rotate A
            for k in range(n):
                if k != p and k != q:
                    akp = A[k][p]
                    akq = A[k][q]
                    A[k][p] = akp * c - akq * s
                    A[p][k] = A[k][p]
                    A[k][q] = akp * s + akq * c
                    A[q][k] = A[k][q]

            A[p][p] = app * c * c - 2.0 * apq * c * s + aqq * s * s
            A[q][q] = app * s * s + 2.0 * apq * c * s + aqq * c * c
            A[p][q] = 0.0
            A[q][p] = 0.0

            # rotate V
            for k in range(n):
                vkp = V[k][p]
                vkq = V[k][q]
                V[k][p] = vkp * c - vkq * s
                V[k][q] = vkp * s + vkq * c

        evals = [A[i][i] for i in range(n)]
        # Ensure non-negative (numerical)
        for i in range(n):
            if evals[i] < eps:
                evals[i] = eps
        return evals, V  # V columns are eigenvectors

    # multiply B * (D * z) where D is vector of stddevs
    def BDz(B, D, z):
        # w = D * z
        w = [D[i] * z[i] for i in range(n)]
        # return B*w
        return mat_vec(B, w)

    # solve B^T * x then divide by D: invsqrtC * x = B * ( (B^T x) / D )
    def invsqrtC_mul(B, D, x):
        # y = B^T x
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for k in range(n):
                s += B[k][i] * x[k]
            y[i] = s / (D[i] + eps)
        # return B*y
        return mat_vec(B, y)

    # -------------------- local pattern refinement --------------------
    def pattern_refine(u0, f0, step_u):
        u = u0[:]
        f = f0
        order = list(range(n))
        random.shuffle(order)
        improved = False
        for i in order:
            s = step_u[i]
            if s <= 1e-18:
                continue
            for sign in (+1.0, -1.0):
                uu = u[:]
                uu[i] = reflect01(uu[i] + sign * s)
                ff, uur = eval_u(uu)
                if ff < f:
                    u, f = uur, ff
                    improved = True
            if now() >= deadline:
                break
        return u, f, improved

    # -------------------- initialization --------------------
    best = float("inf")
    best_u = None

    init_n = max(120, 40 * n)
    for _ in range(init_n):
        if now() >= deadline:
            return best
        r = random.random()
        if r < 0.70:
            u = halton_u(hal_k); hal_k += 1
        elif r < 0.92:
            u = [random.random() for _ in range(n)]
        else:
            if best_u is None:
                u = [random.random() for _ in range(n)]
            else:
                u = [reflect01(best_u[i] + random.gauss(0.0, 0.15)) for i in range(n)]
        fx, ur = eval_u(u)
        if fx < best:
            best, best_u = fx, ur

    if best_u is None:
        return best

    # -------------------- CMA-ES parameters --------------------
    lam0 = max(10, 4 + int(3 * math.log(n + 1.0)))
    lam = max(lam0, 4 * n // 3 + 8)
    mu = lam // 2

    ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(ws)
    ws = [w / wsum for w in ws]
    mueff = 1.0 / sum(w * w for w in ws)

    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    m = best_u[:]               # mean in [0,1]
    sigma = 0.25                # global step-size in [0,1]
    # covariance stored as lower triangle
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = 1.0

    pc = [0.0] * n
    ps = [0.0] * n

    # eigensystem (B, D) for C; initialize as identity
    B = [[0.0] * n for _ in range(n)]
    for i in range(n):
        B[i][i] = 1.0
    D = [1.0] * n

    # update eigen decomposition occasionally
    eigeneval = 0
    evals_count = 0  # track function evals in main phase for scheduling

    no_imp = 0
    stagnate = 25 + 8 * n
    heavy_tail_p = 0.05

    # local step in u-space
    pstep_u = [0.08 for _ in range(n)]

    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    gen = 0
    while True:
        if now() >= deadline:
            return best
        gen += 1

        # eigen decomposition schedule like CMA-ES:
        # recompute every ~ (n / (c1+cmu)) / 10 generations (clamped)
        if gen == 1 or gen - eigeneval > max(1, int(0.1 * lam / (c1 + cmu + eps) / n) + 1):
            A = symmetrize_lower_to_full(C)
            evals, V = jacobi_eigh(A, max_sweeps=30 if n <= 25 else 20)
            # D = sqrt(evals), B = V
            D = [math.sqrt(max(eps, evals[i])) for i in range(n)]
            B = V
            eigeneval = gen

        # sample population
        pop = []  # (fx, z, y, u)
        for _ in range(lam):
            if now() >= deadline:
                return best

            # occasional global injections
            if random.random() < 0.10:
                u = halton_u(hal_k); hal_k += 1
                fx, ur = eval_u(u)
                # derive y ~ (ur - m)/sigma (approx)
                y = [(ur[i] - m[i]) / (sigma + eps) for i in range(n)]
                z = invsqrtC_mul(B, D, y)  # approx corresponding z
                pop.append((fx, z, y, ur))
                evals_count += 1
                continue

            z = [random.gauss(0.0, 1.0) for _ in range(n)]
            if random.random() < heavy_tail_p:
                ht = 0.35 * cauchy()
                z = [z[i] + ht * random.gauss(0.0, 1.0) for i in range(n)]

            y = BDz(B, D, z)
            u = [m[i] + sigma * y[i] for i in range(n)]
            # reflect into [0,1]
            u = [reflect01(u[i]) for i in range(n)]

            fx, ur = eval_u(u)
            pop.append((fx, z, y, ur))
            evals_count += 1

        pop.sort(key=lambda t: t[0])

        if pop[0][0] < best:
            best = pop[0][0]
            best_u = pop[0][3]
            no_imp = 0
        else:
            no_imp += 1

        m_old = m[:]

        # weighted recombination in y-space (normalized)
        y_w = [0.0] * n
        z_w = [0.0] * n
        for i in range(mu):
            _, z_i, y_i, _ = pop[i]
            wi = ws[i]
            for j in range(n):
                y_w[j] += wi * y_i[j]
                z_w[j] += wi * z_i[j]

        # update mean
        m = [reflect01(m_old[j] + sigma * y_w[j]) for j in range(n)]

        # update ps using invsqrtC * (m-m_old)/sigma == invsqrtC * y_w
        invC_yw = invsqrtC_mul(B, D, y_w)
        c_sig = math.sqrt(cs * (2.0 - cs) * mueff)
        for i in range(n):
            ps[i] = (1.0 - cs) * ps[i] + c_sig * invC_yw[i]

        # sigma
        psn = norm(ps)
        sigma *= math.exp((cs / damps) * (psn / (chi_n + eps) - 1.0))
        sigma = max(1e-12, min(0.7, sigma))

        # hsig
        hsig_cond = psn / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen) + eps)
        hsig = 1.0 if hsig_cond < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

        # pc
        c_cum = math.sqrt(cc * (2.0 - cc) * mueff)
        for i in range(n):
            pc[i] = (1.0 - cc) * pc[i] + hsig * c_cum * y_w[i]

        # covariance update (lower triangle)
        a = 1.0 - c1 - cmu
        if a < 0.0:
            a = 0.0

        # scale old C
        for i in range(n):
            Ci = C[i]
            for j in range(i + 1):
                Ci[j] *= a

        # rank-1
        c1_fac = c1 * (0.5 if hsig == 0.0 else 1.0)
        for i in range(n):
            for j in range(i + 1):
                C[i][j] += c1_fac * pc[i] * pc[j]

        # rank-mu using y_i
        for k in range(mu):
            _, _, yk, _ = pop[k]
            wk = cmu * ws[k]
            for i in range(n):
                for j in range(i + 1):
                    C[i][j] += wk * yk[i] * yk[j]

        # keep diagonals sane
        for i in range(n):
            if C[i][i] < 1e-18:
                C[i][i] = 1e-18

        # occasional local refinement around current best
        if gen % 5 == 0 and now() < deadline and best_u is not None:
            u2, f2, imp = pattern_refine(best_u, best, pstep_u)
            if f2 < best:
                best, best_u = f2, u2
                no_imp = 0
                for i in range(n):
                    pstep_u[i] = max(1e-18, pstep_u[i] * 0.75)
                m = best_u[:]  # pull mean
            else:
                for i in range(n):
                    pstep_u[i] = max(1e-18, pstep_u[i] * 0.92)

        # restart on stagnation
        if no_imp >= stagnate:
            no_imp = 0
            if best_u is not None:
                m = best_u[:]
            pc = [0.0] * n
            ps = [0.0] * n
            C = [[0.0] * n for _ in range(n)]
            for i in range(n):
                C[i][i] = 1.0

            sigma = min(0.5, max(0.18, sigma * 1.9))
            lam = min(max(12, lam * 2), 220 + 10 * n)
            mu = lam // 2

            ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            wsum = sum(ws)
            ws = [w / wsum for w in ws]
            mueff = 1.0 / sum(w * w for w in ws)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
            chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

            # reset eigensystem
            B = [[0.0] * n for _ in range(n)]
            for i in range(n):
                B[i][i] = 1.0
            D = [1.0] * n
            eigeneval = gen
