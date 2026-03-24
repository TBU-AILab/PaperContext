import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only, self-contained).

    Improvement focus vs last attempt:
      - More reliable core optimizer: CMA-ES (robust, fast on many continuous problems)
      - Asynchronous ask/tell with evaluation cache (avoid wasted re-evals)
      - Restarts with increasing population (IPOP-like) to escape local minima
      - Small elite local search (coordinate + gaussian) near the end
      - Bound handling in normalized space [0,1]^d with reflection (better than clamp)
    Returns:
      best fitness (float)
    """

    t_end = time.time() + float(max_time)

    # ---------- bounds / normalization ----------
    lo = [0.0] * dim
    hi = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
    span = [hi[i] - lo[i] for i in range(dim)]
    active = [span[i] > 0.0 for i in range(dim)]
    act_idx = [i for i in range(dim) if active[i]]
    adim = len(act_idx)

    if adim == 0:
        x = [lo[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    def to_real(z):
        x = [0.0] * dim
        for i in range(dim):
            if active[i]:
                x[i] = lo[i] + z[i] * span[i]
            else:
                x[i] = lo[i]
        return x

    # ---------- RNG helpers ----------
    def rand01():
        return random.random()

    def gauss():
        u1 = rand01()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = rand01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---------- bounded handling (reflection) ----------
    def reflect01(z):
        # reflect each coordinate into [0,1]
        # reflection tends to preserve step statistics better than hard clamping
        for i in range(dim):
            if not active[i]:
                z[i] = 0.0
                continue
            v = z[i]
            # reflect using period-2 folding
            if v < 0.0 or v > 1.0:
                v = v % 2.0
                if v > 1.0:
                    v = 2.0 - v
            # just in case of tiny numeric drift
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            z[i] = v
        return z

    # ---------- evaluation cache + best tracking ----------
    cache = {}
    best = float("inf")
    best_z = [0.5] * dim

    def qstep(n_eval):
        # coarse early, finer later
        if n_eval < 200:
            return 2e-6
        if n_eval < 2000:
            return 8e-7
        return 3e-7

    n_eval = 0

    def z_key(z):
        q = qstep(n_eval)
        inv = 1.0 / q
        return tuple(int(v * inv + 0.5) for v in z)

    def eval_z(z):
        nonlocal best, best_z, n_eval
        reflect01(z)
        k = z_key(z)
        if k in cache:
            return cache[k]
        fx = float(func(to_real(z)))
        cache[k] = fx
        n_eval += 1
        if fx < best:
            best = fx
            best_z = z[:]
        return fx

    # ---------- initialization: LHS-like + opposition + random ----------
    def lhs_like(n):
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            z = [0.0] * dim
            for j in range(dim):
                if not active[j]:
                    z[j] = 0.0
                else:
                    z[j] = (perms[j][i] + rand01()) / n
            pts.append(z)
        return pts

    def opposite(z):
        return [1.0 - v if active[i] else 0.0 for i, v in enumerate(z)]

    # Seed pool
    seed_n = max(12, min(80, 10 + 4 * dim))
    seeds = lhs_like(seed_n)
    seeds = seeds + [opposite(z) for z in seeds]
    for _ in range(min(30, 2 * seed_n)):
        seeds.append([rand01() if active[i] else 0.0 for i in range(dim)])

    for z in seeds:
        if time.time() >= t_end:
            return best
        eval_z(z[:])

    # ---------- CMA-ES implementation (active dims only) ----------
    # Minimal, robust version with diagonalization via eigen-decomposition of C.
    # We keep full covariance but do eigen-decomp with a simple Jacobi method (no numpy).

    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def mat_vec(M, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Mi = M[i]
            for j in range(n):
                s += Mi[j] * v[j]
            out[i] = s
        return out

    def mat_mul(A, B):
        n = len(A)
        m = len(B[0])
        p = len(B)
        out = [[0.0] * m for _ in range(n)]
        for i in range(n):
            Ai = A[i]
            Oi = out[i]
            for k in range(p):
                aik = Ai[k]
                if aik == 0.0:
                    continue
                Bk = B[k]
                for j in range(m):
                    Oi[j] += aik * Bk[j]
        return out

    def mat_T(M):
        n = len(M)
        m = len(M[0])
        out = [[0.0] * n for _ in range(m)]
        for i in range(n):
            Mi = M[i]
            for j in range(m):
                out[j][i] = Mi[j]
        return out

    def identity(n):
        I = [[0.0] * n for _ in range(n)]
        for i in range(n):
            I[i][i] = 1.0
        return I

    def jacobi_eigh(A, max_sweeps=18):
        # Eigen-decomposition for symmetric matrix A (small/medium adim)
        n = len(A)
        V = identity(n)
        M = [row[:] for row in A]

        def max_offdiag(M):
            n = len(M)
            p = 0
            q = 1
            mv = 0.0
            for i in range(n):
                Mi = M[i]
                for j in range(i + 1, n):
                    v = abs(Mi[j])
                    if v > mv:
                        mv = v
                        p, q = i, j
            return p, q, mv

        for _ in range(max_sweeps):
            p, q, mv = max_offdiag(M)
            if mv < 1e-14:
                break
            app = M[p][p]
            aqq = M[q][q]
            apq = M[p][q]

            tau = (aqq - app) / (2.0 * apq) if apq != 0.0 else 0.0
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)) if apq != 0.0 else 0.0
            if tau < 0.0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t * t) if apq != 0.0 else 1.0
            s = t * c if apq != 0.0 else 0.0

            # rotate M
            for k in range(n):
                mkp = M[k][p]
                mkq = M[k][q]
                M[k][p] = c * mkp - s * mkq
                M[k][q] = s * mkp + c * mkq
            for k in range(n):
                mpk = M[p][k]
                mqk = M[q][k]
                M[p][k] = c * mpk - s * mqk
                M[q][k] = s * mpk + c * mqk

            M[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
            M[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
            M[p][q] = 0.0
            M[q][p] = 0.0

            # rotate V
            for k in range(n):
                vkp = V[k][p]
                vkq = V[k][q]
                V[k][p] = c * vkp - s * vkq
                V[k][q] = s * vkp + c * vkq

        eigvals = [M[i][i] for i in range(n)]
        # clamp small negatives (numerical)
        for i in range(n):
            if eigvals[i] < 1e-18:
                eigvals[i] = 1e-18
        return eigvals, V  # A ≈ V diag(eigvals) V^T

    # start mean from best seed
    m = [0.0] * adim
    for t, j in enumerate(act_idx):
        m[t] = best_z[j]

    # sigma schedule: initial based on dimension
    sigma0 = 0.30
    sigma_min = 1e-10

    # restart scheme
    restart = 0
    base_lambda = max(10, 4 + int(3.0 * math.log(adim + 1.0)) + 2 * adim)

    # small local search near end
    def local_polish(budget=10):
        nonlocal best, best_z
        z0 = best_z[:]
        used = 0
        step = 0.10
        # gaussian
        for _ in range(min(4, budget)):
            if time.time() >= t_end:
                return
            z = z0[:]
            for j in act_idx:
                z[j] += 0.15 * gauss()
            reflect01(z)
            eval_z(z)
            used += 1
            if used >= budget:
                return
        # coordinate
        order = act_idx[:]
        random.shuffle(order)
        for j in order:
            if time.time() >= t_end or used >= budget:
                return
            for d in (-step, +step):
                z = best_z[:]
                z[j] += d
                reflect01(z)
                eval_z(z)
                used += 1
                if used >= budget:
                    return

    # ---------- main optimization loop with CMA-ES restarts ----------
    while time.time() < t_end:
        # set lambda, mu
        lam = int(base_lambda * (2 ** restart))
        lam = max(lam, 8)
        lam = min(lam, 60 + 6 * adim)  # keep bounded (time)
        mu = lam // 2

        # recombination weights (log)
        w = [0.0] * mu
        for i in range(mu):
            w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w)

        # parameters
        cc = (4.0 + mu_eff / adim) / (adim + 4.0 + 2.0 * mu_eff / adim)
        cs = (mu_eff + 2.0) / (adim + mu_eff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((adim + 2.0) ** 2 + mu_eff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (adim + 1.0)) - 1.0) + cs

        # evolution paths
        pc = [0.0] * adim
        ps = [0.0] * adim

        # covariance
        C = identity(adim)
        # decomposition
        eigvals, B = jacobi_eigh(C)
        D = [math.sqrt(v) for v in eigvals]
        invD = [1.0 / d for d in D]

        # expectation of ||N(0,I)||
        chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

        sigma = max(sigma_min, sigma0 * (0.85 ** restart))
        if restart == 0:
            sigma = max(sigma, 0.20)

        # run this restart until stagnation or time
        no_improve = 0
        best_restart = best

        gen = 0
        while time.time() < t_end:
            gen += 1
            # occasional local polish very late
            time_left = t_end - time.time()
            if time_left < 0.12 * max_time and (gen % 6 == 0):
                local_polish(budget=8)

            # ask: sample population
            arz = []   # z ~ N(0,I)
            ary = []   # y = B*D*z
            arx = []   # candidate points in full dim (normalized)

            # rebuild eigendecomp occasionally (or every few gens)
            if gen == 1 or gen % max(3, 1 + adim // 4) == 0:
                # ensure symmetry
                for i in range(adim):
                    for j in range(i):
                        v = 0.5 * (C[i][j] + C[j][i])
                        C[i][j] = v
                        C[j][i] = v
                eigvals, B = jacobi_eigh(C, max_sweeps=10 + min(20, adim))
                D = [math.sqrt(v) for v in eigvals]
                invD = [1.0 / d for d in D]

            # precompute BD = B * diag(D)
            BD = [[B[i][j] * D[j] for j in range(adim)] for i in range(adim)]

            for _ in range(lam):
                if time.time() >= t_end:
                    return best
                z = [gauss() for _ in range(adim)]
                y = mat_vec(BD, z)
                # construct full-dim point
                x = best_z[:]  # use full vector as template
                for t, j in enumerate(act_idx):
                    x[j] = m[t] + sigma * y[t]
                reflect01(x)
                arz.append(z)
                ary.append(y)
                arx.append(x)

            # evaluate
            fit = []
            for x in arx:
                if time.time() >= t_end:
                    return best
                fit.append(eval_z(x))

            # sort by fitness
            idx = list(range(lam))
            idx.sort(key=lambda i: fit[i])

            # track improvement
            if fit[idx[0]] < best_restart - 1e-15:
                best_restart = fit[idx[0]]
                no_improve = 0
            else:
                no_improve += 1

            # recombination
            m_old = m[:]
            m = [0.0] * adim
            y_w = [0.0] * adim
            z_w = [0.0] * adim
            for k in range(mu):
                i = idx[k]
                wi = w[k]
                yi = ary[i]
                zi = arz[i]
                for t in range(adim):
                    y_w[t] += wi * yi[t]
                    z_w[t] += wi * zi[t]
            for t in range(adim):
                m[t] = m_old[t] + sigma * y_w[t]

            # update ps: ps = (1-cs)ps + sqrt(cs(2-cs)mu_eff) * B * z_w
            c_fac = math.sqrt(cs * (2.0 - cs) * mu_eff)
            Bzw = mat_vec(B, z_w)
            for t in range(adim):
                ps[t] = (1.0 - cs) * ps[t] + c_fac * Bzw[t]

            # hsig
            ps_norm = math.sqrt(dot(ps, ps))
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chiN) < (1.4 + 2.0 / (adim + 1.0)) else 0.0

            # update pc
            c_fac2 = math.sqrt(cc * (2.0 - cc) * mu_eff)
            for t in range(adim):
                pc[t] = (1.0 - cc) * pc[t] + hsig * c_fac2 * y_w[t]

            # update C
            # C = (1-c1-cmu)C + c1*(pc pc^T + (1-hsig)cc(2-cc)C) + cmu*sum(w_i y_i y_i^T)
            # rank-one
            oldC = C
            C = [[(1.0 - c1 - cmu) * oldC[i][j] for j in range(adim)] for i in range(adim)]

            # add rank-one term
            rank1 = [[pc[i] * pc[j] for j in range(adim)] for i in range(adim)]
            if hsig < 0.5:
                adj = cc * (2.0 - cc)
                for i in range(adim):
                    Ci = C[i]
                    oldCi = oldC[i]
                    for j in range(adim):
                        Ci[j] += c1 * (rank1[i][j] + adj * oldCi[j])
            else:
                for i in range(adim):
                    Ci = C[i]
                    r1i = rank1[i]
                    for j in range(adim):
                        Ci[j] += c1 * r1i[j]

            # add rank-mu term
            for k in range(mu):
                i = idx[k]
                wi = w[k]
                yi = ary[i]
                for a in range(adim):
                    Cia = C[a]
                    vya = yi[a]
                    for b in range(adim):
                        Cia[b] += cmu * wi * vya * yi[b]

            # update sigma
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
            if sigma < sigma_min:
                sigma = sigma_min
            if sigma > 0.7:
                sigma = 0.7

            # stop criteria for this restart
            if no_improve > 18 + 2 * adim:
                break
            if sigma < 1e-12:
                break
            if time.time() >= t_end:
                return best

        # restart with larger population, shift mean to best found so far
        restart += 1
        # set new mean to current best_z
        for t, j in enumerate(act_idx):
            m[t] = best_z[j]

    return best
