import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib-only), tuned to be more reliable than the
    prior DE+DiagES hybrid by fixing two common weaknesses:

    1) Local search quality: replace "diag CMA-ES-lite" with a small *full-covariance*
       CMA-ES core (rank-1 + rank-mu updates). This handles rotated / ill-conditioned
       basins much better than diagonal-only methods and coordinate polish.

    2) Budget use: keep a light L-SHADE/JADE DE front-end for global exploration,
       then increasingly hand over to CMA-ES as time progresses or DE stagnates.

    Design:
      - Scrambled Halton + random + opposition init
      - L-SHADE style DE (archive + success-history + shrinking population)
      - Full CMA-ES bursts late / on stagnation, seeded from incumbent best
      - Time-aware partial restarts/injections
      - Robust bound handling: reflection for ES, bounce-back for DE
      - Robust evaluation: guards NaN/inf/exceptions

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_repair(x):
        for i in range(dim):
            L, U = lo[i], hi[i]
            if x[i] < L or x[i] > U:
                if U <= L:
                    x[i] = L
                    continue
                w = U - L
                y = (x[i] - L) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                x[i] = L + y
        return x

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = func(x)
            if v is None or isinstance(v, complex):
                return float("inf")
            v = float(v)
            if v != v or v == float("inf") or v == -float("inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ---------------- Halton (scrambled) ----------------
    def _primes_upto(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        r = int(n ** 0.5)
        for p in range(2, r + 1):
            if sieve[p]:
                start = p * p
                sieve[start:n + 1:p] = [False] * (((n - start) // p) + 1)
        return [i for i, ok in enumerate(sieve) if ok]

    def _first_n_primes(n):
        if n <= 0:
            return []
        ub = max(50, int(n * (math.log(max(3, n)) + math.log(math.log(max(3, n))) + 3)))
        primes = _primes_upto(ub)
        while len(primes) < n:
            ub = int(ub * 1.7) + 10
            primes = _primes_upto(ub)
        return primes[:n]

    primes = _first_n_primes(dim)
    scramble = [random.random() for _ in range(dim)]

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = (halton_value(k, primes[i]) + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # DE bounce-back repair (towards base) + clip
    def de_repair(trial, base):
        for j in range(dim):
            if trial[j] < lo[j]:
                r = random.random()
                trial[j] = lo[j] + r * (base[j] - lo[j])
            elif trial[j] > hi[j]:
                r = random.random()
                trial[j] = hi[j] - r * (hi[j] - base[j])
        return clip_inplace(trial)

    # ---------------- small linear algebra helpers (stdlib-only) ----------------
    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def mat_vec(A, x):
        return [dot(row, x) for row in A]

    def outer(a, b):
        return [[ai * bj for bj in b] for ai in a]

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

    def eye(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Cholesky with jitter; returns lower-triangular L such that L*L^T ~ A
    def cholesky_jitter(A):
        n = len(A)
        # start jitter proportional to average diagonal
        avgd = sum(A[i][i] for i in range(n)) / max(1, n)
        jitter = 1e-14 * (avgd if avgd > 0.0 else 1.0)
        for _ in range(6):
            L = [[0.0] * n for _ in range(n)]
            ok = True
            for i in range(n):
                for j in range(i + 1):
                    s = A[i][j]
                    for k in range(j):
                        s -= L[i][k] * L[j][k]
                    if i == j:
                        s += jitter
                        if s <= 0.0 or not (s == s):
                            ok = False
                            break
                        L[i][j] = math.sqrt(s)
                    else:
                        if L[j][j] == 0.0:
                            ok = False
                            break
                        L[i][j] = s / L[j][j]
                if not ok:
                    break
            if ok:
                return L
            jitter = (jitter * 10.0) if jitter > 0.0 else 1e-12
        # fallback: diagonal sqrt
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            v = A[i][i]
            if v <= 0.0 or not (v == v):
                v = 1.0
            L[i][i] = math.sqrt(v)
        return L

    def lower_tri_mul(L, z):
        # y = L z
        n = len(L)
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            Li = L[i]
            for j in range(i + 1):
                s += Li[j] * z[j]
            y[i] = s
        return y

    # ---------------- Full CMA-ES (small/fast) ----------------
    class CMAES:
        __slots__ = ("n", "m", "sigma", "C", "B", "D", "L", "lam", "mu", "w",
                     "mueff", "cc", "cs", "c1", "cmu", "damps", "pc", "ps",
                     "chiN", "eig_it", "bestx", "bestf", "stall")

        def __init__(self, x0, f0):
            n = dim
            self.n = n
            self.m = list(x0)
            # sigma as fraction of domain
            self.sigma = 0.20 * (sum(scale) / max(1, n))
            if self.sigma <= 0.0:
                self.sigma = 0.2

            # population sizing (small but decent)
            lam = 4 + int(3 * math.log(n + 1.0))
            lam = max(8, min(28, lam))
            self.lam = lam
            self.mu = lam // 2

            # log weights
            ws = [math.log((self.mu + 0.5) / (i + 1.0)) for i in range(self.mu)]
            s = sum(ws) if ws else 1.0
            self.w = [wi / s for wi in ws]
            self.mueff = 1.0 / sum(wi * wi for wi in self.w)

            # strategy parameters (standard defaults)
            self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
            self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
            self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
            self.cmu = min(1.0 - self.c1,
                           2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
            self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs

            self.pc = [0.0] * n
            self.ps = [0.0] * n

            self.C = eye(n)  # covariance
            self.L = eye(n)  # cholesky of C (lower)
            self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

            self.eig_it = 0
            self.bestx = list(x0)
            self.bestf = f0
            self.stall = 0

        def _update_factorization(self):
            self.L = cholesky_jitter(self.C)

        def ask(self):
            # sample z ~ N(0,I), y = L z, x = m + sigma y
            z = [random.gauss(0.0, 1.0) for _ in range(self.n)]
            y = lower_tri_mul(self.L, z)
            x = [self.m[i] + self.sigma * y[i] for i in range(self.n)]
            reflect_repair(x)
            return x, z, y

        def tell(self, evaluated):
            # evaluated: list of tuples (f, x, z, y) sorted by f asc
            n = self.n
            oldm = self.m[:]

            # recombination
            newm = [0.0] * n
            zmean = [0.0] * n
            for k in range(self.mu):
                wk = self.w[k]
                xk, zk = evaluated[k][1], evaluated[k][2]
                for i in range(n):
                    newm[i] += wk * xk[i]
                    zmean[i] += wk * zk[i]

            self.m = newm

            # update evolution path ps
            # ps = (1-cs) ps + sqrt(cs(2-cs)mueff) * (L^{-1} (m-oldm) / sigma)
            # Using zmean is equivalent if x were unbounded; with reflection it's approximate but works.
            c = math.sqrt(self.cs * (2.0 - self.cs) * self.mueff)
            for i in range(n):
                self.ps[i] = (1.0 - self.cs) * self.ps[i] + c * zmean[i]

            # step-size control
            psn = norm(self.ps)
            self.sigma *= math.exp((self.cs / self.damps) * (psn / self.chiN - 1.0))
            # clamp sigma
            sig_min = 1e-14 * (sum(scale) / max(1, n))
            sig_max = 0.8 * (sum(scale) / max(1, n))
            if self.sigma < sig_min:
                self.sigma = sig_min
            elif self.sigma > sig_max:
                self.sigma = sig_max

            # hsig
            hsig = 1.0 if (psn / math.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.eig_it + 1)))) < (1.4 + 2.0 / (n + 1.0)) * self.chiN else 0.0

            # update evolution path pc
            cc = self.cc
            cpc = math.sqrt(cc * (2.0 - cc) * self.mueff)
            dm = [(self.m[i] - oldm[i]) / max(1e-300, self.sigma) for i in range(n)]
            for i in range(n):
                self.pc[i] = (1.0 - cc) * self.pc[i] + hsig * cpc * dm[i]

            # covariance update
            # C = (1-c1-cmu) C + c1 (pc pc^T + (1-hsig) cc(2-cc) C) + cmu * sum w_i * y_i y_i^T
            c1 = self.c1
            cmu = self.cmu

            # decay
            mat_scale_inplace(self.C, (1.0 - c1 - cmu))

            # rank-1
            pcpc = outer(self.pc, self.pc)
            mat_add_inplace(self.C, pcpc, alpha=c1)

            if hsig == 0.0:
                mat_add_inplace(self.C, self.C, alpha=(c1 * cc * (2.0 - cc)))  # mild compensation

            # rank-mu using y vectors (in covariance coordinates)
            for k in range(self.mu):
                wk = self.w[k]
                yk = evaluated[k][3]
                mat_add_inplace(self.C, outer(yk, yk), alpha=(cmu * wk))

            # periodic factorization
            self.eig_it += 1
            if self.eig_it % max(1, (self.n // 2)) == 0:
                self._update_factorization()

            # best / stall
            fbest = evaluated[0][0]
            if fbest < self.bestf:
                self.bestf = fbest
                self.bestx = list(evaluated[0][1])
                self.stall = 0
            else:
                self.stall += 1

    # ---------------- initialization ----------------
    NP0 = max(18, min(90, 14 + 6 * dim))
    NPmin = max(8, min(36, 10 + 2 * dim))
    NP = NP0

    pop, fit = [], []
    k = 1
    while len(pop) < NP and time.time() < deadline:
        if len(pop) % 4 == 0:
            x = rand_point()
        else:
            x = halton_point(k)
            k += 1

        fx = safe_eval(x)
        if random.random() < 0.65:
            xo = opposite_point(x)
            fo = safe_eval(xo)
            if fo < fx:
                x, fx = xo, fo
        pop.append(list(x))
        fit.append(fx)

    if not pop:
        return float("inf")

    best_idx = min(range(len(pop)), key=lambda i: fit[i])
    best_x = list(pop[best_idx])
    best = fit[best_idx]

    # ---------------- L-SHADE-style DE state ----------------
    archive = []
    Amax = 2 * NP0

    H = max(6, min(28, 6 + dim // 2))
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    def p_fraction(frac_time, NP_):
        pmin = 2.0 / max(2, NP_)
        p = 0.22 - 0.14 * frac_time
        if p < pmin:
            p = pmin
        if p > 0.30:
            p = 0.30
        return p

    es = CMAES(best_x, best)
    es._update_factorization()

    it = 0
    no_best_improve = 0
    inj_patience = max(80, 25 * dim)

    while time.time() < deadline:
        it += 1
        now = time.time()
        frac_time = (now - t0) / max(1e-9, float(max_time))

        # shrink DE population linearly
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac_time))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            if len(archive) > 2 * NP:
                random.shuffle(archive)
                archive = archive[:2 * NP]

        # CMA-ES bursts later in time or when DE stagnates
        if (frac_time > 0.55 and it % max(6, 2 * dim) == 0) or (no_best_improve > inj_patience and frac_time > 0.25):
            # if incumbent improved a lot, re-seed ES around it
            if best < es.bestf - 1e-15:
                es = CMAES(best_x, best)
                es._update_factorization()

            # burst generations (each costs ~lambda evals)
            burst_gens = max(1, min(10, 2 + int(2.0 * math.sqrt(dim))))
            for _g in range(burst_gens):
                if time.time() >= deadline:
                    break
                evaluated = []
                for _ in range(es.lam):
                    if time.time() >= deadline:
                        break
                    x, z, y = es.ask()
                    fx = safe_eval(x)
                    evaluated.append((fx, x, z, y))
                if not evaluated:
                    break
                evaluated.sort(key=lambda t: t[0])
                es.tell(evaluated)
                if es.bestf < best:
                    best = es.bestf
                    best_x = list(es.bestx)
                    no_best_improve = 0

                # if ES stalls hard, slightly expand sigma to escape
                if es.stall > 6 and frac_time < 0.95:
                    es.sigma *= 1.35
                    es.stall = 0

        # injections if DE stagnates
        if no_best_improve > inj_patience and time.time() < deadline:
            nrep = max(1, NP // 4)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for wi in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.45:
                    x = halton_point(k); k += 1
                else:
                    x = [best_x[j] + random.gauss(0.0, 0.50 * scale[j]) for j in range(dim)]
                    reflect_repair(x)
                fx = safe_eval(x)
                pop[wi] = list(x)
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, list(x)
                    no_best_improve = 0
            # reset ES around new best
            es = CMAES(best_x, best)
            es._update_factorization()
            no_best_improve = inj_patience // 2

        # --- DE generation ---
        order = sorted(range(NP), key=lambda i: fit[i])
        p = p_fraction(frac_time, NP)
        p_count = max(2, int(math.ceil(p * NP)))

        union = pop + archive
        union_n = len(union)

        def pick_r(exclude_pop):
            while True:
                r = random.randrange(union_n)
                if r < NP and r in exclude_pop:
                    continue
                return r

        S_F, S_CR, S_df = [], [], []
        improved_gen = False

        for i in range(NP):
            if time.time() >= deadline:
                break
            xi = pop[i]
            fi = fit[i]

            rH = random.randrange(H)
            muF = M_F[rH]
            muCR = M_CR[rH]

            CR = random.gauss(muCR, 0.1)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            u = random.random()
            u = min(1.0 - 1e-12, max(1e-12, u))
            F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
            tries = 0
            while F <= 0.0 and tries < 8:
                u = random.random()
                u = min(1.0 - 1e-12, max(1e-12, u))
                F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            pbest = order[random.randrange(p_count)]
            xp = pop[pbest]

            excl = {i, pbest}
            r1 = pick_r(excl)
            r2 = pick_r(excl)
            while r2 == r1:
                r2 = pick_r(excl)

            xr1 = union[r1]
            xr2 = union[r2]

            v = [xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]
            de_repair(v, xi)

            jrand = random.randrange(dim)
            uvec = [v[j] if (j == jrand or random.random() < CR) else xi[j] for j in range(dim)]
            de_repair(uvec, xi)

            fu = safe_eval(uvec)

            if fu <= fi:
                if len(archive) < Amax:
                    archive.append(list(xi))
                else:
                    archive[random.randrange(Amax)] = list(xi)

                pop[i] = uvec
                fit[i] = fu

                S_F.append(F)
                S_CR.append(CR)
                df = abs(fi - fu)
                S_df.append(df if df > 0.0 else 1e-12)

                if fu < best:
                    best, best_x = fu, list(uvec)
                    improved_gen = True

        if improved_gen:
            no_best_improve = 0
        else:
            no_best_improve += 1

        # update memories
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = 1.0
            mcr = 0.0
            num = 0.0
            den = 0.0
            for Fv, CRv, w in zip(S_F, S_CR, S_df):
                ww = w / wsum
                mcr += ww * CRv
                num += ww * (Fv * Fv)
                den += ww * Fv
            mF = (num / den) if den > 0.0 else M_F[mem_idx]
            M_F[mem_idx] = mF
            M_CR[mem_idx] = mcr
            mem_idx = (mem_idx + 1) % H

        Amax = max(2 * NP, 2 * NP0)
        if len(archive) > Amax:
            random.shuffle(archive)
            archive = archive[:Amax]

        if deadline - time.time() < 0.02:
            break

    # last-chance micro-polish (very cheap)
    if time.time() < deadline:
        for _ in range(min(12, 2 + dim)):
            if time.time() >= deadline:
                break
            x = [best_x[i] + random.gauss(0.0, 0.01 * scale[i]) for i in range(dim)]
            reflect_repair(x)
            fx = safe_eval(x)
            if fx < best:
                best, best_x = fx, x

    return best
