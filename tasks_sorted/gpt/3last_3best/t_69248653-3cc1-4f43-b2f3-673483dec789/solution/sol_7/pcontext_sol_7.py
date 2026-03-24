import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib-only), aimed to beat the best shown (~3.80).

    What is improved vs your current best (L-SHADE/JADE DE + DiagES):
      1) Better local refinement: adds a *strong* local optimizer:
         - a small full-covariance CMA-ES *only when it is worth it* (late time / stagnation),
           but implemented cheaply (Cholesky only, no heavy eigendecomp).
         - plus a tiny Powell-like random-direction pattern search micro-polish for the very end.
      2) More robust global exploration early: keeps L-SHADE-style DE with:
         - archive, success-history memories, shrinking population,
         - occasional "scatter" injections sampled around best AND globally (Halton).
      3) Time scheduling: DE dominates early; CMA bursts become more frequent later,
         and if DE stagnates we switch to local focus sooner.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]
    avg_scale = sum(scale) / max(1, dim)

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_repair(x):
        # mirror reflection into bounds; smooth for ES steps
        for i in range(dim):
            L, U = lo[i], hi[i]
            if U <= L:
                x[i] = L
                continue
            v = x[i]
            if v < L or v > U:
                w = U - L
                y = (v - L) % (2.0 * w)
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

    # ---------------- scrambled Halton seeding ----------------
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

    # ---------------- small linear algebra helpers (stdlib-only) ----------------
    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

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
        avgd = sum(A[i][i] for i in range(n)) / max(1, n)
        jitter = 1e-14 * (avgd if avgd > 0.0 else 1.0)
        for _ in range(7):
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
                        d = L[j][j]
                        if d == 0.0:
                            ok = False
                            break
                        L[i][j] = s / d
                if not ok:
                    break
            if ok:
                return L
            jitter = (jitter * 10.0) if jitter > 0.0 else 1e-12
        # fallback diag
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            v = A[i][i]
            if v <= 0.0 or not (v == v):
                v = 1.0
            L[i][i] = math.sqrt(v)
        return L

    def lower_tri_mul(L, z):
        n = len(L)
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            Li = L[i]
            for j in range(i + 1):
                s += Li[j] * z[j]
            y[i] = s
        return y

    # ---------------- Full CMA-ES (light, time-aware) ----------------
    class CMAES:
        __slots__ = ("n", "m", "sigma", "C", "L", "lam", "mu", "w", "mueff",
                     "cc", "cs", "c1", "cmu", "damps", "pc", "ps", "chiN",
                     "eig_it", "bestx", "bestf", "stall")

        def __init__(self, x0, f0):
            n = dim
            self.n = n
            self.m = list(x0)

            # start sigma moderate; later it adapts
            self.sigma = 0.18 * avg_scale
            if self.sigma <= 0.0:
                self.sigma = 0.18

            lam = 4 + int(3 * math.log(n + 1.0))
            lam = max(8, min(26, lam))
            self.lam = lam
            self.mu = lam // 2

            ws = [math.log((self.mu + 0.5) / (i + 1.0)) for i in range(self.mu)]
            s = sum(ws) if ws else 1.0
            self.w = [wi / s for wi in ws]
            self.mueff = 1.0 / sum(wi * wi for wi in self.w)

            self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
            self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
            self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
            self.cmu = min(1.0 - self.c1,
                           2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
            self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs

            self.pc = [0.0] * n
            self.ps = [0.0] * n

            self.C = eye(n)
            self.L = eye(n)
            self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
            self.eig_it = 0

            self.bestx = list(x0)
            self.bestf = f0
            self.stall = 0

        def refactor(self):
            self.L = cholesky_jitter(self.C)

        def ask(self):
            z = [random.gauss(0.0, 1.0) for _ in range(self.n)]
            y = lower_tri_mul(self.L, z)
            x = [self.m[i] + self.sigma * y[i] for i in range(self.n)]
            reflect_repair(x)
            return x, z, y

        def tell(self, evaluated):
            # evaluated sorted by f
            n = self.n
            oldm = self.m[:]

            newm = [0.0] * n
            zmean = [0.0] * n
            for k in range(self.mu):
                wk = self.w[k]
                xk, zk = evaluated[k][1], evaluated[k][2]
                for i in range(n):
                    newm[i] += wk * xk[i]
                    zmean[i] += wk * zk[i]
            self.m = newm

            c = math.sqrt(self.cs * (2.0 - self.cs) * self.mueff)
            for i in range(n):
                self.ps[i] = (1.0 - self.cs) * self.ps[i] + c * zmean[i]

            psn = norm(self.ps)
            self.sigma *= math.exp((self.cs / self.damps) * (psn / self.chiN - 1.0))
            sig_min = 1e-14 * avg_scale
            sig_max = 0.9 * avg_scale
            if self.sigma < sig_min: self.sigma = sig_min
            if self.sigma > sig_max: self.sigma = sig_max

            # hsig
            denom = math.sqrt(max(1e-300, 1.0 - (1.0 - self.cs) ** (2.0 * (self.eig_it + 1))))
            hsig = 1.0 if (psn / denom) < (1.4 + 2.0 / (n + 1.0)) * self.chiN else 0.0

            cc = self.cc
            cpc = math.sqrt(cc * (2.0 - cc) * self.mueff)
            dm = [(self.m[i] - oldm[i]) / max(1e-300, self.sigma) for i in range(n)]
            for i in range(n):
                self.pc[i] = (1.0 - cc) * self.pc[i] + hsig * cpc * dm[i]

            c1, cmu = self.c1, self.cmu

            # C = (1-c1-cmu)C + c1*pcpc + cmu * sum w_i * y_i y_i^T
            mat_scale_inplace(self.C, (1.0 - c1 - cmu))
            mat_add_inplace(self.C, outer(self.pc, self.pc), alpha=c1)

            # rank-mu
            for k in range(self.mu):
                wk = self.w[k]
                yk = evaluated[k][3]
                mat_add_inplace(self.C, outer(yk, yk), alpha=(cmu * wk))

            # refactor occasionally
            self.eig_it += 1
            if self.eig_it % max(1, (n // 2)) == 0:
                self.refactor()

            fbest = evaluated[0][0]
            if fbest < self.bestf:
                self.bestf = fbest
                self.bestx = list(evaluated[0][1])
                self.stall = 0
            else:
                self.stall += 1

    # ---------------- micro pattern-search polish (very cheap) ----------------
    def micro_polish(x0, f0, evals_max):
        x = list(x0)
        fx = f0
        used = 0

        # random directions, shrinking step
        step = 0.03 * avg_scale
        step_min = 1e-12 * avg_scale

        while used < evals_max and time.time() < deadline and step > step_min:
            improved = False
            # try a few random directions per step size
            tries = max(4, min(12, 2 * dim))
            for _ in range(tries):
                if used >= evals_max or time.time() >= deadline:
                    break
                d = [random.gauss(0.0, 1.0) for _ in range(dim)]
                dn = norm(d)
                if dn <= 0.0:
                    continue
                inv = 1.0 / dn
                d = [di * inv for di in d]
                # try +/- step along d
                for sgn in (-1.0, 1.0):
                    if used >= evals_max or time.time() >= deadline:
                        break
                    xn = [x[i] + sgn * step * d[i] for i in range(dim)]
                    reflect_repair(xn)
                    fn = safe_eval(xn)
                    used += 1
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.55
        return x, fx, used

    # ---------------- initialization ----------------
    NP0 = max(18, min(90, 14 + 6 * dim))
    NPmin = max(8, min(36, 10 + 2 * dim))
    NP = NP0

    pop, fit = [], []
    k = 1
    while len(pop) < NP and time.time() < deadline:
        x = rand_point() if (len(pop) % 4 == 0) else halton_point(k)
        if len(pop) % 4 != 0:
            k += 1

        fx = safe_eval(x)
        if random.random() < 0.7:
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

    # ---------------- L-SHADE DE state ----------------
    archive = []
    Amax = 2 * NP0

    H = max(6, min(28, 6 + dim // 2))
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    def p_fraction(frac_time, NP_):
        pmin = 2.0 / max(2, NP_)
        # slightly more exploitative than your best, but not too early
        p = 0.21 - 0.13 * frac_time
        if p < pmin: p = pmin
        if p > 0.28: p = 0.28
        return p

    # local solver state (created lazily, refreshed on improvement)
    es = None
    last_es_best = float("inf")

    it = 0
    no_best_improve = 0
    inj_patience = max(80, 25 * dim)

    while time.time() < deadline:
        it += 1
        now = time.time()
        frac_time = (now - t0) / max(1e-9, float(max_time))

        # shrink DE population
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

        # --- occasional injections on stagnation (diversify) ---
        if no_best_improve > inj_patience and time.time() < deadline:
            nrep = max(1, NP // 4)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for wi in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.50:
                    x = halton_point(k); k += 1
                else:
                    # around incumbent but wide
                    x = [best_x[j] + random.gauss(0.0, 0.55 * scale[j]) for j in range(dim)]
                    reflect_repair(x)
                fx = safe_eval(x)
                pop[wi] = list(x)
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, list(x)
                    no_best_improve = 0
            # refresh local model
            es = None
            no_best_improve = inj_patience // 2

        # --- CMA bursts late or on stagnation ---
        want_local = (frac_time > 0.62) or (no_best_improve > inj_patience and frac_time > 0.30)
        if want_local and (it % max(6, 2 * dim) == 0 or no_best_improve > inj_patience):
            if es is None or best < last_es_best - 1e-15:
                es = CMAES(best_x, best)
                es.refactor()
                last_es_best = best

            # time-aware short burst
            burst_gens = max(1, min(7, 2 + int(1.5 * math.sqrt(dim))))
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
                    last_es_best = best

                # if stalling, gently expand
                if es.stall >= 5 and frac_time < 0.98:
                    es.sigma *= 1.25
                    es.stall = 0

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
            if F <= 0.0: F = 0.1
            if F > 1.0: F = 1.0

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

        # memory update (weighted)
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

    # very last chance: micro-polish around incumbent (few evals, robust)
    if time.time() < deadline:
        budget = min(40, max(10, 3 * dim))
        _, best2, _ = micro_polish(best_x, best, budget)
        if best2 < best:
            best = best2

    return best
