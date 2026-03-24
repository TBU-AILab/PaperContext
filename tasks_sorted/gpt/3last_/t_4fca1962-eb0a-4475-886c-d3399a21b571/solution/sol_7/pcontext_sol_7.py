import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvement focus vs your best (#1):
      1) Keep strong global engine: L-SHADE current-to-pbest/1 + archive.
      2) Add *CMA-ES-lite* local search on the best-so-far:
           - learns a full covariance (via Cholesky) from successful steps
           - sampled steps are evaluated and used to update mean/cov
         This is a big upgrade over diagonal/quadratic surrogate on rotated valleys.
      3) Better late-stage intensification: adaptive trust radius + restarts
         around best with shrinking sigma.
      4) Still fully self-contained, time-safe (checks deadline often).

    Returns:
        best (float): best fitness found
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    # ---------------- basic helpers ----------------
    def eval_f(x):
        return float(func(x))

    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def repair_midpoint(x, ref):
        # If out of bounds, put it midway between violated bound and reference
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = 0.5 * (lows[i] + ref[i])
            elif x[i] > highs[i]:
                x[i] = 0.5 * (highs[i] + ref[i])
        return x

    def gauss01():
        # approx N(0,1)
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_clip():
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        if v > 30.0: v = 30.0
        if v < -30.0: v = -30.0
        return v

    # ---------------- scrambled Halton seeding ----------------
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
              137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
              199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
              277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
              359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433]

    def is_prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        r = int(n ** 0.5)
        f = 3
        while f <= r:
            if n % f == 0:
                return False
            f += 2
        return True

    def ensure_primes(k):
        nonlocal PRIMES
        if len(PRIMES) >= k:
            return
        p = PRIMES[-1] + 2
        while len(PRIMES) < k:
            if is_prime(p):
                PRIMES.append(p)
            p += 2

    ensure_primes(dim)
    digit_perms = []
    for i in range(dim):
        base = PRIMES[i]
        perm = list(range(base))
        random.shuffle(perm)
        digit_perms.append(perm)

    def halton_scrambled(idx, base, perm):
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_scrambled(k, PRIMES[i], digit_perms[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- small linear algebra (no numpy) ----------------
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

    def symmetrize_inplace(A):
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                v = 0.5 * (A[i][j] + A[j][i])
                A[i][j] = v
                A[j][i] = v

    def cholesky_spd(A):
        # returns lower-triangular L such that A ~ L L^T
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 1e-18:
                        return None
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / (L[j][j] + 1e-300)
        return L

    def enforce_spd(C, diag_floor):
        # force symmetry and add diag if needed
        symmetrize_inplace(C)
        n = len(C)
        for i in range(n):
            if C[i][i] < diag_floor:
                C[i][i] = diag_floor
        # try a couple of diagonal shifts if cholesky fails
        shift = 0.0
        for _ in range(6):
            L = cholesky_spd(C)
            if L is not None:
                return L
            shift = diag_floor if shift == 0.0 else shift * 10.0
            for i in range(n):
                C[i][i] += shift
        # last resort: diagonal
        for i in range(n):
            for j in range(n):
                C[i][j] = 0.0
            C[i][i] = max(C[i][i], diag_floor)
        return cholesky_spd(C)

    # ---------------- L-SHADE parameters ----------------
    NP0 = 10 + 6 * int(math.sqrt(max(1, dim)))
    NP0 = max(24, min(NP0, 90))
    NPmin = max(10, min(24, NP0))

    # init: oversample then keep best NP0
    pop, fit = [], []
    best = float("inf")
    best_x = None

    k_hal = 1
    overs = 3 * NP0
    cand, candf = [], []
    while len(cand) < overs and time.time() < deadline:
        x = halton_point(k_hal); k_hal += 1
        fx = eval_f(x)
        cand.append(x); candf.append(fx)

        xo = [lows[j] + highs[j] - x[j] for j in range(dim)]
        clip(xo)
        fo = eval_f(xo)
        cand.append(xo); candf.append(fo)

        xm = [(x[j] + center[j]) * 0.5 for j in range(dim)]
        clip(xm)
        fm = eval_f(xm)
        cand.append(xm); candf.append(fm)

    if not cand:
        return best

    order0 = list(range(len(cand)))
    order0.sort(key=lambda i: candf[i])
    order0 = order0[:NP0]
    for idx in order0:
        pop.append(cand[idx])
        fit.append(candf[idx])
        if candf[idx] < best:
            best = candf[idx]
            best_x = cand[idx][:]

    archive = []
    arch_cap = 2 * NP0

    H = 10
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mptr = 0

    def sample_F(m):
        F = m + 0.1 * cauchy_clip()
        tries = 0
        while F <= 0.0 and tries < 10:
            F = m + 0.1 * cauchy_clip()
            tries += 1
        if F <= 0.0:
            F = 0.1
        if F > 1.0:
            F = 1.0
        return F

    def sample_CR(m):
        CR = m + 0.1 * gauss01()
        if CR < 0.0: CR = 0.0
        if CR > 1.0: CR = 1.0
        return CR

    # ---------------- CMA-ES-lite local search on best ----------------
    # We keep a small covariance in full dim (works up to moderate dims);
    # if dim is large, we reduce the update frequency & sample count.
    class CMALite:
        def __init__(self, x0):
            self.mean = x0[:]  # in original coordinates
            # initial sigma relative to domain
            avg_span = 0.0
            for s in spans:
                avg_span += abs(s)
            avg_span = avg_span / max(1, dim)
            self.sigma = 0.15 * (avg_span if avg_span > 0 else 1.0)
            # covariance (start isotropic)
            self.C = [[0.0] * dim for _ in range(dim)]
            base = 1.0
            for i in range(dim):
                self.C[i][i] = base
            self.L = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                self.L[i][i] = 1.0
            self.last_update = 0
            self.succ_steps = []  # store a few successful normalized steps

        def set_mean(self, x):
            self.mean = x[:]

        def maybe_update_chol(self):
            # keep diag floor to avoid degeneracy
            diag_floor = 1e-12
            L = enforce_spd(self.C, diag_floor)
            if L is None:
                # should not happen due to enforce_spd fallback
                return
            self.L = L

        def sample(self, lam):
            # return list of candidates and their steps z (in whitened space)
            xs = []
            zs = []
            for _ in range(lam):
                z = [gauss01() for _ in range(dim)]
                y = mat_vec(self.L, z)  # correlated in C space
                x = [self.mean[i] + self.sigma * y[i] for i in range(dim)]
                repair_midpoint(x, self.mean)
                xs.append(x)
                zs.append(z)
            return xs, zs

        def tell(self, z_steps, x_steps, weights):
            # Rank-1 / rank-m update on covariance using successful steps in y-space
            # We store successful normalized steps y = (x-mean)/sigma
            if not z_steps:
                self.sigma *= 0.93
                if self.sigma < 1e-18:
                    self.sigma = 1e-18
                return

            # compute weighted mean step in y-space (using actual step)
            ybar = [0.0] * dim
            wsum = 0.0
            for y, w in zip(x_steps, weights):
                wsum += w
                for i in range(dim):
                    ybar[i] += w * y[i]
            if wsum <= 0.0:
                return
            inv = 1.0 / wsum
            for i in range(dim):
                ybar[i] *= inv

            # covariance update: C = (1-a)C + a * sum w * (y y^T)
            a = 0.22 if dim <= 30 else 0.12
            mat_scale_inplace(self.C, 1.0 - a)
            for y, w in zip(x_steps, weights):
                yy = outer(y, y)
                mat_add_inplace(self.C, yy, alpha=a * (w / (wsum + 1e-300)))

            # add a small rank-1 on ybar to stabilize directionality
            yyb = outer(ybar, ybar)
            mat_add_inplace(self.C, yyb, alpha=0.12 * a)

            # step-size adaptation: if steps exist, slightly increase
            self.sigma *= 1.08
            # clamp sigma to domain scale
            max_span = max([abs(s) for s in spans] + [1.0])
            if self.sigma > 0.6 * max_span:
                self.sigma = 0.6 * max_span

            self.maybe_update_chol()

    cma = CMALite(best_x if best_x is not None else center)

    def cma_polish(xb, fb, frac_left):
        if xb is None:
            return xb, fb
        # time-adaptive lambda
        if dim <= 12:
            lam = 8
        elif dim <= 30:
            lam = 10
        else:
            lam = 6
        if frac_left < 0.12:
            lam = max(4, lam - 3)
        if frac_left < 0.06:
            lam = max(3, lam - 1)

        cma.set_mean(xb)
        # ensure chol exists
        if cma.L is None or len(cma.L) != dim:
            cma.maybe_update_chol()

        xs, zs = cma.sample(lam)

        # evaluate and keep best few
        scored = []
        for x, z in zip(xs, zs):
            if time.time() >= deadline:
                break
            fx = eval_f(x)
            scored.append((fx, x, z))
        if not scored:
            return xb, fb

        scored.sort(key=lambda t: t[0])
        # accept the best if improves
        fbest = scored[0][0]
        xbest = scored[0][1]
        if fbest < fb:
            xb2, fb2 = xbest[:], fbest
        else:
            xb2, fb2 = xb, fb

        # build covariance update from top mu
        mu = max(2, len(scored) // 2)
        top = scored[:mu]

        # steps in y-space: y = (x - mean)/sigma
        y_steps = []
        z_steps = []
        weights = []
        # log weights
        wsum = 0.0
        for k in range(mu):
            w = math.log(mu + 0.5) - math.log(k + 1.0)
            if w < 0.0: w = 0.0
            weights.append(w)
            wsum += w
        if wsum <= 0.0:
            weights = [1.0] * mu

        for (fx, x, z), w in zip(top, weights):
            y = [(x[i] - xb[i]) / (cma.sigma + 1e-300) for i in range(dim)]
            y_steps.append(y)
            z_steps.append(z)

        cma.tell(z_steps, y_steps, weights)

        # if no improvement, try shrinking sigma a bit (trust-region behavior)
        if fb2 >= fb - 1e-15:
            cma.sigma *= 0.85

        return xb2, fb2

    # ---------------- main loop ----------------
    gen = 0
    last_best = best
    stall = 0

    while time.time() < deadline:
        gen += 1
        rem = deadline - time.time()
        frac_left = rem / max(1e-12, float(max_time))

        # population reduction
        NP_target = int(round(NPmin + (NP0 - NPmin) * frac_left))
        if NP_target < NPmin: NP_target = NPmin
        if NP_target > NP0: NP_target = NP0

        while len(pop) > NP_target:
            worst = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(worst); fit.pop(worst)

        NP = len(pop)
        arch_cap = 2 * max(1, NP)
        if len(archive) > arch_cap:
            while len(archive) > arch_cap:
                archive.pop(random.randrange(len(archive)))

        # occasional injection early
        if frac_left > 0.55 and gen % 10 == 0 and time.time() < deadline:
            xg = halton_point(k_hal); k_hal += 1
            fg = eval_f(xg)
            worst = max(range(NP), key=lambda i: fit[i])
            if fg < fit[worst]:
                if len(archive) < arch_cap:
                    archive.append(pop[worst][:])
                else:
                    archive[random.randrange(arch_cap)] = pop[worst][:]
                pop[worst] = xg
                fit[worst] = fg
                if fg < best:
                    best = fg
                    best_x = xg[:]

        order = list(range(NP))
        order.sort(key=lambda i: fit[i])

        p = 0.26 if frac_left > 0.65 else (0.18 if frac_left > 0.25 else 0.10)
        pbest_count = max(2, int(p * NP))

        S_F, S_CR, dF = [], [], []

        pool = pop + archive
        pool_n = len(pool)

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            rmem = random.randrange(H)
            F = sample_F(M_F[rmem])
            CR = sample_CR(M_CR[rmem])

            pbest = order[random.randrange(pbest_count)]
            xp = pop[pbest]

            r1 = random.randrange(NP)
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            if pool_n <= 2:
                xr2 = pop[random.randrange(NP)]
            else:
                r2 = random.randrange(pool_n)
                tries = 0
                while tries < 20 and (pool[r2] is xi or pool[r2] is xr1):
                    r2 = random.randrange(pool_n)
                    tries += 1
                xr2 = pool[r2]

            # mutation current-to-pbest/1
            vi = [0.0] * dim
            for j in range(dim):
                if spans[j] <= 0.0:
                    vi[j] = xi[j]
                else:
                    vi[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            # crossover
            ui = xi[:]
            if dim == 1:
                ui[0] = vi[0]
            else:
                jrand = random.randrange(dim)
                for j in range(dim):
                    if j == jrand or random.random() < CR:
                        ui[j] = vi[j]

            repair_midpoint(ui, xi)
            fu = eval_f(ui)

            if fu <= fi:
                if len(archive) < arch_cap:
                    archive.append(xi[:])
                else:
                    if arch_cap > 0:
                        archive[random.randrange(arch_cap)] = xi[:]

                pop[i] = ui
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = ui[:]

                S_F.append(F)
                S_CR.append(CR)
                df = fi - fu
                if df < 0.0: df = 0.0
                dF.append(df)
            else:
                # tiny late mobility
                if frac_left < 0.10 and random.random() < 0.015:
                    pop[i] = ui
                    fit[i] = fu
                    if fu < best:
                        best = fu
                        best_x = ui[:]

        # update SHADE memories
        if dF:
            wsum = sum(dF) + 1e-300
            numF = 0.0
            denF = 0.0
            numCR = 0.0
            for fval, crval, w in zip(S_F, S_CR, dF):
                ww = w / wsum
                numF += ww * (fval * fval)
                denF += ww * fval
                numCR += ww * crval
            newMF = (numF / (denF + 1e-300)) if denF > 0.0 else M_F[mptr]
            newMCR = numCR
            if newMF <= 0.0: newMF = 0.1
            if newMF > 1.0: newMF = 1.0
            if newMCR < 0.0: newMCR = 0.0
            if newMCR > 1.0: newMCR = 1.0
            M_F[mptr] = newMF
            M_CR[mptr] = newMCR
            mptr = (mptr + 1) % H

        # stagnation
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        # --- improved local polish: CMA-lite triggers ---
        if best_x is not None:
            # trigger more when stalled or late
            if (frac_left < 0.28 and random.random() < 0.45) or (stall >= 8 and random.random() < 0.55):
                xb, fb = cma_polish(best_x, best, frac_left)
                if fb < best:
                    best = fb
                    best_x = xb[:]
                    stall = 0

            # late aggressive restart around best with smaller sigma if still stalled
            if stall >= 18 and frac_left < 0.18 and random.random() < 0.35:
                # shrink sigma and re-polish quickly
                cma.sigma *= 0.6
                xb, fb = cma_polish(best_x, best, frac_left)
                if fb < best:
                    best = fb
                    best_x = xb[:]
                    stall = 0

    return best
