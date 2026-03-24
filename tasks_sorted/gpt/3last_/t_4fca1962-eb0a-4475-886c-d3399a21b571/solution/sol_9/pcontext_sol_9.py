import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improved vs the provided variants:
      - Stronger global phase: JADE-style current-to-pbest with *external archive*,
        plus *eigencoordinate crossover* (rotationally-invariant mixing without heavy math).
      - Stronger local phase: bounded *Powell-like direction set* search around best,
        with adaptive step + occasional coordinate refresh (good on smooth/ill-conditioned problems).
      - Better robustness: mixed initialization (Halton + opposition + center jitter),
        multiple restarts triggered by stall, and careful bounds repair.

    Returns:
        best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    # ---------- helpers ----------
    def eval_f(x):
        return float(func(x))

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def repair_midpoint_inplace(x, ref):
        # If out of bounds, bring halfway back toward ref (keeps feasibility & avoids sticking on bounds).
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = 0.5 * (lows[i] + ref[i])
            elif x[i] > highs[i]:
                x[i] = 0.5 * (highs[i] + ref[i])
        return x

    def gauss01():
        # approx N(0,1) via CLT
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_clip():
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        if v > 30.0:
            v = 30.0
        elif v < -30.0:
            v = -30.0
        return v

    def norm2(v):
        s = 0.0
        for a in v:
            s += a * a
        return s

    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    # ---------- scrambled Halton seeding ----------
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
              137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
              199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
              277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
              359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433]

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
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

    # ---------- population sizes ----------
    # Slightly larger initial pop than previous (but still capped) improves robustness.
    NP0 = 12 + 7 * int(math.sqrt(max(1, dim)))
    NP0 = max(28, min(NP0, 110))
    NPmin = max(10, min(26, NP0))

    # ---------- init population (oversample + keep best) ----------
    pop, fit = [], []
    best = float("inf")
    best_x = None

    k_hal = 1
    overs = 3 * NP0
    cand, candf = [], []

    # center jitter helper
    def jitter_center(scale):
        x = center[:]
        for j in range(dim):
            if spans[j] > 0.0:
                x[j] += (2.0 * random.random() - 1.0) * scale * spans[j]
        clip_inplace(x)
        return x

    while len(cand) < overs and time.time() < deadline:
        x = halton_point(k_hal); k_hal += 1
        fx = eval_f(x)
        cand.append(x); candf.append(fx)

        xo = [lows[j] + highs[j] - x[j] for j in range(dim)]
        clip_inplace(xo)
        fo = eval_f(xo)
        cand.append(xo); candf.append(fo)

        xm = [(x[j] + center[j]) * 0.5 for j in range(dim)]
        clip_inplace(xm)
        fm = eval_f(xm)
        cand.append(xm); candf.append(fm)

        if len(cand) < overs:
            xj = jitter_center(0.20)
            fj = eval_f(xj)
            cand.append(xj); candf.append(fj)

    if not cand:
        return best

    order0 = list(range(len(cand)))
    order0.sort(key=lambda i: candf[i])
    order0 = order0[:NP0]
    for idx in order0:
        pop.append(cand[idx][:])
        fit.append(candf[idx])
        if candf[idx] < best:
            best = candf[idx]
            best_x = cand[idx][:]

    # ---------- archive for JADE ----------
    archive = []
    arch_cap = 2 * NP0

    # ---------- JADE adaptation ----------
    mu_F = 0.55
    mu_CR = 0.55
    c_adapt = 0.10

    def sample_F(mu):
        F = mu + 0.1 * cauchy_clip()
        tries = 0
        while F <= 0.0 and tries < 12:
            F = mu + 0.1 * cauchy_clip()
            tries += 1
        if F <= 0.0:
            F = 0.05
        if F > 1.0:
            F = 1.0
        return F

    def sample_CR(mu):
        CR = mu + 0.1 * gauss01()
        if CR < 0.0:
            CR = 0.0
        if CR > 1.0:
            CR = 1.0
        return CR

    # ---------- eigencoordinate crossover (cheap random orthonormal basis via Gram-Schmidt) ----------
    # Rebuilt occasionally from random vectors; used to do crossover in rotated coordinates sometimes.
    def random_orthonormal_basis(m):
        # returns list of m vectors of length m, orthonormal
        basis = []
        for _ in range(m):
            v = [gauss01() for _ in range(m)]
            # subtract projections
            for b in basis:
                proj = dot(v, b)
                for i in range(m):
                    v[i] -= proj * b[i]
            n = math.sqrt(norm2(v))
            if n < 1e-12:
                # fallback: standard basis vector
                v = [0.0] * m
                v[len(basis)] = 1.0
                n = 1.0
            inv = 1.0 / n
            for i in range(m):
                v[i] *= inv
            basis.append(v)
        return basis

    # Keep a small rotating block (subspace) for eigencrossover to avoid O(dim^3).
    eig_idxs = None
    eig_basis = None
    eig_refresh_gen = 0

    def maybe_refresh_eig(gen, frac_left):
        nonlocal eig_idxs, eig_basis, eig_refresh_gen
        # refresh more often early; also when stalled later (handled outside by forcing refresh)
        period = 7 if frac_left > 0.5 else (11 if frac_left > 0.2 else 17)
        if eig_basis is None or (gen - eig_refresh_gen) >= period:
            m = 0
            if dim <= 8:
                m = dim
            elif dim <= 25:
                m = 8
            else:
                m = 10
            m = min(m, dim)
            idxs = list(range(dim))
            random.shuffle(idxs)
            eig_idxs = idxs[:m]
            eig_basis = random_orthonormal_basis(m)
            eig_refresh_gen = gen

    # ---------- local search: bounded Powell-like direction set ----------
    def powell_refine(x0, f0, frac_left):
        if x0 is None or time.time() >= deadline:
            return x0, f0

        # number of directions
        m = dim if dim <= 10 else (10 if dim <= 30 else 12)
        m = min(m, dim)

        # pick working indices and initialize directions as coordinate axes in that subspace
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:m]

        dirs = []
        for t in range(m):
            d = [0.0] * dim
            d[idxs[t]] = 1.0
            dirs.append(d)

        # step scale
        base = 0.08 if frac_left > 0.25 else (0.05 if frac_left > 0.12 else 0.03)
        step0 = base

        x = x0[:]
        fx = f0

        # limited iterations (time-safe)
        iters = 1 if frac_left < 0.08 else (2 if frac_left < 0.18 else 3)
        if dim <= 12 and frac_left > 0.25:
            iters = 4

        def line_search(xb, fb, d, step_frac):
            # try steps along +/-d with backtracking
            bestl_x = xb
            bestl_f = fb
            # scale by average span on active components
            scale = 0.0
            cnt = 0
            for j in range(dim):
                if d[j] != 0.0 and spans[j] > 0.0:
                    scale += spans[j]
                    cnt += 1
            if cnt == 0:
                return bestl_x, bestl_f
            scale /= cnt
            step = step_frac * scale

            for sgn in (1.0, -1.0):
                alpha = 1.0
                for _ in range(6):
                    if time.time() >= deadline:
                        return bestl_x, bestl_f
                    xt = bestl_x[:] if bestl_x is not xb else xb[:]
                    for j in range(dim):
                        if d[j] != 0.0:
                            xt[j] = xb[j] + sgn * alpha * step * d[j]
                    repair_midpoint_inplace(xt, xb)
                    ft = eval_f(xt)
                    if ft < bestl_f:
                        bestl_x, bestl_f = xt, ft
                        # try a bit bigger (one expansion)
                        alpha *= 1.6
                        continue
                    alpha *= 0.5
            return bestl_x, bestl_f

        for _ in range(iters):
            if time.time() >= deadline:
                break
            x_start = x[:]
            f_start = fx

            # sweep directions
            for d in dirs:
                if time.time() >= deadline:
                    break
                xn, fn = line_search(x, fx, d, step0)
                if fn < fx:
                    x, fx = xn, fn

            # construct new direction (Powell): x - x_start
            if time.time() >= deadline:
                break
            delta = [x[i] - x_start[i] for i in range(dim)]
            if norm2(delta) > 1e-18:
                # normalize delta direction in subspace-ish sense
                n = math.sqrt(norm2(delta))
                inv = 1.0 / (n + 1e-300)
                for i in range(dim):
                    delta[i] *= inv
                # replace the worst direction (randomly, cheap) with delta
                dirs[random.randrange(len(dirs))] = delta

            # if no progress, reduce step
            if fx >= f_start - 1e-12:
                step0 *= 0.65
                if step0 < 0.01:
                    break
            else:
                step0 *= 1.08
                if step0 > 0.18:
                    step0 = 0.18

        return x, fx

    # ---------- main loop ----------
    gen = 0
    last_best = best
    stall = 0

    while time.time() < deadline:
        gen += 1
        rem = deadline - time.time()
        frac_left = rem / max(1e-12, float(max_time))

        # population reduction (linear)
        NP_target = int(round(NPmin + (NP0 - NPmin) * frac_left))
        NP_target = max(NPmin, min(NP0, NP_target))
        while len(pop) > NP_target:
            worst = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(worst)
            fit.pop(worst)
        NP = len(pop)

        # archive size management
        arch_cap = 2 * max(1, NP)
        while len(archive) > arch_cap:
            archive.pop(random.randrange(len(archive)))

        # refresh eigencrossover subspace/basis
        maybe_refresh_eig(gen, frac_left)

        # occasional injection (diversity)
        if gen % 12 == 0 and frac_left > 0.35 and time.time() < deadline:
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
                    stall = 0

        # rank for pbest
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = 0.22 if frac_left > 0.6 else (0.15 if frac_left > 0.25 else 0.10)
        pbest_count = max(2, int(p * NP))

        pool = pop + archive
        pool_n = len(pool)

        SF, SCR, dF = [], [], []

        # use eigencrossover sometimes (helps rotated problems)
        use_eig = (eig_basis is not None) and (random.random() < (0.35 if frac_left > 0.35 else 0.25))

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            F = sample_F(mu_F)
            CR = sample_CR(mu_CR)

            # choose pbest and r1, r2
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
                while tries < 24 and (pool[r2] is xi or pool[r2] is xr1):
                    r2 = random.randrange(pool_n)
                    tries += 1
                xr2 = pool[r2]

            # mutation: current-to-pbest/1 with archive
            v = [0.0] * dim
            for j in range(dim):
                if spans[j] <= 0.0:
                    v[j] = xi[j]
                else:
                    v[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            # crossover (either normal binomial or eigencrossover in a subspace)
            u = xi[:]
            if dim == 1:
                u[0] = v[0]
            else:
                if use_eig and eig_idxs and eig_basis:
                    # transform subspace coords: y = Q^T x, apply crossover there, then back x = Q y
                    m = len(eig_idxs)
                    # extract sub-vectors
                    xs = [xi[j] for j in eig_idxs]
                    vs = [v[j] for j in eig_idxs]

                    # y = Q^T x  (basis vectors are rows of Q here)
                    yx = [0.0] * m
                    yv = [0.0] * m
                    for a in range(m):
                        ba = eig_basis[a]
                        s1 = 0.0
                        s2 = 0.0
                        for b in range(m):
                            s1 += ba[b] * xs[b]
                            s2 += ba[b] * vs[b]
                        yx[a] = s1
                        yv[a] = s2

                    # binomial crossover in y-space
                    jrand = random.randrange(m)
                    yu = yx[:]
                    for a in range(m):
                        if a == jrand or random.random() < CR:
                            yu[a] = yv[a]

                    # back transform: x = Q y  (columns)
                    # since basis rows are orthonormal, Q^{-1}=Q^T, and x = Q^T y using rows:
                    # For orthonormal rows, reconstruct via sum_a y[a]*basis[a][b]
                    us = [0.0] * m
                    for b in range(m):
                        s = 0.0
                        for a in range(m):
                            s += yu[a] * eig_basis[a][b]
                        us[b] = s

                    for t, j in enumerate(eig_idxs):
                        u[j] = us[t]
                else:
                    jrand = random.randrange(dim)
                    for j in range(dim):
                        if j == jrand or random.random() < CR:
                            u[j] = v[j]

            repair_midpoint_inplace(u, xi)
            fu = eval_f(u)

            if fu <= fi:
                # archive add old xi
                if len(archive) < arch_cap:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_cap)] = xi[:]

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = u[:]

                SF.append(F)
                SCR.append(CR)
                df = fi - fu
                if df < 0.0:
                    df = 0.0
                dF.append(df)
            else:
                # tiny late acceptance to keep movement if stuck
                if frac_left < 0.08 and random.random() < 0.012:
                    pop[i] = u
                    fit[i] = fu
                    if fu < best:
                        best = fu
                        best_x = u[:]

        # JADE update (Lehmer mean for F, arithmetic for CR)
        if dF:
            wsum = sum(dF) + 1e-300
            # mu_F update via Lehmer mean
            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for fval, crval, w in zip(SF, SCR, dF):
                ww = w / wsum
                num += ww * fval * fval
                den += ww * fval
                cr_mean += ww * crval
            if den > 0.0:
                lehmer = num / (den + 1e-300)
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * cr_mean

            # keep in range
            if mu_F < 0.05:
                mu_F = 0.05
            if mu_F > 0.95:
                mu_F = 0.95
            if mu_CR < 0.05:
                mu_CR = 0.05
            if mu_CR > 0.95:
                mu_CR = 0.95

        # stall tracking
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        # local refinement: Powell-like
        if best_x is not None:
            trig = (stall >= 7 and random.random() < 0.55) or (frac_left < 0.22 and random.random() < 0.45)
            if trig and time.time() < deadline:
                xb, fb = powell_refine(best_x, best, frac_left)
                if fb < best:
                    best = fb
                    best_x = xb[:]
                    stall = 0
                else:
                    # if stuck hard, refresh eig basis and do a small restart around best
                    if stall >= 16 and frac_left < 0.18 and time.time() < deadline:
                        # force eig refresh next gen
                        eig_basis = None
                        # restart a few worst near best
                        krep = max(2, NP // 6)
                        for _ in range(krep):
                            if time.time() >= deadline:
                                break
                            w = max(range(NP), key=lambda ii: fit[ii])
                            xj = best_x[:]
                            rad = 0.05 if frac_left > 0.10 else 0.03
                            for j in range(dim):
                                if spans[j] > 0.0:
                                    xj[j] += (2.0 * random.random() - 1.0) * rad * spans[j]
                            repair_midpoint_inplace(xj, best_x)
                            fj = eval_f(xj)
                            if len(archive) < arch_cap:
                                archive.append(pop[w][:])
                            else:
                                archive[random.randrange(arch_cap)] = pop[w][:]
                            pop[w] = xj
                            fit[w] = fj
                            if fj < best:
                                best = fj
                                best_x = xj[:]
                                stall = 0

    return best
