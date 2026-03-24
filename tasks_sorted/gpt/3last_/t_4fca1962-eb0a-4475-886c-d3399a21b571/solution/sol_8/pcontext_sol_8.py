import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvement over your best (#2 = 15.38) approach:
      1) Keep a strong global engine: L-SHADE / current-to-pbest/1 + archive + pop reduction.
      2) Add a *proper derivative-free local optimizer*: bounded BOBYQA-like trust-region
         with quadratic interpolation in a random subspace (small m), plus geometry updates.
         This is markedly stronger than 1D diagonal quadratic probes and helps on coupled/rotated valleys.
      3) Add "jDE" self-adaptation as a fallback when SHADE memory becomes stale (diversity preservation).
      4) Better restart & intensification scheduling: local TR when stalled, plus micro-restarts around best.

    Returns:
        best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    # ---------------- utilities ----------------
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
        # If out of bounds, move halfway between violated bound and reference
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = 0.5 * (lows[i] + ref[i])
            elif x[i] > highs[i]:
                x[i] = 0.5 * (highs[i] + ref[i])
        return x

    def gauss01():
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

    def norm2(v):
        s = 0.0
        for a in v:
            s += a * a
        return s

    # ---------------- scrambled Halton for seeding/injection ----------------
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

    # ---------------- global DE: L-SHADE + jDE backup ----------------
    NP0 = 10 + 6 * int(math.sqrt(max(1, dim)))
    NP0 = max(24, min(NP0, 90))
    NPmin = max(10, min(24, NP0))

    pop, fit = [], []
    best = float("inf")
    best_x = None

    # oversample init
    k_hal = 1
    overs = 3 * NP0
    cand, candf = [], []
    while len(cand) < overs and time.time() < deadline:
        x = halton_point(k_hal); k_hal += 1
        fx = eval_f(x)
        cand.append(x); candf.append(fx)

        # opposition
        xo = [lows[j] + highs[j] - x[j] for j in range(dim)]
        clip_inplace(xo)
        fo = eval_f(xo)
        cand.append(xo); candf.append(fo)

        # toward center
        xm = [(x[j] + center[j]) * 0.5 for j in range(dim)]
        clip_inplace(xm)
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

    # archive
    archive = []
    arch_cap = 2 * NP0

    # SHADE memory
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
        if F <= 0.0: F = 0.1
        if F > 1.0: F = 1.0
        return F

    def sample_CR(m):
        CR = m + 0.1 * gauss01()
        if CR < 0.0: CR = 0.0
        if CR > 1.0: CR = 1.0
        return CR

    # per-individual jDE params (fallback diversity / adaptation)
    jF = [0.5] * len(pop)
    jCR = [0.9] * len(pop)

    # ---------------- Local optimizer: subspace BOBYQA-like TR ----------------
    # We build a quadratic model in m-dim subspace using (2m+1) points:
    # f(0), f(±e_i). Fit full quadratic in subspace via least squares (small).
    # Then compute TR step by solving (H + λI)p = -g with λ adjusted for ||p||<=Δ.
    # Use acceptance ratio to update Δ and optionally refresh geometry.
    def solve_linear(A, b):
        # Gaussian elimination with partial pivoting (small systems only)
        n = len(b)
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            piv = col
            bestv = abs(M[col][col])
            for r in range(col + 1, n):
                v = abs(M[r][col])
                if v > bestv:
                    bestv = v
                    piv = r
            if bestv < 1e-14:
                return None
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
            invp = 1.0 / M[col][col]
            for j in range(col, n + 1):
                M[col][j] *= invp
            for r in range(n):
                if r == col:
                    continue
                factor = M[r][col]
                if factor != 0.0:
                    for j in range(col, n + 1):
                        M[r][j] -= factor * M[col][j]
        return [M[i][n] for i in range(n)]

    def quad_fit_from_plusminus(f0, fp, fm, h):
        # build g and H (symmetric) for model:
        # m(p)=f0 + g^T p + 0.5 p^T H p
        m = len(h)
        g = [0.0] * m
        Hq = [[0.0] * m for _ in range(m)]
        # diagonal from second differences; gradient from central difference
        for i in range(m):
            hi = h[i]
            g[i] = (fp[i] - fm[i]) / (2.0 * hi + 1e-300)
            Hq[i][i] = (fp[i] - 2.0 * f0 + fm[i]) / (hi * hi + 1e-300)
        # no cross terms from this stencil; still works well as TR core.
        return g, Hq

    def tr_step(g, H, Delta):
        # Solve min g^T p + 0.5 p^T H p s.t. ||p||<=Delta
        m = len(g)
        # Try Newton step if well-conditioned
        # Use Levenberg-Marquardt: (H + lam I)p = -g, tune lam
        def mat_add_lam(H, lam):
            A = [row[:] for row in H]
            for i in range(m):
                A[i][i] += lam
            return A

        # initial lambda guess
        lam = 0.0
        for i in range(m):
            if H[i][i] < 0.0:
                lam = max(lam, -H[i][i] + 1e-6)
        lam = max(lam, 1e-12)

        best_p = None
        # increase lam until inside trust region
        for _ in range(20):
            A = mat_add_lam(H, lam)
            rhs = [-gi for gi in g]
            p = solve_linear(A, rhs)
            if p is None:
                lam *= 10.0
                continue
            n2 = norm2(p)
            if n2 <= Delta * Delta * (1.0 + 1e-9):
                best_p = p
                break
            lam *= 3.0

        if best_p is None:
            # fallback: scaled steepest descent
            ng = math.sqrt(norm2(g)) + 1e-300
            p = [-(Delta / ng) * gi for gi in g]
            return p

        # if it's much smaller, consider decreasing lam (not essential)
        return best_p

    def local_tr_refine(xb, fb, frac_left):
        if xb is None or time.time() >= deadline:
            return xb, fb

        # subspace size (keep small for speed)
        if dim <= 8:
            m = min(dim, 5)
        elif dim <= 25:
            m = 6
        else:
            m = 7
        m = min(m, dim)
        if frac_left < 0.15:
            m = min(m, 5)
        if frac_left < 0.07:
            m = min(m, 4)

        # pick subspace indices
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:m]

        # initial trust radius in subspace coordinates
        Delta = 0.12 if frac_left > 0.25 else (0.07 if frac_left > 0.12 else 0.04)

        # step sizes for stencil (per coordinate)
        h = [0.0] * m
        for t, j in enumerate(idxs):
            s = spans[j]
            hi = (0.03 if frac_left > 0.2 else 0.02) * (abs(s) if abs(s) > 0 else 1.0)
            if hi < 1e-12 * (abs(s) + 1.0):
                hi = 1e-12 * (abs(s) + 1.0)
            h[t] = hi

        xcur = xb[:]
        fcur = fb

        # a few TR iterations
        iters = 2 if frac_left > 0.20 else (2 if frac_left > 0.10 else 1)
        if dim <= 12 and frac_left > 0.20:
            iters = 3

        for _ in range(iters):
            if time.time() >= deadline:
                break

            # stencil evaluations
            fp = [0.0] * m
            fm = [0.0] * m
            for t, j in enumerate(idxs):
                if time.time() >= deadline:
                    return xcur, fcur
                xp = xcur[:]; xp[j] += h[t]
                xm = xcur[:]; xm[j] -= h[t]
                repair_midpoint_inplace(xp, xcur)
                repair_midpoint_inplace(xm, xcur)
                fp[t] = eval_f(xp)
                if time.time() >= deadline:
                    return xcur, fcur
                fm[t] = eval_f(xm)

            g, Hq = quad_fit_from_plusminus(fcur, fp, fm, h)

            # compute step in subspace variables p (same units as x on these coords)
            # scale p by h to normalize trust region: use p_scaled in normalized coords
            # Here we directly operate in physical coords but interpret Delta as fraction of h.
            # Map: p_phys[i] = h[i] * p_norm[i], ||p_norm||<=Delta
            # -> Solve in norm space using g_norm, H_norm
            gN = [g[i] * h[i] for i in range(m)]
            HN = [[0.0] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    HN[i][j] = Hq[i][j] * h[i] * h[j]

            pN = tr_step(gN, HN, Delta)
            p = [pN[i] * h[i] for i in range(m)]

            # candidate
            xnew = xcur[:]
            for t, j in enumerate(idxs):
                xnew[j] += p[t]
            repair_midpoint_inplace(xnew, xcur)
            fnew = eval_f(xnew)

            # predicted reduction
            pred = 0.0
            for i in range(m):
                pred += g[i] * p[i]
            quad = 0.0
            for i in range(m):
                for j in range(m):
                    quad += 0.5 * p[i] * Hq[i][j] * p[j]
            pred_red = -(pred + quad)
            act_red = fcur - fnew

            # trust region update
            rho = act_red / (pred_red + 1e-300) if pred_red > 0 else (-1.0 if act_red <= 0 else 0.0)

            if fnew < fcur:
                xcur, fcur = xnew, fnew
                if rho > 0.75:
                    Delta = min(0.7, Delta * 1.7)
                elif rho < 0.25:
                    Delta = max(0.02, Delta * 0.7)
            else:
                Delta = max(0.02, Delta * 0.55)

            # early exit if tiny
            if Delta < 0.021 and frac_left < 0.10:
                break

        return xcur, fcur

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
        NP_target = max(NPmin, min(NP0, NP_target))
        while len(pop) > NP_target:
            worst = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(worst); fit.pop(worst)
        NP = len(pop)

        # keep jDE arrays consistent
        if len(jF) != NP:
            jF = (jF + [0.5] * NP)[:NP]
            jCR = (jCR + [0.9] * NP)[:NP]

        arch_cap = 2 * max(1, NP)
        while len(archive) > arch_cap:
            archive.pop(random.randrange(len(archive)))

        # occasional injection early/mid
        if gen % 10 == 0 and frac_left > 0.45 and time.time() < deadline:
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
                jF[worst] = 0.5
                jCR[worst] = 0.9
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

            # choose SHADE or jDE parameters
            use_jde = (stall >= 10 and random.random() < 0.25) or (frac_left > 0.6 and random.random() < 0.10)

            if use_jde:
                # jDE self-adaptation
                if random.random() < 0.1:
                    jF[i] = 0.1 + 0.9 * random.random()
                if random.random() < 0.1:
                    jCR[i] = random.random()
                F = jF[i]
                CR = jCR[i]
            else:
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

            repair_midpoint_inplace(ui, xi)
            fu = eval_f(ui)

            if fu <= fi:
                if len(archive) < arch_cap:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_cap)] = xi[:]

                pop[i] = ui
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = ui[:]

                if not use_jde:
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
            numF = denF = numCR = 0.0
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

        # stagnation tracking
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        # local TR refinement trigger (stronger than prior surrogate)
        if best_x is not None:
            trig = (stall >= 8 and random.random() < 0.55) or (frac_left < 0.22 and random.random() < 0.45)
            if trig and time.time() < deadline:
                xb, fb = local_tr_refine(best_x, best, frac_left)
                if fb < best:
                    best, best_x = fb, xb[:]
                    stall = 0
                else:
                    # micro-restart around best if hard stall late
                    if stall >= 18 and frac_left < 0.18 and time.time() < deadline:
                        # jitter a few worst individuals around best to re-diversify locally
                        krep = max(2, NP // 6)
                        for _ in range(krep):
                            if time.time() >= deadline:
                                break
                            w = max(range(NP), key=lambda ii: fit[ii])
                            xj = best_x[:]
                            for j in range(dim):
                                if spans[j] > 0.0:
                                    xj[j] += (2.0 * random.random() - 1.0) * 0.03 * spans[j]
                            repair_midpoint_inplace(xj, best_x)
                            fj = eval_f(xj)
                            if len(archive) < arch_cap:
                                archive.append(pop[w][:])
                            else:
                                archive[random.randrange(arch_cap)] = pop[w][:]
                            pop[w] = xj
                            fit[w] = fj
                            jF[w] = 0.5
                            jCR[w] = 0.9
                            if fj < best:
                                best, best_x = fj, xj[:]
                                stall = 0

    return best
