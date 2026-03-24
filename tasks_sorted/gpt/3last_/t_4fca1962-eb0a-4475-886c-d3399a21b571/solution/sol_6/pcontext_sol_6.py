import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Upgrade vs your best (#1, L-SHADE-ish):
      - Keeps L-SHADE current-to-pbest/1 + archive (strong global engine)
      - Adds *surrogate-guided local trust region* around best:
          builds a cheap quadratic model in a random subspace (k dims),
          proposes a step, then does a small line-search/backtracking.
        This often beats pure pattern search polishing on smooth-ish tasks.
      - Uses *population reduction* (L-SHADE style) to shift evals to exploitation.
      - Adds *multiple small local refinements* triggered by stagnation + late time.
      - More robust bounds handling (midpoint repair) + occasional re-centering.

    Returns:
        best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def repair_midpoint(x, ref):
        # If out of bounds, place between bound and reference coordinate
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = 0.5 * (lows[i] + ref[i])
            elif x[i] > highs[i]:
                x[i] = 0.5 * (highs[i] + ref[i])
        return x

    def eval_f(x):
        return float(func(x))

    def gauss01():
        # approx N(0,1)
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_clip():
        # clipped Cauchy
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        if v > 30.0: v = 30.0
        if v < -30.0: v = -30.0
        return v

    # --- scrambled Halton for seeding/injection ---
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

    # --- initial population size (with later reduction) ---
    NP0 = 10 + 6 * int(math.sqrt(max(1, dim)))
    NP0 = max(24, min(NP0, 90))
    NPmin = max(10, min(24, NP0))

    # --- init population: oversample then select best ---
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    k_hal = 1
    overs = 3 * NP0
    cand = []
    candf = []

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

    order = list(range(len(cand)))
    order.sort(key=lambda i: candf[i])
    order = order[:NP0]

    for idx in order:
        pop.append(cand[idx])
        fit.append(candf[idx])
        if candf[idx] < best:
            best = candf[idx]
            best_x = cand[idx][:]

    # --- archive ---
    archive = []
    arch_cap = 2 * NP0

    # --- SHADE memories ---
    H = 10
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mptr = 0

    def sample_F(m):
        F = m + 0.1 * cauchy_clip()
        tries = 0
        while F <= 0.0 and tries < 8:
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

    # --- cheap subspace quadratic local model around best ---
    def local_surrogate_refine(xb, fb, frac_left):
        # Build in a random subspace of size k and fit diagonal quadratic:
        # f(x + t e_i) for +/- steps -> estimate curvature and gradient (per dim).
        # Then propose step with trust region + backtracking.
        if dim <= 0:
            return xb, fb
        if time.time() >= deadline:
            return xb, fb

        # choose subspace size
        k = 2 if dim < 8 else (3 if dim < 20 else 5)
        k = min(k, dim)
        # late time: smaller/cheaper
        if frac_left < 0.12:
            k = min(k, 3)
        if frac_left < 0.06:
            k = min(k, 2)

        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:k]

        # trust radius as fraction of span
        base = 0.06 if frac_left > 0.25 else (0.035 if frac_left > 0.10 else 0.02)
        # if near bounds, reduce a bit
        base *= 1.0 / (1.0 + 0.15 * k)

        # gather samples (2k evals)
        g = [0.0] * k
        h = [0.0] * k
        step = [0.0] * k

        f0 = fb
        for t, j in enumerate(idxs):
            if time.time() >= deadline:
                return xb, fb
            if spans[j] <= 0.0:
                step[t] = 0.0
                g[t] = 0.0
                h[t] = 0.0
                continue

            s = base * spans[j]
            # ensure step not too tiny
            if s < 1e-14 * (abs(spans[j]) + 1.0):
                s = 1e-14 * (abs(spans[j]) + 1.0)

            x1 = xb[:]; x1[j] += s
            x2 = xb[:]; x2[j] -= s
            repair_midpoint(x1, xb)
            repair_midpoint(x2, xb)

            f1 = eval_f(x1)
            if time.time() >= deadline:
                return xb, fb
            f2 = eval_f(x2)

            # gradient and curvature estimates (1D)
            # g ≈ (f1 - f2)/(2s), h ≈ (f1 - 2f0 + f2)/s^2
            g[t] = (f1 - f2) / (2.0 * s + 1e-300)
            h[t] = (f1 - 2.0 * f0 + f2) / (s * s + 1e-300)
            step[t] = s

        # propose Newton-like step with trust region and damping
        xprop = xb[:]
        for t, j in enumerate(idxs):
            if spans[j] <= 0.0:
                continue
            curv = h[t]
            # if curvature non-positive/flat, use small gradient step
            if curv <= 1e-12:
                delta = -0.15 * step[t] * (1.0 if g[t] > 0 else -1.0) if abs(g[t]) > 0 else 0.0
            else:
                delta = -g[t] / curv
                # trust region clip
                tr = 2.2 * step[t]
                if delta > tr: delta = tr
                if delta < -tr: delta = -tr
            xprop[j] += delta

        repair_midpoint(xprop, xb)

        # backtracking / multi-try in the same direction
        # try alpha in {1, 0.5, 0.25}
        direction = [xprop[i] - xb[i] for i in range(dim)]
        bestl_x = xb[:]
        bestl_f = fb

        for alpha in (1.0, 0.5, 0.25):
            if time.time() >= deadline:
                break
            xt = xb[:]
            for i in range(dim):
                xt[i] += alpha * direction[i]
            repair_midpoint(xt, xb)
            ft = eval_f(xt)
            if ft < bestl_f:
                bestl_f = ft
                bestl_x = xt[:]

        return bestl_x, bestl_f

    # --- fallback mini pattern search (very cheap) ---
    def micro_pattern(xb, fb, frac_left):
        step_frac = 0.02 if frac_left > 0.15 else 0.01
        tries = min(10 + dim, 30)
        xs = xb[:]
        fs = fb
        for _ in range(tries):
            if time.time() >= deadline:
                break
            j = random.randrange(dim)
            if spans[j] <= 0.0:
                continue
            s = step_frac * spans[j]
            for sgn in (1.0, -1.0):
                xt = xs[:]
                xt[j] += sgn * s
                repair_midpoint(xt, xs)
                ft = eval_f(xt)
                if ft < fs:
                    xs, fs = xt, ft
                    break
        return xs, fs

    # --- main loop ---
    gen = 0
    last_best = best
    stall = 0

    while time.time() < deadline:
        gen += 1
        rem = deadline - time.time()
        frac_left = rem / max(1e-12, float(max_time))

        # population reduction (linear schedule)
        NP_target = int(round(NPmin + (NP0 - NPmin) * frac_left))
        if NP_target < NPmin: NP_target = NPmin
        if NP_target > NP0: NP_target = NP0

        # reduce by removing worst
        while len(pop) > NP_target:
            worst = max(range(len(pop)), key=lambda i: fit[i])
            pop.pop(worst)
            fit.pop(worst)
        NP = len(pop)
        arch_cap = 2 * max(NP, 1)
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

            # mutation: current-to-pbest/1
            vi = [0.0] * dim
            for j in range(dim):
                if spans[j] <= 0.0:
                    vi[j] = xi[j]
                else:
                    vi[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            # crossover: binomial
            ui = xi[:]
            if dim == 1:
                ui[0] = vi[0]
            else:
                jrand = random.randrange(dim)
                for j in range(dim):
                    if j == jrand or random.random() < CR:
                        ui[j] = vi[j]

            # bounds repair
            repair_midpoint(ui, xi)

            fu = eval_f(ui)

            if fu <= fi:
                # archive replaced
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

        # stagnation tracking
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        # local refinement triggers:
        # - late time, or
        # - stagnation for a few generations
        if best_x is not None:
            if (frac_left < 0.22 and random.random() < 0.35) or (stall >= 10 and random.random() < 0.45):
                xb, fb = local_surrogate_refine(best_x, best, frac_left)
                if fb < best:
                    best = fb
                    best_x = xb[:]
                    stall = 0
            # very cheap backup
            if (frac_left < 0.12 and random.random() < 0.30) or (stall >= 18 and random.random() < 0.35):
                xb, fb = micro_pattern(best_x, best, frac_left)
                if fb < best:
                    best = fb
                    best_x = xb[:]
                    stall = 0

    return best
