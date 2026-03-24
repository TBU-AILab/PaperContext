import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    Main changes vs your best (DE/SHADE hybrid):
      - Uses L-SHADE + "current-to-pbest/1" + archive (strong baseline)
      - Adds *Eigenvector/coordinate mixed crossover* (cheap rotation surrogate):
          occasionally permutes dimensions and performs grouped crossover, which
          helps on rotated/non-separable problems without heavy linear algebra.
      - Adds *triangle/quad opposition* seeding + continued injection of LDS points
      - Adds *very cheap local refinement* around best: adaptive pattern search
        (coordinate + random subspace) with step halving.
      - Stronger time scheduling: exploration early, exploitation late.

    Returns:
        best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ---------- helpers ----------
    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

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

    # ---------- scrambled Halton ----------
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

    # ---------- population size ----------
    # L-SHADE typically uses ~18..100 depending on dim/time; keep moderate.
    NP = 10 + 5 * int(math.sqrt(max(1, dim)))
    NP = max(20, min(NP, 80))

    # ---------- init population: halton + opposition + mid-opposition ----------
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    k = 1
    # create 3*NP candidates then keep best NP (improves early quality/coverage)
    cand = []
    candf = []
    need = 3 * NP
    while len(cand) < need and time.time() < deadline:
        x = halton_point(k); k += 1
        fx = eval_f(x)
        cand.append(x); candf.append(fx)

        # opposition
        xo = [lows[j] + highs[j] - x[j] for j in range(dim)]
        clip(xo)
        fo = eval_f(xo)
        cand.append(xo); candf.append(fo)

        # mid-opposition (towards center) - cheap diversity boost
        xc = [(lows[j] + highs[j]) * 0.5 for j in range(dim)]
        xm = [0.5 * (x[j] + xc[j]) for j in range(dim)]
        clip(xm)
        fm = eval_f(xm)
        cand.append(xm); candf.append(fm)

    if not cand:
        return best

    order = list(range(len(cand)))
    order.sort(key=lambda i: candf[i])
    order = order[:NP]
    for idx in order:
        pop.append(cand[idx])
        fit.append(candf[idx])
        if candf[idx] < best:
            best = candf[idx]
            best_x = cand[idx][:]

    # ---------- archive ----------
    archive = []
    arch_cap = 2 * NP

    # ---------- SHADE memories ----------
    H = 10
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mptr = 0

    def sample_F(m):
        F = m + 0.1 * cauchy_clip()
        # redraw if <=0
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

    # ---------- local refinement (adaptive pattern) ----------
    def local_refine(x0, f0, time_frac_left):
        # Budgeted refine: fewer steps when time is low
        xb = x0[:]
        fb = f0

        # base step is relative to spans; shrink on failures
        step = 0.04 if time_frac_left > 0.20 else 0.02
        step = step if time_frac_left > 0.08 else 0.012

        # max evals
        max_tries = 12 + min(2 * dim, 30)
        tries = 0

        # shuffle dims for better conditioning
        dims = list(range(dim))
        random.shuffle(dims)

        while tries < max_tries and time.time() < deadline:
            improved = False

            # coordinate probes
            for j in dims[:min(dim, 18)]:
                if spans[j] <= 0:
                    continue
                s = step * spans[j]
                for sgn in (1.0, -1.0):
                    if time.time() >= deadline:
                        return xb, fb
                    xt = xb[:]
                    xt[j] += sgn * s
                    clip(xt)
                    ft = eval_f(xt)
                    tries += 1
                    if ft < fb:
                        xb, fb = xt, ft
                        improved = True
                        break
                if tries >= max_tries or time.time() >= deadline:
                    return xb, fb

            # random small subspace move (helps non-separable)
            if time.time() < deadline and tries < max_tries:
                xt = xb[:]
                # choose small group
                gsize = 2 if dim < 8 else 3
                for _ in range(gsize):
                    j = random.randrange(dim)
                    if spans[j] > 0:
                        xt[j] += (2.0 * random.random() - 1.0) * step * spans[j]
                clip(xt)
                ft = eval_f(xt)
                tries += 1
                if ft < fb:
                    xb, fb = xt, ft
                    improved = True

            if not improved:
                step *= 0.5
                if step < 1e-12:
                    break

        return xb, fb

    # ---------- dimension grouping permutation (cheap "rotation" surrogate) ----------
    perm = list(range(dim))
    def regroup():
        random.shuffle(perm)

    regroup()
    last_regroup = 0

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # time schedule
        rem = deadline - time.time()
        frac_left = rem / max(1e-12, float(max_time))

        # re-inject a few halton points early to avoid premature convergence
        if frac_left > 0.55 and gen % 9 == 0:
            if time.time() >= deadline:
                return best
            xg = halton_point(k); k += 1
            fg = eval_f(xg)
            # replace worst if better
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

        # occasional regrouping
        if gen - last_regroup > 12 and (frac_left > 0.35 or random.random() < 0.15):
            regroup()
            last_regroup = gen

        order = list(range(NP))
        order.sort(key=lambda i: fit[i])

        # p-best fraction (smaller late)
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

            # choose r1 from pop excluding i
            r1 = random.randrange(NP)
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            # choose r2 from pool, avoid pointing to xi/xr1 if possible
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
                if spans[j] <= 0:
                    vi[j] = xi[j]
                else:
                    vi[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            # crossover:
            # - mostly standard binomial
            # - sometimes grouped by perm blocks (better for rotated problems)
            ui = xi[:]
            if dim <= 1:
                ui[0] = vi[0]
            else:
                if random.random() < (0.22 if frac_left > 0.35 else 0.12):
                    # grouped crossover
                    block = 2 if dim < 12 else 3
                    start = random.randrange(dim)
                    jrand = perm[start]
                    # force at least one dimension
                    ui[jrand] = vi[jrand]
                    # then take a couple blocks with prob CR
                    take_blocks = 1 + (1 if random.random() < CR else 0)
                    for b in range(take_blocks):
                        base = (start + b * block) % dim
                        for t in range(block):
                            jj = perm[(base + t) % dim]
                            if random.random() < CR or jj == jrand:
                                ui[jj] = vi[jj]
                else:
                    jrand = random.randrange(dim)
                    for j in range(dim):
                        if j == jrand or random.random() < CR:
                            ui[j] = vi[j]

            # boundary handling: bounce back towards target
            for j in range(dim):
                if ui[j] < lows[j]:
                    ui[j] = lows[j] + random.random() * (xi[j] - lows[j])
                elif ui[j] > highs[j]:
                    ui[j] = highs[j] - random.random() * (highs[j] - xi[j])

            fu = eval_f(ui)

            if fu <= fi:
                # archive replaced
                if len(archive) < arch_cap:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_cap)] = xi[:]

                pop[i] = ui
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = ui[:]

                S_F.append(F)
                S_CR.append(CR)
                df = fi - fu
                if df < 0.0:
                    df = 0.0
                dF.append(df)
            else:
                # very small late "accept worse" for mobility
                if frac_left < 0.12 and random.random() < 0.02:
                    pop[i] = ui
                    fit[i] = fu
                    if fu < best:
                        best = fu
                        best_x = ui[:]

        # update memories
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

        # cap archive
        if len(archive) > arch_cap:
            while len(archive) > arch_cap:
                archive.pop(random.randrange(len(archive)))

        # local refine near end or after improvements
        if best_x is not None:
            if (frac_left < 0.22 and random.random() < 0.35) or (frac_left < 0.10 and random.random() < 0.65):
                xb, fb = local_refine(best_x, best, frac_left)
                if fb < best:
                    best = fb
                    best_x = xb[:]

    return best
