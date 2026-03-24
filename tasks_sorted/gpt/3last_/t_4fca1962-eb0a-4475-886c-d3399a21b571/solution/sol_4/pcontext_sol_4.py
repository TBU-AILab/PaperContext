import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libs).
    Improvements vs your previous hybrids:
      - Stronger global exploration early: scrambled Halton + opposition points
      - Main engine: DE (current-to-pbest/1 + bin crossover) with SHADE-style
        memory of F/CR + p-best archive (very strong on many black-box tasks)
      - Polishing: bounded coordinate/diagonal pattern + cheap 2-eval SPSA step
      - Automatic time-splitting between explore/exploit; works for any dim/bounds

    Returns:
        best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ---------- helpers ----------
    def clip_inplace(x):
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

    def cauchy(scale=1.0):
        # heavy-tailed; clipped
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        if v > 30.0: v = 30.0
        if v < -30.0: v = -30.0
        return scale * v

    # ---------- scrambled Halton (fast LDS) ----------
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
              137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
              199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
              277, 281, 283, 293, 307, 311, 313]

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

    # ---------- init population ----------
    # Keep population moderate to preserve evaluations for evolution.
    NP = 8 + 6 * int(math.sqrt(max(1, dim)))
    NP = max(18, min(NP, 70))

    # Initial seeding: scrambled Halton + opposition (often boosts coverage a lot)
    pop = []
    fit = []

    best = float("inf")
    best_x = None

    k = 1
    for i in range(NP):
        if time.time() >= deadline:
            return best

        x = halton_point(k); k += 1
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition point (mirror in bounds)
        if time.time() >= deadline:
            return best
        xo = [lows[j] + highs[j] - x[j] for j in range(dim)]
        clip_inplace(xo)
        fo = eval_f(xo)
        pop.append(xo)
        fit.append(fo)
        if fo < best:
            best, best_x = fo, xo[:]

    # trim to NP (keep best NP)
    idx = list(range(len(pop)))
    idx.sort(key=lambda i: fit[i])
    idx = idx[:NP]
    pop = [pop[i] for i in idx]
    fit = [fit[i] for i in idx]

    # External archive for DE (stores replaced individuals)
    archive = []
    arch_cap = 2 * NP

    # ---------- SHADE-style parameter memory ----------
    H = 8
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mptr = 0

    def sample_F(m):
        # Cauchy around memory mean (like SHADE); ensure in (0,1]
        F = m + 0.1 * cauchy(1.0)
        while F <= 0.0:
            F = m + 0.1 * cauchy(1.0)
        if F > 1.0:
            F = 1.0
        return F

    def sample_CR(m):
        # Normal around memory mean; clip to [0,1]
        CR = m + 0.1 * gauss01()
        if CR < 0.0: CR = 0.0
        if CR > 1.0: CR = 1.0
        return CR

    # ---------- polishing steps ----------
    # small 2-eval SPSA-like step (works well as cheap local refinement)
    def spsa_step(xc, fc, scale):
        if time.time() >= deadline:
            return None, None
        # direction vector with +/-1
        d = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
        eps = 0.0
        for i in range(dim):
            if spans[i] > 0:
                eps = max(eps, scale * spans[i])
        if eps <= 0.0:
            return None, None

        x1 = xc[:]
        x2 = xc[:]
        for i in range(dim):
            if spans[i] > 0:
                x1[i] += eps * d[i]
                x2[i] -= eps * d[i]
        clip_inplace(x1)
        clip_inplace(x2)
        f1 = eval_f(x1)
        if time.time() >= deadline:
            return None, None
        f2 = eval_f(x2)

        gdir = (f1 - f2) / (2.0 * eps + 1e-300)
        eta = 0.7 * eps / max(1.0, math.sqrt(dim))
        xn = xc[:]
        for i in range(dim):
            if spans[i] > 0:
                xn[i] += -eta * gdir * d[i]
        clip_inplace(xn)
        fn = eval_f(xn)
        return xn, fn

    def coord_polish(xc, fc):
        # lightweight coordinate/diagonal polish, few evals
        xb = xc[:]
        fb = fc
        base_step = 0.02  # fraction of span
        # coordinate probes
        trials = min(2 * dim, 30)
        for _ in range(trials):
            if time.time() >= deadline:
                return xb, fb
            j = random.randrange(dim)
            if spans[j] <= 0:
                continue
            s = base_step * spans[j]
            for sgn in (1.0, -1.0):
                xt = xb[:]
                xt[j] += sgn * s
                clip_inplace(xt)
                ft = eval_f(xt)
                if ft < fb:
                    xb, fb = xt, ft
        # diagonal probe
        if time.time() < deadline and dim > 1:
            xt = xb[:]
            for j in range(dim):
                if spans[j] > 0:
                    xt[j] += (1.0 if random.random() < 0.5 else -1.0) * base_step * spans[j]
            clip_inplace(xt)
            ft = eval_f(xt)
            if ft < fb:
                xb, fb = xt, ft
        return xb, fb

    # ---------- main loop: DE with p-best + archive + occasional polish ----------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # sort indices by fitness once per generation
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])

        # exploration/exploitation schedule by remaining time
        rem = deadline - time.time()
        frac = rem / max(1e-12, float(max_time))
        # p in current-to-pbest/1: smaller -> more exploit
        p = 0.25 if frac > 0.6 else (0.18 if frac > 0.25 else 0.10)
        pbest_count = max(2, int(p * NP))

        S_F = []
        S_CR = []
        dF = []

        # for each target
        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            # choose pbest from top pbest_count
            pbest = order[random.randrange(pbest_count)]
            xp = pop[pbest]

            # choose r1 != i, r2 from union(pop + archive) and != i,r1
            def pick_index_excluding(n, ex1, ex2=-1):
                j = random.randrange(n)
                while j == ex1 or j == ex2:
                    j = random.randrange(n)
                return j

            r1 = pick_index_excluding(NP, i)
            # pool for r2: pop + archive
            pool = pop + archive
            pool_n = len(pool)
            # ensure pool has enough diversity
            if pool_n < 2:
                pool = pop
                pool_n = NP

            # pick r2 not pointing to same vector as xi or pop[r1] when possible
            r2 = random.randrange(pool_n)
            tries = 0
            while tries < 20 and (pool[r2] is xi or pool[r2] is pop[r1]):
                r2 = random.randrange(pool_n)
                tries += 1
            xr1 = pop[r1]
            xr2 = pool[r2]

            # mutation: current-to-pbest/1
            vi = [0.0] * dim
            for j in range(dim):
                if spans[j] <= 0:
                    vi[j] = xi[j]
                else:
                    vi[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            # binomial crossover
            ui = xi[:]
            jrand = random.randrange(dim) if dim > 1 else 0
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    ui[j] = vi[j]

            # boundary handling: bounce-back mix
            for j in range(dim):
                if ui[j] < lows[j]:
                    ui[j] = lows[j] + random.random() * (xi[j] - lows[j])
                elif ui[j] > highs[j]:
                    ui[j] = highs[j] - random.random() * (highs[j] - xi[j])

            fu = eval_f(ui)

            # selection
            if fu <= fi:
                # success: put old into archive
                if len(archive) < arch_cap:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_cap)] = xi[:]

                pop[i] = ui
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = ui[:]

                # store successful params for SHADE update
                S_F.append(F)
                S_CR.append(CR)
                d = (fi - fu)
                if d < 0.0:
                    d = 0.0
                dF.append(d)
            else:
                # occasional "soft" move to keep diversity late
                if frac < 0.15 and random.random() < 0.03:
                    pop[i] = ui
                    fit[i] = fu
                    if fu < best:
                        best, best_x = fu, ui[:]

        # update memories if any successes
        if dF:
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F, weighted arithmetic for CR
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
            # keep within reasonable bounds
            if newMF <= 0.0: newMF = 0.1
            if newMF > 1.0: newMF = 1.0
            if newMCR < 0.0: newMCR = 0.0
            if newMCR > 1.0: newMCR = 1.0

            M_F[mptr] = newMF
            M_CR[mptr] = newMCR
            mptr = (mptr + 1) % H

        # occasional polish near end or after some generations
        if best_x is not None:
            # use time-based trigger rather than generation count
            if frac < 0.35 and random.random() < 0.25:
                xb, fb = coord_polish(best_x, best)
                if fb < best:
                    best, best_x = fb, xb[:]
            if frac < 0.20 and random.random() < 0.35:
                xn, fn = spsa_step(best_x, best, scale=0.005)
                if xn is not None and fn < best:
                    best, best_x = fn, xn[:]

        # keep archive bounded (in case NP changes not, but safe)
        if len(archive) > arch_cap:
            # random downsample
            while len(archive) > arch_cap:
                archive.pop(random.randrange(len(archive)))

    return best
