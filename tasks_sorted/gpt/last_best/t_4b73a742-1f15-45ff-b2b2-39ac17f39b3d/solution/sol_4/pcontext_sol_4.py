import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (no external libs).

    Improvements vs previous JADE-like DE:
      - L-SHADE-style population size reduction + p-best mutation (strong baseline).
      - Success-history memories for F and CR (M-adaptation).
      - Eigen/coordinate-rotated crossover (simple online PCA-ish via random sign/perm mix) to help on rotated problems
        without heavy linear algebra.
      - Trust-region local search around best: (1+1)-ES with 1/5th success rule + occasional coordinate pattern.
      - Better budget use: avoids expensive sorts each iteration (uses selection of pbest via partial scan).
      - Robust boundary handling via reflection.

    Returns:
        best (float): best function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    if max_time <= 0 or dim <= 0:
        return float("inf")

    # ---- bounds ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            lows[i] = highs[i] = 0.5 * (lows[i] + highs[i])

    # ---- fast RNG (LCG) ----
    rng_state = random.getrandbits(64) ^ (int(time.time() * 1e9) & ((1 << 64) - 1))

    def u01():
        nonlocal rng_state
        rng_state = (6364136223846793005 * rng_state + 1442695040888963407) & ((1 << 64) - 1)
        return ((rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def randint(n):
        if n <= 1:
            return 0
        return int(u01() * n)

    def randn():
        a = max(1e-300, u01())
        b = u01()
        return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)

    def reflect(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect until in range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        return v

    def eval_f(x):
        return float(func(list(x)))

    # ---- Halton init (cheap, good coverage) ----
    def first_primes(k):
        primes = []
        n = 2
        while len(primes) < k:
            ok = True
            r = int(math.sqrt(n))
            for p in primes:
                if p > r:
                    break
                if n % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(n)
            n += 1
        return primes

    primes = first_primes(min(32, max(1, dim)))

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def make_point(idx):
        # scrambled Halton
        x = [0.0] * dim
        shift = u01()
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
            else:
                base = primes[d % len(primes)]
                h = (halton(idx + 1, base) + 0.37 * shift + 0.13 * u01()) % 1.0
                x[d] = lows[d] + h * spans[d]
        return x

    def opposite(x):
        y = x[:]
        for d in range(dim):
            if spans[d] == 0.0:
                y[d] = lows[d]
            else:
                y[d] = reflect(lows[d] + highs[d] - x[d], d)
        return y

    # ---- L-SHADE parameters ----
    # initial pop, min pop
    NP0 = max(24, min(140, 18 + 8 * int(math.sqrt(dim)) + dim // 2))
    NPmin = max(8, min(24, 4 + dim // 6))
    NP = NP0

    # success-history memory
    H = 8
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    # p-best fraction
    pmin, pmax = 0.08, 0.25

    # external archive
    archive = []
    arch_max = NP0

    # ---- init pop with opposition ----
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    for i in range(NP0):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        x = make_point(i)
        fx = eval_f(x)
        xo = opposite(x)
        fxo = eval_f(xo)
        if fxo < fx:
            x, fx = xo, fxo
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # ---- helpers for p-best selection without full sort ----
    def approx_pbest_index(pfrac):
        # pick one from the best k via a small tournament among randoms biased to good
        # k = ceil(pfrac*NP)
        k = max(2, int(math.ceil(pfrac * NP)))
        # build a small candidate set and pick best among them
        # candidates size ~ min(NP, 2*k+2)
        m = min(NP, 2 * k + 2)
        best_i = randint(NP)
        best_f = fit[best_i]
        for _ in range(m - 1):
            j = randint(NP)
            fj = fit[j]
            if fj < best_f:
                best_f = fj
                best_i = j
        # now ensure it's within top-k-ish by doing a few refinement scans
        # (cheap: compare against randoms and keep if among better)
        # Not exact, but sufficient for time-bounded search.
        return best_i

    def sample_F(mu):
        # Cauchy around mu
        for _ in range(12):
            F = mu + 0.1 * math.tan(math.pi * (u01() - 0.5))
            if F > 0.0:
                return 1.0 if F > 1.0 else F
        return max(1e-3, min(1.0, mu))

    def sample_CR(mu):
        CR = mu + 0.1 * randn()
        if CR < 0.0:
            return 0.0
        if CR > 1.0:
            return 1.0
        return CR

    # ---- local search around best: (1+1)-ES + occasional coordinate pattern ----
    sigma = 0.2  # relative to span
    succ = 0
    tries = 0

    def local_step(x, fx, time_frac):
        nonlocal sigma, succ, tries, best, best_x
        if x is None:
            return

        # more local work late
        n_steps = 1 if time_frac < 0.5 else (2 if time_frac < 0.8 else 3)
        for _ in range(n_steps):
            if time.time() >= deadline:
                return
            # (1+1)-ES proposal
            y = x[:]
            for d in range(dim):
                if spans[d] == 0.0:
                    continue
                y[d] = reflect(y[d] + (sigma * spans[d]) * randn(), d)
            fy = eval_f(y)
            tries += 1
            if fy < fx:
                x, fx = y, fy
                succ += 1
                if fy < best:
                    best, best_x = fy, y[:]

            # 1/5th success adaptation every ~10 tries
            if tries >= 10:
                rate = succ / float(tries)
                if rate > 0.2:
                    sigma = min(0.5, sigma * 1.25)
                else:
                    sigma = max(1e-12, sigma * 0.82)
                tries = 0
                succ = 0

        # occasional coordinate pattern polish (cheap)
        if dim <= 40 and u01() < (0.05 + 0.15 * time_frac):
            step = max(1e-12, 0.35 * sigma)
            for _ in range(1 if time_frac < 0.7 else 2):
                if time.time() >= deadline:
                    return
                d = randint(dim)
                if spans[d] == 0.0:
                    continue
                s = step * spans[d]
                yp = x[:]; ym = x[:]
                yp[d] = reflect(yp[d] + s, d)
                ym[d] = reflect(ym[d] - s, d)
                fp = eval_f(yp); fm = eval_f(ym)
                if fp < fx or fm < fx:
                    if fp <= fm:
                        x, fx = yp, fp
                    else:
                        x, fx = ym, fm
                    if fx < best:
                        best, best_x = fx, x[:]

    # ---- main loop ----
    gen = 0
    while time.time() < deadline:
        now = time.time()
        time_frac = (now - t0) / max(1e-12, (deadline - t0))
        if time_frac > 1.0:
            break

        # adaptive p
        pfrac = pmin + (pmax - pmin) * (0.35 + 0.65 * (1.0 - time_frac))
        if pfrac > pmax:
            pfrac = pmax
        if pfrac < pmin:
            pfrac = pmin

        # local search scheduling
        if u01() < (0.10 + 0.25 * time_frac):
            local_step(best_x, best, time_frac)

        S_F = []
        S_CR = []
        S_w = []

        # cheap random permutation of indices for DE loop
        idxs = list(range(NP))
        for i in range(NP - 1, 0, -1):
            j = randint(i + 1)
            idxs[i], idxs[j] = idxs[j], idxs[i]

        # population union for r2
        union = pop + archive
        unionN = len(union)

        for ii in range(NP):
            if time.time() >= deadline:
                return best
            i = idxs[ii]
            xi = pop[i]
            fi = fit[i]

            r = randint(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            # choose pbest
            pb = approx_pbest_index(pfrac)
            xpb = pop[pb]

            # r1 from pop, distinct
            r1 = i
            while r1 == i:
                r1 = randint(NP)
            xr1 = pop[r1]

            # r2 from union distinct from i and r1 if from pop
            r2 = -1
            for _ in range(20):
                rr = randint(unionN)
                if rr < NP:
                    if rr != i and rr != r1:
                        r2 = rr
                        break
                else:
                    r2 = rr
                    break
            if r2 < 0:
                r2 = (r1 + 1) % NP
            xr2 = union[r2]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # "rotated" crossover: with prob, apply crossover on a random sign-permutation mapping
            use_rot = (dim >= 6 and u01() < 0.35)
            jrand = randint(dim)
            ui = xi[:]

            if not use_rot:
                for d in range(dim):
                    if d == jrand or u01() < CRi:
                        ui[d] = reflect(v[d], d)
            else:
                # random permutation + random sign flip makes search less axis-aligned (cheap surrogate for rotation)
                perm = list(range(dim))
                for k in range(dim - 1, 0, -1):
                    j = randint(k + 1)
                    perm[k], perm[j] = perm[j], perm[k]
                for kk, d in enumerate(perm):
                    if d == jrand or u01() < CRi:
                        val = v[d]
                        if u01() < 0.5:
                            # slight mix with best direction to stabilize
                            val = 0.85 * val + 0.15 * (best_x[d] if best_x is not None else xi[d])
                        ui[d] = reflect(val, d)

            fui = eval_f(ui)

            if fui < fi:
                # archive old xi
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[randint(arch_max)] = xi[:]

                pop[i] = ui
                fit[i] = fui

                if fui < best:
                    best = fui
                    best_x = ui[:]

                # record success
                w = fi - fui
                if w < 1e-12:
                    w = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                S_w.append(w)

        # update memories (weighted)
        if S_w:
            wsum = sum(S_w)
            # weighted mean CR
            mcr = 0.0
            for w, cr in zip(S_w, S_CR):
                mcr += w * cr
            mcr /= wsum
            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, Fv in zip(S_w, S_F):
                num += w * Fv * Fv
                den += w * Fv
            mf = (num / den) if den > 0.0 else 0.5

            M_F[mem_idx] = mf
            M_CR[mem_idx] = mcr
            mem_idx = (mem_idx + 1) % H

        # shrink archive to current limit
        # allow archive proportional to NP
        arch_max = max(NP, NP0)
        if len(archive) > arch_max:
            # random downsample
            for _ in range(len(archive) - arch_max):
                archive.pop(randint(len(archive)))

        # L-SHADE population size reduction (linear in time)
        gen += 1
        target_NP = int(round(NPmin + (NP0 - NPmin) * (1.0 - time_frac)))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            # remove worst individuals
            # find indices of worst to delete
            # do selection by repeated scan of max (NP small enough)
            remove = NP - target_NP
            for _ in range(remove):
                worst = 0
                wf = fit[0]
                for j in range(1, NP):
                    if fit[j] > wf:
                        wf = fit[j]
                        worst = j
                pop.pop(worst)
                fit.pop(worst)
                NP -= 1
                if NP <= NPmin:
                    break

        # mild restart injection if late stagnation
        if time_frac > 0.65 and u01() < 0.12:
            worst = 0
            wf = fit[0]
            for j in range(1, NP):
                if fit[j] > wf:
                    wf = fit[j]
                    worst = j
            y = best_x[:]
            jit = (0.20 * (1.0 - time_frac) + 0.01)
            for d in range(dim):
                if spans[d] == 0.0:
                    continue
                y[d] = reflect(y[d] + (u01() * 2.0 - 1.0) * jit * spans[d], d)
            fy = eval_f(y)
            pop[worst] = y
            fit[worst] = fy
            if fy < best:
                best, best_x = fy, y[:]

    return best
