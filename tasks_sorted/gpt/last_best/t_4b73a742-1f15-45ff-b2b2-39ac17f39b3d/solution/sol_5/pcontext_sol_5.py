import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over the provided JADE/L-SHADE variants:
      - Keeps the strong DE core (current-to-pbest/1 + archive + adaptive F/CR with success history)
      - Uses *cheap but effective* p-best selection (no full sort each gen)
      - Adds an *evaluations-aware* surrogate of CMA-like local search:
            * Diagonal covariance (per-dimension step sizes) updated from successful best-centered moves
            * Occasional 2D subspace random-rotation local probes (captures variable interactions cheaply)
      - Adds *restart / diversification* via (a) best-jitter, (b) quasi-random fresh points
      - More robust boundary handling (reflect + clamp)
      - Careful time checks to return within max_time

    Returns:
        best (float): best function value found.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    if max_time is None or max_time <= 0 or dim <= 0:
        return float("inf")

    # -------- bounds ----------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            m = 0.5 * (lows[i] + highs[i])
            lows[i] = highs[i] = m

    # -------- fast RNG (LCG) ----------
    rng_state = random.getrandbits(64) ^ (int(time.time() * 1e9) & ((1 << 64) - 1))

    def u01():
        nonlocal rng_state
        rng_state = (6364136223846793005 * rng_state + 1442695040888963407) & ((1 << 64) - 1)
        return ((rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def randint(n):
        if n <= 1:
            return 0
        x = int(u01() * n)
        return x if x < n else (n - 1)

    def randn():
        a = max(1e-300, u01())
        b = u01()
        return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)

    def reflect(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect into range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        # final clamp for numeric issues
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        return v

    def eval_f(x):
        return float(func(list(x)))

    # -------- Halton init ----------
    def first_primes(k):
        ps = []
        n = 2
        while len(ps) < k:
            ok = True
            r = int(math.sqrt(n))
            for p in ps:
                if p > r:
                    break
                if n % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(n)
            n += 1
        return ps

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

    # -------- DE parameters ----------
    NP0 = max(28, min(160, 18 + 8 * int(math.sqrt(dim)) + dim // 2))
    NPmin = max(10, min(30, 6 + dim // 8))
    NP = NP0

    # success-history memory (L-SHADE style)
    H = 10
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    # p-best fraction schedule
    pmin, pmax = 0.06, 0.25

    archive = []
    arch_max = NP0

    # -------- init pop (Halton + opposition) ----------
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    # also track 2nd best to help local steps in some landscapes
    second_best = float("inf")
    second_x = None

    # cap initial evaluations if time is tiny
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
            second_best, second_x = best, (best_x[:] if best_x is not None else None)
            best, best_x = fx, x[:]
        elif fx < second_best:
            second_best, second_x = fx, x[:]

    # -------- helpers ----------
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

    # Fast-ish pbest selection without sorting:
    # tournament among m samples; return best among them.
    def pbest_index(pfrac):
        k = max(2, int(math.ceil(pfrac * NP)))
        # larger m early to bias stronger toward good; smaller later to save time
        m = min(NP, max(8, 2 * k))
        bi = randint(NP)
        bf = fit[bi]
        for _ in range(m - 1):
            j = randint(NP)
            fj = fit[j]
            if fj < bf:
                bf = fj
                bi = j
        return bi

    # -------- local search (best-centered, diagonal step sizes) ----------
    # per-dimension step size as fraction of span (like diagonal covariance)
    diag = [0.18] * dim
    diag_min = 1e-14
    diag_max = 0.6

    # 1/5th-like success control
    ls_succ = 0
    ls_tries = 0

    def local_search(time_frac):
        nonlocal best, best_x, second_best, second_x, ls_succ, ls_tries

        if best_x is None:
            return
        if time.time() >= deadline:
            return

        # how many local attempts this call
        # more late, but keep bounded
        L = 1 if time_frac < 0.45 else (2 if time_frac < 0.8 else 3)

        # anchor: mostly best, sometimes second-best to escape small basins
        anchor = best_x if (second_x is None or u01() < 0.8) else second_x
        anchor_f = best if anchor is best_x else second_best

        for _ in range(L):
            if time.time() >= deadline:
                return

            # 80% diagonal gaussian step
            # 20% 2D random-rotation step (captures interactions cheaply)
            use_2d = (dim >= 2 and u01() < 0.20)

            y = anchor[:]
            if not use_2d:
                for d in range(dim):
                    if spans[d] == 0.0:
                        continue
                    y[d] = reflect(y[d] + (diag[d] * spans[d]) * randn(), d)
            else:
                i = randint(dim)
                j = randint(dim - 1)
                if j >= i:
                    j += 1
                # random angle, two independent scales
                a = 2.0 * math.pi * u01()
                ci, si = math.cos(a), math.sin(a)
                si1 = (diag[i] * spans[i]) * randn()
                sj1 = (diag[j] * spans[j]) * randn()
                di = ci * si1 - si * sj1
                dj = si * si1 + ci * sj1
                if spans[i] != 0.0:
                    y[i] = reflect(y[i] + di, i)
                if spans[j] != 0.0:
                    y[j] = reflect(y[j] + dj, j)

            fy = eval_f(y)
            ls_tries += 1

            if fy < anchor_f:
                # accept to anchor; also update best bookkeeping
                anchor = y
                anchor_f = fy
                ls_succ += 1

                # update diag from move magnitude (increase where we moved)
                # small, stable update
                for d in range(dim):
                    if spans[d] == 0.0:
                        continue
                    md = abs(y[d] - best_x[d]) / (spans[d] if spans[d] != 0.0 else 1.0)
                    # push step up a bit where movement occurred
                    if md > 0.0:
                        diag[d] = min(diag_max, max(diag_min, 0.90 * diag[d] + 0.10 * min(diag_max, 2.5 * md + 1e-12)))

                if fy < best:
                    second_best, second_x = best, (best_x[:] if best_x is not None else None)
                    best, best_x = fy, y[:]
                elif fy < second_best and (best_x is None or y != best_x):
                    second_best, second_x = fy, y[:]

            # adapt global diag scale occasionally
            if ls_tries >= 12:
                rate = ls_succ / float(ls_tries)
                if rate > 0.2:
                    # expand slightly
                    for d in range(dim):
                        diag[d] = min(diag_max, diag[d] * 1.12)
                else:
                    # contract
                    for d in range(dim):
                        diag[d] = max(diag_min, diag[d] * 0.82)
                ls_tries = 0
                ls_succ = 0

    # -------- main loop ----------
    gen = 0
    while time.time() < deadline:
        now = time.time()
        time_frac = (now - t0) / max(1e-12, (deadline - t0))
        if time_frac >= 1.0:
            break

        # local search schedule (slightly more frequent late)
        if u01() < (0.10 + 0.35 * time_frac):
            local_search(time_frac)

        # p schedule (more exploit early, a bit more explore late)
        pfrac = pmin + (pmax - pmin) * (0.70 - 0.55 * time_frac)
        if pfrac < pmin:
            pfrac = pmin
        if pfrac > pmax:
            pfrac = pmax

        # shuffled indices
        idxs = list(range(NP))
        for i in range(NP - 1, 0, -1):
            j = randint(i + 1)
            idxs[i], idxs[j] = idxs[j], idxs[i]

        S_F, S_CR, S_w = [], [], []

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

            pb = pbest_index(pfrac)
            xpb = pop[pb]

            r1 = i
            while r1 == i:
                r1 = randint(NP)
            xr1 = pop[r1]

            # r2 from union, distinct when from pop
            r2 = -1
            for _ in range(25):
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

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover: mostly binomial; occasionally "best-mix" to stabilize
            jrand = randint(dim)
            ui = xi[:]
            mix_best = (best_x is not None and dim >= 6 and u01() < 0.20)
            for d in range(dim):
                if d == jrand or u01() < CRi:
                    val = v[d]
                    if mix_best:
                        val = 0.90 * val + 0.10 * best_x[d]
                    ui[d] = reflect(val, d)

            fui = eval_f(ui)

            if fui < fi:
                # archive
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[randint(arch_max)] = xi[:]

                pop[i] = ui
                fit[i] = fui

                if fui < best:
                    second_best, second_x = best, (best_x[:] if best_x is not None else None)
                    best, best_x = fui, ui[:]
                elif fui < second_best and (best_x is None or ui != best_x):
                    second_best, second_x = fui, ui[:]

                w = fi - fui
                if w < 1e-12:
                    w = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                S_w.append(w)

        # update memory
        if S_w:
            wsum = sum(S_w)
            # weighted mean CR
            mcr = 0.0
            for w, cr in zip(S_w, S_CR):
                mcr += w * cr
            mcr /= wsum
            # Lehmer mean F
            num = 0.0
            den = 0.0
            for w, Fv in zip(S_w, S_F):
                num += w * Fv * Fv
                den += w * Fv
            mf = (num / den) if den > 0.0 else 0.5

            M_F[mem_idx] = mf
            M_CR[mem_idx] = mcr
            mem_idx = (mem_idx + 1) % H

        # archive size control
        arch_max = max(NP, NP0)
        if len(archive) > arch_max:
            for _ in range(len(archive) - arch_max):
                archive.pop(randint(len(archive)))

        # population size reduction (linear with time)
        gen += 1
        target_NP = int(round(NPmin + (NP0 - NPmin) * (1.0 - time_frac)))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            remove = NP - target_NP
            # remove worst repeatedly (NP is modest; scan ok)
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

        # diversification / restart if late:
        # replace worst with (a) best-jitter or (b) quasi-random
        if time_frac > 0.55 and u01() < 0.18 and best_x is not None:
            worst = 0
            wf = fit[0]
            for j in range(1, NP):
                if fit[j] > wf:
                    wf = fit[j]
                    worst = j

            if u01() < 0.65:
                y = best_x[:]
                # jitter scale: small late, larger mid
                jit = (0.22 * (1.0 - time_frac) + 0.01)
                for d in range(dim):
                    if spans[d] == 0.0:
                        continue
                    y[d] = reflect(y[d] + (u01() * 2.0 - 1.0) * jit * spans[d], d)
            else:
                y = make_point(int(1e6 * u01()) + gen + worst)

            fy = eval_f(y)
            pop[worst] = y
            fit[worst] = fy
            if fy < best:
                second_best, second_x = best, (best_x[:] if best_x is not None else None)
                best, best_x = fy, y[:]
            elif fy < second_best and (best_x is None or y != best_x):
                second_best, second_x = fy, y[:]

    return best
