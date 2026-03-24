import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Key upgrades vs previous DE+ES:
      - JADE-style "current-to-pbest/1" differential evolution (stronger, adaptive F/CR memory).
      - External archive (classic JADE) to maintain diversity and avoid premature convergence.
      - Opposition-based / quasi-random (Halton) initialization for better coverage than pure random.
      - Local search: lightweight, adaptive coordinate pattern search around the current best.
      - Robust reflection boundary handling.

    Returns:
        best (float): best function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    if max_time <= 0 or dim <= 0:
        return float("inf")

    # ---- bounds preproc ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            lows[i] = highs[i] = (lows[i] + highs[i]) * 0.5

    # ---- fast RNG (LCG) + helpers ----
    rng_state = random.getrandbits(64) ^ (int(time.time() * 1e9) & ((1 << 64) - 1))

    def u01():
        nonlocal rng_state
        rng_state = (6364136223846793005 * rng_state + 1442695040888963407) & ((1 << 64) - 1)
        return ((rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def randint(n):
        # 0..n-1
        return int(u01() * n)

    def randn():
        # Box-Muller
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
        if v > hi:
            v = hi
        return v

    def eval_f(x):
        return float(func(list(x)))

    # ---- Halton sequence for better initialization ----
    def first_primes(k):
        primes = []
        n = 2
        while len(primes) < k:
            is_p = True
            r = int(math.sqrt(n))
            for p in primes:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(n)
            n += 1
        return primes

    primes = first_primes(min(dim, 32))  # enough; for dim>32, reuse with offsets

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def make_halton_point(idx):
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
            else:
                base = primes[d % len(primes)]
                # slight scrambling via additive shift from RNG
                h = (halton(idx + 1, base) + 0.37 * u01()) % 1.0
                x[d] = lows[d] + h * spans[d]
        return x

    def opposite_point(x):
        y = x[:]
        for d in range(dim):
            if spans[d] == 0.0:
                y[d] = lows[d]
            else:
                y[d] = lows[d] + highs[d] - x[d]
                y[d] = reflect(y[d], d)
        return y

    # ---- population size ----
    # JADE typically works with ~10*dim, but time-limited: cap it.
    pop_size = max(20, min(80, 10 + 6 * int(math.sqrt(dim)) + (dim // 4)))
    # ---- init population (Halton + opposition) ----
    pop = []
    fit = []
    for i in range(pop_size):
        if time.time() >= deadline:
            return float("inf")
        x = make_halton_point(i)
        fx = eval_f(x)
        xo = opposite_point(x)
        fxo = eval_f(xo)
        if fxo < fx:
            x, fx = xo, fxo
        pop.append(x)
        fit.append(fx)

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # ---- JADE parameters + memory ----
    mu_F = 0.5
    mu_CR = 0.5
    c = 0.1                 # learning rate
    p = 0.15                # p-best fraction
    archive = []            # external archive of replaced solutions
    arch_max = pop_size     # keep archive size bounded

    def sample_F(mu):
        # JADE uses Cauchy; implement light-tailed-ish Cauchy via tan(pi*(u-0.5))
        # and resample if nonpositive.
        for _ in range(10):
            F = mu + 0.1 * math.tan(math.pi * (u01() - 0.5))
            if F > 0:
                return min(1.0, F)
        return max(1e-3, min(1.0, mu))

    def sample_CR(mu):
        CR = mu + 0.1 * randn()
        if CR < 0.0:
            CR = 0.0
        elif CR > 1.0:
            CR = 1.0
        return CR

    # ---- lightweight local search: coordinate pattern steps on best ----
    # adaptive step as fraction of span; shrinks when no improvement
    step_frac = 0.2
    min_step_frac = 1e-12

    def local_refine(x, fx, time_frac):
        nonlocal step_frac
        # attempt a few coordinate moves; more near the end
        tries = 1 if time_frac < 0.4 else (2 if time_frac < 0.75 else 3)
        curx, curf = x[:], fx
        # choose subset for high dims
        if dim <= 24:
            coords = list(range(dim))
        else:
            k = max(10, dim // 3)
            coords = [randint(dim) for _ in range(k)]

        # shuffle a bit
        for j in range(len(coords) - 1, 0, -1):
            r = randint(j + 1)
            coords[j], coords[r] = coords[r], coords[j]

        improved = False
        base_step = step_frac
        for _ in range(tries):
            if time.time() >= deadline:
                break
            for d in coords:
                if spans[d] == 0.0:
                    continue
                step = base_step * spans[d]
                if step <= 0.0:
                    continue
                # try + and -
                xp = curx[:]
                xm = curx[:]
                xp[d] = reflect(xp[d] + step, d)
                xm[d] = reflect(xm[d] - step, d)
                fp = eval_f(xp)
                fm = eval_f(xm)
                if fp < curf or fm < curf:
                    if fp <= fm:
                        curx, curf = xp, fp
                    else:
                        curx, curf = xm, fm
                    improved = True
        # adapt step
        if improved:
            step_frac = min(0.5, step_frac * 1.15)
        else:
            step_frac = max(min_step_frac, step_frac * 0.7)
        return curx, curf

    # ---- main loop ----
    while time.time() < deadline:
        now = time.time()
        time_frac = (now - t0) / max(1e-12, (deadline - t0))
        if time_frac > 1.0:
            break

        # local refine best sometimes, especially later
        if u01() < (0.10 + 0.25 * time_frac):
            bx, bf = local_refine(best_x, best, time_frac)
            if bf < best:
                best_x, best = bx, bf

        # sort indices by fitness for p-best selection
        # partial approach: find threshold by selecting top k indices
        kbest = max(2, int(math.ceil(p * pop_size)))
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        S_F = []
        S_CR = []
        wS = []  # weights (fitness improvements)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # choose p-best individual
            pb = idx_sorted[randint(kbest)]
            xpb = pop[pb]

            # choose r1 from population != i
            r1 = i
            while r1 == i:
                r1 = randint(pop_size)

            # choose r2 from (population U archive) != i, != r1
            union_size = pop_size + len(archive)
            r2 = -1
            for _ in range(20):
                rr = randint(union_size)
                # map rr to either pop or archive
                if rr < pop_size:
                    if rr != i and rr != r1:
                        r2 = rr
                        break
                else:
                    # archive index always ok (distinctness from i/r1 not needed)
                    r2 = rr
                    break
            if r2 < 0:
                r2 = (r1 + 1) % pop_size

            xr1 = pop[r1]
            if r2 < pop_size:
                xr2 = pop[r2]
            else:
                xr2 = archive[r2 - pop_size]

            Fi = sample_F(mu_F)
            CRi = sample_CR(mu_CR)

            # current-to-pbest/1 mutation:
            # v = xi + F*(xpb - xi) + F*(xr1 - xr2)
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # binomial crossover
            jrand = randint(dim)
            ui = xi[:]
            for d in range(dim):
                if d == jrand or u01() < CRi:
                    ui[d] = reflect(v[d], d)

            fui = eval_f(ui)

            # selection
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

                # store successful parameters
                imp = fi - fui
                S_F.append(Fi)
                S_CR.append(CRi)
                wS.append(max(1e-12, imp))

        # JADE parameter update (weighted)
        if wS:
            wsum = sum(wS)
            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for w, Fv in zip(wS, S_F):
                num += w * Fv * Fv
                den += w * Fv
            F_lehmer = (num / den) if den > 0 else mu_F

            CR_mean = 0.0
            for w, crv in zip(wS, S_CR):
                CR_mean += w * crv
            CR_mean = CR_mean / wsum

            mu_F = (1.0 - c) * mu_F + c * F_lehmer
            mu_CR = (1.0 - c) * mu_CR + c * CR_mean

        # keep archive bounded (already bounded, but shrink if pop shrank etc.)
        if len(archive) > arch_max:
            # random downsample
            for _ in range(len(archive) - arch_max):
                archive.pop(randint(len(archive)))

        # mild diversification if stagnating late: inject a few random/halton points
        if time_frac > 0.6 and u01() < 0.15:
            # replace worst with a fresh point near best + random
            worst = max(range(pop_size), key=lambda j: fit[j])
            if u01() < 0.6:
                y = best_x[:]
                # jitter decreases with time
                jitter = (0.15 * (1.0 - time_frac) + 0.01)
                for d in range(dim):
                    if spans[d] == 0.0:
                        continue
                    y[d] = reflect(y[d] + (u01() * 2.0 - 1.0) * jitter * spans[d], d)
            else:
                y = make_halton_point(int(1e6 * u01()) + worst + int(1000 * time_frac))
            fy = eval_f(y)
            pop[worst] = y
            fit[worst] = fy
            if fy < best:
                best, best_x = fy, y[:]

    return best
