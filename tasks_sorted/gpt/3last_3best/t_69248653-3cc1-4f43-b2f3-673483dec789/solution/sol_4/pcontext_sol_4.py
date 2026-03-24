import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (stdlib-only): L-SHADE-style DE + SPG polish.

    Improvements over your current best (JADE-like DE + coord polish):
    - Uses an external archive (L-SHADE idea) to increase diversity and avoid premature convergence.
    - Uses a short memory of successful F/CR pairs (per-iteration sampling from memory),
      rather than only global mu_F/mu_CR.
    - Uses p-best current-to-pbest mutation (as before) but with archive difference vector.
    - Uses gradual population size reduction (keeps exploration early, exploitation later).
    - Replaces weak coordinate-only polish with a fast "stochastic projected gradient" (SPG)
      finite-difference local step + fallback coordinate probing.
    - More careful time-aware budgeting: avoids heavy phases near the deadline.

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

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

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------------- Halton seeding (scrambled) ----------------
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
            u = halton_value(k, primes[i])
            u = (u + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # Repair: bounce-back towards base (better than pure clipping for DE)
    def repair(trial, base):
        for j in range(dim):
            if trial[j] < lo[j]:
                r = random.random()
                trial[j] = lo[j] + r * (base[j] - lo[j])
            elif trial[j] > hi[j]:
                r = random.random()
                trial[j] = hi[j] - r * (hi[j] - base[j])
        return clip_inplace(trial)

    # ---------------- SPG polish (finite-diff, projected) ----------------
    def spg_polish(x0, f0, eval_budget):
        """
        Stochastic projected gradient-ish polish:
        - Approx grad via a few random coordinate finite differences.
        - Take a projected step with backtracking.
        - Fallback small coordinate probes if gradient step fails.
        """
        x = list(x0)
        fx = f0
        used = 0

        # step length in variable units
        base_alpha = 0.08
        min_alpha = 1e-6
        # finite diff epsilon per dim
        eps_base = 1e-7

        # number of sampled coordinates for gradient estimate
        m = min(dim, max(4, int(math.sqrt(dim) + 2)))

        while used < eval_budget and time.time() < deadline:
            # sample coordinates
            coords = list(range(dim))
            random.shuffle(coords)
            coords = coords[:m]

            # estimate partial gradient (forward differences)
            g = [0.0] * dim
            # choose eps per coordinate proportional to scale
            # Use current point x, evaluate only m forward points
            for j in coords:
                if used >= eval_budget:
                    break
                eps = max(eps_base * scale[j], 1e-12 * scale[j])
                xp = list(x)
                xp[j] += eps
                if xp[j] > hi[j]:
                    xp[j] = hi[j]
                fp = safe_eval(xp)
                used += 1
                # if fp is inf, just skip that coordinate
                if fp < float("inf") and fx < float("inf"):
                    g[j] = (fp - fx) / max(eps, 1e-300)

            # if gradient is all zeros, do a tiny random probe
            gn2 = sum(gg * gg for gg in g)
            if gn2 <= 0.0:
                # coordinate probe
                j = random.randrange(dim)
                step = 0.02 * scale[j]
                for sgn in (-1.0, 1.0):
                    if used >= eval_budget:
                        break
                    xn = list(x)
                    xn[j] += sgn * step
                    clip_inplace(xn)
                    fn = safe_eval(xn)
                    used += 1
                    if fn < fx:
                        x, fx = xn, fn
                        break
                # if no improvement, exit early
                break

            gn = math.sqrt(gn2)
            # normalized descent direction
            d = [(-g[j] / gn) for j in range(dim)]

            # backtracking along d
            alpha = base_alpha
            improved = False
            while alpha >= min_alpha and used < eval_budget and time.time() < deadline:
                xn = [x[j] + alpha * d[j] * scale[j] for j in range(dim)]
                clip_inplace(xn)
                fn = safe_eval(xn)
                used += 1
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    # slightly increase base step when successful
                    base_alpha = min(0.25, base_alpha * 1.15)
                    break
                alpha *= 0.5

            if not improved:
                base_alpha = max(0.01, base_alpha * 0.7)
                # small fallback coordinate probes
                if used < eval_budget:
                    j = random.randrange(dim)
                    step = 0.01 * scale[j]
                    xn = list(x)
                    xn[j] += (step if random.random() < 0.5 else -step)
                    clip_inplace(xn)
                    fn = safe_eval(xn)
                    used += 1
                    if fn < fx:
                        x, fx = xn, fn
                    else:
                        break

        return x, fx, used

    # ---------------- Initialization ----------------
    # Initial pop size and minimum pop size (L-SHADE shrink)
    NP0 = max(18, min(80, 12 + 6 * dim))
    NPmin = max(8, min(30, 8 + 2 * dim))
    NP = NP0

    pop, fit = [], []
    k = 1
    while len(pop) < NP and time.time() < deadline:
        if len(pop) % 4 == 0:
            x = rand_point()
        else:
            x = halton_point(k)
            k += 1

        fx = safe_eval(x)
        # cheap opposition try
        if random.random() < 0.6:
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

    # External archive (stores replaced individuals)
    archive = []
    Amax = 2 * NP0

    # Success-history memories for F and CR (L-SHADE style)
    H = max(6, min(25, 5 + dim // 2))
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    # p-best fraction
    def p_fraction(frac_time):
        # more exploitation later
        # keep within [2/NP, 0.2]
        pmin = 2.0 / max(2, NP)
        p = 0.18 - 0.10 * frac_time
        if p < pmin:
            p = pmin
        if p > 0.2:
            p = 0.2
        return p

    # ---------------- Main loop ----------------
    it = 0
    no_best_improve = 0
    inj_patience = max(120, 40 * dim)

    while time.time() < deadline:
        it += 1
        frac_time = (time.time() - t0) / max(1e-9, float(max_time))

        # population size reduction (linear)
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac_time))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            # remove worst to shrink
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            # shrink archive max as well
            Amax = max(Amax, 2 * NP)
            if len(archive) > Amax:
                archive = archive[:Amax]

            best_idx = min(range(NP), key=lambda i: fit[i])
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = list(pop[best_idx])

        # periodic local polish, time-aware
        if (it % max(25, 8 * dim) == 0) and frac_time > 0.15:
            budget = max(10, 2 * dim)
            px, pf, _ = spg_polish(best_x, best, budget)
            if pf < best:
                best, best_x = pf, px
                no_best_improve = 0

        # if stuck, inject a few global points
        if no_best_improve > inj_patience and time.time() < deadline:
            nrep = max(1, NP // 5)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for wi in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.5:
                    x = halton_point(k); k += 1
                else:
                    x = [best_x[j] + random.gauss(0.0, 0.45 * scale[j]) for j in range(dim)]
                    clip_inplace(x)
                fx = safe_eval(x)
                pop[wi] = list(x)
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, list(x)
                    no_best_improve = 0
            no_best_improve = inj_patience // 2

        order = sorted(range(NP), key=lambda i: fit[i])
        p = p_fraction(frac_time)
        p_count = max(2, int(math.ceil(p * NP)))

        # generation success lists
        S_F, S_CR, S_df = [], [], []

        improved_gen = False

        # union for difference vector (pop + archive)
        union = pop + archive
        union_n = len(union)

        def pick_r(exclude_set):
            # pick an index in [0, union_n) not in exclude_set
            # (exclude_set contains indices in population space [0, NP) for convenience,
            #  for union indices, we offset archive by NP)
            while True:
                r = random.randrange(union_n)
                # map to pop index if within NP, else archive index
                if r < NP:
                    if r in exclude_set:
                        continue
                # archive indices are always allowed
                return r

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            # choose memory slot
            rH = random.randrange(H)
            muF = M_F[rH]
            muCR = M_CR[rH]

            # sample CR ~ N(mu, 0.1)
            CR = random.gauss(muCR, 0.1)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # sample F ~ Cauchy(mu, 0.1) and resample if <=0
            u = random.random()
            u = min(1.0 - 1e-12, max(1e-12, u))
            F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
            tries = 0
            while F <= 0.0 and tries < 6:
                u = random.random()
                u = min(1.0 - 1e-12, max(1e-12, u))
                F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # choose pbest from top p_count
            pbest = order[random.randrange(p_count)]
            xp = pop[pbest]

            # choose r1, r2 from union, excluding i and pbest (in population)
            excl = {i, pbest}
            r1 = pick_r(excl)
            # for r2, exclude r1 if it is a pop index too (if archive, doesn't matter but keep distinct)
            while True:
                r2 = pick_r(excl)
                if r2 != r1:
                    break

            xr1 = union[r1]
            xr2 = union[r2]

            # mutation: current-to-pbest/1 with archive difference
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            repair(v, xi)

            # crossover (binomial)
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    uvec[j] = v[j]
                else:
                    uvec[j] = xi[j]
            repair(uvec, xi)

            fu = safe_eval(uvec)

            if fu <= fi:
                # add replaced parent to archive
                if len(archive) < Amax:
                    archive.append(list(xi))
                else:
                    archive[random.randrange(Amax)] = list(xi)

                pop[i] = uvec
                fit[i] = fu

                S_F.append(F)
                S_CR.append(CR)
                # fitness gain weight
                df = abs(fi - fu)
                S_df.append(df if df > 0.0 else 1e-12)

                if fu < best:
                    best = fu
                    best_x = list(uvec)
                    improved_gen = True
            # else reject

        if improved_gen:
            no_best_improve = 0
        else:
            no_best_improve += 1

        # update memories using weighted means
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = 1.0
            # weighted arithmetic mean for CR, weighted Lehmer mean for F
            mcr = 0.0
            num = 0.0
            den = 0.0
            for F, CR, w in zip(S_F, S_CR, S_df):
                ww = w / wsum
                mcr += ww * CR
                num += ww * (F * F)
                den += ww * F
            mF = (num / den) if den > 0.0 else M_F[mem_idx]

            M_F[mem_idx] = mF
            M_CR[mem_idx] = mcr
            mem_idx = (mem_idx + 1) % H

        # keep archive bounded (extra safety)
        if len(archive) > Amax:
            # random truncation
            random.shuffle(archive)
            archive = archive[:Amax]

    return best
