import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (stdlib-only).

    Improvements vs your current best (L-SHADE/JADE DE + DiagES):
      - Uses *multi-start trust-region Nelder–Mead* as a deterministic(ish) local workhorse
        late in the budget + on stagnation (excellent when gradients are unavailable).
      - Keeps a lighter L-SHADE DE front-end for exploration, but:
          * adds "jittered-best" and "recenter" injections,
          * uses triangular dithering for CR and Cauchy for F (more robust),
          * uses tighter time scheduling and stagnation detection.
      - Always keeps an incumbent; any component can improve it.
      - Robust bounds: DE uses bounce-back-to-parent; local search uses reflection.

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]
    avg_scale = sum(scale) / max(1, dim)

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_repair(x):
        # mirror reflection into bounds (smooth for local moves)
        for i in range(dim):
            L, U = lo[i], hi[i]
            if U <= L:
                x[i] = L
                continue
            v = x[i]
            if v < L or v > U:
                w = U - L
                y = (v - L) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                x[i] = L + y
        return x

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

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def de_repair(trial, base):
        # bounce-back towards base, then clip
        for j in range(dim):
            if trial[j] < lo[j]:
                r = random.random()
                trial[j] = lo[j] + r * (base[j] - lo[j])
            elif trial[j] > hi[j]:
                r = random.random()
                trial[j] = hi[j] - r * (hi[j] - base[j])
        return clip_inplace(trial)

    def norm2(a):
        return sum(v * v for v in a)

    # ---------------- scrambled Halton seeding ----------------
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
            u = (halton_value(k, primes[i]) + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # ---------------- local optimizer: bounded Nelder–Mead (time/feval bounded) ----------------
    def nelder_mead_bounded(x0, f0, feval_budget, time_budget):
        """
        Simple Nelder–Mead with reflection bound handling.
        Designed as a strong "polisher" when DE stalls / late in time.
        """
        if feval_budget <= 0 or time_budget <= 0.0:
            return x0, f0, 0

        start = time.time()
        n = dim
        # build initial simplex around x0
        simplex = [list(x0)]
        fx = [f0]
        used = 0

        # step sizes: relative to scale, not too tiny
        base_step = 0.08 * avg_scale
        for i in range(n):
            xi = list(x0)
            step = base_step * (scale[i] / (avg_scale if avg_scale > 0 else 1.0))
            if step <= 0.0:
                step = base_step
            xi[i] += step
            reflect_repair(xi)
            fi = safe_eval(xi)
            used += 1
            simplex.append(xi)
            fx.append(fi)
            if fi < f0:
                x0, f0 = xi, fi

            if used >= feval_budget or (time.time() - start) >= time_budget or time.time() >= deadline:
                # return best known
                bi = min(range(len(fx)), key=lambda k: fx[k])
                return simplex[bi], fx[bi], used

        # coefficients
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        def centroid(exclude_last=True):
            m = [0.0] * n
            count = n if exclude_last else (n + 1)
            upto = n if exclude_last else (n + 1)
            for k in range(upto):
                xk = simplex[k]
                for i in range(n):
                    m[i] += xk[i]
            inv = 1.0 / float(count)
            for i in range(n):
                m[i] *= inv
            return m

        # main loop
        while used < feval_budget and (time.time() - start) < time_budget and time.time() < deadline:
            # sort by fitness
            order = sorted(range(n + 1), key=lambda k: fx[k])
            simplex = [simplex[k] for k in order]
            fx = [fx[k] for k in order]

            bestx, bestf = simplex[0], fx[0]
            worstx, worstf = simplex[-1], fx[-1]
            second_worstf = fx[-2]

            # stopping: simplex size small (scaled)
            size = 0.0
            for k in range(1, n + 1):
                d = 0.0
                xk = simplex[k]
                for i in range(n):
                    t = (xk[i] - bestx[i]) / (scale[i] if scale[i] > 0 else 1.0)
                    d += t * t
                if d > size:
                    size = d
            if size < 1e-16:
                break

            c = centroid(exclude_last=True)

            # reflection
            xr = [c[i] + alpha * (c[i] - worstx[i]) for i in range(n)]
            reflect_repair(xr)
            fr = safe_eval(xr)
            used += 1
            if fr < bestf:
                # expansion
                xe = [c[i] + gamma * (xr[i] - c[i]) for i in range(n)]
                reflect_repair(xe)
                fe = safe_eval(xe)
                used += 1
                if fe < fr:
                    simplex[-1], fx[-1] = xe, fe
                else:
                    simplex[-1], fx[-1] = xr, fr
            elif fr < second_worstf:
                simplex[-1], fx[-1] = xr, fr
            else:
                # contraction
                if fr < worstf:
                    # outside contraction
                    xc = [c[i] + rho * (xr[i] - c[i]) for i in range(n)]
                else:
                    # inside contraction
                    xc = [c[i] + rho * (worstx[i] - c[i]) for i in range(n)]
                reflect_repair(xc)
                fc = safe_eval(xc)
                used += 1

                if fc < worstf:
                    simplex[-1], fx[-1] = xc, fc
                else:
                    # shrink
                    b = simplex[0]
                    for k in range(1, n + 1):
                        xs = [b[i] + sigma * (simplex[k][i] - b[i]) for i in range(n)]
                        reflect_repair(xs)
                        fs = safe_eval(xs)
                        used += 1
                        simplex[k], fx[k] = xs, fs
                        if used >= feval_budget or (time.time() - start) >= time_budget or time.time() >= deadline:
                            break

        bi = min(range(len(fx)), key=lambda k: fx[k])
        return simplex[bi], fx[bi], used

    # ---------------- initialization ----------------
    NP0 = max(18, min(96, 14 + 7 * dim))
    NPmin = max(8, min(40, 10 + 2 * dim))
    NP = NP0

    pop, fit = [], []
    k = 1
    while len(pop) < NP and time.time() < deadline:
        x = rand_point() if (len(pop) % 5 == 0) else halton_point(k)
        if len(pop) % 5 != 0:
            k += 1
        fx = safe_eval(x)
        if random.random() < 0.7:
            xo = opposite_point(x)
            fo = safe_eval(xo)
            if fo < fx:
                x, fx = xo, fo
        pop.append(list(x))
        fit.append(fx)

    if not pop:
        return float("inf")

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = list(pop[best_i])
    best = fit[best_i]

    # ---------------- L-SHADE memories ----------------
    archive = []
    Amax = 2 * NP0

    H = max(6, min(30, 6 + dim // 2))
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    # stagnation
    no_best = 0
    inj_patience = max(70, 22 * dim)

    def p_fraction(frac_time, NP_):
        pmin = 2.0 / max(2, NP_)
        # slightly more exploitative later
        p = 0.24 - 0.16 * frac_time
        if p < pmin:
            p = pmin
        if p > 0.35:
            p = 0.35
        return p

    it = 0
    last_local_t = 0.0

    while time.time() < deadline:
        it += 1
        now = time.time()
        frac_time = (now - t0) / max(1e-9, float(max_time))
        time_left = deadline - now

        # shrink population with time
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac_time))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            if len(archive) > 2 * NP:
                random.shuffle(archive)
                archive = archive[:2 * NP]

        # injections on stagnation
        if no_best > inj_patience and time_left > 0.03:
            nrep = max(1, NP // 4)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for wi in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.5:
                    x = halton_point(k); k += 1
                else:
                    # jittered around best (wide -> escape)
                    x = [best_x[j] + random.gauss(0.0, 0.55 * scale[j]) for j in range(dim)]
                    reflect_repair(x)
                fx = safe_eval(x)
                pop[wi] = list(x)
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, list(x)
                    no_best = 0
            no_best = inj_patience // 2

        # local NM polish late or on stagnation (rate-limited)
        if (frac_time > 0.70 or no_best > inj_patience) and time_left > 0.05:
            if (now - last_local_t) > max(0.15, 0.03 * max_time):
                last_local_t = now
                fe_budget = max(10, min(120, 12 + 8 * dim))
                # keep a strict wall-time slice
                slice_time = min(0.12 * max_time, 0.35 * time_left)
                bx, bf, _ = nelder_mead_bounded(best_x, best, fe_budget, slice_time)
                if bf < best:
                    best, best_x = bf, list(bx)
                    no_best = 0

        # --- DE generation ---
        order = sorted(range(NP), key=lambda i: fit[i])
        p = p_fraction(frac_time, NP)
        p_count = max(2, int(math.ceil(p * NP)))

        union = pop + archive
        union_n = len(union)

        def pick_r(excl):
            while True:
                r = random.randrange(union_n)
                if r < NP and r in excl:
                    continue
                return r

        S_F, S_CR, S_df = [], [], []
        improved = False

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            rH = random.randrange(H)
            muF = M_F[rH]
            muCR = M_CR[rH]

            # CR: triangular around muCR (bounded, fewer outliers than Gaussian)
            u = random.random()
            if u < 0.5:
                CR = muCR + math.sqrt(u * 0.5) * (1.0 - muCR)
            else:
                CR = muCR - math.sqrt((1.0 - u) * 0.5) * (muCR - 0.0)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # F: cauchy around muF
            uu = min(1.0 - 1e-12, max(1e-12, random.random()))
            F = muF + 0.1 * math.tan(math.pi * (uu - 0.5))
            tries = 0
            while F <= 0.0 and tries < 6:
                uu = min(1.0 - 1e-12, max(1e-12, random.random()))
                F = muF + 0.1 * math.tan(math.pi * (uu - 0.5))
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            pbest = order[random.randrange(p_count)]
            xp = pop[pbest]

            excl = {i, pbest}
            r1 = pick_r(excl)
            r2 = pick_r(excl)
            while r2 == r1:
                r2 = pick_r(excl)

            xr1 = union[r1]
            xr2 = union[r2]

            # current-to-pbest/1
            v = [xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]
            de_repair(v, xi)

            jrand = random.randrange(dim)
            uvec = [v[j] if (j == jrand or random.random() < CR) else xi[j] for j in range(dim)]
            de_repair(uvec, xi)

            fu = safe_eval(uvec)

            if fu <= fi:
                # archive parent
                if len(archive) < Amax:
                    archive.append(list(xi))
                else:
                    archive[random.randrange(Amax)] = list(xi)

                pop[i] = uvec
                fit[i] = fu

                S_F.append(F)
                S_CR.append(CR)
                df = abs(fi - fu)
                S_df.append(df if df > 0.0 else 1e-12)

                if fu < best:
                    best, best_x = fu, list(uvec)
                    improved = True

        if improved:
            no_best = 0
        else:
            no_best += 1

        # update memories (weighted Lehmer mean for F; weighted mean for CR)
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = 1.0

            mcr = 0.0
            num = 0.0
            den = 0.0
            for Fv, CRv, w in zip(S_F, S_CR, S_df):
                ww = w / wsum
                mcr += ww * CRv
                num += ww * (Fv * Fv)
                den += ww * Fv

            M_CR[mem_idx] = mcr
            M_F[mem_idx] = (num / den) if den > 0.0 else M_F[mem_idx]
            mem_idx = (mem_idx + 1) % H

        Amax = max(2 * NP, 2 * NP0)
        if len(archive) > Amax:
            random.shuffle(archive)
            archive = archive[:Amax]

        if time_left < 0.02:
            break

    # last tiny greedy probe near best (ultra-cheap)
    for _ in range(min(20, 3 + dim)):
        if time.time() >= deadline:
            break
        x = [best_x[i] + random.gauss(0.0, 0.01 * scale[i]) for i in range(dim)]
        reflect_repair(x)
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x

    return best
