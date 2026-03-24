import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (self-contained, no external libs).

    What changed vs your best (JADE-like DE + coordinate polish):
      - Better initialization: scrambled Halton (quasi-random) + optional best-centered samples
      - More robust DE core: "current-to-pbest/1" with archive + optional "rand/1" injection
      - Adaptive p-best fraction (more exploration early, more exploitation late)
      - Two-stage local search:
          (1) fast coordinate/pattern search
          (2) occasional Nelder–Mead simplex around incumbent (very effective in low/medium dim)
      - Smarter restart/stagnation detection; end-of-time intensified polishing

    Notes:
      - func must accept a list of floats (len=dim) and return a float.
      - bounds is list of (lo, hi) for each dimension.
      - returns best fitness (float).
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    span_max = max(spans) if spans else 1.0
    if span_max <= 0.0:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # ------------------------- helpers -------------------------
    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect a few times then clip
            for _ in range(4):
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
                else:
                    break
            if v < lo: v = lo
            elif v > hi: v = hi
            y[i] = v
        return y

    def eval_vec(x):
        return float(func(reflect_into_bounds(x)))

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # --- scrambled Halton (base primes) for initialization ---
    def _first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k**0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    PRIMES = _first_primes(dim)

    def _radical_inverse(index, base, scramble_shift):
        # index >= 1
        f = 1.0 / base
        x = 0.0
        i = index + scramble_shift
        while i > 0:
            x += (i % base) * f
            i //= base
            f /= base
        return x

    def halton_point(k, shifts):
        # k starts at 1
        u = [0.0] * dim
        for d in range(dim):
            u[d] = _radical_inverse(k, PRIMES[d], shifts[d])  # in [0,1)
        return u

    def halton_population(n, jitter=0.001):
        shifts = [random.randrange(1, 1000) for _ in range(dim)]
        pop = []
        # skip a few to avoid early correlation
        start_k = random.randrange(5, 50)
        for t in range(n):
            u = halton_point(start_k + t + 1, shifts)
            x = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    x[d] = lows[d]
                else:
                    # tiny jitter to break ties / symmetries
                    uj = u[d] + (random.random() - 0.5) * jitter
                    if uj < 0.0: uj = 0.0
                    if uj > 1.0: uj = 1.0
                    x[d] = lows[d] + uj * spans[d]
            pop.append(x)
        return pop

    # ------------------------- local searches -------------------------
    def local_coordinate_pattern(x, fx, time_slice):
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        xb, fb = x[:], fx

        step = [0.07 * s if s > 0 else 1.0 for s in spans]
        min_step = [1e-14 * (s if s > 0 else 1.0) for s in spans]

        no_imp = 0
        while time.time() < t_end:
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if time.time() >= t_end:
                    break
                if step[i] <= min_step[i]:
                    continue

                base = xb[i]
                best_i = base
                best_f = fb

                for mult in (1.0, 0.5, 1.5, 2.2):
                    delta = step[i] * mult

                    x1 = xb[:]
                    x1[i] = base + delta
                    f1 = eval_vec(x1)
                    if f1 < best_f:
                        best_f = f1
                        best_i = x1[i]

                    x2 = xb[:]
                    x2[i] = base - delta
                    f2 = eval_vec(x2)
                    if f2 < best_f:
                        best_f = f2
                        best_i = x2[i]

                if best_f < fb:
                    xb[i] = best_i
                    fb = best_f
                    improved = True
                else:
                    step[i] *= 0.72

            # occasional 2D pattern move (helps rotated basins)
            if dim >= 2 and time.time() < t_end and random.random() < 0.35:
                i = random.randrange(dim)
                j = random.randrange(dim - 1)
                if j >= i:
                    j += 1
                if step[i] > min_step[i] and step[j] > min_step[j]:
                    xt = xb[:]
                    xt[i] += step[i] * (1.0 if random.random() < 0.5 else -1.0)
                    xt[j] += step[j] * (1.0 if random.random() < 0.5 else -1.0)
                    ft = eval_vec(xt)
                    if ft < fb:
                        xb, fb = reflect_into_bounds(xt), ft
                        improved = True

            if improved:
                no_imp = 0
                for d in range(dim):
                    step[d] *= 1.04
            else:
                no_imp += 1
                for d in range(dim):
                    step[d] *= 0.88
                if no_imp >= 3:
                    break

        return xb, fb

    def nelder_mead(x0, f0, time_slice):
        """
        Small Nelder–Mead around incumbent.
        Uses dim+1 simplex; good when dim is not huge.
        """
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        if dim == 0:
            return x0[:], f0

        # Build simplex
        simplex = [x0[:]]
        fvals = [f0]
        # scale: relative to bounds
        scale = [0.06 * s if s > 0 else 1.0 for s in spans]

        for i in range(dim):
            x = x0[:]
            x[i] = x[i] + scale[i] if spans[i] > 0 else x[i]
            x = reflect_into_bounds(x)
            simplex.append(x)
            fvals.append(eval_vec(x))
            if time.time() >= t_end:
                # return best so far
                bi = min(range(len(fvals)), key=lambda k: fvals[k])
                return simplex[bi][:], fvals[bi]

        # coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        def sort_simplex():
            nonlocal simplex, fvals
            idx = sorted(range(len(fvals)), key=lambda k: fvals[k])
            simplex = [simplex[k] for k in idx]
            fvals = [fvals[k] for k in idx]

        sort_simplex()

        while time.time() < t_end:
            # centroid of all but worst
            centroid = [0.0] * dim
            for k in range(dim):  # first dim points (exclude worst at dim)
                xk = simplex[k]
                for d in range(dim):
                    centroid[d] += xk[d]
            inv = 1.0 / dim
            for d in range(dim):
                centroid[d] *= inv

            best = simplex[0]
            worst = simplex[-1]
            fbest = fvals[0]
            fworst = fvals[-1]
            fsecond = fvals[-2]

            # reflect
            xr = [centroid[d] + alpha * (centroid[d] - worst[d]) for d in range(dim)]
            xr = reflect_into_bounds(xr)
            fr = eval_vec(xr)
            if time.time() >= t_end:
                break

            if fr < fbest:
                # expand
                xe = [centroid[d] + gamma * (xr[d] - centroid[d]) for d in range(dim)]
                xe = reflect_into_bounds(xe)
                fe = eval_vec(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fsecond:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contract
                if fr < fworst:
                    # outside contraction
                    xc = [centroid[d] + rho * (xr[d] - centroid[d]) for d in range(dim)]
                else:
                    # inside contraction
                    xc = [centroid[d] - rho * (centroid[d] - worst[d]) for d in range(dim)]
                xc = reflect_into_bounds(xc)
                fc = eval_vec(xc)

                if fc < fworst:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink towards best
                    for k in range(1, dim + 1):
                        xk = simplex[k]
                        xs = [best[d] + sigma * (xk[d] - best[d]) for d in range(dim)]
                        xs = reflect_into_bounds(xs)
                        simplex[k] = xs
                        fvals[k] = eval_vec(xs)
                        if time.time() >= t_end:
                            break

            sort_simplex()

        return simplex[0][:], fvals[0]

    # ------------------------- DE (JADE/SHADE-like) -------------------------
    pop_size = max(16, min(96, 10 * dim))
    H = 8
    MF = [0.6] * H
    MCR = [0.85] * H
    hist_idx = 0
    c = 0.12

    archive = []
    arch_max = pop_size

    best = float("inf")
    best_x = None
    last_improve = time.time()
    min_progress = 1e-12

    stall_seconds = max(0.14 * float(max_time), 0.45)
    end_polish_reserved = 0.12 * float(max_time)

    def update_best(x, fx):
        nonlocal best, best_x, last_improve
        if fx + min_progress < best:
            best = fx
            best_x = x[:]
            last_improve = time.time()

    # multi-start loop
    while time.time() < deadline:
        # init: quasi-random + some around incumbent
        pop = []
        if best_x is not None:
            k = min(pop_size // 4, 12)
            for _ in range(k):
                x = best_x[:]
                for d in range(dim):
                    if spans[d] > 0:
                        x[d] += random.gauss(0.0, 0.16 * spans[d])
                pop.append(reflect_into_bounds(x))

        rem = pop_size - len(pop)
        pop.extend(halton_population(rem, jitter=0.002))

        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = eval_vec(x)
            fit.append(fx)
            update_best(x, fx)

        while time.time() < deadline:
            # adaptive p-best fraction: explore early, exploit late
            remaining = deadline - time.time()
            frac_left = remaining / max(1e-9, float(max_time))
            p = 0.25 if frac_left > 0.6 else (0.18 if frac_left > 0.25 else 0.10)

            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            pcount = max(2, int(math.ceil(p * pop_size)))
            pbest_idx = idx_sorted[:pcount]

            S_F, S_CR, dF = [], [], []

            new_pop = pop[:]
            new_fit = fit[:]

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi, fi = pop[i], fit[i]

                r = random.randrange(H)
                muF, muCR = MF[r], MCR[r]

                CR = random.gauss(muCR, 0.1)
                if CR < 0.0: CR = 0.0
                if CR > 1.0: CR = 1.0

                # Cauchy for F
                F = -1.0
                for _ in range(12):
                    u = random.random()
                    F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                    if F > 0.0:
                        break
                if F <= 0.0:
                    F = 0.5
                if F > 1.0:
                    F = 1.0

                # choose pbest
                pb = pop[random.choice(pbest_idx)]

                # choose r1 != i
                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)

                # choose r2 from pop U archive
                use_arch = (archive and random.random() < 0.5)
                if use_arch:
                    pool = pop + archive
                    xr2 = pool[random.randrange(len(pool))]
                else:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    xr2 = pop[r2]
                xr1 = pop[r1]

                # mostly current-to-pbest/1, sometimes rand/1 for escape
                if random.random() < 0.84:
                    v = [xi[d] + F * (pb[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
                else:
                    # rand/1: a + F*(b-c)
                    a = random.randrange(pop_size)
                    b = random.randrange(pop_size)
                    cidx = random.randrange(pop_size)
                    xa, xb, xc = pop[a], pop[b], pop[cidx]
                    v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

                jrand = random.randrange(dim)
                uvec = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        uvec[d] = v[d]

                uvec = reflect_into_bounds(uvec)
                fu = eval_vec(uvec)

                if fu <= fi:
                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))

                    new_pop[i] = uvec
                    new_fit[i] = fu

                    S_F.append(F)
                    S_CR.append(CR)
                    df = abs(fi - fu)
                    dF.append(df if df > 0.0 else 1e-16)

                    update_best(uvec, fu)

            pop, fit = new_pop, new_fit

            # memory update
            if S_F:
                wsum = sum(dF)
                inv = 1.0 / wsum if wsum > 0 else 1.0 / len(dF)
                weights = [(df * inv) if wsum > 0 else inv for df in dF]

                num = 0.0
                den = 0.0
                for w, f in zip(weights, S_F):
                    num += w * f * f
                    den += w * f
                F_new = num / den if den > 0 else sum(S_F) / len(S_F)

                CR_new = 0.0
                for w, cr in zip(weights, S_CR):
                    CR_new += w * cr

                MF[hist_idx] = (1.0 - c) * MF[hist_idx] + c * F_new
                MCR[hist_idx] = (1.0 - c) * MCR[hist_idx] + c * CR_new
                hist_idx = (hist_idx + 1) % H

            # polishing schedule
            remaining = deadline - time.time()
            if best_x is not None and remaining > 0.02:
                # more polishing near the end
                if remaining > end_polish_reserved:
                    if random.random() < 0.06:
                        bx, bf = local_coordinate_pattern(best_x, best, min(0.06, 0.12 * remaining))
                        update_best(bx, bf)
                    if dim <= 25 and random.random() < 0.015:
                        bx, bf = nelder_mead(best_x, best, min(0.08, 0.10 * remaining))
                        update_best(bx, bf)
                else:
                    # end-game: intensify
                    if random.random() < 0.22:
                        bx, bf = local_coordinate_pattern(best_x, best, min(0.10, 0.22 * remaining))
                        update_best(bx, bf)
                    if dim <= 30 and random.random() < 0.10:
                        bx, bf = nelder_mead(best_x, best, min(0.12, 0.26 * remaining))
                        update_best(bx, bf)

            # restart if stalled
            if time.time() - last_improve > stall_seconds:
                # one last attempt before restarting
                remaining = deadline - time.time()
                if best_x is not None and remaining > 0.03:
                    bx, bf = local_coordinate_pattern(best_x, best, min(0.12, 0.25 * remaining))
                    update_best(bx, bf)
                    if dim <= 25 and remaining > 0.05:
                        bx, bf = nelder_mead(best_x, best, min(0.14, 0.25 * remaining))
                        update_best(bx, bf)
                break

    return best
