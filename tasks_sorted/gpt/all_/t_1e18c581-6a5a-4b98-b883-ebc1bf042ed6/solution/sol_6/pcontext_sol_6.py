import random, time, math

def run(func, dim, bounds, max_time):
    """
    Hybrid time-bounded derivative-free minimizer (self-contained).

    Improvements vs previous versions:
      - Better global-to-local scheduling: early space-filling + late intensification
      - Triangular-reflection boundary handling (stable, unbiased vs clamping)
      - Multi-basin search: elite archive + probabilistic basin selection
      - Two local optimizers:
          * Nelder-Mead (robust local simplex, no gradients)
          * Hooke-Jeeves pattern search (cheap coordinate exploitation)
      - Lightweight surrogate-free "trust region" sampling around elites
      - Automatic restarts on stall with opposition + widened trust region

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def reflect(v, lo, hi):
        # triangular reflection into [lo, hi]
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def proj(x):
        return [reflect(x[i], lows[i], highs[i]) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------- low-discrepancy Halton ----------
    def nth_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    bases = nth_primes(max(1, dim))
    h_idx = 1
    h_shift = [random.random() for _ in range(dim)]

    def radical_inverse(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    def halton_vec():
        nonlocal h_idx
        k = h_idx
        h_idx += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (radical_inverse(k, bases[i]) + h_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite archive ----------
    elite_cap = max(14, 6 + int(5.0 * math.sqrt(dim + 1.0)))
    elites = []  # (f, x) sorted

    def push_elite(f, x):
        nonlocal elites
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ---------- local search: Hooke-Jeeves ----------
    def hooke_jeeves(x0, f0, step0, max_passes=2):
        x = x0[:]
        f = f0
        step = step0[:]
        min_step = [1e-15 * spans_safe[i] for i in range(dim)]

        def clamp01(v):  # for cheap comparisons
            return v

        for _ in range(max_passes):
            if time.time() >= deadline:
                break

            base = x[:]
            base_f = f
            improved = False

            # exploratory
            for j in range(dim):
                if time.time() >= deadline:
                    break
                sj = step[j]
                if sj <= min_step[j]:
                    continue

                xp = x[:]
                xp[j] = reflect(xp[j] + sj, lows[j], highs[j])
                fp = evaluate(xp)
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if time.time() >= deadline:
                    break

                xm = x[:]
                xm[j] = reflect(xm[j] - sj, lows[j], highs[j])
                fm = evaluate(xm)
                if fm < f:
                    x, f = xm, fm
                    improved = True

            if time.time() >= deadline:
                break

            if improved and f < base_f:
                # pattern move
                patt = [reflect(x[i] + (x[i] - base[i]), lows[i], highs[i]) for i in range(dim)]
                fp = evaluate(patt)
                if fp < f:
                    x, f = patt, fp
            else:
                # shrink
                shrunk = 0
                for j in range(dim):
                    if step[j] > min_step[j]:
                        step[j] *= 0.5
                        if step[j] < min_step[j]:
                            step[j] = min_step[j]
                        shrunk += 1
                if shrunk == 0:
                    break

        return x, f, step

    # ---------- local search: Nelder-Mead ----------
    def nelder_mead(x_start, f_start, scale, max_iter=30):
        # Classic NM with reflection/expansion/contraction/shrink.
        # Uses reflection projection for bounds.
        if dim == 0:
            return x_start, f_start

        # create simplex
        simplex = [(f_start, x_start[:])]
        for i in range(dim):
            if time.time() >= deadline:
                break
            x = x_start[:]
            x[i] = reflect(x[i] + scale[i], lows[i], highs[i])
            fx = evaluate(x)
            simplex.append((fx, x))

        if len(simplex) < 2:
            return x_start, f_start

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_iter):
            if time.time() >= deadline:
                break

            simplex.sort(key=lambda t: t[0])
            best_f, best_x = simplex[0]
            worst_f, worst_x = simplex[-1]
            second_worst_f, _ = simplex[-2]

            # centroid of all but worst
            centroid = [0.0] * dim
            m = len(simplex) - 1
            for _, x in simplex[:-1]:
                for i in range(dim):
                    centroid[i] += x[i]
            for i in range(dim):
                centroid[i] /= float(m)

            # reflect
            xr = [reflect(centroid[i] + alpha * (centroid[i] - worst_x[i]), lows[i], highs[i]) for i in range(dim)]
            fr = evaluate(xr)

            if fr < best_f:
                # expand
                xe = [reflect(centroid[i] + gamma * (xr[i] - centroid[i]), lows[i], highs[i]) for i in range(dim)]
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = (fe, xe)
                else:
                    simplex[-1] = (fr, xr)
            elif fr < second_worst_f:
                simplex[-1] = (fr, xr)
            else:
                # contract
                if fr < worst_f:
                    # outside
                    xc = [reflect(centroid[i] + rho * (xr[i] - centroid[i]), lows[i], highs[i]) for i in range(dim)]
                else:
                    # inside
                    xc = [reflect(centroid[i] - rho * (centroid[i] - worst_x[i]), lows[i], highs[i]) for i in range(dim)]
                fc = evaluate(xc)

                if fc < worst_f:
                    simplex[-1] = (fc, xc)
                else:
                    # shrink towards best
                    new_simplex = [simplex[0]]
                    bx = simplex[0][1]
                    for (fi, xi) in simplex[1:]:
                        if time.time() >= deadline:
                            break
                        xs = [reflect(bx[d] + sigma * (xi[d] - bx[d]), lows[d], highs[d]) for d in range(dim)]
                        fs = evaluate(xs)
                        new_simplex.append((fs, xs))
                    simplex = new_simplex

        simplex.sort(key=lambda t: t[0])
        return simplex[0][1], simplex[0][0]

    # ---------- initialization ----------
    best = float("inf")
    best_x = None

    init_n = max(40, 16 * dim)
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_vec() if random.random() < 0.85 else rand_vec()
        f = evaluate(x)
        if f < best:
            best, best_x = f, x
        push_elite(f, x)

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = evaluate(xo)
        if fo < best:
            best, best_x = fo, xo
        push_elite(fo, xo)

    if best_x is None:
        best_x = rand_vec()
        best = evaluate(best_x)
        push_elite(best, best_x)

    # quick polish
    hj_step = [0.10 * spans_safe[i] for i in range(dim)]
    best_x, best, hj_step = hooke_jeeves(best_x, best, hj_step, max_passes=1)
    push_elite(best, best_x)

    # ---------- main loop (elite trust-region + NM/HJ intensification) ----------
    it = 0
    stall = 0
    last_best = best
    patience = max(80, 28 * dim)

    # trust-region radii (absolute)
    rad = [0.25 * spans_safe[i] for i in range(dim)]
    min_rad = [1e-15 * spans_safe[i] for i in range(dim)]
    max_rad = [2.0 * spans_safe[i] for i in range(dim)]

    pop = max(26, 12 + int(10.0 * math.log(dim + 1.0)))
    polish_every = max(14, 4 * dim)
    nm_every = max(35, 10 * dim)
    inject_every = max(28, 7 * dim)

    while time.time() < deadline:
        it += 1

        # choose basin (biased to better elites)
        if elites:
            kmax = min(len(elites), max(5, elite_cap))
            idx = int((random.random() ** 2.5) * kmax)
            anchor = elites[idx][1]
        else:
            anchor = best_x

        # occasional global injection
        if it % inject_every == 0:
            n = max(6, dim // 2)
            for _ in range(n):
                if time.time() >= deadline:
                    return best
                xg = halton_vec() if random.random() < 0.85 else rand_vec()
                fg = evaluate(xg)
                push_elite(fg, xg)
                if fg < best:
                    best, best_x = fg, xg
                    stall = 0

        # sample around anchor with decaying/adjustable radii (trust-region)
        samples = []
        for _ in range(pop):
            if time.time() >= deadline:
                break
            if random.random() < 0.12:
                x = halton_vec() if random.random() < 0.8 else rand_vec()
            else:
                x = [0.0] * dim
                # mixture: mostly local, sometimes broader
                broad = 1.0 if random.random() < 0.80 else 2.2
                for i in range(dim):
                    x[i] = anchor[i] + broad * random.gauss(0.0, rad[i])
                x = proj(x)
            f = evaluate(x)
            samples.append((f, x))
            push_elite(f, x)
            if f < best:
                best, best_x = f, x

        if samples:
            samples.sort(key=lambda t: t[0])
            # adapt trust region: if we see strong progress, shrink a bit (focus), else expand a bit
            # based on relative improvement of best-in-sample vs current best before sampling
            best_in = samples[0][0]
            if best_in < last_best - 1e-12:
                for i in range(dim):
                    rad[i] = max(min_rad[i], rad[i] * 0.92)
            else:
                for i in range(dim):
                    rad[i] = min(max_rad[i], rad[i] * 1.03)

        # periodic polishing
        if it % polish_every == 0 and time.time() < deadline:
            # Hooke-Jeeves polish around best using trust radii as steps
            steps = [max(min_rad[i], 0.8 * rad[i]) for i in range(dim)]
            best_x, best, _ = hooke_jeeves(best_x, best, steps, max_passes=2)
            push_elite(best, best_x)

        if it % nm_every == 0 and time.time() < deadline:
            # Nelder-Mead from best (stronger local basin search)
            nm_scale = [max(min_rad[i], 0.6 * rad[i]) for i in range(dim)]
            bx, bf = nelder_mead(best_x, best, nm_scale, max_iter=18)
            if bf < best:
                best, best_x = bf, bx
            push_elite(bf, bx)

        # stall / restart
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= patience and time.time() < deadline:
            stall = 0
            # restart from a decent elite or global point, and widen region
            if elites and random.random() < 0.85:
                kmax = min(len(elites), max(6, elite_cap))
                idx = int((random.random() ** 2.2) * kmax)
                base = elites[idx][1][:]
            else:
                base = halton_vec() if random.random() < 0.8 else rand_vec()

            # opposition try
            xop = opposite(base)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop

            # widen trust region for re-exploration
            for i in range(dim):
                rad[i] = min(max_rad[i], 0.35 * spans_safe[i])

    return best
