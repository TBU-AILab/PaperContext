import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Hybrid strategy:
      1) Space-filling exploration: scrambled Halton + occasional pure random.
      2) Maintain an elite pool (best K points).
      3) From elites, run a budgeted local optimizer:
           - Adaptive pattern/coordinate search (very robust on noisy/non-smooth)
           - Occasional simplex-like local refinement (lightweight NM variant)
      4) Stagnation handling: restarts + larger perturbations around elites.

    Returns:
        best (float): minimum function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        return float("inf")
    if bounds is None or len(bounds) != dim:
        raise ValueError("bounds must be a list of (low, high) pairs, one per dimension")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if not (s > 0.0):
            raise ValueError("Each bound must satisfy high > low")

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_point(x):
        try:
            v = func(x)
            return float(v)
        except Exception:
            return float("inf")

    def push_elite(elites, x, fx, K):
        if not math.isfinite(fx):
            return
        if len(elites) < K:
            elites.append((fx, x[:]))
            elites.sort(key=lambda t: t[0])
        else:
            if fx < elites[-1][0]:
                elites[-1] = (fx, x[:])
                elites.sort(key=lambda t: t[0])

    # ---------------- Halton (scrambled) ----------------
    def first_primes(n):
        primes = []
        c = 2
        while len(primes) < n:
            is_p = True
            r = int(c ** 0.5)
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(c)
            c += 1
        return primes

    primes = first_primes(dim)

    def van_der_corput(n, base):
        vdc = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    # Cranley-Patterson rotation (simple scramble): add a random shift in [0,1)
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = []
        for i in range(dim):
            u = van_der_corput(idx, primes[i])
            u = u + halton_shift[i]
            u -= int(u)  # frac
            x.append(lows[i] + u * spans[i])
        return x

    # ---------------- Local search: adaptive pattern ----------------
    def pattern_search(x0, f0, time_limit, eval_limit):
        """
        Robust coordinate/pattern search with:
          - adaptive per-dimension steps
          - occasional multi-dim "pattern" move when improvement found
          - random coordinate order
        """
        x = x0[:]
        fx = f0
        evals = 0

        step = [0.18 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]
        max_step = [0.45 * spans[i] for i in range(dim)]

        # direction memory for pattern moves
        last_dx = [0.0] * dim

        noimp = 0
        while evals < eval_limit and time.time() < time_limit:
            improved = False
            coords = list(range(dim))
            random.shuffle(coords)

            base = x[:]
            base_fx = fx

            for i in coords:
                if evals >= eval_limit or time.time() >= time_limit:
                    break

                # try +step
                xi = x[:]
                xi[i] += step[i]
                clip_inplace(xi)
                f1 = eval_point(xi); evals += 1
                if f1 < fx:
                    last_dx = [xi[j] - x[j] for j in range(dim)]
                    x, fx = xi, f1
                    improved = True
                    continue

                if evals >= eval_limit or time.time() >= time_limit:
                    break

                # try -step
                xi = x[:]
                xi[i] -= step[i]
                clip_inplace(xi)
                f2 = eval_point(xi); evals += 1
                if f2 < fx:
                    last_dx = [xi[j] - x[j] for j in range(dim)]
                    x, fx = xi, f2
                    improved = True

            # If we improved during coordinate moves, try a pattern move (accelerates along valley)
            if improved and evals < eval_limit and time.time() < time_limit:
                xp = [x[i] + 0.8 * last_dx[i] for i in range(dim)]
                clip_inplace(xp)
                fp = eval_point(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp

            # Step adaptation
            if fx < base_fx:
                noimp = 0
                for i in range(dim):
                    step[i] = min(max_step[i], step[i] * 1.22)
            else:
                noimp += 1
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.55)

                # Stagnation kick: small random perturbation (local escape)
                if noimp >= 3 and evals < eval_limit and time.time() < time_limit:
                    noimp = 0
                    xr = x[:]
                    # "Gaussian-ish" via sum of uniforms
                    for i in range(dim):
                        z = (sum(random.random() for _ in range(8)) - 4.0)
                        xr[i] += z * (0.06 * spans[i])
                    clip_inplace(xr)
                    fr = eval_point(xr); evals += 1
                    if fr < fx:
                        x, fx = xr, fr

        return x, fx, evals

    # ---------------- Lightweight NM-like refinement ----------------
    def simplex_refine(x0, f0, time_limit, eval_limit):
        """
        Small-budget Nelder-Mead-style refinement.
        Kept lightweight and safe; helps on smooth-ish basins.
        """
        n = dim
        evals = 0

        # Build simplex
        simplex = [x0[:]]
        fvals = [f0]
        delta = [0.10 * spans[i] for i in range(n)]

        for i in range(n):
            if evals >= eval_limit or time.time() >= time_limit:
                break
            xi = x0[:]
            xi[i] += (delta[i] if delta[i] > 0 else 1e-6)
            clip_inplace(xi)
            fi = eval_point(xi); evals += 1
            simplex.append(xi)
            fvals.append(fi)

        if len(simplex) < 2:
            return x0, f0, evals

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        def order():
            idx = list(range(len(simplex)))
            idx.sort(key=lambda k: fvals[k])
            s2 = [simplex[k] for k in idx]
            f2 = [fvals[k] for k in idx]
            simplex[:] = s2
            fvals[:] = f2

        def centroid_excl_last():
            m = len(simplex) - 1
            c = [0.0] * n
            for k in range(m):
                xk = simplex[k]
                for i in range(n):
                    c[i] += xk[i]
            inv = 1.0 / m
            for i in range(n):
                c[i] *= inv
            return c

        order()
        bestx = simplex[0][:]
        bestf = fvals[0]

        while evals < eval_limit and time.time() < time_limit:
            order()
            if fvals[0] < bestf:
                bestf, bestx = fvals[0], simplex[0][:]

            xh, fh = simplex[-1], fvals[-1]
            xg, fg = simplex[-2], fvals[-2]
            xl, fl = simplex[0], fvals[0]

            c = centroid_excl_last()

            # reflection
            xr = [c[i] + alpha * (c[i] - xh[i]) for i in range(n)]
            clip_inplace(xr)
            fr = eval_point(xr); evals += 1

            if fl <= fr < fg:
                simplex[-1], fvals[-1] = xr, fr
                continue

            if fr < fl:
                # expansion
                xe = [c[i] + gamma * (xr[i] - c[i]) for i in range(n)]
                clip_inplace(xe)
                fe = eval_point(xe); evals += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
                continue

            # contraction
            if fr < fh:
                xc = [c[i] + rho * (xr[i] - c[i]) for i in range(n)]  # outside
            else:
                xc = [c[i] + rho * (xh[i] - c[i]) for i in range(n)]  # inside
            clip_inplace(xc)
            fc = eval_point(xc); evals += 1
            if fc < min(fh, fr):
                simplex[-1], fvals[-1] = xc, fc
                continue

            # shrink
            xb = simplex[0][:]
            for k in range(1, len(simplex)):
                if evals >= eval_limit or time.time() >= time_limit:
                    break
                xs = [xb[i] + sigma * (simplex[k][i] - xb[i]) for i in range(n)]
                clip_inplace(xs)
                fs = eval_point(xs); evals += 1
                simplex[k], fvals[k] = xs, fs

        order()
        if fvals[0] < bestf:
            return simplex[0], fvals[0], evals
        return bestx, bestf, evals

    # ---------------- Initialization ----------------
    best = float("inf")
    best_x = None

    elites = []
    # slightly larger elite pool than your "best" code, but bounded
    K = 8 if dim <= 10 else 5

    # time-safe initial sampling: mix halton + random + "corners" probing
    init_n = max(40, min(220, 40 + 10 * dim))

    # probe a few bound corners (sometimes objective likes edges)
    if time.time() < deadline:
        for _ in range(min(2 * dim + 2, 30)):
            x = []
            for i in range(dim):
                x.append(lows[i] if random.random() < 0.5 else highs[i])
            fx = eval_point(x)
            if fx < best:
                best, best_x = fx, x[:]
            push_elite(elites, x, fx, K)
            if time.time() >= deadline:
                return best

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        r = random.random()
        if r < 0.75:
            x = halton_point()
        else:
            x = rand_point()
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(elites, x, fx, K)

    if best_x is None:
        return best

    # ---------------- Main loop ----------------
    stagnation = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        elites.sort(key=lambda t: t[0])
        best = elites[0][0]
        best_x = elites[0][1][:]

        # --- global exploration burst ---
        # more when stuck
        burst = 10 + (8 if stagnation >= 4 else 0) + (8 if dim > 8 else 0)
        for _ in range(burst):
            if time.time() >= deadline:
                return best

            r = random.random()
            if r < 0.50:
                x = halton_point()
            elif r < 0.70:
                x = rand_point()
            else:
                # around a random elite with adaptive radius
                fx_e, xe = elites[random.randrange(len(elites))]
                x = xe[:]
                rad = (0.10 if stagnation < 4 else 0.22)
                for i in range(dim):
                    z = (sum(random.random() for _ in range(10)) - 5.0)
                    x[i] += z * (rad * spans[i])
                clip_inplace(x)

            fx = eval_point(x)
            if fx < best:
                best, best_x = fx, x[:]
                stagnation = 0
            else:
                stagnation += 1
            push_elite(elites, x, fx, K)

        elites.sort(key=lambda t: t[0])

        # --- local refinement on top elites ---
        # small time slices to avoid getting stuck in one local run
        # refine 1-2 elites depending on remaining time
        refine_count = 2 if (remaining > 0.35 * max_time and len(elites) > 1) else 1
        for j in range(refine_count):
            if time.time() >= deadline:
                return best
            fx0, x0 = elites[j]

            # allocate a small but meaningful slice
            slice1 = min(deadline, time.time() + max(0.01, 0.10 * remaining))
            x1, f1, _ = pattern_search(x0, fx0, time_limit=slice1, eval_limit=30 + 8 * dim)
            push_elite(elites, x1, f1, K)

            if f1 < best:
                best, best_x = f1, x1[:]
                stagnation = 0

            if time.time() >= deadline:
                return best

            # occasionally do simplex refine (more expensive, but powerful)
            if random.random() < (0.65 if dim <= 12 else 0.35):
                slice2 = min(deadline, time.time() + max(0.01, 0.16 * (deadline - time.time())))
                x2, f2, _ = simplex_refine(x1, f1, time_limit=slice2, eval_limit=35 + 10 * dim)
                push_elite(elites, x2, f2, K)
                if f2 < best:
                    best, best_x = f2, x2[:]
                    stagnation = 0

        # periodic "hard restart" injection if very stuck: re-seed elite pool with fresh points
        if stagnation >= 18 and time.time() < deadline:
            stagnation = 0
            for _ in range(max(10, 3 * dim)):
                if time.time() >= deadline:
                    return best
                x = rand_point()
                fx = eval_point(x)
                push_elite(elites, x, fx, K)
                if fx < best:
                    best, best_x = fx, x[:]

    return best
