import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Main changes vs your previous attempts:
      - Multi-start + elite archive (keeps several good basins alive).
      - Robust local optimizer: bounded Nelder–Mead with periodic restarts.
      - Occasional coordinate/pattern polishing around the best.
      - Time-aware scheduling (more global early, more local late).
      - Mirrored bounds handling (less bias than clipping).

    Returns:
      best (float): minimum objective value found within max_time.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    spans_nz = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def repair(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    # Box-Muller normal
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        t = 2.0 * math.pi * u2
        z0 = r * math.cos(t)
        z1 = r * math.sin(t)
        _spare = z1
        _has_spare = True
        return z0

    # ---- Halton for global coverage ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---- elite archive ----
    elite_size = max(10, min(40, 12 + int(3.5 * math.sqrt(dim))))
    elites = []  # sorted list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        lo, hi = 0, len(elites)
        while lo < hi:
            mid = (lo + hi) // 2
            if fx < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > elite_size:
            elites.pop()

    def best_from_elite():
        if not elites:
            return float("inf"), None
        return elites[0][0], elites[0][1][:]

    # ---- Initialization (global probes + opposition) ----
    best = float("inf")
    best_x = None

    init_n = max(32, min(320, 40 + 18 * int(math.sqrt(dim))))
    for _ in range(init_n):
        if now() >= deadline:
            return best

        if random.random() < 0.82:
            x = halton_point(hal_k)
            hal_k += 1
        else:
            x = rand_uniform_point()

        fx = evaluate(x)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        if now() >= deadline:
            return best

        # opposition point
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        repair(xo)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]
        push_elite(best, best_x)

    # ---- Bounded Nelder–Mead (with restarts) ----
    def nm_optimize(x0, f0, max_evals, start_scale):
        """
        Local Nelder–Mead using mirrored bounds (repair on every vertex).
        Returns (best_f, best_x, evals_used).
        """
        if dim == 1:
            # 1D: just do a tiny pattern search
            xbest = x0[:]
            fbest = f0
            evals = 0
            step = min(0.25 * spans_nz[0], max(1e-12, start_scale * spans_nz[0]))
            for _ in range(8):
                if evals >= max_evals:
                    break
                xp = [xbest[0] + step]
                repair(xp)
                fp = evaluate(xp); evals += 1
                if fp < fbest:
                    xbest, fbest = xp, fp
                    continue
                if evals >= max_evals:
                    break
                xm = [xbest[0] - step]
                repair(xm)
                fm = evaluate(xm); evals += 1
                if fm < fbest:
                    xbest, fbest = xm, fm
                else:
                    step *= 0.5
            return fbest, xbest, evals

        # NM coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # initial simplex
        simplex = []
        simplex.append((f0, x0[:]))
        # build n additional points
        for i in range(dim):
            xi = x0[:]
            step = start_scale * spans_nz[i]
            if step <= 0.0:
                step = start_scale
            # signed step to avoid always hitting the same boundary direction
            xi[i] += step if (random.random() < 0.5) else -step
            repair(xi)
            fi = evaluate(xi)
            simplex.append((fi, xi))

        evals = dim  # already evaluated dim new points
        # if time is super tight, still return the best of these
        simplex.sort(key=lambda t: t[0])

        def centroid(exclude_last=True):
            m = [0.0] * dim
            count = len(simplex) - 1 if exclude_last else len(simplex)
            for j in range(count):
                xj = simplex[j][1]
                for k in range(dim):
                    m[k] += xj[k]
            inv = 1.0 / float(count)
            for k in range(dim):
                m[k] *= inv
            return m

        # iterate
        last_improve_eval = 0
        while evals < max_evals:
            simplex.sort(key=lambda t: t[0])
            fbest, xbest = simplex[0]
            fworst, xworst = simplex[-1]
            f2, x2 = simplex[-2]

            # stagnation inside local run
            if evals - last_improve_eval > (10 + 3 * dim):
                break

            c = centroid(exclude_last=True)

            # reflection
            xr = [c[i] + alpha * (c[i] - xworst[i]) for i in range(dim)]
            repair(xr)
            fr = evaluate(xr)
            evals += 1

            if fr < fbest:
                # expansion
                xe = [c[i] + gamma * (xr[i] - c[i]) for i in range(dim)]
                repair(xe)
                fe = evaluate(xe)
                evals += 1
                if fe < fr:
                    simplex[-1] = (fe, xe)
                    last_improve_eval = evals
                else:
                    simplex[-1] = (fr, xr)
                    last_improve_eval = evals
            elif fr < f2:
                simplex[-1] = (fr, xr)
                last_improve_eval = evals if fr < fworst else last_improve_eval
            else:
                # contraction
                if fr < fworst:
                    # outside contraction
                    xc = [c[i] + rho * (xr[i] - c[i]) for i in range(dim)]
                else:
                    # inside contraction
                    xc = [c[i] - rho * (c[i] - xworst[i]) for i in range(dim)]
                repair(xc)
                fc = evaluate(xc)
                evals += 1

                if fc < fworst:
                    simplex[-1] = (fc, xc)
                    last_improve_eval = evals
                else:
                    # shrink towards best
                    new_simplex = [simplex[0]]
                    xb = simplex[0][1]
                    for j in range(1, len(simplex)):
                        xj = simplex[j][1]
                        xs = [xb[i] + sigma * (xj[i] - xb[i]) for i in range(dim)]
                        repair(xs)
                        fs = evaluate(xs)
                        evals += 1
                        new_simplex.append((fs, xs))
                        if evals >= max_evals:
                            break
                    simplex = new_simplex

        simplex.sort(key=lambda t: t[0])
        return simplex[0][0], simplex[0][1][:], evals

    # ---- Polishing: cheap coordinate pattern around current best ----
    def coord_polish(x, fx, max_steps, base_step):
        xbest = x[:]
        fbest = fx
        idxs = list(range(dim))
        idxs.sort(key=lambda i: spans_nz[i], reverse=True)
        idxs = idxs[:max(1, min(dim, 14))]
        evals = 0
        step_mul = base_step
        for _ in range(max_steps):
            improved = False
            for i in idxs:
                if evals >= max_steps * len(idxs):
                    return fbest, xbest, evals
                if spans[i] == 0.0:
                    continue
                delta = min(0.25 * spans_nz[i], max(1e-12, step_mul * spans_nz[i]))
                xp = xbest[:]
                xp[i] += delta
                repair(xp)
                fp = evaluate(xp); evals += 1
                if fp < fbest:
                    xbest, fbest = xp, fp
                    improved = True
                    continue
                xm = xbest[:]
                xm[i] -= delta
                repair(xm)
                fm = evaluate(xm); evals += 1
                if fm < fbest:
                    xbest, fbest = xm, fm
                    improved = True
            if not improved:
                step_mul *= 0.5
                if step_mul < 1e-12:
                    break
        return fbest, xbest, evals

    # ---- Main time loop: alternate global injections and local NM ----
    nm_restart = 0
    last_best = best
    no_improve_rounds = 0

    while now() < deadline:
        time_left = deadline - now()
        if time_left <= 0:
            break

        # choose a start point for local search
        if elites and random.random() < 0.85:
            # bias to best few
            top = min(len(elites), 10)
            idx = int((random.random() ** 2) * top)
            x0 = elites[idx][1][:]
            f0 = elites[idx][0]
        else:
            x0 = halton_point(hal_k) if random.random() < 0.6 else rand_uniform_point()
            hal_k += 1 if x0 is not None else 0
            f0 = evaluate(x0)
            push_elite(f0, x0)
            if f0 < best:
                best, best_x = f0, x0[:]

        # local budget based on remaining time (keep it safe)
        # assume objective evaluations dominate; keep fixed-ish budgets
        local_evals = max(40, min(500, 80 + 25 * int(math.sqrt(dim)) + 40 * nm_restart))
        # also scale down late if time is tiny
        if time_left < 0.15 * float(max_time):
            local_evals = max(30, local_evals // 2)

        # start scale schedule (restart-dependent)
        start_scale = 0.10 * (2.0 ** min(3, nm_restart))  # broader with restarts
        start_scale = min(0.6, max(1e-6, start_scale))

        # run NM
        if now() >= deadline:
            break
        f_loc, x_loc, used = nm_optimize(x0, f0, local_evals, start_scale)
        push_elite(f_loc, x_loc)

        if f_loc < best:
            best, best_x = f_loc, x_loc[:]

        # endgame polish sometimes
        if now() < deadline and (time_left < 0.35 * float(max_time) or random.random() < 0.25):
            f_pol, x_pol, _ = coord_polish(best_x, best, max_steps=3, base_step=0.05)
            push_elite(f_pol, x_pol)
            if f_pol < best:
                best, best_x = f_pol, x_pol[:]

        # progress / restart logic
        if best < last_best - 1e-12 * (1.0 + abs(last_best)):
            last_best = best
            no_improve_rounds = 0
            nm_restart = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= (2 + int(1.5 * math.sqrt(dim))):
                nm_restart += 1
                no_improve_rounds = 0

        # occasional global injection into elites (helps multimodal)
        if now() < deadline and random.random() < 0.30:
            xg = halton_point(hal_k) if random.random() < 0.75 else rand_uniform_point()
            hal_k += 1
            fg = evaluate(xg)
            push_elite(fg, xg)
            if fg < best:
                best, best_x = fg, xg[:]

    return best
