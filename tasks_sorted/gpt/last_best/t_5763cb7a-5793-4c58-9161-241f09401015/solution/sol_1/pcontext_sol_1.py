import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization (self-contained, no numpy).

    Key changes vs prior version:
    - Multi-start population: keep several elites instead of a single incumbent.
    - Nelder–Mead simplex local search on elites (strong local optimizer).
    - Stochastic global exploration via quasi-uniform (Halton) sampling + random sampling.
    - On-the-fly coordinate/pattern search polishing with adaptive steps.
    - Evaluation budget adapts to time; always returns best found within max_time.

    Returns:
        best (float): best (minimum) function value found.
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

    # ---------- utilities ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Halton sequence for better space-filling than plain random
    # (Deterministic low-discrepancy; implemented without external libs)
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

    halton_index = 1  # start from 1
    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = []
        for i in range(dim):
            u = van_der_corput(idx, primes[i])
            x.append(lows[i] + u * spans[i])
        return x

    # Wrap func to float; also robust to exceptions (treat as bad)
    def eval_point(x):
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # Keep K best points (elite set)
    def push_elite(elites, x, fx, K):
        # elites: list of (fx, x)
        if not math.isfinite(fx):
            return
        if len(elites) < K:
            elites.append((fx, x[:]))
            elites.sort(key=lambda t: t[0])
        else:
            if fx < elites[-1][0]:
                elites[-1] = (fx, x[:])
                elites.sort(key=lambda t: t[0])

    # ---------- initialization (global) ----------
    best = float("inf")
    best_x = None

    K = 6 if dim <= 8 else 4  # keep fewer elites in higher dims for speed
    elites = []

    # initial sample count: mix Halton + random
    # scaled modestly with dim; still time-safe.
    init_n = max(30, min(140, 30 + 8 * dim))

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        if k % 3 == 0:
            x = rand_point()
        else:
            x = halton_point()
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(elites, x, fx, K)

    if not elites:
        return best

    # ---------- local optimizers ----------
    # 1) Pattern/coordinate search (cheap polish; adaptive per-elite)
    def pattern_polish(x0, f0, max_evals, time_limit):
        x = x0[:]
        fx = f0
        step = [0.20 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]
        evals = 0
        noimp = 0

        while evals < max_evals and time.time() < time_limit:
            improved = False
            # random coord order
            coords = list(range(dim))
            random.shuffle(coords)
            for i in coords:
                if evals >= max_evals or time.time() >= time_limit:
                    break
                # try +, then -
                xi = x[:]
                xi[i] += step[i]
                clip_inplace(xi)
                f1 = eval_point(xi); evals += 1
                if f1 < fx:
                    x, fx = xi, f1
                    improved = True
                    continue
                if evals >= max_evals or time.time() >= time_limit:
                    break
                xi = x[:]
                xi[i] -= step[i]
                clip_inplace(xi)
                f2 = eval_point(xi); evals += 1
                if f2 < fx:
                    x, fx = xi, f2
                    improved = True

            if improved:
                noimp = 0
                for i in range(dim):
                    step[i] = min(0.35 * spans[i], step[i] * 1.25)
            else:
                noimp += 1
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.55)
                # if stuck, do a small random kick
                if noimp >= 3:
                    noimp = 0
                    j = random.randrange(dim)
                    xi = x[:]
                    xi[j] += (random.random() * 2.0 - 1.0) * (0.15 * spans[j])
                    clip_inplace(xi)
                    f3 = eval_point(xi); evals += 1
                    if f3 < fx:
                        x, fx = xi, f3
        return x, fx, evals

    # 2) Nelder–Mead (strong local search, no derivatives)
    def nelder_mead(x_start, f_start, max_evals, time_limit):
        # Build initial simplex around x_start
        n = dim
        simplex = [x_start[:]]
        fvals = [f_start]
        evals = 0

        # step sizes: relative to spans, but ensure not too tiny
        delta = [0.10 * spans[i] for i in range(n)]
        for i in range(n):
            xi = x_start[:]
            xi[i] += delta[i] if delta[i] > 0 else 1e-6
            clip_inplace(xi)
            fi = eval_point(xi); evals += 1
            simplex.append(xi)
            fvals.append(fi)
            if evals >= max_evals or time.time() >= time_limit:
                break

        if len(simplex) < 2:
            return x_start, f_start, evals

        # NM parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

        def order():
            nonlocal simplex, fvals
            idx = list(range(len(simplex)))
            idx.sort(key=lambda i: fvals[i])
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

        def centroid(exclude_last=True):
            m = len(simplex) - (1 if exclude_last else 0)
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

        while evals < max_evals and time.time() < time_limit:
            order()
            if fvals[0] < bestf:
                bestf = fvals[0]
                bestx = simplex[0][:]

            x0 = centroid(exclude_last=True)
            xh = simplex[-1]
            fh = fvals[-1]
            xl = simplex[0]
            fl = fvals[0]
            xg = simplex[-2]
            fg = fvals[-2]

            # Reflection
            xr = [x0[i] + alpha * (x0[i] - xh[i]) for i in range(n)]
            clip_inplace(xr)
            fr = eval_point(xr); evals += 1
            if not (time.time() < time_limit and evals < max_evals + 1):
                # allow loop to break naturally
                pass

            if fl <= fr < fg:
                simplex[-1] = xr
                fvals[-1] = fr
                continue

            if fr < fl:
                # Expansion
                xe = [x0[i] + gamma * (xr[i] - x0[i]) for i in range(n)]
                clip_inplace(xe)
                fe = eval_point(xe); evals += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                continue

            # Contraction
            if fr < fh:
                # outside contraction
                xc = [x0[i] + rho * (xr[i] - x0[i]) for i in range(n)]
            else:
                # inside contraction
                xc = [x0[i] + rho * (xh[i] - x0[i]) for i in range(n)]
            clip_inplace(xc)
            fc = eval_point(xc); evals += 1
            if fc < min(fh, fr):
                simplex[-1] = xc
                fvals[-1] = fc
                continue

            # Shrink towards best
            xb = simplex[0][:]
            for k in range(1, len(simplex)):
                xs = [xb[i] + sigma * (simplex[k][i] - xb[i]) for i in range(n)]
                clip_inplace(xs)
                fs = eval_point(xs); evals += 1
                simplex[k] = xs
                fvals[k] = fs
                if evals >= max_evals or time.time() >= time_limit:
                    break

        # return best in simplex
        order()
        if fvals[0] < bestf:
            return simplex[0], fvals[0], evals
        return bestx, bestf, evals

    # ---------- main loop ----------
    # Allocate time between global exploration and local refinement
    while time.time() < deadline:
        now = time.time()
        remaining = deadline - now
        if remaining <= 0:
            break

        # 1) Global exploration burst to refresh elites
        # Use more exploration when dim is high.
        burst = 12 if dim <= 6 else 18
        for _ in range(burst):
            if time.time() >= deadline:
                return best
            # biased sampling: mostly Halton, sometimes random, sometimes around elite
            r = random.random()
            if r < 0.55:
                x = halton_point()
            elif r < 0.80:
                x = rand_point()
            else:
                # sample around a random elite with gaussian-like perturbation (via sum of uniforms)
                fx_e, xe = elites[random.randrange(len(elites))]
                x = xe[:]
                for i in range(dim):
                    # approx normal(0,1): sum of 12 U - 6
                    z = sum(random.random() for _ in range(12)) - 6.0
                    x[i] += z * (0.08 * spans[i])
                clip_inplace(x)

            fx = eval_point(x)
            if fx < best:
                best, best_x = fx, x[:]
            push_elite(elites, x, fx, K)

        if not elites:
            continue

        # 2) Local refinement on the best elite (and occasionally second best)
        elites.sort(key=lambda t: t[0])
        targets = [0]
        if len(elites) > 1 and random.random() < 0.35:
            targets.append(1)

        for ti in targets:
            if time.time() >= deadline:
                return best
            fx0, x0 = elites[ti]

            # Small per-call time slice so we don't get stuck in one local search
            local_deadline = min(deadline, time.time() + max(0.01, 0.20 * remaining))

            # First do a short pattern polish (cheap)
            x1, f1, _ = pattern_polish(x0, fx0, max_evals=25 + 6 * dim, time_limit=local_deadline)
            if f1 < best:
                best, best_x = f1, x1[:]
            push_elite(elites, x1, f1, K)

            if time.time() >= deadline:
                return best

            # Then do Nelder-Mead if we have time (more powerful)
            local_deadline2 = min(deadline, time.time() + max(0.01, 0.35 * (deadline - time.time())))
            x2, f2, _ = nelder_mead(x1, f1, max_evals=40 + 12 * dim, time_limit=local_deadline2)
            if f2 < best:
                best, best_x = f2, x2[:]
            push_elite(elites, x2, f2, K)

    return best
