import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Main changes vs prior hybrids:
      1) Better global exploration early: scrambled Halton + (optional) opposition points.
      2) Stronger local optimizer: compact CMA-ES (diagonal covariance) with boundary-aware handling.
      3) Two-stage local refinement: (a) best-1D quadratic line search (very cheap), then
         (b) pattern/coordinate search with step shrinking.
      4) Robust restarts: IPOP-like population growth + elite archive + stagnation triggers.

    Returns:
      best (float): best objective value found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    nonzero = [spans[i] > 0.0 for i in range(dim)]
    var_idx = [i for i in range(dim) if nonzero[i]]
    if not var_idx:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    def now():
        return time.time()

    def clip_inplace(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def eval_f(x):
        return float(func(x))

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if nonzero[i]:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for i in range(dim):
            xo[i] = lows[i] + highs[i] - x[i]
        return xo

    # ---------- scrambled Halton ----------
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite archive ----------
    elite_max = 8 if dim <= 30 else 6
    elites = []  # list of (f, x)

    def norm_dist2(a, b):
        s = 0.0
        for i in var_idx:
            sp = spans[i]
            if sp > 0:
                d = (a[i] - b[i]) / sp
                s += d * d
        return s

    def push_elite(fx, x):
        nonlocal elites
        # avoid near-duplicates
        for fe, xe in elites:
            if abs(fe - fx) < 1e-14 and norm_dist2(xe, x) < 1e-10:
                return
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_max:
            elites = elites[:elite_max]

    # ---------- 1D quadratic line search along coordinate ----------
    # Uses 3 samples to fit a parabola; falls back safely if degenerate.
    def quad_min_1d(fa, fb, fc, a, b, c):
        # points (a,fa),(b,fb),(c,fc)
        # return argmin within [min(a,c), max(a,c)] if stable, else b
        denom = (a - b) * (a - c) * (b - c)
        if denom == 0.0:
            return b
        # vertex formula for interpolating parabola
        A = (a * (fc - fb) + b * (fa - fc) + c * (fb - fa)) / denom
        B = (a * a * (fb - fc) + b * b * (fc - fa) + c * c * (fa - fb)) / denom
        if A <= 0.0:
            return b
        x = -B / (2.0 * A)
        lo = min(a, c)
        hi = max(a, c)
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        return x

    def best_coord_quadratic_refine(x0, f0, trials):
        # Try a few promising coordinates; for each, do a small bracket and quadratic step.
        x = x0[:]
        fx = f0

        # prioritize by span
        order = var_idx[:]
        order.sort(key=lambda i: spans[i], reverse=True)

        # limit which coords to try
        m = min(len(order), max(2, int(math.sqrt(len(order)) + 2)))
        order = order[:m]

        for _ in range(trials):
            if now() >= deadline:
                break
            i = order[random.randrange(m)]
            sp = spans[i]
            if sp <= 0:
                continue

            # bracket size relative to span; smaller if close to boundary
            base = 0.06 * sp
            # clamp bracket to remain in bounds
            xi = x[i]
            a = max(lows[i], xi - base)
            c = min(highs[i], xi + base)
            b = xi

            # ensure distinct points
            if c - a < 1e-16 * (sp + 1.0):
                continue
            if abs(b - a) < 1e-16:
                b = a + 0.5 * (c - a)

            xa = x[:]
            xb = x[:]
            xc = x[:]
            xa[i] = a
            xb[i] = b
            xc[i] = c

            fa = eval_f(xa)
            if now() >= deadline:
                break
            fb = eval_f(xb)
            if now() >= deadline:
                break
            fc = eval_f(xc)
            if now() >= deadline:
                break

            # pick best among a,b,c
            if fa < fx:
                fx, x = fa, xa
            if fb < fx:
                fx, x = fb, xb
            if fc < fx:
                fx, x = fc, xc

            # quadratic proposal
            xstar = quad_min_1d(fa, fb, fc, a, b, c)
            xs = x[:]
            xs[i] = xstar
            fs = eval_f(xs)
            if fs < fx:
                fx, x = fs, xs
        return fx, x

    # ---------- pattern refine ----------
    def pattern_refine(x0, f0, base_step, eval_budget):
        x = x0[:]
        fx = f0

        order = var_idx[:]
        order.sort(key=lambda i: spans[i], reverse=True)

        step = [0.0] * dim
        for i in var_idx:
            step[i] = base_step * spans[i]

        evals = 0
        while evals < eval_budget and now() < deadline:
            improved = False
            for i in order:
                si = step[i]
                if si <= 0.0:
                    continue

                best_local = fx
                best_x = None

                # try +/- si
                for d in (-1.0, 1.0):
                    cand = x[:]
                    cand[i] += d * si
                    if cand[i] < lows[i]:
                        cand[i] = lows[i]
                    elif cand[i] > highs[i]:
                        cand[i] = highs[i]
                    fc = eval_f(cand)
                    evals += 1
                    if fc < best_local:
                        best_local = fc
                        best_x = cand
                    if evals >= eval_budget or now() >= deadline:
                        break

                if best_x is not None:
                    x, fx = best_x, best_local
                    improved = True

                if evals >= eval_budget or now() >= deadline:
                    break

            if not improved:
                # shrink
                any_nontrivial = False
                for i in var_idx:
                    step[i] *= 0.5
                    if step[i] > 1e-14 * (spans[i] + 1.0):
                        any_nontrivial = True
                if not any_nontrivial:
                    break

        return fx, x

    # ---------- initialize ----------
    x_best = rand_point()
    f_best = eval_f(x_best)
    push_elite(f_best, x_best)

    # quick global warmup
    warm = int(max(50, min(1200, 60 + 40 * dim)))
    if max_time <= 0.3:
        warm = min(warm, 50)

    for k in range(warm):
        if now() >= deadline:
            return f_best
        x = halton_point() if (k % 4 != 1) else rand_point()
        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]
        push_elite(fx, x)

        # occasional opposition (cheap global boost)
        if (k % 7 == 0) and now() < deadline:
            xo = opposite_point(x)
            clip_inplace(xo)
            fo = eval_f(xo)
            if fo < f_best:
                f_best, x_best = fo, xo[:]
            push_elite(fo, xo)

    # ---------- diagonal CMA-ES with restarts (IPOP-ish) ----------
    # strategy parameters
    lam0 = int(max(12, min(90, 12 + 4 * dim)))
    lam = lam0
    mu = max(3, lam // 2)

    def make_weights(mu):
        ws = 0.0
        w = [0.0] * mu
        for i in range(mu):
            wi = math.log(mu + 0.5) - math.log(i + 1.0)
            w[i] = wi
            ws += wi
        inv = 1.0 / ws
        for i in range(mu):
            w[i] *= inv
        mueff = 1.0 / sum(wi * wi for wi in w)
        return w, mueff

    weights, mueff = make_weights(mu)

    # CMA params
    cs = (mueff + 2.0) / (len(var_idx) + mueff + 5.0)
    cc = (4.0 + mueff / len(var_idx)) / (len(var_idx) + 4.0 + 2.0 * mueff / len(var_idx))
    c1 = 2.0 / ((len(var_idx) + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((len(var_idx) + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (len(var_idx) + 1.0)) - 1.0) + cs
    chiN = math.sqrt(len(var_idx)) * (1.0 - 1.0 / (4.0 * len(var_idx)) + 1.0 / (21.0 * len(var_idx) * len(var_idx)))

    # state
    m = x_best[:]
    sigma = 0.22 * (sum(spans[i] for i in var_idx) / max(1, len(var_idx)))
    sigma = max(1e-14, sigma)

    diag = [1.0] * dim
    pc = [0.0] * dim
    ps = [0.0] * dim

    last_improve = now()
    restart_count = 0
    best_at_restart = f_best

    def restart(center=None):
        nonlocal m, sigma, diag, pc, ps, lam, mu, weights, mueff
        nonlocal cs, cc, c1, cmu, damps, chiN, restart_count, best_at_restart

        restart_count += 1
        if center is None:
            if elites and random.random() < 0.75:
                center = elites[0][1][:]
            else:
                center = halton_point() if random.random() < 0.6 else rand_point()

        m = center[:]
        clip_inplace(m)

        # IPOP: increase population sometimes
        if restart_count % 2 == 0:
            lam = min(200, int(lam * 1.6))
        else:
            lam = max(lam0, int(lam * 1.1))
        mu = max(3, lam // 2)
        weights, mueff = make_weights(mu)

        d = len(var_idx)
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
        c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        chiN = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

        # reset covariance and paths
        for i in var_idx:
            diag[i] = 1.0
            pc[i] = 0.0
            ps[i] = 0.0

        # reset sigma moderately wide
        span_mean = sum(spans[i] for i in var_idx) / max(1, len(var_idx))
        sigma = max(1e-14, (0.30 if restart_count <= 2 else 0.38) * span_mean)

        best_at_restart = f_best

    # ---------- main loop ----------
    while now() < deadline:
        # local refinements around current best (cheap + high impact)
        if now() < deadline:
            fb2, xb2 = best_coord_quadratic_refine(x_best, f_best, trials=2)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]
                push_elite(f_best, x_best)
                m = x_best[:]
                last_improve = now()

        if now() < deadline:
            fb3, xb3 = pattern_refine(x_best, f_best, base_step=0.025, eval_budget=2 * len(var_idx) + 10)
            if fb3 < f_best:
                f_best, x_best = fb3, xb3[:]
                push_elite(f_best, x_best)
                m = x_best[:]
                last_improve = now()

        # CMA generation
        pop = []
        for _ in range(lam):
            if now() >= deadline:
                return f_best

            z = [0.0] * dim
            y = [0.0] * dim
            x = m[:]

            heavy = (random.random() < 0.10)
            for i in var_idx:
                zi = random.gauss(0.0, 1.0)
                if heavy:
                    u = random.random()
                    zi += 0.45 * math.tan(math.pi * (u - 0.5))
                z[i] = zi
                yi = diag[i] * zi
                y[i] = yi
                x[i] = x[i] + sigma * yi

            clip_inplace(x)
            fx = eval_f(x)
            pop.append((fx, x, z, y))

            if fx < f_best:
                f_best, x_best = fx, x[:]
                push_elite(f_best, x_best)
                last_improve = now()

        pop.sort(key=lambda t: t[0])

        # recombination (new mean)
        m_old = m[:]
        for i in var_idx:
            s = 0.0
            for k in range(mu):
                s += weights[k] * pop[k][1][i]
            m[i] = s
        clip_inplace(m)

        # weighted z and y
        z_w = [0.0] * dim
        y_w = [0.0] * dim
        for i in var_idx:
            sz = 0.0
            sy = 0.0
            for k in range(mu):
                sz += weights[k] * pop[k][2][i]
                sy += weights[k] * pop[k][3][i]
            z_w[i] = sz
            y_w[i] = sy

        # update ps and sigma
        for i in var_idx:
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * z_w[i]

        norm_ps = math.sqrt(sum(ps[i] * ps[i] for i in var_idx))
        sigma *= math.exp((cs / damps) * (norm_ps / chiN - 1.0))

        # clamp sigma to box scale
        span_mean = sum(spans[i] for i in var_idx) / max(1, len(var_idx))
        sigma = max(1e-16, min(sigma, 0.9 * span_mean))

        # update pc
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** 2) / chiN) < (1.4 + 2.0 / (len(var_idx) + 1.0)) else 0.0
        for i in var_idx:
            pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

        # diagonal covariance update
        for i in var_idx:
            v = diag[i] * diag[i]
            rank_one = pc[i] * pc[i]
            rank_mu = 0.0
            for k in range(mu):
                yi = pop[k][3][i]
                rank_mu += weights[k] * (yi * yi)
            v = (1.0 - c1 - cmu) * v + c1 * rank_one + cmu * rank_mu
            if v < 1e-24:
                v = 1e-24
            elif v > 1e24:
                v = 1e24
            diag[i] = math.sqrt(v)

        # track elites from top few each gen (helps with restarts)
        for k in range(min(3, len(pop))):
            push_elite(pop[k][0], pop[k][1])

        # stagnation / restart logic
        if now() < deadline:
            no_improve = now() - last_improve

            # Also restart if improvement since last restart is tiny
            tiny_progress = (best_at_restart - f_best) < (1e-12 * (abs(best_at_restart) + 1.0))

            # time-adaptive thresholds
            thr = 0.25 if max_time <= 2.0 else (0.7 if max_time <= 10.0 else 1.2)

            if no_improve > thr or (tiny_progress and no_improve > 0.5 * thr):
                # restart center: best elite, else global point
                if elites and random.random() < 0.85:
                    base = elites[0][1][:]
                else:
                    base = halton_point() if random.random() < 0.7 else rand_point()

                # perturb base a bit to avoid exact repeats
                for i in var_idx:
                    base[i] += random.gauss(0.0, 0.06 * spans[i])
                clip_inplace(base)

                # accept base as new candidate
                fb = eval_f(base)
                if fb < f_best:
                    f_best, x_best = fb, base[:]
                    push_elite(f_best, x_best)

                restart(center=base)
                last_improve = now()

        # occasional global injection
        if now() < deadline and random.random() < 0.10:
            xg = halton_point() if random.random() < 0.7 else rand_point()
            fg = eval_f(xg)
            if fg < f_best:
                f_best, x_best = fg, xg[:]
                push_elite(f_best, x_best)
                m = x_best[:]
                last_improve = now()

    return f_best
