import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Key upgrades vs previous hybrid:
      1) Better global coverage: scrambled Halton + opposition points + occasional random injections.
      2) Stronger local progress: compact CMA-ES-like update (rank-mu, evolution paths, diagonal covariance).
      3) Cheap local refinement: adaptive coordinate/pattern search with step halving.
      4) Multi-start handling: maintain a small elite set and restart around elites when stagnating.

    Returns:
      best (float): best objective value found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    nonzero = [spans[i] > 0.0 for i in range(dim)]

    # ---- helpers ----
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
        # opposition-based point in box: x' = lo+hi-x
        xo = [0.0] * dim
        for i in range(dim):
            xo[i] = lows[i] + highs[i] - x[i]
        return xo

    # ---- scrambled Halton for global sampling ----
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

    # ---- Elite set (small) for multi-start ----
    elite_max = 6 if dim <= 30 else 4
    elites = []  # list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        # keep unique-ish by simple distance check on normalized space
        def norm_dist(a, b):
            s = 0.0
            for i in range(dim):
                sp = spans[i]
                if sp > 0:
                    d = (a[i] - b[i]) / sp
                    s += d * d
            return s

        for (fe, xe) in elites:
            if abs(fe - fx) < 1e-12 and norm_dist(xe, x) < 1e-6:
                return
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_max:
            elites = elites[:elite_max]

    # ---- Initial sampling: halton + random + opposition ----
    x_best = rand_point()
    f_best = eval_f(x_best)
    push_elite(f_best, x_best)

    # Budget warmup depending on time
    warm = int(max(30, min(900, 40 + 35 * dim)))
    if max_time < 0.2:
        warm = min(warm, 40)

    for k in range(warm):
        if time.time() >= deadline:
            return f_best

        if k % 3 == 0:
            x = halton_point()
        else:
            x = rand_point()

        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]
        push_elite(fx, x)

        # opposition evaluation sometimes (often improves early)
        if k % 4 == 0 and time.time() < deadline:
            xo = opposite_point(x)
            clip_inplace(xo)
            fo = eval_f(xo)
            if fo < f_best:
                f_best, x_best = fo, xo[:]
            push_elite(fo, xo)

    # ---- Local search: adaptive coordinate/pattern ----
    def pattern_refine(x0, f0, base_step, eval_budget):
        x = x0[:]
        fx = f0
        # prioritize larger spans
        order = list(range(dim))
        order.sort(key=lambda i: spans[i], reverse=True)

        # per-dim step
        step = [0.0] * dim
        for i in range(dim):
            step[i] = (base_step * spans[i]) if nonzero[i] else 0.0

        evals = 0
        while evals < eval_budget and time.time() < deadline:
            improved = False
            for i in order:
                if not nonzero[i]:
                    continue
                si = step[i]
                if si <= 0.0:
                    continue

                best_local = fx
                best_x = None

                # try +/- step (and a slightly larger jump if close to boundary)
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
                    if evals >= eval_budget or time.time() >= deadline:
                        break

                if best_x is not None:
                    x, fx = best_x, best_local
                    improved = True

                if evals >= eval_budget or time.time() >= deadline:
                    break

            if not improved:
                # reduce step sizes
                shrink = 0.5
                any_nontrivial = False
                for i in range(dim):
                    step[i] *= shrink
                    if step[i] > 1e-12 * (spans[i] if spans[i] > 0 else 1.0):
                        any_nontrivial = True
                if not any_nontrivial:
                    break
        return fx, x

    # ---- Diagonal CMA-ES-like optimizer state ----
    # Center at best
    m = x_best[:]
    f_m = f_best

    # Strategy parameters
    lam = int(max(10, min(80, 10 + 4 * dim)))
    mu = max(3, lam // 2)

    # log weights
    weights = [0.0] * mu
    ws = 0.0
    for i in range(mu):
        w = math.log(mu + 0.5) - math.log(i + 1.0)
        weights[i] = w
        ws += w
    weights = [w / ws for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # diag "covariance" as std dev per dimension
    sigma0 = 0.22
    sigma = max(1e-12, sigma0)
    diag = [1.0] * dim  # scaling per coordinate

    # time-safe min/max scaling
    diag_min = 1e-12
    diag_max = 1e12

    # evolution paths (diagonal form)
    pc = [0.0] * dim
    ps = [0.0] * dim

    # learning rates (standard-ish diagonal CMA)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    # expected norm of N(0, I)
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # precompute which dims are variable
    var_idx = [i for i in range(dim) if nonzero[i]]
    if not var_idx:
        return f_best

    # stagnation / restart control
    last_improve_t = time.time()
    best_seen = f_best
    restart_count = 0

    # ---- main loop ----
    while time.time() < deadline:
        # occasional local refine of current best (cheap, strong)
        if time.time() < deadline:
            fb2, xb2 = pattern_refine(x_best, f_best, base_step=0.03, eval_budget=2 * dim + 12)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]
                push_elite(f_best, x_best)
                m = x_best[:]
                f_m = f_best
                best_seen = f_best
                last_improve_t = time.time()

        # sample offspring
        pop = []
        for _ in range(lam):
            if time.time() >= deadline:
                return f_best

            # sample z ~ N(0, I), y = diag_scale * z
            z = [0.0] * dim
            y = [0.0] * dim
            x = m[:]

            # occasional heavy-tail jump to escape local traps
            heavy = (random.random() < 0.08)

            for i in var_idx:
                zi = random.gauss(0.0, 1.0)
                if heavy:
                    # mixture: add Cauchy-ish component
                    u = random.random()
                    zi += 0.35 * math.tan(math.pi * (u - 0.5))
                z[i] = zi
                yi = (diag[i] * zi)
                y[i] = yi
                x[i] = x[i] + sigma * yi

            clip_inplace(x)
            fx = eval_f(x)
            pop.append((fx, x, z, y))

            if fx < f_best:
                f_best, x_best = fx, x[:]
                push_elite(f_best, x_best)
                best_seen = f_best
                last_improve_t = time.time()

        pop.sort(key=lambda t: t[0])

        # recombination: new mean
        m_old = m[:]
        m = m[:]  # overwrite
        for i in range(dim):
            if nonzero[i]:
                s = 0.0
                for k in range(mu):
                    s += weights[k] * pop[k][1][i]
                m[i] = s
            else:
                m[i] = lows[i]
        clip_inplace(m)

        # compute y_w and z_w (weighted)
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

        # update evolution path for sigma (ps)
        for i in var_idx:
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * z_w[i]

        # sigma update using ||ps||
        norm_ps = 0.0
        for i in var_idx:
            norm_ps += ps[i] * ps[i]
        norm_ps = math.sqrt(norm_ps)

        sigma *= math.exp((cs / damps) * (norm_ps / chiN - 1.0))

        # keep sigma in a sensible range relative to box
        # (avoid collapse too early; also avoid enormous steps)
        span_mean = 0.0
        cnt = 0
        for i in var_idx:
            span_mean += spans[i]
            cnt += 1
        span_mean = span_mean / max(1, cnt)
        sigma = max(1e-14, min(sigma, 1.0 * span_mean))

        # update evolution path for covariance (pc), with diagonal approximation
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0)) / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0
        for i in var_idx:
            pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

        # diagonal covariance update (diag holds std-like multipliers)
        # We update diag^2 implicitly via diag, keeping it positive.
        for i in var_idx:
            # rank-one + rank-mu update on variance
            # v <- (1-c1-cmu)*v + c1*pc^2 + cmu*sum(w_k*y_k^2)
            v = diag[i] * diag[i]
            rank_one = pc[i] * pc[i]
            rank_mu = 0.0
            for k in range(mu):
                yi = pop[k][3][i]
                rank_mu += weights[k] * (yi * yi)
            v = (1.0 - c1 - cmu) * v + c1 * rank_one + cmu * rank_mu
            if v < diag_min:
                v = diag_min
            elif v > diag_max:
                v = diag_max
            diag[i] = math.sqrt(v)

        # ensure best tracking
        if f_m < f_best:
            f_best, x_best = f_m, m_old[:]
            push_elite(f_best, x_best)

        # Stagnation restart: if no improvement for a while, restart around an elite/global point
        if time.time() < deadline:
            no_improve_for = time.time() - last_improve_t
            # adapt threshold to time budget; small budgets restart sooner
            thresh = 0.35 if max_time <= 2.0 else 0.9
            if no_improve_for > thresh:
                restart_count += 1

                # choose a restart center: best elite with some probability, else halton/random
                if elites and random.random() < 0.7:
                    base = elites[random.randrange(min(len(elites), elite_max))][1][:]
                else:
                    base = halton_point() if random.random() < 0.6 else rand_point()

                # perturb base
                m = base[:]
                for i in var_idx:
                    m[i] += random.gauss(0.0, 0.12 * spans[i])
                clip_inplace(m)
                f_m = eval_f(m)
                if f_m < f_best:
                    f_best, x_best = f_m, m[:]
                    push_elite(f_best, x_best)
                    best_seen = f_best
                last_improve_t = time.time()

                # reset strategy a bit (wider after many restarts)
                sigma = min(0.35 * span_mean, (0.18 + 0.03 * restart_count) * span_mean)
                for i in var_idx:
                    diag[i] = 1.0
                    pc[i] = 0.0
                    ps[i] = 0.0

        # occasional global injection evaluations (very cheap)
        if time.time() < deadline and random.random() < 0.12:
            xg = halton_point() if random.random() < 0.7 else rand_point()
            fg = eval_f(xg)
            if fg < f_best:
                f_best, x_best = fg, xg[:]
                push_elite(f_best, x_best)
                m = x_best[:]
                last_improve_t = time.time()

    return f_best
