import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvement over your best (#2, ES+SA+pattern+restarts):
      - Adds *true* local optimization: a bounded, gradient-free L-BFGS-like method
        using finite-difference gradients + backtracking line-search (cheap, strong exploitation).
      - Keeps global exploration: Halton + stratified init + (mu,lambda)-ES with heavy tails + restarts.
      - Adds a small "tabu/too-close" filter to avoid wasting evaluations.
      - Better schedule: global search early, then increasingly frequent local L-BFGS bursts.

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # --------- basic guards ----------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0

    avg_span = sum(spans) / float(dim)
    # diagonal in *normalized* space is sqrt(dim); use spans for scaling elsewhere
    inv_spans = [1.0 / s for s in spans]

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def clip_vec(x):
        return [clamp(x[i], i) for i in range(dim)]

    def eval_x(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    # --------- RNG helpers ----------
    def randn():
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = random.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # --------- Halton + stratified init ----------
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

    primes = first_primes(dim)

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        n = index
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(k):
        x = []
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x.append(lows[i] + u * (highs[i] - lows[i]))
        return x

    def stratified_points(n):
        strata = []
        for i in range(dim):
            vals = [((j + random.random()) / n) for j in range(n)]
            random.shuffle(vals)
            strata.append([lows[i] + v * (highs[i] - lows[i]) for v in vals])
        pts = []
        for j in range(n):
            pts.append([strata[i][j] for i in range(dim)])
        return pts

    # --------- distance + "too close" rejection ----------
    def dist2_norm(a, b):
        # normalized squared distance
        s = 0.0
        for i in range(dim):
            d = (a[i] - b[i]) * inv_spans[i]
            s += d * d
        return s

    recent = []
    recent_cap = 120
    close_thr2 = (2e-5) ** 2  # tighter than before; avoids exact duplicates and tiny perturbations

    def accept_new_point(x):
        for y in recent:
            if dist2_norm(x, y) < close_thr2:
                return False
        recent.append(x[:])
        if len(recent) > recent_cap:
            del recent[0:len(recent) - recent_cap]
        return True

    # --------- pattern search (cheap deterministic intensification) ----------
    def pattern_search(x0, f0, base_scale):
        x = x0[:]
        fx = f0
        step = [base_scale * spans[i] for i in range(dim)]
        for _round in range(2):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - eps_time:
                    return fx, x
                si = step[i]
                xi = x[i]

                cand = x[:]
                cand[i] = clamp(xi + si, i)
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        continue

                cand = x[:]
                cand[i] = clamp(xi - si, i)
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True

            step = [v * (0.55 if improved else 0.35) for v in step]
            if not improved:
                break
        return fx, x

    # --------- finite-difference gradient + limited-memory BFGS local solver ----------
    def fd_grad(x, fx=None):
        # central differences in each coordinate; scaled step by span and magnitude
        if fx is None:
            fx = eval_x(x)
        g = [0.0] * dim

        # step in absolute space
        # larger than machine eps, smaller than spans: tuned for black-box objectives
        for i in range(dim):
            if time.time() >= deadline - eps_time:
                return None, fx
            h = 1e-6 * spans[i] + 1e-12
            # make sure h is not too small relative to x
            h *= (1.0 + 0.25 * abs((x[i] - lows[i]) * inv_spans[i] - 0.5))

            xp = x[:]
            xm = x[:]
            xp[i] = clamp(x[i] + h, i)
            xm[i] = clamp(x[i] - h, i)

            # if clamp collapses, fallback to one-sided
            if xp[i] == xm[i]:
                # variable effectively pinned
                g[i] = 0.0
                continue

            # Evaluate
            fp = eval_x(xp)
            if time.time() >= deadline - eps_time:
                return None, fx
            fm = eval_x(xm)

            denom = (xp[i] - xm[i])
            if denom != 0.0:
                g[i] = (fp - fm) / denom
            else:
                g[i] = 0.0
        return g, fx

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(dim))

    def norm2(a):
        return math.sqrt(dot(a, a))

    def axpy(a, x, y):
        # a*x + y
        return [a * x[i] + y[i] for i in range(dim)]

    def vec_sub(a, b):
        return [a[i] - b[i] for i in range(dim)]

    def lbfgs_local(x_start, f_start, m_hist=6, max_iter=40):
        """
        Very small L-BFGS with backtracking Armijo line-search.
        Bounded by clamping after steps.
        Uses FD gradients -> expensive; called sparingly.
        """
        x = x_start[:]
        fx = f_start

        g, fx = fd_grad(x, fx)
        if g is None:
            return fx, x
        gnorm = norm2(g)
        if not (gnorm > 0.0):
            return fx, x

        S = []  # s vectors
        Y = []  # y vectors
        RHO = []

        for _ in range(max_iter):
            if time.time() >= deadline - eps_time:
                break

            gnorm = norm2(g)
            # stop if gradient is tiny in normalized terms
            if gnorm <= 1e-10 * (1.0 + abs(fx)):
                break

            # two-loop recursion to get direction p = -H*g
            q = g[:]
            alpha = [0.0] * len(S)
            for k in range(len(S) - 1, -1, -1):
                a = RHO[k] * dot(S[k], q)
                alpha[k] = a
                q = vec_sub(q, [a * Y[k][i] for i in range(dim)])

            # initial H0 scaling
            if len(S) > 0:
                ys = dot(Y[-1], S[-1])
                yy = dot(Y[-1], Y[-1])
                gamma = ys / yy if yy > 1e-30 else 1.0
            else:
                gamma = 1.0

            r = [gamma * q[i] for i in range(dim)]

            for k in range(len(S)):
                b = RHO[k] * dot(Y[k], r)
                r = axpy(alpha[k] - b, S[k], r)

            p = [-r[i] for i in range(dim)]
            # if p is not a descent direction, fall back to steepest descent
            if dot(p, g) >= 0.0:
                p = [-g[i] for i in range(dim)]

            # backtracking line-search (Armijo)
            gp = dot(g, p)
            if gp >= 0.0:
                break

            # initial step: proportional to span
            step0 = 1.0
            # clamp-aware: if p is huge, scale down
            pscale = max(1e-18, max(abs(p[i]) / spans[i] for i in range(dim)))
            if pscale > 10.0:
                step0 = 1.0 / pscale

            c1 = 1e-4
            t = step0
            ok = False
            best_ls_fx = fx
            best_ls_x = x[:]

            # try a few step sizes
            for _ls in range(14):
                if time.time() >= deadline - eps_time:
                    break
                xn = [clamp(x[i] + t * p[i], i) for i in range(dim)]
                if not accept_new_point(xn):
                    t *= 0.5
                    continue
                fn = eval_x(xn)
                if fn < best_ls_fx:
                    best_ls_fx = fn
                    best_ls_x = xn[:]
                if fn <= fx + c1 * t * gp:
                    ok = True
                    x_new = xn
                    f_new = fn
                    break
                t *= 0.5

            if not ok:
                # accept best seen in line search if it improved at all
                if best_ls_fx < fx:
                    x_new = best_ls_x
                    f_new = best_ls_fx
                else:
                    break

            g_new, _ = fd_grad(x_new, f_new)
            if g_new is None:
                x, fx = x_new, f_new
                break

            s = vec_sub(x_new, x)
            y = vec_sub(g_new, g)
            ys = dot(y, s)

            x, fx, g = x_new, f_new, g_new

            if ys > 1e-18:
                rho = 1.0 / ys
                S.append(s)
                Y.append(y)
                RHO.append(rho)
                if len(S) > m_hist:
                    S.pop(0); Y.pop(0); RHO.pop(0)

        return fx, x

    # --------- initialization ----------
    best = float("inf")
    best_x = None

    n_lhs = max(18, min(90, 9 * dim))
    n_hal = max(18, min(160, 16 * dim))
    n_rnd = max(10, min(80, 7 * dim))

    init_pts = []
    init_pts.extend(stratified_points(n_lhs))
    init_pts.extend(halton_point(k) for k in range(1, n_hal + 1))
    init_pts.extend(rand_point() for _ in range(n_rnd))
    for _ in range(min(2 * dim, 24)):
        init_pts.append([highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)])

    # elites
    mu = max(6, min(26, 3 + dim // 2))
    elites = []  # list of (f,x)

    def push_elite(fx, x):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > mu:
            elites = elites[:mu]

    for x in init_pts:
        if time.time() >= deadline - eps_time:
            return best
        if not accept_new_point(x):
            continue
        fx = eval_x(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(fx, x)

    if best_x is None:
        return best

    # --------- ES/SA parameters ----------
    lam = max(16, min(100, 8 + 4 * dim))
    sigma = 0.22 * avg_span
    sigma_min = 1e-14 * avg_span + 1e-18
    sigma_max = 1.8 * avg_span

    # SA temperature
    T = 1.0
    T_min = 1e-9
    cool = 0.995

    stall = 0
    restart_after = max(90, 18 * dim)
    last_best = best

    # local-search scheduling
    next_local_time = t0 + 0.18  # start a bit after init
    local_period = 0.22          # will shrink as time passes

    # --------- main loop ----------
    while time.time() < deadline - eps_time:
        now = time.time()
        frac = (now - t0) / max(1e-9, max_time)  # 0..1

        # increasingly exploit later: more frequent local search
        local_period = max(0.08, 0.22 * (1.0 - 0.6 * frac))

        elites.sort(key=lambda t: t[0])
        parents = [x for (_, x) in elites[:max(3, min(mu, len(elites)))]]
        parents.append(best_x[:])

        # --- ES batch ---
        for _ in range(lam):
            if time.time() >= deadline - eps_time:
                break
            p = parents[random.randrange(len(parents))]
            child = p[:]

            s = sigma * math.exp(0.25 * randn())
            if s < sigma_min:
                s = sigma_min
            elif s > sigma_max:
                s = sigma_max

            heavy = (random.random() < 0.18)
            for i in range(dim):
                z = cauchy() if heavy else randn()
                child[i] = clamp(child[i] + z * s, i)

            if not accept_new_point(child):
                continue
            fchild = eval_x(child)
            push_elite(fchild, child)

            if fchild < best:
                best, best_x = fchild, child[:]
                stall = 0

        # --- SA move around best ---
        if time.time() < deadline - eps_time:
            x = best_x[:]
            step_scale = max(sigma_min, min(sigma_max, 0.75 * sigma))
            for i in range(dim):
                z = cauchy() if random.random() < 0.35 else randn()
                x[i] = clamp(x[i] + z * step_scale, i)
            if accept_new_point(x):
                fx = eval_x(x)
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x[:]
                    stall = 0
                else:
                    df = fx - best
                    if T > T_min and df < 700.0 * T and random.random() < math.exp(-df / max(T, T_min)):
                        # accepted move only affects elite pool already
                        pass

        # --- periodic pattern + L-BFGS local burst ---
        if now >= next_local_time and time.time() < deadline - eps_time:
            # quick pattern first (often gets "free" improvements)
            f2, x2 = pattern_search(best_x, best, base_scale=0.06)
            if f2 < best:
                best, best_x = f2, x2[:]
                push_elite(best, best_x)
                stall = 0
                last_best = best

            # then L-BFGS local optimize from best (strong exploitation)
            # keep iterations modest to control FD cost
            # do more iterations later in time budget
            iters = 10 if frac < 0.5 else 16
            f3, x3 = lbfgs_local(best_x, best, m_hist=6, max_iter=iters)
            if f3 < best:
                best, best_x = f3, x3[:]
                push_elite(best, best_x)
                stall = 0
                last_best = best
                sigma = max(sigma_min, sigma * 0.85)

            next_local_time = now + local_period

        # --- adapt sigma / stall ---
        if best < last_best:
            sigma = max(sigma_min, sigma * 0.92)
            last_best = best
        else:
            stall += 1
            if stall % max(10, 2 * dim) == 0:
                sigma = min(sigma_max, sigma * 1.08)

        T = max(T_min, T * cool)

        # --- restart on stagnation ---
        if stall >= restart_after and time.time() < deadline - eps_time:
            stall = 0
            T = 1.0
            sigma = max(sigma_min, min(sigma_max, 0.26 * avg_span))

            new_elites = [(best, best_x[:])]
            m = max(8, mu)
            for _ in range(m - 1):
                if time.time() >= deadline - eps_time:
                    break
                if random.random() < 0.55:
                    y = halton_point(random.randint(1, max(80, 24 * dim)))
                else:
                    y = rand_point()
                a = 0.25 + 0.70 * random.random()
                x = [clamp(a * best_x[i] + (1.0 - a) * y[i], i) for i in range(dim)]
                if not accept_new_point(x):
                    continue
                fx = eval_x(x)
                new_elites.append((fx, x[:]))
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
            new_elites.sort(key=lambda t: t[0])
            elites = new_elites[:mu]

    return best
