import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Key improvements over the shown best (Powell+ES) style code:
      1) Stronger local exploitation: a compact L-BFGS-like (limited-memory) quasi-Newton
         step with finite-difference gradients, guarded by backtracking line-search.
         (Still derivative-free: gradients are approximated.)
      2) Robustness: automatically switches between (a) quasi-Newton steps,
         (b) SPSA (cheap stochastic gradient) steps when dim is large or time is tight,
         (c) adaptive (1+1)-ES with heavy-tail jumps for global exploration.
      3) Better time usage: evaluation budgeting and time-sliced phases, plus elite restarts.

    Returns:
      best (float): best fitness found within time limit
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
    fixed = [spans[i] == 0.0 for i in range(dim)]
    span_max = max(spans) if spans else 0.0

    # ---------- helpers ----------
    def now():
        return time.time()

    def clamp(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def evalf(x):
        return float(func(x))

    def rand_point():
        return [lows[i] if fixed[i] else (lows[i] + random.random() * spans[i]) for i in range(dim)]

    def opposition(x):
        return [lows[i] if fixed[i] else (lows[i] + highs[i] - x[i]) for i in range(dim)]

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm2(a):
        return dot(a, a)

    def add_scaled(x, a, v):
        # x + a*v, then clamp
        y = [0.0] * dim
        for i in range(dim):
            y[i] = x[i] + a * v[i]
        return clamp(y)

    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- low-discrepancy Halton (scrambled) ----------
    def first_primes(k):
        primes = []
        c = 2
        while len(primes) < k:
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

    bases = first_primes(max(1, dim))
    scr = [random.random() for _ in range(dim)]

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton(idx):
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                u = (vdc(idx + 1, bases[i]) + scr[i]) % 1.0
                x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite pool ----------
    elite_k = max(10, min(30, 12 + dim // 2))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites.pop()

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    idx = 0

    # Use more structured global coverage early, but cap.
    init_n = max(160, min(2200, 180 + 55 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best

        if j % 13 == 0:
            x = rand_point()
        else:
            x = halton(idx)
            idx += 1

        x = clamp(x)
        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]
        push_elite(f, x)

        if now() >= deadline:
            return best
        xo = opposition(x)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]
        push_elite(fo, xo)

    if best_x is None:
        best_x = rand_point()
        best = evalf(best_x)
        push_elite(best, best_x)

    # ---------- finite-difference gradient (central where possible) ----------
    # Keep it cheap: only use on selected dims if dim huge.
    eps_base = 1e-8 + 1e-7 * (span_max + 1.0)

    def fd_grad(x, fx, budget_dims=None, time_end=None):
        # returns (g, used_dims_idx) where g has full dim entries (zeros if not used)
        if time_end is not None and now() >= time_end:
            return [0.0] * dim, []

        if budget_dims is None:
            # for very large dim, sample subset
            if dim <= 60:
                idxs = list(range(dim))
            else:
                m = max(18, min(60, 10 + dim // 8))
                idxs = random.sample(range(dim), m)
        else:
            idxs = budget_dims

        g = [0.0] * dim
        used = []
        for i in idxs:
            if fixed[i]:
                continue
            if time_end is not None and now() >= time_end:
                break

            # step proportional to span and scale
            h = eps_base * (spans[i] if spans[i] > 0.0 else 1.0)
            if h <= 0.0:
                continue

            xi_p = x[:]
            xi_m = x[:]
            xi_p[i] = min(highs[i], x[i] + h)
            xi_m[i] = max(lows[i], x[i] - h)

            # if clamped both ways to same point, skip
            if xi_p[i] == xi_m[i]:
                continue

            fp = evalf(xi_p)
            if time_end is not None and now() >= time_end:
                break
            fm = evalf(xi_m)

            denom = (xi_p[i] - xi_m[i])
            if denom != 0.0:
                g[i] = (fp - fm) / denom
                used.append(i)

        return g, used

    # ---------- SPSA gradient (2 evals) ----------
    def spsa_grad(x, c):
        # returns g approx with 2 evaluations
        delta = [0.0] * dim
        xp = x[:]
        xm = x[:]
        for i in range(dim):
            if fixed[i]:
                delta[i] = 0.0
                continue
            d = -1.0 if random.random() < 0.5 else 1.0
            delta[i] = d
            xp[i] = x[i] + c * d
            xm[i] = x[i] - c * d
        xp = clamp(xp)
        xm = clamp(xm)
        fp = evalf(xp)
        fm = evalf(xm)
        g = [0.0] * dim
        diff = (fp - fm)
        for i in range(dim):
            if fixed[i] or delta[i] == 0.0:
                continue
            denom = (xp[i] - xm[i])
            if denom != 0.0:
                # scale using actual denom after clamping
                g[i] = diff / denom * delta[i]  # delta[i] keeps sign convention stable
        return g

    # ---------- L-BFGS two-loop recursion (limited-memory) ----------
    # Store s = x_{k+1}-x_k, y = g_{k+1}-g_k, and rho = 1/(y^T s)
    m_hist = 6 if dim > 25 else 8
    s_hist = []
    y_hist = []
    rho_hist = []

    def lbfgs_direction(g):
        # returns p = -H*g
        q = g[:]
        alphas = []
        for k in range(len(s_hist) - 1, -1, -1):
            s = s_hist[k]
            y = y_hist[k]
            rho = rho_hist[k]
            a = rho * dot(s, q)
            alphas.append(a)
            # q -= a*y
            for i in range(dim):
                q[i] -= a * y[i]

        # initial Hessian scaling
        if y_hist:
            ys = dot(y_hist[-1], s_hist[-1])
            yy = dot(y_hist[-1], y_hist[-1])
            gamma = ys / yy if yy > 1e-30 else 1.0
            if gamma <= 0.0 or not math.isfinite(gamma):
                gamma = 1.0
        else:
            gamma = 1.0

        r = [gamma * qi for qi in q]

        # second loop
        alphas.reverse()
        for k in range(len(s_hist)):
            s = s_hist[k]
            y = y_hist[k]
            rho = rho_hist[k]
            b = rho * dot(y, r)
            a = alphas[k]
            coeff = (a - b)
            for i in range(dim):
                r[i] += coeff * s[i]

        # p = -r
        for i in range(dim):
            r[i] = -r[i]
        return r

    # ---------- line search (Armijo backtracking) ----------
    def armijo(x, fx, g, p, step0, time_end):
        # Ensure descent-ish: if g^T p >= 0, flip to negative gradient
        gp = dot(g, p)
        if not math.isfinite(gp):
            return fx, x[:], False
        if gp >= 0.0:
            p = [-gi for gi in g]
            gp = -dot(g, g)
            if gp >= 0.0:
                return fx, x[:], False

        c1 = 1e-4
        a = step0
        best_local_f = fx
        best_local_x = x[:]
        improved = False

        # A few backtracking steps only (time bounded)
        for _ in range(10):
            if now() >= time_end:
                break
            xn = add_scaled(x, a, p)
            fn = evalf(xn)
            if fn < best_local_f:
                best_local_f, best_local_x = fn, xn
                improved = True
            if fn <= fx + c1 * a * gp:
                return fn, xn, True
            a *= 0.5
            if a < 1e-18:
                break

        return best_local_f, best_local_x, improved

    # ---------- ES parameters ----------
    sigma_floor = 1e-12 + 1e-10 * (span_max + 1.0)
    sigma0 = [0.18 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
    sigma = sigma0[:]
    window = max(30, 14 + 2 * dim)
    succ = 0
    trials = 0
    no_improve = 0
    restart_after = 110 + 18 * dim

    x_cur = best_x[:]
    f_cur = best

    # scheduling
    next_qn = now()          # quasi-Newton slice
    next_global = now()      # occasional global reseed

    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # ---- occasional global reseed to refresh elites ----
        if t >= next_global:
            # 1-3 global samples
            tries = 2 if dim < 25 else 1
            for _ in range(tries):
                if now() >= deadline:
                    break
                if random.random() < 0.7:
                    xg = halton(idx)
                    idx += 1
                else:
                    xg = rand_point()
                fg = evalf(xg)
                push_elite(fg, xg)
                if fg < best:
                    best, best_x = fg, xg[:]
                    x_cur, f_cur = xg[:], fg
                    no_improve = 0
            next_global = now() + (0.15 + 0.01 * dim) * (0.35 + 0.85 * frac)

        # ---- quasi-Newton / gradient-based intensification (time-sliced) ----
        if t >= next_qn:
            slice_len = min(deadline - t, 0.040 + 0.0025 * dim)
            time_end = t + max(0.0, slice_len)

            # seed from elite best most of the time
            if elites and random.random() < 0.85:
                x_seed = elites[0][1][:]
            elif elites:
                x_seed = random.choice(elites)[1][:]
            else:
                x_seed = x_cur[:]
            x_seed = clamp(x_seed)
            f_seed = evalf(x_seed)

            # choose gradient strategy:
            # - FD (more accurate) for small/moderate dim
            # - SPSA (2 evals) for large dim or tight slice
            use_spsa = (dim > 80) or (slice_len < 0.02)

            # do a few quasi-Newton iterations within slice
            xq = x_seed[:]
            fq = f_seed
            g_prev = None
            x_prev = None

            it = 0
            while now() < time_end:
                it += 1
                # gradient
                if use_spsa:
                    c = (0.02 * (1.0 - 0.8 * frac) + 0.002) * (span_max + 1.0)
                    g = spsa_grad(xq, c)
                else:
                    g, _ = fd_grad(xq, fq, time_end=time_end)
                # if gradient is near zero, stop
                gg = norm2(g)
                if not math.isfinite(gg) or gg < 1e-30:
                    break

                # update L-BFGS memory if we have previous
                if g_prev is not None and x_prev is not None:
                    s = [xq[i] - x_prev[i] for i in range(dim)]
                    y = [g[i] - g_prev[i] for i in range(dim)]
                    ys = dot(y, s)
                    if math.isfinite(ys) and ys > 1e-18:
                        rho = 1.0 / ys
                        s_hist.append(s)
                        y_hist.append(y)
                        rho_hist.append(rho)
                        if len(s_hist) > m_hist:
                            s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)

                # direction
                if s_hist:
                    p = lbfgs_direction(g)
                else:
                    p = [-gi for gi in g]

                # step size heuristic based on bounds and time (smaller late)
                step0 = (0.35 * (1.0 - 0.85 * frac) + 0.05) * (span_max + 1.0) / (math.sqrt(gg) + 1e-12)

                # line search
                f_new, x_new, ok = armijo(xq, fq, g, p, step0, time_end)

                x_prev, g_prev = xq, g
                if ok or (f_new < fq):
                    xq, fq = x_new, f_new
                else:
                    break

                if fq < best:
                    best, best_x = fq, xq[:]
                    push_elite(best, best_x)
                    x_cur, f_cur = xq[:], fq
                    no_improve = 0

                # keep QN iteration count modest
                if it >= (5 if dim < 40 else 3):
                    break

            # schedule next intensification: more frequent early
            next_qn = now() + (0.06 + 0.01 * dim) * (0.35 + 0.95 * frac)

            if now() >= deadline:
                break

        # ---- stochastic ES step (global + local blend) ----
        # choose base
        if elites and random.random() < 0.25:
            x_base = random.choice(elites)[1][:]
        else:
            x_base = x_cur

        r = random.random()
        if r < 0.76:
            # gaussian-ish
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                g = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random() - 3.0) / 3.0
                x_new[i] += g * s
        elif r < 0.92:
            # heavy tail escape
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                x_new[i] += 0.50 * s * cauchy()
        else:
            # global
            if random.random() < 0.75:
                x_new = halton(idx)
                idx += 1
            else:
                x_new = rand_point()

        x_new = clamp(x_new)
        f_new = evalf(x_new)

        trials += 1
        if f_new <= f_cur:
            x_cur, f_cur = x_new, f_new
            succ += 1

        if f_new < best:
            best, best_x = f_new, x_new[:]
            push_elite(best, best_x)
            no_improve = 0
        else:
            no_improve += 1

        # 1/5 success adaptation
        if trials >= window:
            rate = succ / float(trials)
            if rate > 0.22:
                factor = 1.25
            elif rate < 0.18:
                factor = 0.78
            else:
                factor = 1.0
            if factor != 1.0:
                for i in range(dim):
                    if fixed[i]:
                        continue
                    sigma[i] *= factor
                    if sigma[i] > 0.90 * spans[i]:
                        sigma[i] = 0.90 * spans[i]
                    if sigma[i] < sigma_floor:
                        sigma[i] = sigma_floor
            succ = 0
            trials = 0

        # stagnation restart
        if no_improve >= restart_after:
            if elites and random.random() < 0.92:
                x_cur = (elites[0][1] if random.random() < 0.7 else random.choice(elites)[1])[:]
            else:
                x_cur = halton(idx); idx += 1

            # jitter (smaller later)
            jitter = (0.16 * (1.0 - 0.85 * frac) + 0.015)
            for i in range(dim):
                if fixed[i]:
                    continue
                x_cur[i] += (random.random() * 2.0 - 1.0) * jitter * spans[i]
            x_cur = clamp(x_cur)
            f_cur = evalf(x_cur)
            push_elite(f_cur, x_cur)
            if f_cur < best:
                best, best_x = f_cur, x_cur[:]

            # reset sigmas
            shrink = 0.55 * (1.0 - 0.85 * frac) + 0.12
            for i in range(dim):
                sigma[i] = 0.0 if fixed[i] else max(sigma_floor, min(0.70 * spans[i], shrink * sigma0[i]))

            # clear some quasi-newton history to avoid stale curvature after a jump
            if random.random() < 0.6:
                s_hist[:] = []
                y_hist[:] = []
                rho_hist[:] = []

            no_improve = 0

    return best
