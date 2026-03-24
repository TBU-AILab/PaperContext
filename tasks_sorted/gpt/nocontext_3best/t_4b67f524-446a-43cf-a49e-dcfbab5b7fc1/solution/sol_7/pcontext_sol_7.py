import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Main changes vs the provided best (~49.03):
      - Switches to a compact "CMA-ES-like" search (diagonal covariance + rank-1 update)
        for strong global+local performance with few hyperparameters.
      - Uses mirrored sampling (antithetic) to reduce noise and improve sample efficiency.
      - Uses elite recombination and evolution-path step-size adaptation (CMA-style),
        plus occasional heavy-tail injections and restarts from an elite archive.
      - Keeps a short coordinate pattern search polish when time allows.

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
    fixed = [spans[i] <= 0.0 for i in range(dim)]
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
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                x[i] = lows[i] + random.random() * spans[i]
        return x

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                y[i] = lows[i]
            else:
                y[i] = lows[i] + highs[i] - x[i]
        return y

    # approx standard normal via sum of uniforms (fast, no libs)
    def randn():
        # mean 0, var ~1 using 12 uniforms - 6
        return (random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() - 6.0)

    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- scrambled Halton for initial coverage ----------
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

    # ---------- elite archive (best-so-far points) ----------
    elite_k = max(16, min(64, 18 + dim))
    elites = []  # list of (f, x)

    # diversity by coarse cell hashing
    q_levels = 14 if dim <= 30 else 10
    seen_cells = set()

    def cell_key(x):
        key = []
        for i in range(dim):
            if fixed[i]:
                key.append(0)
                continue
            s = spans[i]
            if s <= 0.0:
                key.append(0)
            else:
                u = (x[i] - lows[i]) / s
                b = int(u * q_levels)
                if b < 0:
                    b = 0
                elif b >= q_levels:
                    b = q_levels - 1
                key.append(b)
        return tuple(key)

    def push_elite(f, x):
        nonlocal elites
        ck = cell_key(x)
        if ck in seen_cells and elites:
            # skip near-duplicates unless very good
            if f > elites[0][0] * 1.02 + 1e-12:
                return
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]
        seen_cells.clear()
        for ff, xx in elites:
            seen_cells.add(cell_key(xx))

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    hidx = 0

    init_n = max(120, min(2600, 220 + 70 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best

        if j % 19 == 0:
            x = rand_point()
        else:
            x = halton(hidx)
            hidx += 1
        x = clamp(x)
        fx = evalf(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(fx, x)

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

    # ---------- diagonal CMA-ES-like state ----------
    # population size (small to keep evals cheap)
    lam = max(8, min(28, 8 + dim // 3))
    if lam % 2 == 1:
        lam += 1  # easier to mirror pairs
    mu = max(2, lam // 2)

    # recombination weights (log)
    w = []
    for i in range(mu):
        w.append(math.log(mu + 0.5) - math.log(i + 1.0))
    w_sum = sum(w)
    w = [wi / w_sum for wi in w]
    mu_eff = 1.0 / sum(wi * wi for wi in w)

    # step-size and diagonal covariance
    # start moderately large; shrink later via CSA
    sigma = 0.22 * (span_max + 1.0)
    sigma_min = 1e-15 + 1e-12 * (span_max + 1.0)

    # diagonal std factors (D), keep in log space for stability
    logD = [0.0] * dim  # D starts at 1
    logD_min = math.log(1e-9)
    logD_max = math.log(1e3)

    # evolution path for step-size (CSA)
    ps = [0.0] * dim
    cs = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    ds = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + cs

    # diagonal covariance learning rate (rank-1-ish on z-variance)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    c1 = min(0.10, max(0.002, c1))

    # expected ||N(0,I)||
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)) if dim > 0 else 0.0

    # mean in normalized coordinates y in [0,1]^dim; operate there for easier scaling
    def to_unit(x):
        y = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                y[i] = 0.0
            else:
                s = spans[i]
                y[i] = (x[i] - lows[i]) / s if s > 0.0 else 0.0
                if y[i] < 0.0:
                    y[i] = 0.0
                elif y[i] > 1.0:
                    y[i] = 1.0
        return y

    def from_unit(y):
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                v = y[i]
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                x[i] = lows[i] + v * spans[i]
        return x

    m = to_unit(best_x)

    # ---------- cheap local polish: coordinate pattern with adaptive step ----------
    def local_polish(x0, f0, time_end):
        x = x0[:]
        fx = f0
        # step in unit-space
        step = 0.15
        step_floor = 1e-12
        # only subset if very large dim
        if dim <= 60:
            active = list(range(dim))
        else:
            active = random.sample(range(dim), max(24, min(90, 12 + dim // 6)))

        it = 0
        while now() < time_end and it < (5 if dim <= 40 else 3):
            it += 1
            improved = False
            random.shuffle(active)
            for i in active:
                if fixed[i]:
                    continue
                if now() >= time_end:
                    break
                xi = x[i]
                # try +step
                x[i] = min(1.0, xi + step)
                f1 = evalf(from_unit(x))
                if f1 < fx:
                    fx = f1
                    improved = True
                    continue
                # try -step
                x[i] = max(0.0, xi - step)
                f2 = evalf(from_unit(x))
                if f2 < fx:
                    fx = f2
                    improved = True
                    continue
                # restore
                x[i] = xi
            if not improved:
                step *= 0.55
                if step < step_floor:
                    break
            else:
                step = min(0.25, step * 1.05)
        return fx, x

    # ---------- main loop ----------
    no_improve = 0
    restart_after = 80 + 16 * dim
    next_polish = now()

    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # occasional local polish slice
        if t >= next_polish and (deadline - t) > 1e-6:
            slice_len = min(deadline - t, (0.025 + 0.0016 * dim) * (1.10 - 0.55 * frac))
            time_end = t + max(0.0, slice_len)

            seed = None
            if elites and random.random() < 0.75:
                seed = to_unit(elites[0][1])
            elif elites and random.random() < 0.6:
                seed = to_unit(random.choice(elites)[1])
            else:
                seed = m[:]

            fseed = evalf(from_unit(seed))
            push_elite(fseed, from_unit(seed))
            if fseed < best:
                best, best_x = fseed, from_unit(seed)
                m = seed[:]
                no_improve = 0

            fP, yP = local_polish(seed, fseed, time_end)
            xP = from_unit(yP)
            push_elite(fP, xP)
            if fP < best:
                best, best_x = fP, xP
                m = yP[:]
                no_improve = 0
            else:
                no_improve += 1

            next_polish = now() + (0.06 + 0.004 * dim) * (0.35 + 0.95 * frac)
            if now() >= deadline:
                break

        # build population around mean in unit space
        D = [math.exp(ld) for ld in logD]
        # time-aware sigma decay cap (helps tighten near end)
        sigma_eff = max(sigma_min, sigma * (1.0 - 0.35 * frac))

        pop = []  # list of (f, y, z)
        # mirrored sampling: create lam/2 z then use +/-z
        for _ in range(lam // 2):
            if now() >= deadline:
                break
            z = [0.0] * dim
            for i in range(dim):
                if fixed[i]:
                    z[i] = 0.0
                else:
                    z[i] = randn()
            for sgn in (1.0, -1.0):
                y = [0.0] * dim
                for i in range(dim):
                    if fixed[i]:
                        y[i] = 0.0
                    else:
                        # unit-space step size proportional to sigma_eff/spans
                        step_unit = (sigma_eff / (spans[i] + 1e-300)) * D[i]
                        y[i] = m[i] + sgn * step_unit * z[i]
                        # reflect into [0,1] to avoid sticking to bounds
                        if y[i] < 0.0:
                            y[i] = -y[i]
                        if y[i] > 1.0:
                            y[i] = 2.0 - y[i]
                        if y[i] < 0.0:
                            y[i] = 0.0
                        elif y[i] > 1.0:
                            y[i] = 1.0
                x = from_unit(y)
                f = evalf(x)
                pop.append((f, y, [sgn * zi for zi in z]))
                push_elite(f, x)
                if f < best:
                    best, best_x = f, x
                    no_improve = 0
                else:
                    no_improve += 1
                if now() >= deadline:
                    break

        if not pop:
            break

        pop.sort(key=lambda t: t[0])

        # recombination: new mean
        m_old = m[:]
        m = [0.0] * dim
        for j in range(mu):
            yj = pop[j][1]
            wj = w[j]
            for i in range(dim):
                m[i] += wj * yj[i]

        # update evolution path ps (in z-space approx)
        # approximate "zmean" from selected steps in normalized coordinates
        zmean = [0.0] * dim
        for j in range(mu):
            zj = pop[j][2]
            wj = w[j]
            for i in range(dim):
                zmean[i] += wj * zj[i]

        csa = math.sqrt(cs * (2.0 - cs) * mu_eff)
        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + csa * zmean[i]

        # step-size adaptation
        ps_norm = math.sqrt(sum(pi * pi for pi in ps))
        if chiN > 0.0 and math.isfinite(ps_norm):
            sigma *= math.exp((cs / ds) * ((ps_norm / chiN) - 1.0))
        if sigma < sigma_min:
            sigma = sigma_min
        if sigma > 2.0 * (span_max + 1.0):
            sigma = 2.0 * (span_max + 1.0)

        # diagonal covariance update: track per-dim z-variance among elites
        # (cheap rank-1-ish: update logD toward observed std of selected z)
        for i in range(dim):
            if fixed[i]:
                continue
            v = 0.0
            for j in range(mu):
                zi = pop[j][2][i]
                v += w[j] * (zi * zi)
            # v ~ variance in z; desired D scales with sqrt(v)
            # update logD smoothly
            target = 0.5 * math.log(max(1e-12, v))
            logD[i] = (1.0 - c1) * logD[i] + c1 * target
            if logD[i] < logD_min:
                logD[i] = logD_min
            elif logD[i] > logD_max:
                logD[i] = logD_max

        # occasional heavy-tail injection to escape
        if random.random() < (0.05 + 0.10 * (1.0 - frac)) and (deadline - now()) > 0.0:
            base = m[:] if random.random() < 0.6 else to_unit(random.choice(elites)[1]) if elites else m[:]
            y = base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                y[i] += 0.08 * cauchy()
                if y[i] < 0.0:
                    y[i] = -y[i]
                if y[i] > 1.0:
                    y[i] = 2.0 - y[i]
                if y[i] < 0.0:
                    y[i] = 0.0
                elif y[i] > 1.0:
                    y[i] = 1.0
            x = from_unit(y)
            f = evalf(x)
            push_elite(f, x)
            if f < best:
                best, best_x = f, x
                m = y[:]
                no_improve = 0

        # stagnation restart: re-center mean from elites, reset covariance/paths
        if no_improve >= restart_after:
            if elites and random.random() < 0.9:
                if random.random() < 0.65:
                    m = to_unit(elites[0][1])
                else:
                    # crossover in unit space
                    a = 0.25 + 0.5 * random.random()
                    y1 = to_unit(elites[0][1])
                    y2 = to_unit(random.choice(elites)[1])
                    m = [a * y1[i] + (1.0 - a) * y2[i] for i in range(dim)]
                    for i in range(dim):
                        if fixed[i]:
                            m[i] = 0.0
                        else:
                            m[i] = min(1.0, max(0.0, m[i]))
            else:
                m = to_unit(halton(hidx))
                hidx += 1

            ps = [0.0] * dim
            logD = [0.0] * dim
            sigma = 0.25 * (span_max + 1.0) * (0.95 - 0.55 * frac)
            if sigma < sigma_min:
                sigma = sigma_min
            no_improve = 0

    return best
