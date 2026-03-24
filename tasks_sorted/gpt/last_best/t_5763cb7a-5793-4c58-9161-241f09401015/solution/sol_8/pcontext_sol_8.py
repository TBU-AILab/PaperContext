import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Further improved time-bounded derivative-free minimizer (no external libs).

    What changed vs the previous "best":
      - Switch core global optimizer to a compact CMA-ES-like scheme:
          * learns diagonal covariance (per-dim sigma) AND a low-rank direction (rank-1)
          * keeps an evolution path (momentum) for consistent progress directions
      - Better constraint handling with "reflect then clip" to avoid boundary sticking
      - Interleaves three modes automatically:
          (1) CMA-like generation steps (main driver)
          (2) trust-region coordinate/pattern refinement on the incumbent (cheap, robust)
          (3) low-discrepancy/global injections + opposition for diversification
      - Stronger stagnation detection and restarts with sigma reset + new Halton scramble

    Returns:
        best (float): best function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

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
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def reflect_then_clip_inplace(x):
        # reflect at bounds (once), then clip to be safe
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if x[i] < lo:
                x[i] = lo + (lo - x[i])
                if x[i] > hi:
                    x[i] = lo
            elif x[i] > hi:
                x[i] = hi - (x[i] - hi)
                if x[i] < lo:
                    x[i] = hi

            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # approx N(0,1) via CLT
    def gauss01():
        return (sum(random.random() for _ in range(12)) - 6.0)

    # ---------------- Halton sequence ----------------
    def first_primes(n):
        ps = []
        c = 2
        while len(ps) < n:
            ok = True
            r = int(c ** 0.5)
            for p in ps:
                if p > r:
                    break
                if c % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(c)
            c += 1
        return ps

    primes = first_primes(dim)
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
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
            u = vdc(idx, primes[i]) + halton_shift[i]
            u -= int(u)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- lightweight local refinement ----------------
    def local_pattern_refine(x0, f0, time_limit):
        # bounded coordinate search with adaptive steps; very cheap and robust
        x = x0[:]
        fx = f0
        steps = [0.12 * spans[i] for i in range(dim)]
        min_step = 1e-12
        shrink = 0.6
        grow = 1.25

        max_sweeps = 2 + dim // 3
        for _ in range(max_sweeps):
            if time.time() >= time_limit:
                break

            # random order
            order = list(range(dim))
            for k in range(dim - 1, 0, -1):
                j = int(random.random() * (k + 1))
                order[k], order[j] = order[j], order[k]

            improved = False
            for i in order:
                if time.time() >= time_limit:
                    break
                si = steps[i]
                if si < min_step * (spans[i] + 1e-300):
                    continue

                xi = x[i]

                # try plus/minus
                xp = x[:]
                xp[i] = xi + si
                reflect_then_clip_inplace(xp)
                fp = safe_eval(xp)

                xm = x[:]
                xm[i] = xi - si
                reflect_then_clip_inplace(xm)
                fm = safe_eval(xm)

                if fp < fx or fm < fx:
                    if fp <= fm:
                        x, fx = xp, fp
                    else:
                        x, fx = xm, fm
                    improved = True
                    steps[i] = min(0.5 * spans[i], steps[i] * grow)
                else:
                    steps[i] *= shrink

            # stop if steps tiny and no improvement
            if not improved:
                mx = 0.0
                for i in range(dim):
                    rel = steps[i] / (spans[i] + 1e-300)
                    if rel > mx:
                        mx = rel
                if mx < 1e-10:
                    break
        return x, fx

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # a bit of corners/jitter + halton/opposition
    init_n = max(160, min(1400, 220 + 35 * dim))
    for k in range(init_n):
        if time.time() >= deadline:
            return best

        r = random.random()
        if r < 0.15:
            x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
            for _ in range(max(1, dim // 3)):
                i = random.randrange(dim)
                x[i] = lows[i] + random.random() * spans[i]
        elif r < 0.80:
            x = halton_point()
        else:
            x = rand_point()

        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x[:]

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = safe_eval(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        return best

    # ---------------- Compact CMA-ES-like loop ----------------
    # mean in normalized coordinates y in [0,1]^d to stabilize step sizes
    inv_span = [1.0 / (spans[i] + 1e-300) for i in range(dim)]

    def to_unit(x):
        return [(x[i] - lows[i]) * inv_span[i] for i in range(dim)]

    def from_unit(y):
        return [lows[i] + y[i] * spans[i] for i in range(dim)]

    m = to_unit(best_x)

    # diagonal sigma in unit space
    sigma = 0.22  # global step factor
    diag = [0.22] * dim  # per-dim scaling (unit space)
    diag_min, diag_max = 1e-6, 0.6

    # rank-1 direction vector (unit space), and evolution path
    p = [0.0] * dim
    v = [0.0] * dim  # learned predominant direction

    # parameters
    lam = max(18, min(80, 18 + int(8.0 * math.log(dim + 2.0))))
    mu = max(4, lam // 3)

    # recombination weights (log)
    w = [0.0] * mu
    s = 0.0
    for i in range(mu):
        w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        s += w[i]
    for i in range(mu):
        w[i] /= s

    # learning rates
    c_m = 1.0
    c_c = min(0.45, (2.0 / (dim + 2.0)))
    c_v = min(0.25, (1.5 / (dim + 3.0)))
    c_diag = min(0.35, (2.5 / (dim + 5.0)))

    # sigma control via success rate
    target_sr = 0.2
    sr_ema = target_sr

    last_improve_t = time.time()
    last_best = best

    # time slices
    refine_slice = max(0.02, 0.06 * float(max_time))
    inject_every = 3
    gen = 0

    while time.time() < deadline:
        gen += 1

        # offspring
        off = []
        successes = 0
        base_best = best

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # sample z ~ N(0,I), plus rank-1 component along v
            z = [gauss01() for _ in range(dim)]
            g = gauss01()

            y = [0.0] * dim
            for i in range(dim):
                y[i] = m[i] + sigma * (diag[i] * z[i] + 0.35 * v[i] * g)

            # bring back to box by reflecting in real space (more robust near borders)
            x = from_unit(y)
            reflect_then_clip_inplace(x)
            fy = safe_eval(x)
            off.append((fy, x))
            if fy < best:
                best, best_x = fy, x[:]
                last_improve_t = time.time()
            if fy < base_best:
                successes += 1

        off.sort(key=lambda t: t[0])
        elites = off[:mu]

        # update success rate and sigma
        sr = successes / float(lam)
        sr_ema = 0.85 * sr_ema + 0.15 * sr
        if sr_ema > target_sr + 0.05:
            sigma = min(0.6, sigma * 1.12)
        elif sr_ema < target_sr - 0.05:
            sigma = max(1e-6, sigma * 0.88)

        # recombination update of mean in unit space
        m_new = [0.0] * dim
        for j in range(mu):
            yj = to_unit(elites[j][1])
            for i in range(dim):
                m_new[i] += w[j] * yj[i]
        # evolution path
        dm = [m_new[i] - m[i] for i in range(dim)]
        for i in range(dim):
            p[i] = (1.0 - c_c) * p[i] + c_c * (dm[i] / (sigma + 1e-300))
        # update v towards p (rank-1)
        for i in range(dim):
            v[i] = (1.0 - c_v) * v[i] + c_v * p[i]

        # diagonal adaptation from spread of elites around mean
        # estimate per-dim std in unit coords
        var = [0.0] * dim
        for j in range(mu):
            yj = to_unit(elites[j][1])
            for i in range(dim):
                d = (yj[i] - m_new[i])
                var[i] += w[j] * d * d
        for i in range(dim):
            # convert var to scale multiplier; keep bounded
            # (sqrt(var) tends to shrink near optimum)
            sdi = math.sqrt(max(var[i], 1e-20))
            # move diag towards observed dispersion (relative)
            desired = max(diag_min, min(diag_max, 0.65 * sdi + 0.35 * diag[i]))
            diag[i] = (1.0 - c_diag) * diag[i] + c_diag * desired

        # commit mean
        for i in range(dim):
            m[i] = (1.0 - c_m) * m[i] + c_m * m_new[i]
            # keep unit cube
            if m[i] < 0.0:
                m[i] = 0.0
            elif m[i] > 1.0:
                m[i] = 1.0

        # occasional local refine around current best
        if time.time() < deadline and (gen % 2 == 0):
            tl = min(deadline, time.time() + refine_slice)
            xr, fr = local_pattern_refine(best_x, best, tl)
            if fr < best:
                best, best_x = fr, xr[:]
                last_improve_t = time.time()
                # pull mean towards improved best
                m = to_unit(best_x)

        # global injections (Halton/random/opposition)
        if gen % inject_every == 0 and time.time() < deadline:
            for _ in range(3 if dim <= 12 else 4):
                if time.time() >= deadline:
                    return best
                xg = halton_point() if random.random() < 0.7 else rand_point()
                fg = safe_eval(xg)
                if fg < best:
                    best, best_x = fg, xg[:]
                    last_improve_t = time.time()
                    m = to_unit(best_x)
                if time.time() >= deadline:
                    return best
                xo = opposite(xg)
                fo = safe_eval(xo)
                if fo < best:
                    best, best_x = fo, xo[:]
                    last_improve_t = time.time()
                    m = to_unit(best_x)

        # stagnation restart
        now = time.time()
        if best < last_best - 1e-12:
            last_best = best
        else:
            if (now - last_improve_t) > max(0.30 * float(max_time), 1.0):
                last_improve_t = now
                # restart around best with re-expanded steps and new LDS scramble
                halton_shift = [random.random() for _ in range(dim)]
                sigma = min(0.45, max(0.10, sigma * 1.6))
                for i in range(dim):
                    diag[i] = min(diag_max, max(0.10, diag[i] * 1.35))
                    p[i] = 0.0
                    v[i] = 0.0
                # inject some diversified points and possibly move mean
                for _ in range(max(10, 2 * dim)):
                    if time.time() >= deadline:
                        return best
                    x = rand_point() if random.random() < 0.55 else halton_point()
                    fx = safe_eval(x)
                    if fx < best:
                        best, best_x = fx, x[:]
                        m = to_unit(best_x)
                        last_best = best
                    xo = opposite(x)
                    fo = safe_eval(xo)
                    if fo < best:
                        best, best_x = fo, xo[:]
                        m = to_unit(best_x)
                        last_best = best

    return best
