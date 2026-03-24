import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no numpy).

    Upgrade over the provided algorithm:
      1) Stronger GLOBAL search:
         - scrambled Halton + occasional random + opposite points
         - multi-start "incumbent set" (keeps several elites, not just one)
      2) Much stronger LOCAL search:
         - bounded Powell-style coordinate pattern search (very robust)
         - opportunistic 2-point quadratic interpolation per coordinate
      3) Smarter step-size control:
         - per-dimension step sizes with success-based expansion/contraction
         - periodic step-size "reheat" if stagnating
      4) Evaluation budget awareness:
         - always checks deadline, uses cheap operations only

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

    # ---------- utilities ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---------- Halton (scrambled by shift) ----------
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
    halton_idx = 1

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_idx
        idx = halton_idx
        halton_idx += 1
        x = [0.0] * dim
        for i in range(dim):
            u = vdc(idx, primes[i]) + halton_shift[i]
            u -= int(u)  # wrap [0,1)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- Local search: bounded coordinate pattern search + quadratic step ----------
    def local_refine(x0, f0, steps, time_limit, max_sweeps):
        """
        A robust bounded coordinate search:
          - for each coordinate i, try +/- step[i]
          - if both sides evaluated, attempt 1D quadratic interpolation step
          - success expands step; failure shrinks step
        """
        x = x0[:]
        fx = f0
        n = dim

        # Randomized coordinate order each sweep helps on separable/rotated-ish functions
        for _sweep in range(max_sweeps):
            if time.time() >= time_limit:
                break

            order = list(range(n))
            # cheap shuffle
            for k in range(n - 1, 0, -1):
                j = int(random.random() * (k + 1))
                order[k], order[j] = order[j], order[k]

            improved_any = False

            for i in order:
                if time.time() >= time_limit:
                    break

                si = steps[i]
                if si <= 1e-16:
                    continue

                xi = x[i]

                # Try + step
                x_p = x[:]
                x_p[i] = xi + si
                clip_inplace(x_p)
                f_p = safe_eval(x_p)

                # Try - step
                x_m = x[:]
                x_m[i] = xi - si
                clip_inplace(x_m)
                f_m = safe_eval(x_m)

                # Best among {current, plus, minus}
                best1_f = fx
                best1_x = None
                if f_p < best1_f:
                    best1_f = f_p
                    best1_x = x_p
                if f_m < best1_f:
                    best1_f = f_m
                    best1_x = x_m

                # If we have both sides, attempt a quadratic step along this coordinate.
                # Fit parabola through (xi-si,fm), (xi,fx), (xi+si,fp) in local coord t in {-1,0,1}
                # t* = 0.5*(fm - fp)/(fm - 2f0 + fp)
                if time.time() < time_limit:
                    denom = (f_m - 2.0 * fx + f_p)
                    if denom != 0.0 and (not math.isinf(denom)) and (not math.isnan(denom)):
                        t_star = 0.5 * (f_m - f_p) / denom
                        # keep it within a reasonable bracket to avoid wild jumps
                        if t_star > 2.5:
                            t_star = 2.5
                        elif t_star < -2.5:
                            t_star = -2.5

                        if abs(t_star) > 0.15:  # ignore tiny predicted moves
                            x_q = x[:]
                            x_q[i] = xi + t_star * si
                            clip_inplace(x_q)
                            f_q = safe_eval(x_q)
                            if f_q < best1_f:
                                best1_f = f_q
                                best1_x = x_q

                if best1_x is not None:
                    # accept improvement
                    x = best1_x
                    fx = best1_f
                    improved_any = True
                    # expand a bit on success
                    steps[i] = min(0.5 * spans[i], steps[i] * 1.35)
                else:
                    # shrink on failure
                    steps[i] = max(1e-16 * spans[i], steps[i] * 0.65)

            # If a sweep made no improvement, we still continue but the steps have shrunk.
            # Stop when overall steps are tiny.
            if not improved_any:
                max_rel = 0.0
                for i in range(n):
                    r = steps[i] / (spans[i] + 1e-300)
                    if r > max_rel:
                        max_rel = r
                if max_rel < 1e-12:
                    break

        return x, fx, steps

    # ---------- initialization: build an elite set ----------
    best = float("inf")
    best_x = None

    elites = []  # list of (f, x)
    elite_cap = max(8, min(24, 6 + int(2.0 * math.log(dim + 2.0) + dim ** 0.5)))

    def push_elite(fx, x):
        nonlocal best, best_x, elites
        if fx >= float("inf"):
            return
        if best_x is None or fx < best:
            best = fx
            best_x = x[:]
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        # deduplicate-ish: drop very similar points if too many
        if len(elites) > elite_cap * 2:
            elites = elites[:elite_cap * 2]
        if len(elites) > elite_cap:
            elites = elites[:elite_cap]

    # quick probes: corners + halton + random + opposition
    init_budget = max(80, min(1200, 200 + 35 * dim))
    for k in range(init_budget):
        if time.time() >= deadline:
            return best
        r = random.random()
        if r < 0.55:
            x = halton_point()
        elif r < 0.85:
            x = rand_point()
        else:
            # corner-ish point with jitter
            x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
            for _ in range(max(1, dim // 4)):
                i = int(random.random() * dim)
                x[i] = lows[i] + random.random() * spans[i]

        fx = safe_eval(x)
        push_elite(fx, x)

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = safe_eval(xo)
        push_elite(fo, xo)

    if best_x is None:
        return best

    # initial per-dimension step sizes for local search
    base_step = 0.18
    steps0 = [max(1e-16 * spans[i], base_step * spans[i]) for i in range(dim)]

    # ---------- main time-bounded loop: iterate multi-start local refinement + global injections ----------
    last_improve_t = time.time()
    last_best = best

    # Controls
    reheat_period = max(0.25, 0.15 * float(max_time))   # seconds without improvement -> reheat
    refine_slice = max(0.02, 0.06 * float(max_time))    # seconds per refinement burst

    while time.time() < deadline:
        now = time.time()

        # pick a start point: mostly best, sometimes other elite to escape local traps
        if len(elites) > 1 and random.random() < 0.45:
            # bias towards better elites
            idx = int((random.random() ** 2.2) * len(elites))
            x0 = elites[idx][1][:]
            f0 = elites[idx][0]
        else:
            x0 = best_x[:]
            f0 = best

        # local refinement burst
        tl = min(deadline, now + refine_slice)
        steps = steps0[:]  # fresh steps per burst to avoid over-shrinking globally
        x1, f1, _ = local_refine(x0, f0, steps, tl, max_sweeps=4 + dim // 3)
        push_elite(f1, x1)

        if best < last_best - 1e-12:
            last_best = best
            last_improve_t = time.time()

        # global injections each loop (cheap diversification)
        if time.time() >= deadline:
            break

        inj = 2 if dim <= 10 else 3
        for _ in range(inj):
            if time.time() >= deadline:
                break
            # sample near an elite + occasional halton/random far jump
            r = random.random()
            if r < 0.55 and elites:
                # jitter around a random elite with scaled step
                eidx = int((random.random() ** 1.8) * len(elites))
                xe = elites[eidx][1][:]
                # isotropic-ish jitter using sum of uniforms (approx normal)
                for i in range(dim):
                    z = (sum(random.random() for _ in range(8)) - 4.0)  # ~N(0,1) scaled
                    xe[i] += z * 0.12 * spans[i]
                clip_inplace(xe)
                fe = safe_eval(xe)
                push_elite(fe, xe)
            elif r < 0.82:
                xg = halton_point()
                fg = safe_eval(xg)
                push_elite(fg, xg)
            else:
                xr = rand_point()
                fr = safe_eval(xr)
                push_elite(fr, xr)

        # stagnation handling: reheat by expanding step and refreshing Halton scramble
        if (time.time() - last_improve_t) > reheat_period:
            last_improve_t = time.time()
            # reheat base step a bit (encourage bigger moves in next bursts)
            base_step = min(0.35, base_step * 1.25 + 0.02)
            steps0 = [max(1e-16 * spans[i], base_step * spans[i]) for i in range(dim)]
            # change Halton shift to effectively restart LDS
            halton_shift = [random.random() for _ in range(dim)]
            # inject a handful of diversified points incl. opposition
            extra = max(6, 2 + dim // 3)
            for _ in range(extra):
                if time.time() >= deadline:
                    break
                x = halton_point() if random.random() < 0.7 else rand_point()
                fx = safe_eval(x)
                push_elite(fx, x)
                if time.time() >= deadline:
                    break
                xo = opposite(x)
                fo = safe_eval(xo)
                push_elite(fo, xo)
        else:
            # slowly cool step if improving
            base_step = max(0.03, base_step * 0.995)
            steps0 = [max(1e-16 * spans[i], base_step * spans[i]) for i in range(dim)]

    return best
