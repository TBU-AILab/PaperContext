import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only).

    Improvements vs. provided algorithm:
      - Adds a strong *trust-region local search* (Powell/BOBYQA-like spirit) using
        adaptive coordinate directions + quadratic 1D line search (parabolic fit),
        which is very effective for many competition benchmarks.
      - Keeps a light global explorer (Halton + heavy-tail) to escape local minima.
      - Uses an "ask/tell" style with strict deadline checks.
      - Robust bound handling via reflection in normalized [0,1] space.
      - Aggressive caching (quantized) to avoid duplicate evaluations.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time
    n = int(dim)
    if n <= 0:
        return float("inf")

    lo = [float(bounds[i][0]) for i in range(n)]
    hi = [float(bounds[i][1]) for i in range(n)]
    span = [hi[i] - lo[i] for i in range(n)]
    for i in range(n):
        if not (span[i] > 0.0):
            span[i] = 1.0

    eps = 1e-15

    def now():
        return time.time()

    # --- reflection into [0,1] for each coordinate ---
    def reflect01(u):
        if 0.0 <= u <= 1.0:
            return u
        u = u % 2.0
        if u > 1.0:
            u = 2.0 - u
        return u

    def x_from_u(u):
        return [lo[i] + u[i] * span[i] for i in range(n)]

    # --- caching (quantize normalized coordinates) ---
    cache = {}
    # quantization step: coarse enough to help caching, fine enough not to harm search
    q = 2e-10

    def key_of_u(u):
        return tuple(int(u[i] / q) for i in range(n))

    def eval_u(u):
        ur = [reflect01(u[i]) for i in range(n)]
        k = key_of_u(ur)
        v = cache.get(k)
        if v is not None:
            return v, ur
        fx = float(func(x_from_u(ur)))
        cache[k] = fx
        return fx, ur

    # --- low-discrepancy init: Halton sequence ---
    def first_primes(m):
        primes = []
        x = 2
        while len(primes) < m:
            r = int(math.isqrt(x))
            ok = True
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    def vdc(k, base):
        out = 0.0
        denom = 1.0
        while k:
            k, r = divmod(k, base)
            denom *= base
            out += r / denom
        return out

    bases = first_primes(n)
    hal_k = 1

    def halton_u(k):
        return [vdc(k, bases[i]) for i in range(n)]

    # --- helpers ---
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def add_scaled(a, d, s):
        return [a[i] + s * d[i] for i in range(n)]

    def unit_vec(i):
        v = [0.0] * n
        v[i] = 1.0
        return v

    # --- heavy tail sampler (Cauchy-like) ---
    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    # --- 1D line search along direction d in u-space with reflection ---
    # Uses: (a) small bracketing (b) parabolic interpolation if possible (c) fallback.
    def line_search_parabolic(u0, f0, d, step0, max_evals=12):
        # normalize direction to avoid scaling issues
        nd = norm(d)
        if nd < 1e-18:
            return u0, f0, 0
        d = [di / nd for di in d]

        # clamp step0
        step0 = abs(step0)
        if step0 < 1e-12:
            return u0, f0, 0

        # Evaluate at 0, +s, -s
        evals = 0
        s = step0

        u_p = add_scaled(u0, d, +s)
        fp, u_p = eval_u(u_p); evals += 1
        if now() >= deadline:
            return u0, f0, evals

        u_m = add_scaled(u0, d, -s)
        fm, u_m = eval_u(u_m); evals += 1
        if now() >= deadline:
            # return best among evaluated
            if fp < f0 and fp <= fm:
                return u_p, fp, evals
            if fm < f0 and fm < fp:
                return u_m, fm, evals
            return u0, f0, evals

        # pick best among {0, +s, -s}
        ub, fb = u0, f0
        tb = 0.0
        if fp < fb:
            ub, fb, tb = u_p, fp, +s
        if fm < fb:
            ub, fb, tb = u_m, fm, -s

        # If neither side improved, try smaller step quickly
        if fb >= f0:
            s2 = 0.35 * s
            if s2 < 1e-12:
                return u0, f0, evals
            u_p2 = add_scaled(u0, d, +s2)
            fp2, u_p2 = eval_u(u_p2); evals += 1
            if now() >= deadline:
                return (u_p2, fp2, evals) if fp2 < f0 else (u0, f0, evals)
            u_m2 = add_scaled(u0, d, -s2)
            fm2, u_m2 = eval_u(u_m2); evals += 1
            if fp2 < f0 and fp2 <= fm2:
                return u_p2, fp2, evals
            if fm2 < f0 and fm2 < fp2:
                return u_m2, fm2, evals
            return u0, f0, evals

        # We have an improving direction; perform a few refinement steps around tb
        # using parabolic fit from three points: t1 < t2 < t3
        # Build initial triple around tb with spacing s.
        # Ensure we have three points evaluated: reuse f0, fp, fm as needed.
        # Map known points:
        pts = {0.0: (f0, u0)}
        pts[+s] = (fp, u_p)
        pts[-s] = (fm, u_m)

        # helper to evaluate at t if needed
        def get_pt(t):
            nonlocal evals
            if t in pts:
                return pts[t]
            ut = add_scaled(u0, d, t)
            ft, ut = eval_u(ut)
            evals += 1
            pts[t] = (ft, ut)
            return ft, ut

        # expand a bit in the improving direction to get curvature if budget allows
        # (only if tb is on one side)
        if evals < max_evals and now() < deadline:
            t_ext = tb * 2.0
            if abs(t_ext) <= 1.5 * step0:
                ft_ext, ut_ext = get_pt(t_ext)
                if ft_ext < fb:
                    ub, fb, tb = ut_ext, ft_ext, t_ext

        # refinement iterations
        it = 0
        while evals < max_evals and now() < deadline and it < 6:
            it += 1
            # choose three points around current best tb
            cand_ts = sorted(pts.keys())
            # ensure we have neighbors around tb; if not, sample them
            # pick nearest left and right existing; if missing, create
            left = None
            right = None
            for t in cand_ts:
                if t < tb:
                    left = t
                elif t > tb and right is None:
                    right = t
            if left is None:
                left = tb - abs(step0)
                get_pt(left)
            if right is None:
                right = tb + abs(step0)
                get_pt(right)

            # for stability, enforce ordering
            t1, t2, t3 = left, tb, right
            f1, u1 = pts[t1]
            f2, u2 = pts[t2]
            f3, u3 = pts[t3]

            # parabolic minimizer for three points
            # formula using divided differences
            denom = (t1 - t2) * (t1 - t3) * (t2 - t3)
            if abs(denom) < 1e-20:
                # fallback: sample midpoint between best and better neighbor
                t_new = 0.5 * (t2 + (t1 if f1 < f3 else t3))
            else:
                a = (t3 * (f2 - f1) + t2 * (f1 - f3) + t1 * (f3 - f2)) / denom
                b = (t3 * t3 * (f1 - f2) + t2 * t2 * (f3 - f1) + t1 * t1 * (f2 - f3)) / denom
                if a <= 0.0:
                    # not convex -> fallback
                    t_new = 0.5 * (t2 + (t1 if f1 < f3 else t3))
                else:
                    t_new = -b / (2.0 * a)

            # keep t_new within [t1, t3] to avoid instability
            if t_new <= min(t1, t3) or t_new >= max(t1, t3):
                t_new = 0.5 * (t1 + t3)

            f_new, u_new = get_pt(t_new)
            if f_new < fb:
                ub, fb, tb = u_new, f_new, t_new
            else:
                # shrink bracket: replace the worse side
                if t_new < tb:
                    # left side
                    if f1 > f_new:
                        pts[t1] = (f_new, u_new)
                else:
                    if f3 > f_new:
                        pts[t3] = (f_new, u_new)

            # progressively reduce nominal step0
            step0 *= 0.75
            if step0 < 1e-12:
                break

        return ub, fb, evals

    # --- main state ---
    best = float("inf")
    best_u = [random.random() for _ in range(n)]
    best, best_u = eval_u(best_u)

    # --- initial global exploration ---
    init_budget = max(80, 30 * n)
    for _ in range(init_budget):
        if now() >= deadline:
            return best
        r = random.random()
        if r < 0.72:
            u = halton_u(hal_k); hal_k += 1
        elif r < 0.92:
            u = [random.random() for _ in range(n)]
        else:
            # heavy-tailed around incumbent
            u = [reflect01(best_u[i] + 0.18 * cauchy()) for i in range(n)]
        fx, ur = eval_u(u)
        if fx < best:
            best, best_u = fx, ur

    # --- Trust-region local search with evolving directions ---
    # Directions start as coordinate axes; later updated with random orthogonal-ish directions.
    dirs = [unit_vec(i) for i in range(n)]
    step = 0.18  # step in u-space
    min_step = 2e-10
    stall = 0

    # occasional global restart parameters
    global_inject_period = max(12, 4 * n)
    iters = 0

    while now() < deadline:
        iters += 1

        # global injection to avoid local traps
        if iters % global_inject_period == 0:
            # try a few global points quickly
            for _ in range(3):
                if now() >= deadline:
                    return best
                u = halton_u(hal_k); hal_k += 1
                fx, ur = eval_u(u)
                if fx < best:
                    best, best_u = fx, ur
                    stall = 0

        improved_round = False

        # shuffle directions; try line searches
        order = list(range(len(dirs)))
        random.shuffle(order)

        for k in order:
            if now() >= deadline:
                return best
            d = dirs[k]

            u_new, f_new, _ = line_search_parabolic(best_u, best, d, step, max_evals=10)
            if f_new + 0.0 < best:
                # accept
                prev_u = best_u
                best_u, best = u_new, f_new
                improved_round = True
                stall = 0

                # update one direction to approximate "conjugate" direction
                # (move direction = new - old)
                move = [best_u[i] - prev_u[i] for i in range(n)]
                mvn = norm(move)
                if mvn > 1e-12:
                    move = [move[i] / mvn for i in range(n)]
                    dirs[k] = move  # replace used direction with move direction

        if improved_round:
            # if improving, modestly increase step (but keep safe)
            step = min(0.45, step * 1.12)
        else:
            stall += 1
            step *= 0.62

            # if stuck, perturb directions and try a heavy-tailed jump
            if stall >= 3:
                stall = 0
                # refresh a couple directions
                for _ in range(min(2, n)):
                    j = random.randrange(len(dirs))
                    # random direction (Gaussian) normalized
                    v = [random.gauss(0.0, 1.0) for _ in range(n)]
                    vn = norm(v)
                    if vn > 1e-18:
                        dirs[j] = [vi / vn for vi in v]

                # jump
                u_jump = [reflect01(best_u[i] + 0.08 * random.gauss(0.0, 1.0) + 0.10 * cauchy())
                          for i in range(n)]
                fj, uj = eval_u(u_jump)
                if fj < best:
                    best, best_u = fj, uj
                    step = min(0.25, step * 1.4)

        # if step too small: restart trust region around best with some exploration
        if step < min_step:
            step = 0.12
            dirs = [unit_vec(i) for i in range(n)]
            # small exploration cloud around best
            for _ in range(min(10, 2 * n + 2)):
                if now() >= deadline:
                    return best
                u = [reflect01(best_u[i] + random.gauss(0.0, 0.06)) for i in range(n)]
                fx, ur = eval_u(u)
                if fx < best:
                    best, best_u = fx, ur

    return best
