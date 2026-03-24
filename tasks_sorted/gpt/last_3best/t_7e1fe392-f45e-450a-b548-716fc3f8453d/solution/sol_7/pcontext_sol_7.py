import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Key changes vs the last (worse) variant:
      - Switch to a robust "multi-start local search" backbone:
          * strong space-filling init (scrambled Halton + LHS)
          * large-neighborhood exploration via an adaptive ES around elites
          * deterministic local exploitation via Powell-style direction set search
            with bracketing + golden-section line search (bounded + reflect)
      - Remove heavy/fragile caching quantization (can harm when objective is rugged);
        instead do a tiny exact-cache keyed by rounded normalized coords (coarser, safer).
      - Use time-aware scheduling: more global early, more local late.
      - Better restart logic: stagnation triggers bigger exploration and fresh seeds.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        return float("inf")

    # -------------------- bounds / scaling --------------------
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]
    inv_scale = [1.0 / s for s in scale]

    def reflect(v, a, b):
        if a == b:
            return a
        w = b - a
        if w <= 0.0:
            return a
        z = (v - a) % (2.0 * w)
        if z > w:
            z = 2.0 * w - z
        return a + z

    def fix(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = reflect(x[i], lo[i], hi[i])
        return y

    # -------------------- evaluation (robust + tiny cache) --------------------
    # cache key: rounded normalized coordinates (coarse) to avoid too many dups,
    # but not so aggressive that it blocks exploration.
    cache = {}
    def cache_key(x):
        k = []
        for i in range(dim):
            if span[i] <= 0.0:
                k.append(0)
            else:
                u = (x[i] - lo[i]) * inv_scale[i]
                if u < 0.0: u = 0.0
                if u > 1.0: u = 1.0
                k.append(int(u * 200000))  # 2e5 buckets
        return tuple(k)

    def evaluate(x):
        x = fix(x)
        k = cache_key(x)
        if k in cache:
            return cache[k], x
        try:
            v = func(x)
            if v is None:
                v = float("inf")
            v = float(v)
            if not math.isfinite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v, x

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    # -------------------- low-discrepancy init: scrambled Halton + LHS --------------------
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(dim)
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def halton_scrambled(index, base):
        f = 1.0
        r = 0.0
        i = index
        perm = digit_perm[base]
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_scrambled(k, primes[i])
            x[i] = lo[i] + u * (hi[i] - lo[i])
        return x

    def lhs_batch(n):
        per_dim = []
        for i in range(dim):
            arr = [(k + random.random()) / n for k in range(n)]
            random.shuffle(arr)
            per_dim.append(arr)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                x[i] = lo[i] + per_dim[i][k] * (hi[i] - lo[i])
            pts.append(x)
        return pts

    # -------------------- elite archive --------------------
    def dist_norm_l1(a, b):
        s = 0.0
        for i in range(dim):
            s += abs(a[i] - b[i]) * inv_scale[i]
        return s

    def elite_insert(elites, x, fx, max_elites, dedup_eps=1e-9):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        out = []
        for f, p in elites:
            ok = True
            for _, q in out:
                if dist_norm_l1(p, q) < dedup_eps:
                    ok = False
                    break
            if ok:
                out.append((f, p))
            if len(out) >= max_elites:
                break
        return out

    # -------------------- (1+λ)-ES local in normalized space --------------------
    def to_u(x):
        u = [0.5] * dim
        for i in range(dim):
            if span[i] > 0.0:
                t = (x[i] - lo[i]) * inv_scale[i]
                if t < 0.0: t = 0.0
                if t > 1.0: t = 1.0
                u[i] = t
        return u

    def from_u(u):
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] + u[i] * (hi[i] - lo[i])
        return x

    def es_local(x0, f0, sigma_u, lam, iters):
        u = to_u(x0)
        fu = f0
        for _ in range(iters):
            if time.time() >= deadline:
                break
            best_fu = fu
            best_u = None
            for _k in range(lam):
                if time.time() >= deadline:
                    break
                cand = [0.0] * dim
                for i in range(dim):
                    v = u[i] + random.gauss(0.0, sigma_u)
                    # reflect into [0,1]
                    if v < 0.0 or v > 1.0:
                        v = (v % 2.0)
                        if v > 1.0:
                            v = 2.0 - v
                    cand[i] = v
                fx, xfix = evaluate(from_u(cand))
                if fx < best_fu:
                    best_fu = fx
                    best_u = to_u(xfix)
            if best_u is not None:
                u, fu = best_u, best_fu
                sigma_u *= 1.12
            else:
                sigma_u *= 0.88
            if sigma_u < 1e-7: sigma_u = 1e-7
            if sigma_u > 0.6:  sigma_u = 0.6
        return from_u(u), fu

    # -------------------- Powell-style direction set local search --------------------
    # Bounded line search along direction d, using bracketing + golden section.
    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def add_scaled(x, d, alpha):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = x[i] + alpha * d[i]
        return y

    def dir_norm(d):
        return math.sqrt(max(0.0, dot(d, d)))

    def line_search(x, fx, d, step0):
        if time.time() >= deadline:
            return x, fx
        nd = dir_norm(d)
        if nd <= 0.0:
            return x, fx

        # Normalize direction in scaled space to improve conditioning
        dd = [0.0] * dim
        for i in range(dim):
            dd[i] = d[i] / (nd + 1e-300)

        # Objective along line (uses reflect via evaluate)
        def phi(a):
            fa, xa = evaluate(add_scaled(x, dd, a))
            return fa, xa

        # Bracket minimum around 0 using expanding steps
        a0 = 0.0
        f0 = fx
        best_x = x
        best_f = fx

        # try both directions
        a1 = step0
        f1, x1 = phi(a1)
        if f1 < best_f:
            best_f, best_x = f1, x1

        a_1 = -step0
        f_1, x_1 = phi(a_1)
        if f_1 < best_f:
            best_f, best_x = f_1, x_1

        # pick downhill side to expand
        if f1 >= f0 and f_1 >= f0:
            return best_x, best_f

        if f1 < f_1:
            aL, fL = a0, f0
            aM, fM = a1, f1
            direction = 1.0
        else:
            aL, fL = a0, f0
            aM, fM = a_1, f_1
            direction = -1.0

        # expand to find aR with fR > fM
        step = step0
        aR = aM + direction * step
        fR, xR = phi(aR)
        if fR < best_f:
            best_f, best_x = fR, xR

        expand = 0
        while fR < fM and expand < 12 and time.time() < deadline:
            aL, fL = aM, fM
            aM, fM = aR, fR
            step *= 1.7
            aR = aM + direction * step
            fR, xR = phi(aR)
            if fR < best_f:
                best_f, best_x = fR, xR
            expand += 1

        # Golden section search on [aL, aR] with minimum near aM
        left = min(aL, aR)
        right = max(aL, aR)
        # keep it short (time-bounded)
        iters = 10
        gr = 0.6180339887498949
        c = right - gr * (right - left)
        d2 = left + gr * (right - left)
        fc, xc = phi(c)
        fd, xd = phi(d2)
        if fc < best_f:
            best_f, best_x = fc, xc
        if fd < best_f:
            best_f, best_x = fd, xd

        for _ in range(iters):
            if time.time() >= deadline:
                break
            if fc < fd:
                right = d2
                d2 = c
                fd, xd = fc, xc
                c = right - gr * (right - left)
                fc, xc = phi(c)
                if fc < best_f:
                    best_f, best_x = fc, xc
            else:
                left = c
                c = d2
                fc, xc = fd, xd
                d2 = left + gr * (right - left)
                fd, xd = phi(d2)
                if fd < best_f:
                    best_f, best_x = fd, xd

        return best_x, best_f

    def powell_local(x0, f0, passes, step_frac):
        x = list(x0)
        fx = f0

        # initial directions: coordinate axes scaled by bounds span
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = scale[i]
            dirs.append(d)

        base_step = step_frac * (sum(scale) / max(1, dim))
        if base_step <= 0.0:
            base_step = 1e-6

        for _p in range(passes):
            if time.time() >= deadline:
                break
            x_start = x
            f_start = fx

            biggest_drop = 0.0
            biggest_i = -1

            for i, d in enumerate(dirs):
                if time.time() >= deadline:
                    break
                x_new, f_new = line_search(x, fx, d, step0=base_step)
                drop = fx - f_new
                if drop > biggest_drop:
                    biggest_drop = drop
                    biggest_i = i
                x, fx = x_new, f_new

            # update direction set with "net move"
            net = [x[i] - x_start[i] for i in range(dim)]
            if biggest_i >= 0 and dir_norm(net) > 0.0:
                # replace the direction that gave biggest improvement
                dirs[biggest_i] = net

            # stop if no progress
            if f_start - fx <= 1e-12 * (abs(f_start) + 1.0):
                break

            base_step *= 0.7

        return x, fx

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None
    elites = []
    max_elites = max(10, min(60, 4 * dim))

    hal_start = 1 + random.randrange(256)
    n_hal = max(24, min(140, 10 * dim))
    n_lhs = max(10, min(60, 4 * dim))
    n_rand = max(6, min(30, 2 * dim))

    for k in range(n_hal):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        x = halton_point(hal_start + k)
        fx, x = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

    for x in lhs_batch(n_lhs):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        fx, x = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

    for _ in range(n_rand):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        x = rand_uniform()
        fx, x = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

    if best_x is None:
        x = rand_uniform()
        best, best_x = evaluate(x)[0], x
        elites = [(best, best_x)]

    # -------------------- main loop --------------------
    no_improve = 0
    phase = 0
    while time.time() < deadline:
        phase += 1
        m = len(elites) if elites else 1
        if not elites:
            elites = [(best, best_x)]

        # time-aware exploration/exploitation
        remaining = max(0.0, deadline - time.time())
        frac_left = remaining / max(1e-9, max_time)

        # pick elite (best-biased)
        r = random.random()
        idx = int((r * r) * m)
        fx0, x0 = elites[idx]

        improved = False

        # 1) ES exploration (more when lots of time left / stagnating)
        if time.time() < deadline:
            stagn = 1.0 + min(3.0, no_improve / 15.0)
            sigma_u = (0.18 * frac_left + 0.03) * stagn
            lam = min(36, 8 + dim)
            iters = 1 if frac_left < 0.25 else 2
            x1, f1 = es_local(x0, fx0, sigma_u=sigma_u, lam=lam, iters=iters)
            elites = elite_insert(elites, x1, f1, max_elites)
            if f1 < best:
                best, best_x = f1, x1
                improved = True

        # 2) Powell local exploitation (more later / near best)
        if time.time() < deadline:
            # run locally from best or top elite
            base = best_x if random.random() < 0.7 else elites[random.randrange(min(len(elites), max(3, dim)))][1]
            fbase = best if base is best_x else evaluate(base)[0]
            passes = 1 if frac_left > 0.6 else 2
            step_frac = 0.12 * frac_left + 0.01
            x2, f2 = powell_local(base, fbase, passes=passes, step_frac=step_frac)
            elites = elite_insert(elites, x2, f2, max_elites)
            if f2 < best:
                best, best_x = f2, x2
                improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        # 3) Restart injection if stuck
        if no_improve >= 18 and time.time() < deadline:
            # mixture of global points
            batch = max(6, min(24, 2 * dim))
            if random.random() < 0.6:
                pts = lhs_batch(batch)
            else:
                pts = [halton_point(hal_start + n_hal + phase * 7 + j) for j in range(batch)]
            for xg in pts:
                if time.time() >= deadline:
                    break
                fg, xg = evaluate(xg)
                elites = elite_insert(elites, xg, fg, max_elites)
                if fg < best:
                    best, best_x = fg, xg
                    no_improve = 0

            # also add a pure random point occasionally
            if time.time() < deadline:
                xr = rand_uniform()
                fr, xr = evaluate(xr)
                elites = elite_insert(elites, xr, fr, max_elites)
                if fr < best:
                    best, best_x = fr, xr
                    no_improve = 0

    return best
