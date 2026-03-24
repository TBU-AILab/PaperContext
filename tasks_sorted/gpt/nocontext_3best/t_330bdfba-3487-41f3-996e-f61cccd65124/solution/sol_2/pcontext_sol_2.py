import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (self-contained, no numpy):

    Key improvements vs. the provided variants:
      - Halton LDS global sampling + *on-the-fly* spread control
      - Maintains a small elite set with duplicate suppression
      - Two-stage local search:
          (A) coordinate/pattern + random directions (fast)
          (B) 1D line-search along promising directions (cheap refinement)
      - Occasional “large jump” perturbations + adaptive restart logic
      - Careful time-slicing so local search never hogs the entire budget
      - Robust evaluation: clamps, handles exceptions/NaN/inf

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ------------------------- utilities ------------------------- #
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def sanitize_bounds(b):
        out = []
        for lo, hi in b:
            lo = float(lo); hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out

    def safe_eval(x):
        # clamp + safe objective call
        xx = [clamp(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
        try:
            y = func(xx)
            if y is None:
                return float("inf"), xx
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf"), xx
            return y, xx
        except Exception:
            return float("inf"), xx

    # ------------------------- Halton sequence ------------------------- #
    def first_primes(k):
        primes = []
        n = 2
        while len(primes) < k:
            is_p = True
            r = int(n ** 0.5)
            for p in primes:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(n)
            n += 1
        return primes

    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index, bases):
        return [vdc(index, b) for b in bases]

    def to_bounds(u):
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi == lo:
                x.append(lo)
            else:
                x.append(lo + u[i] * (hi - lo))
        return x

    def random_point():
        return [random.uniform(bnds[i][0], bnds[i][1]) for i in range(dim)]

    # ------------------------- vector helpers ------------------------- #
    def norm2(v):
        return sum(t * t for t in v)

    def add_scaled(x, d, s):
        return [x[i] + s * d[i] for i in range(dim)]

    def random_unit_dir():
        v = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        n = math.sqrt(norm2(v)) or 1.0
        return [t / n for t in v]

    # ------------------------- elite set ------------------------- #
    # store a few best solutions; avoid near-duplicates by coarse rounding in normalized space
    def key_of(x):
        # normalize to [0,1] and round coarsely
        k = []
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            if r <= 0:
                k.append(0)
            else:
                u = (x[i] - lo) / r
                # coarse grid: 1e-3 resolution
                k.append(int(u * 1000.0 + 0.5))
        return tuple(k)

    # ------------------------- local search ------------------------- #
    def line_search_1d(x, f, d, base_step, t_end):
        """
        Very small bracketing + ternary-ish refinement on a 1D line.
        Evaluations are bounded and time-checked.
        """
        if time.time() >= t_end:
            return x, f

        # Try a few candidate scales around base_step (both signs)
        # Keep it tiny: this often gives a meaningful gain for little cost.
        scales = [0.0, 1.0, -1.0, 2.5, -2.5, 6.0, -6.0]
        best_x = x
        best_f = f

        for sc in scales:
            if time.time() >= t_end:
                break
            xt = add_scaled(x, d, sc * base_step)
            ft, xt = safe_eval(xt)
            if ft < best_f:
                best_f, best_x = ft, xt

        # If we didn't move, no need to refine
        if best_x is x:
            return x, f

        # Small refinement around the best scale using shrinking bracket
        # (keep evaluation count low)
        # Find approximate best scale among tested
        # Reconstruct best scale:
        best_sc = None
        for sc in scales:
            xt = add_scaled(x, d, sc * base_step)
            # compare by key (since safe_eval clamps)
            if key_of(xt) == key_of(best_x):
                best_sc = sc
                break
        if best_sc is None:
            best_sc = 1.0

        left = best_sc - 1.0
        right = best_sc + 1.0
        for _ in range(4):  # 4 rounds = cheap
            if time.time() >= t_end:
                break
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            x1 = add_scaled(x, d, m1 * base_step)
            f1, x1 = safe_eval(x1)
            x2 = add_scaled(x, d, m2 * base_step)
            f2, x2 = safe_eval(x2)
            if f1 < best_f:
                best_f, best_x = f1, x1
            if f2 < best_f:
                best_f, best_x = f2, x2
            if f1 < f2:
                right = m2
            else:
                left = m1

        return best_x, best_f

    def local_search(x0, f0, t_end):
        x = x0[:]
        f = f0

        ranges = [bnds[i][1] - bnds[i][0] for i in range(dim)]
        # Trust region steps
        step = [0.18 * r for r in ranges]
        min_step = [(1e-12 if r == 0 else max(1e-12, 5e-8 * r)) for r in ranges]
        max_step = [0.6 * r for r in ranges]

        # Direction pool
        dir_pool = [random_unit_dir() for _ in range(10)]

        # Budget: scales with dim but still time bounded
        eval_budget = 80 + 18 * dim
        evals = 0
        stuck_rounds = 0

        while evals < eval_budget and time.time() < t_end:
            improved = False

            # --- coordinate pattern moves (best-of +/-) ---
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if time.time() >= t_end or evals >= eval_budget:
                    break
                si = step[i]
                if si <= 0.0:
                    continue

                lo, hi = bnds[i]
                xi = x[i]

                xp = x[:]
                xp[i] = clamp(xi + si, lo, hi)
                fp, xp = safe_eval(xp); evals += 1

                xm = x[:]
                xm[i] = clamp(xi - si, lo, hi)
                fm, xm = safe_eval(xm); evals += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    improved = True

            # --- random direction probes ---
            if time.time() < t_end and evals < eval_budget:
                # average active step
                ssum = 0.0
                cnt = 0
                for i in range(dim):
                    if step[i] > 0.0:
                        ssum += step[i]
                        cnt += 1
                if cnt:
                    s = ssum / cnt
                    # 2-3 random directions
                    for _ in range(3):
                        if time.time() >= t_end or evals >= eval_budget:
                            break
                        d = dir_pool[random.randrange(len(dir_pool))]
                        xt = add_scaled(x, d, s)
                        ft, xt = safe_eval(xt); evals += 1
                        if ft < f:
                            x, f = xt, ft
                            improved = True

            # --- cheap 1D refinement along last-improvement-like direction ---
            # Try a few line searches from current x on a couple directions.
            if time.time() < t_end and evals < eval_budget:
                # use smaller base step for refinement
                base = 0.0
                cnt = 0
                for i in range(dim):
                    if step[i] > 0.0:
                        base += step[i]
                        cnt += 1
                if cnt:
                    base /= cnt
                    # two line searches maximum
                    for _ in range(2):
                        if time.time() >= t_end or evals >= eval_budget:
                            break
                        d = dir_pool[random.randrange(len(dir_pool))]
                        x2, f2 = line_search_1d(x, f, d, 0.65 * base, t_end)
                        # line_search calls safe_eval internally; not counted in evals
                        # (we keep eval_budget as a soft limiter; time is the true limiter)
                        if f2 < f:
                            x, f = x2, f2
                            improved = True

            # --- adapt trust region ---
            if improved:
                stuck_rounds = 0
                for i in range(dim):
                    if ranges[i] > 0:
                        step[i] = min(max_step[i], step[i] * 1.22)
            else:
                stuck_rounds += 1
                for i in range(dim):
                    step[i] *= 0.55

                if stuck_rounds >= 3:
                    # stop if all steps tiny
                    tiny = True
                    for i in range(dim):
                        if step[i] > min_step[i]:
                            tiny = False
                            break
                    if tiny:
                        break

        return x, f

    # ------------------------- main ------------------------- #
    if dim <= 0:
        return float("inf")
    bnds = sanitize_bounds(bounds)
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    start = time.time()
    deadline = start + float(max_time)

    # Precompute
    bases = first_primes(dim)

    best = float("inf")
    best_x = None

    elite_k = max(6, min(18, 4 + 2 * dim))
    elite = []          # list of (f, x)
    elite_keys = set()  # duplicate suppression

    def elite_add(f, x):
        k = key_of(x)
        if k in elite_keys:
            # still allow if it's strictly better than current best (rare)
            if f >= best:
                return
        elite.append((f, x))
        elite.sort(key=lambda t: t[0])
        # rebuild key set after trimming
        if len(elite) > elite_k:
            elite[:] = elite[:elite_k]
        elite_keys.clear()
        for ff, xx in elite:
            elite_keys.add(key_of(xx))

    # --- Phase 1: LDS global sampling ---
    idx = 1
    # spend about ~20% time (but ensure some minimum points)
    # (no sleeps; just a cap and time checks)
    global_cap = 50 + 18 * dim
    while idx <= global_cap and time.time() < deadline:
        u = halton_point(idx, bases)
        x = to_bounds(u)
        f, x = safe_eval(x)
        if f < best:
            best, best_x = f, x
        elite_add(f, x)
        idx += 1

    # --- Phase 2: iterative multi-start local search ---
    restarts = 0
    no_best_improve = 0
    last_best = best

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        # Choose seed:
        #  - mostly from elite (exploit)
        #  - sometimes new Halton/random (explore)
        r = random.random()

        if elite and r < 0.62:
            # biased towards best
            j = int((random.random() ** 2.3) * len(elite))
            x0 = elite[j][1][:]
            # jitter proportional to domain; larger if stagnating
            jitter = 0.02 + 0.03 * min(1.0, no_best_improve / 8.0)
            for i in range(dim):
                lo, hi = bnds[i]
                rr = hi - lo
                if rr > 0:
                    x0[i] = clamp(x0[i] + random.uniform(-jitter, jitter) * rr, lo, hi)
            f0, x0 = safe_eval(x0)
        elif r < 0.86:
            u = halton_point(idx, bases); idx += 1
            x0 = to_bounds(u)
            f0, x0 = safe_eval(x0)
        else:
            x0 = random_point()
            f0, x0 = safe_eval(x0)

        if f0 < best:
            best, best_x = f0, x0
        elite_add(f0, x0)

        # time slice for local search:
        # - small but non-trivial; slightly more when stagnating
        stagn = min(1.0, no_best_improve / 10.0)
        slice_sec = 0.10 + 0.05 * dim
        slice_sec *= (1.0 + 0.7 * stagn)
        slice_sec = min(slice_sec, 0.33 * remaining)
        slice_sec = max(0.01, slice_sec)
        t_end = min(deadline, time.time() + slice_sec)

        x1, f1 = local_search(x0, f0, t_end)
        if f1 < best:
            best, best_x = f1, x1
        elite_add(f1, x1)

        # occasional big-jump perturbation from best to escape basins
        restarts += 1
        if restarts % 6 == 0 and best_x is not None and time.time() < deadline:
            jump = 0.10 + 0.20 * random.random()  # 10%..30% of range
            xj = best_x[:]
            for i in range(dim):
                lo, hi = bnds[i]
                rr = hi - lo
                if rr > 0:
                    xj[i] = clamp(xj[i] + random.uniform(-jump, jump) * rr, lo, hi)
            fj, xj = safe_eval(xj)
            if fj < best:
                best, best_x = fj, xj
            elite_add(fj, xj)

        # stagnation bookkeeping
        if best < last_best - 1e-15:
            last_best = best
            no_best_improve = 0
        else:
            no_best_improve += 1

        # if severely stagnating, inject extra exploration (LDS + random)
        if no_best_improve >= 10 and time.time() < deadline:
            for _ in range(2):
                if time.time() >= deadline:
                    break
                if random.random() < 0.5:
                    u = halton_point(idx, bases); idx += 1
                    xe = to_bounds(u)
                else:
                    xe = random_point()
                fe, xe = safe_eval(xe)
                if fe < best:
                    best, best_x = fe, xe
                elite_add(fe, xe)
            no_best_improve = 6  # reduce but don't fully reset

    return best
