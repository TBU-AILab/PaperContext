import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries):
      - Low-discrepancy (Halton) global sampling (space-filling)
      - “Best-of-batch” selection + trust-region local search
      - Adaptive per-dimension steps, mixed coordinate + random-direction proposals
      - Lightweight stagnation detection, diversification restarts
      - Safe evaluation (clamp to bounds, handle exceptions/NaN/inf)

    Returns:
        best (float): best objective value found within time limit
    """

    # ------------------------- utilities ------------------------- #
    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def sanitize_bounds(b):
        out = []
        for lo, hi in b:
            lo = float(lo); hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out

    def safe_eval(x):
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

    # ------------------------- local trust-region search ------------------------- #
    def local_search(x0, f0, t_end):
        x = x0[:]
        f = f0

        ranges = [bnds[i][1] - bnds[i][0] for i in range(dim)]
        # initial trust region (relative to domain)
        step = [0.15 * r for r in ranges]
        # minimal meaningful step
        min_step = [(1e-12 if r == 0 else max(1e-12, 1e-7 * r)) for r in ranges]
        # cap to avoid wild jumps
        max_step = [0.5 * r for r in ranges]

        # Pre-generate a small pool of unit random directions
        dirs = []
        for _ in range(10):
            v = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            nrm = math.sqrt(sum(t*t for t in v)) or 1.0
            dirs.append([t / nrm for t in v])

        no_improve_rounds = 0
        # A small budget per call; bounded by time as well
        eval_budget = 60 + 15 * dim
        evals = 0

        while evals < eval_budget and time.time() < t_end:
            improved = False

            # --- coordinate exploration (both signs) ---
            # Shuffle coordinate order to reduce bias
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if time.time() >= t_end or evals >= eval_budget:
                    break
                if step[i] <= 0:
                    continue

                xi = x[i]
                si = step[i]
                lo, hi = bnds[i]

                # Try + and - (choose best among them)
                cand_best = None

                xp = x[:]
                xp[i] = clamp(xi + si, lo, hi)
                fp, xp = safe_eval(xp)
                evals += 1
                cand_best = (fp, xp)

                if time.time() >= t_end or evals >= eval_budget:
                    # accept if better and stop
                    if cand_best[0] < f:
                        x, f = cand_best[1], cand_best[0]
                    break

                xm = x[:]
                xm[i] = clamp(xi - si, lo, hi)
                fm, xm = safe_eval(xm)
                evals += 1
                if fm < cand_best[0]:
                    cand_best = (fm, xm)

                if cand_best[0] < f:
                    x, f = cand_best[1], cand_best[0]
                    improved = True

            # --- random-direction step (helps non-separable problems) ---
            if time.time() < t_end and evals < eval_budget:
                # average step across active dims
                ssum = 0.0
                cnt = 0
                for i in range(dim):
                    if step[i] > 0:
                        ssum += step[i]
                        cnt += 1
                if cnt:
                    s = ssum / cnt
                    # try a couple of directions
                    for _ in range(2):
                        if time.time() >= t_end or evals >= eval_budget:
                            break
                        d = dirs[random.randrange(len(dirs))]
                        xt = [clamp(x[i] + s * d[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
                        ft, xt = safe_eval(xt)
                        evals += 1
                        if ft < f:
                            x, f = xt, ft
                            improved = True

            # --- adapt trust region ---
            if improved:
                no_improve_rounds = 0
                for i in range(dim):
                    if ranges[i] > 0:
                        step[i] = min(max_step[i], step[i] * 1.25)
            else:
                no_improve_rounds += 1
                for i in range(dim):
                    step[i] *= 0.55

                # If we're consistently stuck, stop early (caller can restart elsewhere)
                if no_improve_rounds >= 3:
                    all_tiny = True
                    for i in range(dim):
                        if step[i] > min_step[i]:
                            all_tiny = False
                            break
                    if all_tiny:
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

    bases = first_primes(dim)

    best = float("inf")
    best_x = None

    # Keep a small elite set from global samples for multi-start local search
    elite_k = max(3, min(12, 2 + dim))
    elite = []  # list of (f, x)

    def elite_add(f, x):
        nonlocal elite
        elite.append((f, x))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_k:
            elite.pop()

    # Phase 1: global low-discrepancy sampling
    # (use as many as time allows, but ensure we reach local search quickly)
    idx = 1
    global_cap = 30 + 15 * dim  # modest cap, then switch to intensification
    while idx <= global_cap and time.time() < deadline:
        u = halton_point(idx, bases)
        x = to_bounds(u)
        f, x = safe_eval(x)
        if f < best:
            best, best_x = f, x
        elite_add(f, x)
        idx += 1

    # Phase 2: iterate: choose a seed (elite-biased), run local search, then diversify
    restarts = 0
    while time.time() < deadline:
        now = time.time()
        remaining = deadline - now
        if remaining <= 0:
            break

        # Choose seed:
        #  - mostly from elite (bias toward best)
        #  - sometimes new Halton or random (diversify)
        pick = random.random()
        if elite and pick < 0.65:
            # biased pick: more weight to better ones
            # geometric-ish: pick among first few more often
            j = int((random.random() ** 2.0) * len(elite))
            x0 = elite[j][1][:]
            # small jitter to escape exact basin repeats
            for i in range(dim):
                lo, hi = bnds[i]
                r = hi - lo
                if r > 0:
                    x0[i] = clamp(x0[i] + random.uniform(-0.03, 0.03) * r, lo, hi)
            f0, x0 = safe_eval(x0)
        elif pick < 0.85:
            u = halton_point(idx, bases)
            idx += 1
            x0 = to_bounds(u)
            f0, x0 = safe_eval(x0)
        else:
            x0 = random_point()
            f0, x0 = safe_eval(x0)

        if f0 < best:
            best, best_x = f0, x0
        elite_add(f0, x0)

        # Give local search a controlled slice of remaining time
        # (more time early, less when close to deadline)
        slice_sec = max(0.01, min(0.35 * remaining, 0.25 + 0.03 * dim))
        t_end = min(deadline, time.time() + slice_sec)

        x1, f1 = local_search(x0, f0, t_end)
        if f1 < best:
            best, best_x = f1, x1
        elite_add(f1, x1)

        restarts += 1

        # Occasionally inject extra pure exploration if stagnating
        if restarts % 7 == 0 and time.time() < deadline:
            xg = random_point() if random.random() < 0.5 else to_bounds(halton_point(idx, bases))
            idx += 1
            fg, xg = safe_eval(xg)
            if fg < best:
                best, best_x = fg, xg
            elite_add(fg, xg)

    return best
