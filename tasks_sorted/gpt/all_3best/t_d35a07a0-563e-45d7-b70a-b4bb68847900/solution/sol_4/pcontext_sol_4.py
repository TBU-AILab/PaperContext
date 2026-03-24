import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no numpy).

    Improvements vs your current best:
      - Adds a compact surrogate (RBF) model over evaluated points to propose candidates cheaply.
      - Uses a trust-region around current best + adaptive per-dimension steps (diag sigma).
      - Keeps DE/current-to-best for robust global search, but allocates more budget to
        exploitation when the surrogate is confident.
      - Better bound handling (reflection + clamp) to avoid sticking on borders.
      - More disciplined evaluation budget per loop (generate many cheap candidates, evaluate few).

    Returns:
      best (float): best objective value found within max_time
    """

    # -------------------- helpers --------------------
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def reflect_into_bounds(v, lo, hi):
        # reflect (billiard) to reduce border sticking; then clamp for safety
        if lo == hi:
            return lo
        span = hi - lo
        x = v
        # shift into [0, 2*span) then reflect
        y = (x - lo) % (2.0 * span)
        if y > span:
            y = 2.0 * span - y
        return lo + y

    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    # Box-Muller Gaussian
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Halton sequence for seeding (low discrepancy)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(math.isqrt(k))
        p = 3
        while p <= r:
            if k % p == 0:
                return False
            p += 2
        return True

    def next_prime(n):
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    def pick3(n, exclude):
        # pick 3 distinct indices in [0,n) != exclude
        while True:
            a = random.randrange(n)
            if a != exclude:
                break
        while True:
            b = random.randrange(n)
            if b != exclude and b != a:
                break
        while True:
            c = random.randrange(n)
            if c != exclude and c != a and c != b:
                break
        return a, b, c

    # -------------------- tiny RBF surrogate --------------------
    # Uses a small set of centers (best and diverse) to keep it cheap.
    # Predicts objective with: f_hat(x) = sum_i w_i * exp(-gamma * ||x-ci||^2) + b
    # We fit weights by simple ridge-regularized normal equations via Gauss-Jordan.
    def rbf_fit(centers, y, gamma, ridge):
        m = len(centers)
        if m == 0:
            return None
        # Build A = [K 1; 1^T 0] with ridge on K diagonal
        # Solve for [w; b]
        # K_ij = exp(-gamma * ||ci-cj||^2)
        nvars = m + 1
        A = [[0.0] * nvars for _ in range(nvars)]
        rhs = [0.0] * nvars

        for i in range(m):
            rhs[i] = float(y[i])
            ci = centers[i]
            # row i
            for j in range(m):
                cj = centers[j]
                d2 = 0.0
                for k in range(dim):
                    t = ci[k] - cj[k]
                    d2 += t * t
                A[i][j] = math.exp(-gamma * d2)
            A[i][i] += ridge
            A[i][m] = 1.0  # b
        # constraint row
        for j in range(m):
            A[m][j] = 1.0
        A[m][m] = 0.0
        rhs[m] = 0.0

        # Gauss-Jordan elimination with partial pivoting
        # augmented matrix
        aug = [A[i] + [rhs[i]] for i in range(nvars)]
        for col in range(nvars):
            # pivot
            piv = col
            bestv = abs(aug[col][col])
            for r in range(col + 1, nvars):
                v = abs(aug[r][col])
                if v > bestv:
                    bestv = v
                    piv = r
            if bestv < 1e-14:
                return None
            if piv != col:
                aug[col], aug[piv] = aug[piv], aug[col]

            # normalize
            invp = 1.0 / aug[col][col]
            for c in range(col, nvars + 1):
                aug[col][c] *= invp

            # eliminate
            for r in range(nvars):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor == 0.0:
                    continue
                for c in range(col, nvars + 1):
                    aug[r][c] -= factor * aug[col][c]

        sol = [aug[i][nvars] for i in range(nvars)]
        w = sol[:m]
        b = sol[m]
        return (w, b)

    def rbf_predict(model, centers, x, gamma):
        if model is None:
            return float("inf")
        w, b = model
        s = b
        for i in range(len(centers)):
            ci = centers[i]
            d2 = 0.0
            for k in range(dim):
                t = x[k] - ci[k]
                d2 += t * t
            s += w[i] * math.exp(-gamma * d2)
        return s

    # -------------------- setup --------------------
    start = time.time()
    deadline = start + max(0.0, float(max_time) if max_time is not None else 0.0)
    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # population sizing
    pop_size = max(12, min(48, 6 * dim))
    elite_size = max(4, min(14, pop_size // 2))

    # per-dimension step (trust-region-ish) + global multiplier
    sigma_d = [0.20 * s for s in span]
    min_sigma_d = [1e-15 * s for s in span]
    max_sigma_d = [0.90 * s for s in span]
    sigma_g = 1.0

    # DE params
    F_base = 0.60
    CR_base = 0.85

    # adaptation bookkeeping (success rule)
    accepted = 0
    attempted = 0
    adapt_window = 40

    best = float("inf")
    best_x = None
    last_improve_t = start
    stagnation = max(0.30, 0.10 * float(max_time))

    # evaluation cache for surrogate
    # store (f, x)
    archive = []
    max_archive = max(80, 25 * dim)  # keep modest

    # -------------------- init (Halton + random + opposition) --------------------
    shift = [random.random() for _ in range(dim)]
    pop = []
    k = 1
    init_budget = max(pop_size, 14 * dim)

    i = 0
    while i < init_budget and time.time() < deadline:
        if (i % 4) == 0:
            x = rand_vec()
        else:
            x = halton_point(k, shift)
            k += 1

        fx = eval_f(x)
        pop.append((fx, x))
        archive.append((fx, list(x)))
        if fx < best:
            best, best_x = fx, list(x)
            last_improve_t = time.time()

        if time.time() >= deadline:
            break
        if random.random() < 0.55:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            pop.append((fxo, xo))
            archive.append((fxo, list(xo)))
            if fxo < best:
                best, best_x = fxo, list(xo)
                last_improve_t = time.time()

        i += 1

    if not pop:
        x = rand_vec()
        return eval_f(x)

    pop.sort(key=lambda t: t[0])
    pop = pop[:pop_size]
    best, best_x = pop[0][0], list(pop[0][1])

    # -------------------- main loop --------------------
    def sigma_too_small():
        cnt = 0
        for j in range(dim):
            if sigma_d[j] <= (5e-14 * span[j]):
                cnt += 1
        return cnt >= max(1, int(0.75 * dim))

    def coord_pattern(x, fx, step_scale):
        nonlocal best, best_x, last_improve_t
        m = 1 if dim == 1 else min(dim, 4)
        idxs = random.sample(range(dim), m)
        for j in idxs:
            lo, hi = bounds[j]
            step = step_scale * sigma_g * sigma_d[j]
            if step <= 0.0:
                continue

            xp = list(x)
            xp[j] = clamp(xp[j] + step, lo, hi)
            fp = eval_f(xp)
            if fp < fx:
                if fp < best:
                    best, best_x = fp, list(xp)
                    last_improve_t = time.time()
                return xp, fp

            xm = list(x)
            xm[j] = clamp(xm[j] - step, lo, hi)
            fm = eval_f(xm)
            if fm < fx:
                if fm < best:
                    best, best_x = fm, list(xm)
                    last_improve_t = time.time()
                return xm, fm
        return x, fx

    # surrogate settings (gamma based on dimension and typical span)
    avg_span = sum(span) / float(dim)
    # scale x to roughly comparable magnitude; gamma chosen so kernel isn't flat
    gamma = 1.0 / max(1e-12, (0.35 * avg_span) ** 2)
    ridge = 1e-8

    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0
        if deadline > start:
            tfrac = (now - start) / (deadline - start)
            if tfrac < 0.0:
                tfrac = 0.0
            elif tfrac > 1.0:
                tfrac = 1.0

        # maintain population
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_size]
        if pop[0][0] < best:
            best, best_x = pop[0][0], list(pop[0][1])
            last_improve_t = time.time()

        elites = pop[:elite_size]

        # restart if stagnating or sigma collapse
        if (now - last_improve_t) > stagnation or sigma_too_small():
            keep = elites[:]
            pop = keep[:]
            sigma_d = [0.28 * s for s in span]
            sigma_g = 1.0
            accepted = 0
            attempted = 0
            last_improve_t = now

            inject = max(3, pop_size // 2)
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.70:
                    xr = []
                    for j in range(dim):
                        lo, hi = bounds[j]
                        xr.append(clamp(best_x[j] + gauss() * (0.45 * span[j]), lo, hi))
                else:
                    xr = rand_vec()
                fr = eval_f(xr)
                pop.append((fr, xr))
                archive.append((fr, list(xr)))
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

            continue

        # ---- build surrogate occasionally (cheap) ----
        # Use top elites + a few random archive points for diversity
        model = None
        centers = None
        if len(archive) >= max(12, 3 * dim) and random.random() < (0.35 + 0.25 * tfrac):
            # choose centers
            archive.sort(key=lambda t: t[0])
            m = min(18, max(10, 2 * dim))
            centers = [list(archive[i][1]) for i in range(min(m // 2, len(archive)))]
            y = [float(archive[i][0]) for i in range(min(m // 2, len(archive)))]

            # add diverse random points
            need = m - len(centers)
            if need > 0:
                for _ in range(need):
                    fxr, xr = archive[random.randrange(len(archive))]
                    centers.append(list(xr))
                    y.append(float(fxr))

            model = rbf_fit(centers, y, gamma, ridge)

        # ---- generate candidate pool, evaluate the best few ----
        # more candidates late -> better use of surrogate; but still time-safe
        pool = 10 if dim <= 6 else 14
        pool = int(pool + 10 * tfrac)
        eval_top = 2 if tfrac < 0.5 else 3
        eval_top = min(eval_top, pool)

        # pick a parent
        idx = random.randrange(len(pop))
        fx, x = pop[idx]

        candidates = []

        # operator mix
        # A: DE/current-to-best, B: gaussian around best (TR), C: coord pattern, D: surrogate-guided random
        for _ in range(pool):
            r = random.random()
            if r < (0.28 * (1.0 - tfrac) + 0.12):
                op = "A"
            elif r < 0.72:
                op = "B"
            elif r < 0.88:
                op = "C"
            else:
                op = "D"

            cand = None

            if op == "A" and len(pop) >= 4 and best_x is not None:
                a, b, c = pick3(len(pop), idx)
                xb = pop[b][1]
                xc = pop[c][1]
                F = F_base + 0.30 * (random.random() - 0.5)
                if F < 0.25: F = 0.25
                if F > 0.95: F = 0.95
                CR = CR_base + 0.25 * (random.random() - 0.5)
                if CR < 0.10: CR = 0.10
                if CR > 0.98: CR = 0.98

                v = [0.0] * dim
                for j in range(dim):
                    lo, hi = bounds[j]
                    vj = x[j] + F * (best_x[j] - x[j]) + F * (xb[j] - xc[j])
                    if random.random() < (0.08 * (1.0 - tfrac)):
                        vj += gauss() * (0.02 * span[j])
                    v[j] = reflect_into_bounds(vj, lo, hi)

                u = [0.0] * dim
                jrand = random.randrange(dim)
                for j in range(dim):
                    if random.random() < CR or j == jrand:
                        u[j] = v[j]
                    else:
                        u[j] = x[j]
                cand = u

            elif op == "B" and best_x is not None:
                # trust-region gaussian around best; shrink over time
                shrink = 0.9 - 0.65 * tfrac
                y = [0.0] * dim
                for j in range(dim):
                    lo, hi = bounds[j]
                    step = gauss() * sigma_g * sigma_d[j] * shrink
                    yj = best_x[j] + step
                    y[j] = reflect_into_bounds(yj, lo, hi)
                cand = y

            elif op == "C":
                # make a coordinate move proposal without evaluating yet
                y = list(x)
                m = 1 if dim == 1 else min(dim, 3)
                idxs = random.sample(range(dim), m)
                step_scale = 0.85 - 0.55 * tfrac
                for j in idxs:
                    lo, hi = bounds[j]
                    step = step_scale * sigma_g * sigma_d[j]
                    if random.random() < 0.5:
                        yj = y[j] + step
                    else:
                        yj = y[j] - step
                    y[j] = reflect_into_bounds(yj, lo, hi)
                cand = y

            else:
                # surrogate-guided: random / near-best mix
                if best_x is not None and random.random() < (0.40 + 0.35 * tfrac):
                    y = [0.0] * dim
                    for j in range(dim):
                        lo, hi = bounds[j]
                        yj = best_x[j] + gauss() * (0.60 - 0.45 * tfrac) * span[j]
                        y[j] = reflect_into_bounds(yj, lo, hi)
                    cand = y
                else:
                    cand = rand_vec()

            # score by surrogate if available; otherwise random score
            if model is not None and centers is not None:
                score = rbf_predict(model, centers, cand, gamma)
            else:
                score = random.random()
            candidates.append((score, cand))

        candidates.sort(key=lambda t: t[0])

        # evaluate top few; accept best improving one (or best among them sometimes)
        improved = False
        best_trial = None
        for p in range(eval_top):
            if time.time() >= deadline:
                break
            cand = candidates[p][1]
            fc = eval_f(cand)
            attempted += 1
            archive.append((fc, list(cand)))
            if len(archive) > max_archive:
                # keep archive biased toward good points
                archive.sort(key=lambda t: t[0])
                archive = archive[:max_archive]

            if best_trial is None or fc < best_trial[0]:
                best_trial = (fc, cand)

            if fc <= fx:
                pop[idx] = (fc, cand)
                accepted += 1
                improved = True
                if fc < best:
                    best, best_x = fc, list(cand)
                    last_improve_t = time.time()
                break

        # if no immediate improvement, sometimes replace with best_trial if close (diversify)
        if (not improved) and best_trial is not None and random.random() < 0.10:
            fc, cand = best_trial
            if fc < pop[-1][0]:
                pop[-1] = (fc, cand)
                if fc < best:
                    best, best_x = fc, list(cand)
                    last_improve_t = time.time()

        # occasional direct coordinate refinement on the best late in time
        if tfrac > 0.65 and best_x is not None and random.random() < 0.22 and time.time() < deadline:
            xb = list(best_x)
            fb = float(best)
            xb2, fb2 = coord_pattern(xb, fb, 0.60)
            attempted += 1
            archive.append((fb2, list(xb2)))
            if fb2 < best:
                best, best_x = fb2, list(xb2)
                last_improve_t = time.time()

        # ---- adapt sigmas using success rate ----
        if attempted >= adapt_window:
            rate = accepted / float(attempted) if attempted > 0 else 0.0
            # target success ~0.2
            if rate > 0.26:
                sigma_g *= 1.22
                per = 1.10
            elif rate < 0.12:
                sigma_g *= 0.78
                per = 0.90
            else:
                per = 1.0

            # gentle time-based shrink to focus
            sigma_g *= (0.999 ** (1 + int(8 * tfrac)))

            if sigma_g < 0.06:
                sigma_g = 0.06
            elif sigma_g > 3.0:
                sigma_g = 3.0

            if per != 1.0:
                for j in range(dim):
                    sj = sigma_d[j] * per
                    if sj < min_sigma_d[j]:
                        sj = min_sigma_d[j]
                    elif sj > max_sigma_d[j]:
                        sj = max_sigma_d[j]
                    sigma_d[j] = sj

            accepted = 0
            attempted = 0

        # extra random injection early
        if random.random() < (0.014 * (1.0 - tfrac) + 0.002):
            xr = rand_vec()
            fr = eval_f(xr)
            archive.append((fr, list(xr)))
            pop.sort(key=lambda t: t[0])
            ridx = random.randrange(max(elite_size, 1), len(pop))
            if fr < pop[ridx][0] or random.random() < 0.20:
                pop[ridx] = (fr, xr)
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

    return best
