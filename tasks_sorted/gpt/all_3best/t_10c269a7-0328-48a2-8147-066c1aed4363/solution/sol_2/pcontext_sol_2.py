import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no numpy).

    Strategy (fast + robust):
      1) Diversified initialization: Halton + LHS-like stratified + random
      2) "SPSA-style" simultaneous perturbation local search (2 evals/step)
      3) Small trust-region coordinate/pattern refinement near the best
      4) Multi-start restarts with adaptive radius (scale shrinks on stagnation)

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ---------------- bounds / helpers ----------------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0  # keep math safe; variable effectively fixed

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

    avg_span = sum(spans) / float(dim) if dim > 0 else 1.0

    # ---------------- Halton (cheap LDS) ----------------
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

    primes = first_primes(dim) if dim > 0 else []

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

    # ---------------- LHS-like stratified points ----------------
    def stratified_points(n):
        # For each dim: n strata with jitter, independently permuted
        strata = []
        for i in range(dim):
            vals = [((j + random.random()) / n) for j in range(n)]
            random.shuffle(vals)
            strata.append([lows[i] + v * (highs[i] - lows[i]) for v in vals])
        pts = []
        for j in range(n):
            pts.append([strata[i][j] for i in range(dim)])
        return pts

    def rand_point():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # Keep init modest but effective
    n_lhs = max(10, min(50, 6 * dim))
    n_hal = max(10, min(80, 8 * dim))
    n_rnd = max(6, min(40, 3 * dim))

    init_pts = []
    if dim > 0:
        init_pts.extend(stratified_points(n_lhs))
        init_pts.extend(halton_point(k) for k in range(1, n_hal + 1))
        init_pts.extend(rand_point() for _ in range(n_rnd))
        # Add a couple of "corners-ish" points (sometimes helps)
        for _ in range(min(6, dim + 2)):
            x = [highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)]
            init_pts.append(x)
    else:
        init_pts = [[]]

    # Evaluate init points
    for x in init_pts:
        if time.time() >= deadline - eps_time:
            return best
        fx = eval_x(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        return best

    # ---------------- SPSA local search ----------------
    # SPSA uses two evaluations to approximate gradient along random +/-1 direction.
    # Works well in higher dim and noisy-ish objectives.
    # Step schedules (slower decay is better under short time budgets).
    a0 = 0.15 * avg_span
    c0 = 0.08 * avg_span
    a_min = 1e-14 * avg_span + 1e-18
    c_min = 1e-14 * avg_span + 1e-18

    def spsa_step(x, k, radius_scale):
        # a_k, c_k schedules
        ak = max(a_min, (a0 * radius_scale) / ((k + 10) ** 0.55))
        ck = max(c_min, (c0 * radius_scale) / ((k + 10) ** 0.12))

        # Rademacher direction (+1/-1)
        d = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

        x_plus = [clamp(x[i] + ck * d[i], i) for i in range(dim)]
        x_minus = [clamp(x[i] - ck * d[i], i) for i in range(dim)]

        f_plus = eval_x(x_plus)
        f_minus = eval_x(x_minus)

        # gradient estimate and update
        # g_i ~ (f+ - f-) / (2*c* d_i)
        diff = (f_plus - f_minus) / (2.0 * ck) if ck != 0.0 else 0.0
        x_new = [clamp(x[i] - ak * (diff * (1.0 / d[i])), i) for i in range(dim)]

        f_new = eval_x(x_new)

        # Return best among tried points to be safe
        fx = eval_x(x)
        best_local_f = fx
        best_local_x = x
        if f_plus < best_local_f:
            best_local_f, best_local_x = f_plus, x_plus
        if f_minus < best_local_f:
            best_local_f, best_local_x = f_minus, x_minus
        if f_new < best_local_f:
            best_local_f, best_local_x = f_new, x_new
        return best_local_f, best_local_x[:]

    # ---------------- small pattern refinement ----------------
    def refine_coordinate(x0, f0, base_scale):
        x = x0[:]
        fx = f0
        # few rounds, cheap
        step = [base_scale * spans[i] for i in range(dim)]
        for _ in range(2):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - eps_time:
                    return fx, x
                si = step[i]
                if si <= 0:
                    continue
                xi = x[i]
                # try +/- si
                cand = x[:]
                cand[i] = clamp(xi + si, i)
                fc = eval_x(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
                    continue
                cand = x[:]
                cand[i] = clamp(xi - si, i)
                fc = eval_x(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
            # shrink
            step = [v * (0.55 if improved else 0.35) for v in step]
            if not improved:
                break
        return fx, x

    # ---------------- main loop with restarts ----------------
    # Maintain a small set of "centers" (best-so-far and a few good inits).
    # This helps multi-modal problems under time limits.
    centers = [(best, best_x[:])]
    # seed additional centers from best init points (sample a few)
    random.shuffle(init_pts)
    for x in init_pts[:max(2, min(8, dim))]:
        if time.time() >= deadline - eps_time:
            return best
        fx = eval_x(x)
        centers.append((fx, x[:]))
        if fx < best:
            best, best_x = fx, x[:]
    centers.sort(key=lambda t: t[0])
    centers = centers[:max(3, min(10, 2 + dim // 2))]

    radius_scale = 1.0
    no_improve = 0
    restart_after = max(35, 8 * dim)

    k = 0
    while time.time() < deadline - eps_time:
        k += 1

        # pick center (biased to best but not exclusively)
        if random.random() < 0.70:
            x = best_x[:]
        else:
            x = centers[random.randrange(len(centers))][1][:]

        # occasional "kick" exploration around center
        if random.random() < 0.25:
            kick = 0.25 * radius_scale
            for i in range(dim):
                # uniform kick within scaled span
                x[i] = clamp(x[i] + (2.0 * random.random() - 1.0) * kick * spans[i], i)

        # SPSA step (uses 3 evals + a couple inside selection; but still compact)
        if time.time() >= deadline - eps_time:
            break
        f_try, x_try = spsa_step(x, k, radius_scale)

        if f_try < best:
            best, best_x = f_try, x_try[:]
            centers.append((best, best_x[:]))
            centers.sort(key=lambda t: t[0])
            centers = centers[:max(3, min(12, 3 + dim))]
            no_improve = 0
            # open radius slightly on success
            radius_scale = min(2.0, radius_scale * 1.08)
        else:
            no_improve += 1
            # shrink radius gradually when not improving
            if no_improve % (3 + dim // 3) == 0:
                radius_scale = max(0.05, radius_scale * 0.92)

        # cheap refinement near best sometimes
        if no_improve % max(18, 3 * dim) == 0 and time.time() < deadline - eps_time:
            f2, x2 = refine_coordinate(best_x, best, base_scale=0.06 * radius_scale)
            if f2 < best:
                best, best_x = f2, x2[:]
                centers.append((best, best_x[:]))
                centers.sort(key=lambda t: t[0])
                centers = centers[:max(3, min(12, 3 + dim))]
                no_improve = 0
                radius_scale = min(2.0, radius_scale * 1.05)

        # restart: add new center from Halton/random blended with best
        if no_improve >= restart_after and time.time() < deadline - eps_time:
            no_improve = 0
            radius_scale = max(0.08, radius_scale * 0.7)

            # create a restart point: mix best with diversified point
            if random.random() < 0.55:
                y = halton_point(random.randint(1, max(50, 12 * dim)))
            else:
                y = rand_point()

            a = 0.35 + 0.55 * random.random()  # bias towards best
            xr = [clamp(a * best_x[i] + (1.0 - a) * y[i], i) for i in range(dim)]
            fr = eval_x(xr)
            centers.append((fr, xr))
            centers.sort(key=lambda t: t[0])
            centers = centers[:max(3, min(12, 3 + dim))]
            if fr < best:
                best, best_x = fr, xr[:]

    return best
