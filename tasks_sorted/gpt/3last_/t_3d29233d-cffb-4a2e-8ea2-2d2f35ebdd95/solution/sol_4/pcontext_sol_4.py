import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs) using a stronger hybrid:

    Core ideas (all lightweight, time-aware):
      1) Better initialization: LHS-like + opposite points + corner probes
      2) Multi-start solver portfolio with bandit selection (UCB1):
           - Diagonal "CMA-ish" ES with mirrored sampling + weighted recombination
           - (1+1)-ES with 1/5 success rule (fast, good on smooth basins)
           - Heavy-tail (Cauchy) perturbations for escaping local minima
      3) Occasional coordinate/pattern refinement on global best
      4) Restarts on stagnation with diversified scales

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ---------------- helpers ----------------
    def clip(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        return float(func(x))

    def widths():
        return [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    W = widths()

    def rand_uniform_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
            else:
                y[i] = clip(lo + hi - x[i], lo, hi)
        return y

    def lhs_points(n):
        # Stratified per dimension (LHS-like)
        strata = []
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            strata.append(perm)
        pts = []
        for j in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    x[d] = lo
                else:
                    u = (strata[d][j] + random.random()) / n
                    x[d] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    def corners_subset(k):
        # Probe some corners (useful on many bounded problems).
        # If dim is large, sample k random corners.
        pts = []
        if dim <= 12:
            # Enumerate a limited set: take min(2^dim, k) corners deterministically
            total = 1 << dim
            step = max(1, total // max(1, k))
            for mask in range(0, total, step):
                x = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    x[i] = hi if ((mask >> i) & 1) else lo
                pts.append(x)
                if len(pts) >= k:
                    break
        else:
            for _ in range(k):
                x = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    x[i] = hi if random.getrandbits(1) else lo
                pts.append(x)
        return pts

    def cauchy_step(scale):
        # tan(pi*(u-0.5)) is Cauchy(0,1)
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    def pattern_refine(x0, f0, step, sweeps=2):
        # Cheap coordinate search + pattern move
        x = x0[:]
        fx = f0
        steps = step[:]
        for _ in range(sweeps):
            base = x[:]
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                si = steps[i]
                if si <= 0.0:
                    continue
                lo, hi = bounds[i]
                xi = x[i]

                xp = x[:]
                xp[i] = clip(xi + si, lo, hi)
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[i] = clip(xi - si, lo, hi)
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if improved:
                d = [x[i] - base[i] for i in range(dim)]
                xt = [clip(x[i] + d[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft
            else:
                for i in range(dim):
                    steps[i] *= 0.5
        return x, fx

    # ---------------- time ----------------
    start = time.time()
    deadline = start + float(max_time)

    # ---------------- init ----------------
    best = float("inf")
    best_x = None

    # Initial design: LHS + opposites + corners + randoms
    n0 = max(32, 8 * dim)
    pts = lhs_points(n0)
    pts += [opposite(p) for p in pts[:max(8, n0 // 3)]]
    pts += corners_subset(max(8, 2 * dim))
    for _ in range(max(8, dim)):
        pts.append(rand_uniform_vec())

    for x in pts:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_uniform_vec()
        best = eval_f(best_x)

    # ---------------- portfolio: 3 solvers ----------------
    # Solver A: diagonal CMA-ish ES state
    A_m = best_x[:]
    A_fm = best
    A_p = [0.0] * dim
    A_sigma = [w * 0.25 if w > 0 else 0.0 for w in W]
    A_lam = max(18, 10 + int(6 * math.log(dim + 1.0)))
    if A_lam % 2 == 1:
        A_lam += 1
    A_mu = max(4, A_lam // 3)
    A_w = [math.log(A_mu + 0.5) - math.log(i + 1.0) for i in range(A_mu)]
    s = sum(A_w)
    A_w = [wi / s for wi in A_w]
    A_cpath = 0.22
    A_target_succ = 0.20
    A_succ_ema = 0.0
    A_succ_decay = 0.90
    A_sig_adapt = 0.55
    A_sig_min_frac = 1e-10
    A_sig_max_frac = 0.60
    A_no_improve = 0

    # Solver B: (1+1)-ES state with 1/5 rule
    B_x = best_x[:]
    B_fx = best
    B_sigma = [w * 0.18 if w > 0 else 0.0 for w in W]
    B_succ = 0.0
    B_succ_decay = 0.92
    B_target = 0.20
    B_eta = 0.9  # exponent strength
    B_sig_min_frac = 1e-10
    B_sig_max_frac = 0.60
    B_no_improve = 0

    # Solver C: heavy-tail global perturbations around best (with mild anneal)
    C_T = 1.0
    C_T0 = 1.0
    C_no_improve = 0

    # Bandit (UCB1) selection over solvers
    # reward = positive improvement in global best
    nsol = 3
    pulls = [0] * nsol
    rew = [0.0] * nsol
    total_pulls = 0

    def ucb(i):
        if pulls[i] == 0:
            return float("inf")
        return (rew[i] / pulls[i]) + math.sqrt(2.0 * math.log(max(2, total_pulls)) / pulls[i])

    # controls
    it = 0
    refine_every = 18
    stagn_restart = 70  # iterations without global improvement triggers restart
    no_global = 0

    # ---------------- main loop ----------------
    while time.time() < deadline:
        it += 1

        # choose solver via UCB1, with a small epsilon exploration
        if random.random() < 0.08:
            si = random.randrange(nsol)
        else:
            scores = [ucb(i) for i in range(nsol)]
            si = max(range(nsol), key=lambda i: scores[i])

        before_best = best

        # -------- Solver A step (diag CMA-ish mirrored ES) --------
        if si == 0:
            improved_over_parent = False
            off = []
            # Mirrored sampling reduces noise / variance
            for _ in range(A_lam // 2):
                if time.time() >= deadline:
                    return best
                z = [random.gauss(0.0, 1.0) for _ in range(dim)]

                x1 = [0.0] * dim
                x2 = [0.0] * dim
                for j in range(dim):
                    lo, hi = bounds[j]
                    if hi <= lo:
                        x1[j] = lo
                        x2[j] = lo
                    else:
                        # path-biased sampling
                        step = (A_p[j] * 0.35 + z[j]) * A_sigma[j]
                        x1[j] = clip(A_m[j] + step, lo, hi)
                        x2[j] = clip(A_m[j] - step, lo, hi)

                f1 = eval_f(x1)
                f2 = eval_f(x2)
                off.append((f1, x1))
                off.append((f2, x2))
                if f1 < A_fm or f2 < A_fm:
                    improved_over_parent = True

            off.sort(key=lambda t: t[0])

            top = off[:A_mu]
            new_m = [0.0] * dim
            for j in range(dim):
                s = 0.0
                for k in range(A_mu):
                    s += A_w[k] * top[k][1][j]
                new_m[j] = clip(s, bounds[j][0], bounds[j][1])
            new_f = eval_f(new_m)

            best_off_f, best_off_x = off[0]
            if best_off_f <= new_f:
                cand_m, cand_f = best_off_x[:], best_off_f
            else:
                cand_m, cand_f = new_m, new_f

            old_m = A_m[:]
            if cand_f < A_fm:
                A_m, A_fm = cand_m[:], cand_f
                A_no_improve = 0
            else:
                # still move to best offspring
                A_m, A_fm = best_off_x[:], best_off_f
                A_no_improve += 1

            # update evolution path
            for j in range(dim):
                sj = A_sigma[j]
                stepn = ((A_m[j] - old_m[j]) / sj) if sj > 0.0 else 0.0
                A_p[j] = (1.0 - A_cpath) * A_p[j] + A_cpath * stepn

            # success-based sigma adaptation
            succ = 1.0 if improved_over_parent else 0.0
            A_succ_ema = A_succ_decay * A_succ_ema + (1.0 - A_succ_decay) * succ
            mult = math.exp(A_sig_adapt * (A_succ_ema - A_target_succ))
            for j in range(dim):
                wj = W[j]
                if wj <= 0.0:
                    A_sigma[j] = 0.0
                    continue
                A_sigma[j] *= mult
                smin = wj * A_sig_min_frac
                smax = wj * A_sig_max_frac
                if A_sigma[j] < smin: A_sigma[j] = smin
                elif A_sigma[j] > smax: A_sigma[j] = smax

            # global best update
            if A_fm < best:
                best, best_x = A_fm, A_m[:]
                no_global = 0
            else:
                no_global += 1

        # -------- Solver B step ((1+1)-ES 1/5 rule) --------
        elif si == 1:
            # propose
            y = B_x[:]
            for j in range(dim):
                lo, hi = bounds[j]
                if hi <= lo:
                    y[j] = lo
                else:
                    y[j] = clip(y[j] + random.gauss(0.0, B_sigma[j]), lo, hi)
            fy = eval_f(y)

            if fy < B_fx:
                B_x, B_fx = y, fy
                B_succ = B_succ_decay * B_succ + (1.0 - B_succ_decay) * 1.0
                B_no_improve = 0
            else:
                B_succ = B_succ_decay * B_succ
                B_no_improve += 1

            # 1/5-like adaptation
            # If success rate > target => increase, else decrease
            mult = math.exp(B_eta * (B_succ - B_target))
            for j in range(dim):
                wj = W[j]
                if wj <= 0.0:
                    B_sigma[j] = 0.0
                    continue
                B_sigma[j] *= mult
                smin = wj * B_sig_min_frac
                smax = wj * B_sig_max_frac
                if B_sigma[j] < smin: B_sigma[j] = smin
                elif B_sigma[j] > smax: B_sigma[j] = smax

            if B_fx < best:
                best, best_x = B_fx, B_x[:]
                no_global = 0
            else:
                no_global += 1

        # -------- Solver C step (heavy-tail global shot around best) --------
        else:
            # annealed heavy-tail around current global best
            scale = [max(w * 1e-6, (0.35 * w) * (0.25 + 0.75 * C_T)) for w in W]
            y = best_x[:]
            for j in range(dim):
                lo, hi = bounds[j]
                if hi <= lo:
                    y[j] = lo
                else:
                    step = cauchy_step(scale[j]) if random.random() < 0.40 else random.gauss(0.0, scale[j])
                    y[j] = clip(y[j] + step, lo, hi)
            fy = eval_f(y)

            if fy < best:
                best, best_x = fy, y[:]
                no_global = 0
                C_T = C_T0
                C_no_improve = 0
            else:
                no_global += 1
                C_no_improve += 1
                C_T *= 0.97

        # -------- bandit reward update --------
        improvement = max(0.0, before_best - best)
        pulls[si] += 1
        total_pulls += 1
        rew[si] += improvement

        # -------- occasional refinement --------
        if (it % refine_every) == 0 and time.time() < deadline and best_x is not None:
            # step sizes: mix of current sigmas and a small fraction of box width
            # use the more exploitative of A/B if available
            step = [0.0] * dim
            for j in range(dim):
                wj = W[j]
                base = wj * 0.02
                sA = A_sigma[j] if j < len(A_sigma) else 0.0
                sB = B_sigma[j] if j < len(B_sigma) else 0.0
                step[j] = max(base, min(wj * 0.25, max(sA, sB)))
            rx, rf = pattern_refine(best_x, best, step, sweeps=2)
            if rf < best:
                best, best_x = rf, rx[:]
                # re-anchor solvers on refined best
                A_m, A_fm = best_x[:], best
                B_x, B_fx = best_x[:], best
                for j in range(dim):
                    A_p[j] *= 0.3
                no_global = 0

        # -------- restart on stagnation --------
        if no_global >= stagn_restart and time.time() < deadline and best_x is not None:
            no_global = 0
            # diversify mean around best + random
            r = rand_uniform_vec()
            alpha = 0.80
            anchor = [clip(alpha * best_x[j] + (1.0 - alpha) * r[j], bounds[j][0], bounds[j][1])
                      for j in range(dim)]

            # reset solver states
            A_m = anchor[:]
            A_fm = eval_f(A_m)
            B_x = anchor[:]
            B_fx = A_fm
            C_T = C_T0

            # diversify sigmas
            g = 0.25 + 0.75 * random.random()
            for j in range(dim):
                wj = W[j]
                if wj <= 0.0:
                    A_sigma[j] = 0.0
                    B_sigma[j] = 0.0
                else:
                    A_sigma[j] = clip(wj * (0.10 + 0.40 * g), wj * 1e-6, wj * 0.55)
                    B_sigma[j] = clip(wj * (0.08 + 0.30 * g), wj * 1e-6, wj * 0.55)
                A_p[j] = 0.0
            if A_fm < best:
                best, best_x = A_fm, A_m[:]

    return best
