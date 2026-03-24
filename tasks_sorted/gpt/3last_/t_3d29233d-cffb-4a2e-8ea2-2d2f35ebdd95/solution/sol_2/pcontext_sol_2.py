import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs) using:
      - LHS-style init + a few Cauchy shots
      - Competitive restart portfolio:
          (A) adaptive diagonal ES around incumbent (fast local improvement)
          (B) small-temperature annealed heavy-tail search (escape basins)
      - Best-so-far driven coordinate/pattern refinement (cheap local search)
      - Soft evaluation budget control via time checks

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ---------------- helpers ----------------
    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def eval_f(x):
        return float(func(x))  # list is fine

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def widths_vec():
        return [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    def lhs_init(n):
        # Stratified per dimension (LHS-like)
        strata = []
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            strata.append(perm)
        pts = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (strata[d][i] + random.random()) / n
                lo, hi = bounds[d]
                x[d] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    def cauchy_step(scale):
        # Standard Cauchy via inverse CDF: tan(pi*(u-0.5))
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    def mutate_diag_gauss(x, sig):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
            else:
                y[i] = clip(y[i] + random.gauss(0.0, sig[i]), lo, hi)
        return y

    def mutate_heavytail(x, scale_vec, p_cauchy=0.25):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
                continue
            if random.random() < p_cauchy:
                step = cauchy_step(scale_vec[i])
            else:
                step = random.gauss(0.0, scale_vec[i])
            y[i] = clip(y[i] + step, lo, hi)
        return y

    def pattern_local_search(x0, f0, step0, max_sweeps=2):
        # Cheap coordinate search + one pattern move per sweep
        x = x0[:]
        fx = f0
        steps = step0[:]
        for _ in range(max_sweeps):
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
                # pattern move: x + (x - base)
                d = [x[i] - base[i] for i in range(dim)]
                xt = [clip(x[i] + d[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft
            else:
                for i in range(dim):
                    steps[i] *= 0.5
        return x, fx

    # ---------------- main ----------------
    start = time.time()
    deadline = start + float(max_time)

    W = widths_vec()
    # base scales
    base_sigma = [w * 0.18 if w > 0 else 0.0 for w in W]
    base_jump  = [w * 0.35 if w > 0 else 0.0 for w in W]

    # Initial sampling
    best = float("inf")
    best_x = None

    n0 = max(24, 6 * dim)
    for x in lhs_init(n0):
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    # A few heavy-tail global shots (often helps on rugged functions)
    for _ in range(max(8, 2 * dim)):
        if time.time() >= deadline:
            return best
        x = rand_uniform_vec()
        # one extra Cauchy perturbation from random to cover edges/basins
        x = mutate_heavytail(x, base_jump, p_cauchy=0.8)
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_uniform_vec()
        best = eval_f(best_x)

    # State for local ES track
    x_es = best_x[:]
    f_es = best
    sig = base_sigma[:]  # diagonal step sizes
    succ = [0.2] * dim   # EWMA success proxy per coordinate

    # Portfolio controls
    it = 0
    no_global = 0
    refine_every = 16
    restart_after = 45

    # ES sizing
    lam = max(14, 8 + int(5 * math.log(dim + 1)))
    mu = max(4, lam // 3)
    # recombination weights
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    sW = sum(weights)
    weights = [w / sW for w in weights]

    # adaptation
    target = 0.22
    succ_decay = 0.92
    adapt = 0.35
    sig_min_frac = 1e-8
    sig_max_frac = 0.60

    # Annealed heavy-tail track (for escapes)
    x_ht = best_x[:]
    f_ht = best
    T0 = 1.0
    T = T0

    while time.time() < deadline:
        it += 1

        # ---- choose track (biased to ES, sometimes heavy-tail escape) ----
        use_ht = (no_global > 10 and (it % 3 == 0)) or (it % 11 == 0)
        if use_ht:
            # Annealed heavy-tail step around best (not around ht state only)
            # Temperature shrinks but restarts on improvements.
            scale = [max(W[i] * 1e-6, base_jump[i] * (0.25 + 0.75 * T)) for i in range(dim)]
            cand = mutate_heavytail(best_x, scale, p_cauchy=0.35)
            fc = eval_f(cand)
            # Metropolis accept on ht state to allow drifting
            if fc < f_ht:
                x_ht, f_ht = cand, fc
            else:
                # accept sometimes
                if T > 1e-12:
                    prob = math.exp(-(fc - f_ht) / max(1e-12, T))
                    if random.random() < prob:
                        x_ht, f_ht = cand, fc

            # update global best
            if f_ht < best:
                best, best_x = f_ht, x_ht[:]
                no_global = 0
                T = T0
            else:
                no_global += 1
                T *= 0.97  # cool
            continue

        # ---- ES batch around x_es ----
        offspring = []
        improved_any = False

        for _ in range(lam):
            if time.time() >= deadline:
                return best
            y = mutate_diag_gauss(x_es, sig)
            fy = eval_f(y)
            offspring.append((fy, y))
            if fy < f_es:
                improved_any = True
                for i in range(dim):
                    if y[i] != x_es[i]:
                        succ[i] = succ_decay * succ[i] + (1.0 - succ_decay) * 1.0
            else:
                for i in range(dim):
                    succ[i] = succ_decay * succ[i]

        offspring.sort(key=lambda t: t[0])
        top = offspring[:mu]

        # recombine
        xr = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += weights[k] * top[k][1][i]
            xr[i] = clip(s, bounds[i][0], bounds[i][1])
        fr = eval_f(xr)

        # select new parent as best of (recombined, best offspring)
        fbest_off, xbest_off = offspring[0]
        if fbest_off <= fr:
            x_new, f_new = xbest_off, fbest_off
        else:
            x_new, f_new = xr, fr

        if f_new < f_es:
            x_es, f_es = x_new[:], f_new
        else:
            # still move to best offspring to keep progress pressure
            x_es, f_es = xbest_off[:], fbest_off

        # global best update
        if f_es < best:
            best, best_x = f_es, x_es[:]
            no_global = 0
            T = T0
        else:
            no_global += 1

        # adapt diagonal sigmas based on succ
        for i in range(dim):
            w = W[i]
            if w <= 0.0:
                sig[i] = 0.0
                continue
            sig[i] *= math.exp(adapt * (succ[i] - target))
            smin = w * sig_min_frac
            smax = w * sig_max_frac
            if sig[i] < smin:
                sig[i] = smin
            elif sig[i] > smax:
                sig[i] = smax

        if not improved_any:
            for i in range(dim):
                sig[i] *= 0.88

        # local refinement on best
        if (it % refine_every) == 0 and time.time() < deadline and best_x is not None:
            steps = [max(sig[i], W[i] * 0.025) for i in range(dim)]
            rx, rf = pattern_local_search(best_x, best, steps, max_sweeps=2)
            if rf < best:
                best, best_x = rf, rx[:]
                x_es, f_es = best_x[:], best
                x_ht, f_ht = best_x[:], best
                no_global = 0
                T = T0

        # restart if stagnating: diversify but keep bias to best
        if no_global >= restart_after and time.time() < deadline:
            no_global = 0
            r = rand_uniform_vec()
            alpha = 0.80
            x_es = [clip(alpha * best_x[i] + (1.0 - alpha) * r[i], bounds[i][0], bounds[i][1])
                    for i in range(dim)]
            f_es = eval_f(x_es)

            # reset sigmas to moderately exploratory
            for i in range(dim):
                sig[i] = max(base_sigma[i], W[i] * 1e-4)
                succ[i] = 0.2

            # also reset heavy-tail state near best
            x_ht, f_ht = best_x[:], best
            T = T0

            if f_es < best:
                best, best_x = f_es, x_es[:]

    return best
