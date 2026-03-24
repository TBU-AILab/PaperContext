import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (stdlib only).

    Improvements vs provided versions:
      - Uses a *hybrid* of:
          (1) low-cost quasi-LHS + opposition seeding
          (2) diagonal Evolution Strategy (CEM/ES-style) for global guidance
          (3) Nelder–Mead style simplex search (robust local optimizer, no gradients)
          (4) adaptive 1+1 success-based step-size local mutations (very cheap exploitation)
      - Restarts are scheduled by stagnation and time-left, and reuse best-known point.
      - Careful time-guarding: avoids starting large blocks near the deadline.

    Returns: best objective value found.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if span[i] == 0.0:
            span[i] = 1.0

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def evalf(x):
        return float(func(x))

    def randn():
        # approx N(0,1)
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy():
        u = random.random()
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        if v > 60.0:
            v = 60.0
        elif v < -60.0:
            v = -60.0
        return v

    def rand_uniform_vec():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def time_left():
        return deadline - time.time()

    # -------------------- seeding (quasi-LHS + opposition + center) --------------------
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    clip_inplace(center)
    best_x = center[:]
    best = evalf(best_x)

    # time-aware init size
    init_n = max(14, 7 * dim)
    if max_time < 0.25:
        init_n = max(6, 2 * dim)
    elif max_time > 2.0:
        init_n = max(init_n, 22 + 10 * dim)

    m = init_n
    strata = list(range(m))
    perms = []
    for _ in range(dim):
        p = strata[:]
        random.shuffle(p)
        perms.append(p)

    for k in range(m):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            u = (perms[i][k] + random.random()) / m
            x[i] = lows[i] + u * (highs[i] - lows[i])
        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]

        xo = opposite(x)
        clip_inplace(xo)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    # -------------------- diagonal ES state --------------------
    mean = best_x[:]
    sigma = [0.30 * (highs[i] - lows[i]) for i in range(dim)]
    for i in range(dim):
        if sigma[i] <= 0.0:
            sigma[i] = 1.0

    sig_floor = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    sig_ceil = [0.90 * (highs[i] - lows[i]) if highs[i] != lows[i] else 1.0 for i in range(dim)]

    # smoothing
    alpha_m = 0.35
    alpha_s = 0.25

    # small elite archive for stability
    archive = []
    archive_cap = max(14, 3 * dim)

    def add_archive(f, x):
        archive.append((f, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > archive_cap:
            del archive[archive_cap:]

    add_archive(best, best_x)

    # -------------------- simplex (Nelder–Mead-ish) --------------------
    # Construct an initial simplex around best_x (scaled to bounds)
    # Will be rebuilt on restarts.
    def build_simplex(x0, scale):
        simp = [x0[:]]
        for j in range(dim):
            x = x0[:]
            step = scale * (highs[j] - lows[j])
            if step == 0.0:
                step = scale
            x[j] = x[j] + step
            clip_inplace(x)
            simp.append(x)
        vals = [evalf(s) for s in simp]
        return simp, vals

    # NM coefficients (mildly conservative)
    NM_alpha = 1.0   # reflection
    NM_gamma = 2.0   # expansion
    NM_rho   = 0.5   # contraction
    NM_sigma = 0.5   # shrink

    # -------------------- 1+1 success-step local mutator --------------------
    # global step scale (relative to bounds)
    one1_scale = 0.08

    def one_plus_one_round(x_base, base_f, tries):
        nonlocal best, best_x, one1_scale
        successes = 0
        for _ in range(tries):
            if time.time() >= deadline:
                return
            heavy = (random.random() < 0.10)
            cand = x_base[:]
            # mutate all coords (cheap, good for non-separable)
            for i in range(dim):
                s = (cauchy() if heavy else randn())
                cand[i] += (one1_scale * (highs[i] - lows[i])) * s
            clip_inplace(cand)
            f = evalf(cand)
            if f < best:
                best, best_x = f, cand[:]
                add_archive(best, best_x)
            if f < base_f:
                x_base[:] = cand
                base_f = f
                successes += 1

        # 1/5th success rule-ish adaptation
        if tries > 0:
            rate = successes / float(tries)
            if rate > 0.22:
                one1_scale = min(one1_scale * 1.20, 0.60)
            elif rate < 0.14:
                one1_scale = max(one1_scale * 0.82, 1e-6)
        return x_base, base_f

    # -------------------- main loop --------------------
    no_improve = 0
    last_best = best

    # initial simplex scale
    simplex_scale = 0.07
    simplex, svals = build_simplex(best_x, simplex_scale)

    while True:
        if time.time() >= deadline:
            return best

        tl = time_left()
        if tl <= 0.0:
            return best

        # Adjust batch sizes to time left
        base_lam = 10 + 4 * dim
        if tl < 0.08:
            lam = max(6, 2 * dim)
        elif tl > 1.2 and dim <= 18:
            lam = int(base_lam * 1.6)
        else:
            lam = base_lam

        # ---------------- (A) ES generation ----------------
        pop = []
        half = (lam + 1) // 2

        # anti-drift pull toward best
        if random.random() < 0.20:
            mean = [0.80 * mean[i] + 0.20 * best_x[i] for i in range(dim)]

        for _ in range(half):
            if time.time() >= deadline:
                return best

            heavy = (random.random() < 0.12)
            z = [randn() for _ in range(dim)]
            x1 = [0.0] * dim
            x2 = [0.0] * dim
            for i in range(dim):
                step = sigma[i] * (cauchy() if heavy else z[i])
                x1[i] = mean[i] + step
                x2[i] = mean[i] - step
            clip_inplace(x1)
            clip_inplace(x2)

            f1 = evalf(x1)
            pop.append((f1, x1))
            if f1 < best:
                best, best_x = f1, x1[:]
                add_archive(best, best_x)

            if len(pop) < lam:
                f2 = evalf(x2)
                pop.append((f2, x2))
                if f2 < best:
                    best, best_x = f2, x2[:]
                    add_archive(best, best_x)

        pop.sort(key=lambda t: t[0])

        # elites + archive mixing
        mu = max(4, lam // 5)
        elites = pop[:mu]
        if archive:
            take = min(len(archive), max(1, mu // 3))
            elites = elites[:max(1, mu - take)] + archive[:take]
            elites.sort(key=lambda t: t[0])
            elites = elites[:mu]

        # log-rank weights
        w = []
        wsum = 0.0
        for r in range(mu):
            wr = math.log(mu + 1.0) - math.log(r + 1.0)
            w.append(wr)
            wsum += wr
        if wsum <= 0.0:
            wsum = 1.0

        new_mean = [0.0] * dim
        for wr, (f, x) in zip(w, elites):
            ww = wr / wsum
            for i in range(dim):
                new_mean[i] += ww * x[i]

        new_var = [0.0] * dim
        for wr, (f, x) in zip(w, elites):
            ww = wr / wsum
            for i in range(dim):
                d = x[i] - new_mean[i]
                new_var[i] += ww * d * d

        new_sigma = [0.0] * dim
        for i in range(dim):
            s = math.sqrt(max(new_var[i], sig_floor[i] * sig_floor[i]))
            if s < sig_floor[i]:
                s = sig_floor[i]
            elif s > sig_ceil[i]:
                s = sig_ceil[i]
            new_sigma[i] = s

        for i in range(dim):
            mean[i] = (1.0 - alpha_m) * mean[i] + alpha_m * new_mean[i]
            sigma[i] = (1.0 - alpha_s) * sigma[i] + alpha_s * new_sigma[i]
            if sigma[i] < sig_floor[i]:
                sigma[i] = sig_floor[i]
            elif sigma[i] > sig_ceil[i]:
                sigma[i] = sig_ceil[i]

        # ---------------- (B) Nelder–Mead local steps (few, time-guarded) ----------------
        # Use more NM work when ES seems to have found a good basin (sigma small-ish)
        avg_sig_rel = sum(sigma[i] / (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)) / dim
        nm_steps = 1
        if avg_sig_rel < 0.12:
            nm_steps = 2
        if no_improve > 20 and avg_sig_rel < 0.18:
            nm_steps = 3
        if tl < 0.10:
            nm_steps = 1

        for _ in range(nm_steps):
            if time.time() >= deadline:
                return best

            # Ensure simplex anchored near current best_x occasionally
            if random.random() < 0.15:
                simplex, svals = build_simplex(best_x, simplex_scale)

            # sort simplex
            idx = list(range(len(simplex)))
            idx.sort(key=lambda i: svals[i])
            simplex = [simplex[i] for i in idx]
            svals = [svals[i] for i in idx]

            if svals[0] < best:
                best, best_x = svals[0], simplex[0][:]
                add_archive(best, best_x)

            # centroid of all but worst
            centroid = [0.0] * dim
            for j in range(dim):
                centroid[j] = 0.0
            for k in range(dim):  # first dim+1 points, exclude worst at dim
                xk = simplex[k]
                for j in range(dim):
                    centroid[j] += xk[j]
            inv = 1.0 / dim
            for j in range(dim):
                centroid[j] *= inv

            worst = simplex[-1]
            f_worst = svals[-1]
            f_best = svals[0]
            f_second = svals[-2]

            # reflection
            xr = [centroid[j] + NM_alpha * (centroid[j] - worst[j]) for j in range(dim)]
            clip_inplace(xr)
            fr = evalf(xr)

            if fr < best:
                best, best_x = fr, xr[:]
                add_archive(best, best_x)

            if fr < f_best:
                # expansion
                xe = [centroid[j] + NM_gamma * (xr[j] - centroid[j]) for j in range(dim)]
                clip_inplace(xe)
                fe = evalf(xe)
                if fe < fr:
                    simplex[-1] = xe
                    svals[-1] = fe
                else:
                    simplex[-1] = xr
                    svals[-1] = fr
            elif fr < f_second:
                simplex[-1] = xr
                svals[-1] = fr
            else:
                # contraction
                if fr < f_worst:
                    # outside contraction
                    xc = [centroid[j] + NM_rho * (xr[j] - centroid[j]) for j in range(dim)]
                else:
                    # inside contraction
                    xc = [centroid[j] - NM_rho * (centroid[j] - worst[j]) for j in range(dim)]
                clip_inplace(xc)
                fc = evalf(xc)
                if fc < f_worst:
                    simplex[-1] = xc
                    svals[-1] = fc
                else:
                    # shrink towards best
                    xb = simplex[0]
                    for k in range(1, dim + 1):
                        xs = [xb[j] + NM_sigma * (simplex[k][j] - xb[j]) for j in range(dim)]
                        clip_inplace(xs)
                        simplex[k] = xs
                        svals[k] = evalf(xs)
                    # make shrink slightly smaller over time (more local)
                    simplex_scale = max(simplex_scale * 0.92, 1e-6)

        # ---------------- (C) very cheap 1+1 exploitation around best ----------------
        # Number of tries depends on time left and stagnation.
        if tl > 0.05:
            tries = 2 if no_improve < 10 else 5
            xbase = best_x[:]
            fbase = best
            one_plus_one_round(xbase, fbase, tries)

        # ---------------- stagnation / restart logic ----------------
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # periodic sigma inflation to escape
        if no_improve % 18 == 0:
            for i in range(dim):
                sigma[i] = min(sig_ceil[i], sigma[i] * 1.35)

        # partial restarts: rebuild simplex and/or re-center mean
        if no_improve % 55 == 0:
            if random.random() < 0.80:
                mean = best_x[:]
            else:
                mean = rand_uniform_vec()

            # reset simplex around best (or around mean)
            anchor = best_x[:] if random.random() < 0.85 else mean[:]
            simplex_scale = 0.06 + 0.06 * random.random()
            simplex, svals = build_simplex(anchor, simplex_scale)

            # widen sigma a bit
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.22 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0))

            # nudge 1+1 scale too
            one1_scale = min(max(one1_scale * 1.10, 1e-4), 0.25)
            no_improve = 0
