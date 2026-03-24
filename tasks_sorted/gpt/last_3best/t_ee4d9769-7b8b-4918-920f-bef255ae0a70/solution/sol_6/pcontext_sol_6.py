import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvement goals vs your last version:
    - Make local search *much* more reliable: use a bounded Powell-style line-search
      (direction set) with a robust 1D bracketing + golden-section search.
      This is usually stronger than coordinate probing + quadratic-on-3-points when
      variables are coupled.
    - Keep global exploration: a lean SHADE/JADE-like DE/current-to-pbest/1 + archive,
      but reduce overhead and tune for evaluation efficiency.
    - Better time allocation: alternating short DE bursts with short Powell refinements,
      plus restart when stagnating.

    Returns:
        best fitness (float)
    """

    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def reflect_scalar(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect until in range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def lhs_points(n):
        # cheap LHS-like sampling
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for j in range(dim):
                u = (perms[j][k] + random.random()) / n
                x[j] = lows[j] + u * spans[j]
            pts.append(x)
        return pts

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def norm2(v):
        return math.sqrt(sum(vi * vi for vi in v))

    # ---------------- DE param sampling (light SHADE) ----------------
    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def sample_F(muF):
        for _ in range(10):
            f = rand_cauchy(muF, 0.10)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return max(0.1, min(1.0, muF))

    def sample_CR(muCR):
        return clamp01(random.gauss(muCR, 0.10))

    # ---------------- bounded Powell local search ----------------
    def max_step_along(x, d):
        # Compute maximum feasible step alpha so that x + alpha*d stays in bounds
        # (alpha can be positive or negative; we return symmetric [-a_neg, a_pos] bounds).
        a_pos = float("inf")
        a_neg = float("inf")
        for i in range(dim):
            di = d[i]
            if abs(di) < 1e-18:
                continue
            if di > 0.0:
                a_pos = min(a_pos, (highs[i] - x[i]) / di)
                a_neg = min(a_neg, (x[i] - lows[i]) / di)
            else:
                # di < 0
                a_pos = min(a_pos, (x[i] - lows[i]) / (-di))
                a_neg = min(a_neg, (highs[i] - x[i]) / (-di))
        if a_pos == float("inf"): a_pos = 0.0
        if a_neg == float("inf"): a_neg = 0.0
        return -a_neg, a_pos

    def x_plus_ad(x, d, a):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = reflect_scalar(x[i] + a * d[i], i)
        return y

    def golden_search(phi, a_lo, a_hi, t_end):
        # Minimize phi(a) over [a_lo,a_hi] using golden-section, time bounded.
        gr = 0.6180339887498949
        x1 = a_hi - gr * (a_hi - a_lo)
        x2 = a_lo + gr * (a_hi - a_lo)
        f1 = phi(x1)
        if now() >= t_end:
            return x1, f1
        f2 = phi(x2)

        # stop condition based on interval size
        # (use absolute since alpha scale varies)
        while now() < t_end and abs(a_hi - a_lo) > 1e-9 * (1.0 + abs(a_lo) + abs(a_hi)):
            if f1 <= f2:
                a_hi = x2
                x2, f2 = x1, f1
                x1 = a_hi - gr * (a_hi - a_lo)
                f1 = phi(x1)
            else:
                a_lo = x1
                x1, f1 = x2, f2
                x2 = a_lo + gr * (a_hi - a_lo)
                f2 = phi(x2)
        if f1 <= f2:
            return x1, f1
        return x2, f2

    def line_minimize(x, fx, d, t_end, initial_span=0.25):
        # Bounded 1D minimize f(x + a d), with bracketing then golden-section.
        dn = norm2(d)
        if dn < 1e-18:
            return x, fx

        # normalize direction for stability
        inv = 1.0 / dn
        d = [di * inv for di in d]

        a_min, a_max = max_step_along(x, d)
        if a_max - a_min < 1e-15:
            return x, fx

        # choose a0 range around 0 (scaled by spans)
        # initial step ~ initial_span * average span
        avg_span = sum(spans) / float(dim)
        a0 = initial_span * avg_span
        # in normalized direction units; since d normalized, a0 uses absolute length units
        a0 = clamp(a0, 0.0, max(1e-18, 0.5 * (a_max - a_min)))

        def phi(a):
            return eval_f(x_plus_ad(x, d, a))

        # evaluate at 0, +a0, -a0 (within bounds)
        f0 = fx
        best_a, best_f = 0.0, f0

        ap = min(a_max, a0)
        am = max(a_min, -a0)

        if ap != 0.0 and now() < t_end:
            fp = phi(ap)
            if fp < best_f:
                best_a, best_f = ap, fp
        else:
            fp = float("inf")

        if am != 0.0 and now() < t_end:
            fm = phi(am)
            if fm < best_f:
                best_a, best_f = am, fm
        else:
            fm = float("inf")

        # If neither side improves, stop (Powell will move on)
        if best_a == 0.0:
            return x, fx

        # Bracket minimum on the side that improved
        if best_a > 0.0:
            aL, fL = 0.0, f0
            aM, fM = best_a, best_f
            # expand right until no improvement or hit bounds
            aR = aM
            fR = fM
            step = max(1e-18, aM)
            for _ in range(18):
                if now() >= t_end:
                    break
                cand = aR + step
                if cand > a_max:
                    cand = a_max
                if cand == aR:
                    break
                fc = phi(cand)
                if fc < fR:
                    aL, fL = aM, fM
                    aM, fM = cand, fc
                    aR, fR = cand, fc
                    step *= 1.6
                else:
                    aR, fR = cand, fc
                    break
            lo, hi = aL, aR
        else:
            aR, fR = 0.0, f0
            aM, fM = best_a, best_f
            aL = aM
            fL = fM
            step = max(1e-18, -aM)
            for _ in range(18):
                if now() >= t_end:
                    break
                cand = aL - step
                if cand < a_min:
                    cand = a_min
                if cand == aL:
                    break
                fc = phi(cand)
                if fc < fL:
                    aR, fR = aM, fM
                    aM, fM = cand, fc
                    aL, fL = cand, fc
                    step *= 1.6
                else:
                    aL, fL = cand, fc
                    break
            lo, hi = aL, aR

        # Ensure proper order
        if lo > hi:
            lo, hi = hi, lo

        # Golden-section on [lo,hi]
        a_star, f_star = golden_search(phi, lo, hi, t_end)
        if f_star < fx:
            return x_plus_ad(x, d, a_star), f_star
        return x, fx

    def powell_refine(x0, f0, time_cap, radius_scale):
        # Powell direction-set method (bounded line-search), time bounded.
        t_end = min(deadline, now() + time_cap)
        x = x0[:]
        fx = f0

        # initial directions: scaled coordinate basis
        dirs = []
        base = max(1e-18, radius_scale * (sum(spans) / float(dim)))
        for i in range(dim):
            d = [0.0] * dim
            d[i] = base
            dirs.append(d)

        prev_fx = fx
        it_stall = 0
        while now() < t_end:
            x_start = x[:]
            f_start = fx

            biggest_drop = 0.0
            big_idx = -1

            for k in range(len(dirs)):
                if now() >= t_end:
                    break
                before = fx
                x, fx = line_minimize(x, fx, dirs[k], t_end, initial_span=0.20)
                drop = before - fx
                if drop > biggest_drop:
                    biggest_drop = drop
                    big_idx = k

            if now() >= t_end:
                break

            # extrapolated direction
            d_new = [x[i] - x_start[i] for i in range(dim)]
            if norm2(d_new) < 1e-18:
                it_stall += 1
            else:
                # try one more line minimization along extrapolated direction
                x2, f2 = line_minimize(x, fx, d_new, t_end, initial_span=0.25)
                if f2 < fx:
                    x, fx = x2, f2
                    if big_idx >= 0:
                        dirs[big_idx] = d_new  # replace the direction with most progress
                it_stall = 0

            # stop if marginal improvements
            if abs(f_start - fx) <= 1e-12 * (1.0 + abs(fx)):
                it_stall += 1
            if it_stall >= 2:
                break

            # if no improvement at all, stop early
            if fx >= prev_fx - 1e-14:
                break
            prev_fx = fx

        return x, fx

    # ---------------- sizing ----------------
    pop_size = 10 + 5 * dim
    pop_size = max(22, min(90, pop_size))
    if max_time <= 0.35:
        pop_size = min(pop_size, 34)
    elif max_time <= 0.8:
        pop_size = min(pop_size, 56)

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    pop = []
    fit = []

    init_n = pop_size
    n_lhs = max(6, init_n // 2)
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return best
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        if len(pop) >= init_n:
            break

        ox = [reflect_scalar(opposite_point(x)[i], i) for i in range(dim)]
        if now() >= deadline:
            return best
        fox = eval_f(ox)
        pop.append(ox); fit.append(fox)
        if fox < best:
            best, best_x = fox, ox[:]
        if len(pop) >= init_n:
            break

    # greedy fill
    while len(pop) < pop_size:
        if now() >= deadline:
            return best
        k = 3 if dim <= 12 else 2
        bx, bf = None, float("inf")
        for _ in range(k):
            x = rand_point()
            fx = eval_f(x)
            if fx < bf:
                bx, bf = x, fx
            if fx < best:
                best, best_x = fx, x[:]
            if now() >= deadline:
                return best
        pop.append(bx); fit.append(bf)

    # ---------------- DE memory + archive ----------------
    H = max(6, min(22, int(2 * math.sqrt(dim + 1)) + 6))
    M_F = [0.55] * H
    M_CR = [0.85] * H
    k_mem = 0

    archive = []
    archive_max = pop_size

    def get_from_pool(idx):
        if idx < pop_size:
            return pop[idx]
        return archive[idx - pop_size]

    def pick_distinct(exclude, count, pool_n):
        chosen = set()
        while len(chosen) < count:
            r = random.randrange(pool_n)
            if r != exclude:
                chosen.add(r)
        return list(chosen)

    # ---------------- main loop ----------------
    last_improve_t = start
    last_best = best

    # local refine scheduling
    last_local_t = start
    local_every = max(0.12, 0.06 * max_time)

    while True:
        t = now()
        if t >= deadline:
            return best

        prog = (t - start) / (max_time if max_time > 1e-12 else 1e-12)
        prog = 0.0 if prog < 0.0 else (1.0 if prog > 1.0 else prog)

        # periodic Powell refine (more useful later)
        if best_x is not None and (t - last_local_t) >= local_every:
            remaining = deadline - t
            if remaining > 0.05:
                rad = 0.55 * (1.0 - prog) + 0.08  # shrink radius over time
                cap = min(0.05 + 0.12 * prog, 0.28 * remaining, 0.38)
                rx, rf = powell_refine(best_x, best, cap, rad)
                if rf < best:
                    best, best_x = rf, rx[:]
                    last_improve_t = now()
                    # inject into worst
                    worst = max(range(pop_size), key=lambda i: fit[i])
                    pop[worst] = best_x[:]
                    fit[worst] = best
                last_local_t = now()

        # stagnation -> partial restart (keeps best)
        if best < last_best - 1e-14:
            last_best = best
            last_improve_t = t

        if (t - last_improve_t) > max(0.28, 0.22 * max_time) and (deadline - t) > 0.10:
            # keep top elites, re-seed others around best + some global
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elites = max(2, pop_size // 6)
            new_pop = [pop[i][:] for i in idx_sorted[:elites]]
            new_fit = [fit[i] for i in idx_sorted[:elites]]

            sigma = (0.22 * (1.0 - prog) + 0.05)
            while len(new_pop) < pop_size and now() < deadline:
                if random.random() < 0.70 and best_x is not None:
                    x = best_x[:]
                    for d in range(dim):
                        x[d] = reflect_scalar(x[d] + random.gauss(0.0, sigma) * spans[d], d)
                else:
                    x = rand_point()
                fx = eval_f(x)
                new_pop.append(x); new_fit.append(fx)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_t = now()

            pop, fit = new_pop, new_fit
            archive = []
            for h in range(H):
                M_F[h] = 0.55
                M_CR[h] = 0.85
            k_mem = 0
            last_local_t = now()

        # DE generation
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        # p-best pressure: more exploit later
        p = 0.30 - 0.20 * prog   # 0.30 -> 0.10
        p = 0.08 if p < 0.08 else p
        pbest_count = max(2, int(p * pop_size))

        S_F, S_CR, dF = [], [], []

        pool_n = pop_size + len(archive)

        for i in range(pop_size):
            if now() >= deadline:
                return best

            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            # choose pbest among top p%
            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            pool_n = pop_size + len(archive)
            r1, r2 = pick_distinct(i, 2, pool_n)
            x_r1 = get_from_pool(r1)
            x_r2 = get_from_pool(r2)

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            # binomial crossover with reflection bounds
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect_scalar(v[d], d)

            fu = eval_f(u)

            if fu <= fit[i]:
                # archive parent
                if len(archive) < archive_max:
                    archive.append(pop[i])
                else:
                    archive[random.randrange(archive_max)] = pop[i]

                imp = fit[i] - fu
                if imp < 0.0:
                    imp = 0.0
                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(imp)

                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best, best_x = fu, u[:]
                    last_best = best
                    last_improve_t = now()

        # update memories
        if S_F:
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / wsum for di in dF]

            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            new_MF = (num / den) if den > 1e-18 else M_F[k_mem]

            # weighted mean for CR
            new_MCR = 0.0
            for w, cr in zip(weights, S_CR):
                new_MCR += w * cr

            # smooth a bit for stability
            M_F[k_mem] = min(1.0, max(0.08, 0.85 * M_F[k_mem] + 0.15 * new_MF))
            M_CR[k_mem] = clamp01(0.85 * M_CR[k_mem] + 0.15 * new_MCR)
            k_mem = (k_mem + 1) % H

        if len(archive) > archive_max:
            archive = archive[-archive_max:]
