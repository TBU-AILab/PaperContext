import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libs).

    Improvements vs the provided algorithms:
      - Better global coverage early: scrambled Halton + opposition + occasional global resample
      - Stronger exploitation: bounded Powell-style direction search (derivative-free)
      - Robust local refinement: (1+1)-ES with success rule + heavy-tail escape
      - Restart logic from an elite pool; time-aware step shrink

    Returns:
      best (float): best fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    fixed = [spans[i] == 0.0 for i in range(dim)]

    # ---------- helpers ----------
    def now():
        return time.time()

    def clamp(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def evalf(x):
        return float(func(x))

    def rand_point():
        return [lows[i] if fixed[i] else (lows[i] + random.random() * spans[i]) for i in range(dim)]

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = lows[i] if fixed[i] else (lows[i] + highs[i] - x[i])
        return y

    # heavy-tail (Cauchy)
    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- low-discrepancy: scrambled Halton ----------
    def first_primes(k):
        primes = []
        c = 2
        while len(primes) < k:
            is_p = True
            r = int(c ** 0.5)
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(c)
            c += 1
        return primes

    bases = first_primes(max(1, dim))
    scr = [random.random() for _ in range(dim)]

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton(idx):
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                u = (vdc(idx + 1, bases[i]) + scr[i]) % 1.0
                x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite pool ----------
    elite_k = max(8, min(24, 10 + dim // 2))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites.pop()

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    idx = 0

    # More init for robust global coverage, but bounded
    init_n = max(120, min(1400, 120 + 45 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best

        if j % 11 == 0:
            x = rand_point()
        else:
            x = halton(idx)
            idx += 1
        x = clamp(x)

        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]
        push_elite(f, x)

        # opposition (often helps on bounded domains)
        if now() >= deadline:
            return best
        xo = opposition(x)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]
        push_elite(fo, xo)

    if best_x is None:
        x = rand_point()
        best_x = x
        best = evalf(x)
        push_elite(best, x)

    # ---------- 1D bounded line search along direction (coarse-to-fine + parabolic) ----------
    def line_search(x0, f0, d, step0, time_end):
        # maximize speed: small number of evaluations
        # search alpha in [-A, A], where A determined by bounds and step0
        # then do a few refinements
        if now() >= time_end:
            return f0, x0[:], False

        # Compute maximum feasible alpha range due to bounds
        lo_a = -1e30
        hi_a = 1e30
        for i in range(dim):
            di = d[i]
            if di == 0.0 or fixed[i]:
                continue
            if di > 0.0:
                lo = (lows[i] - x0[i]) / di
                hi = (highs[i] - x0[i]) / di
            else:
                lo = (highs[i] - x0[i]) / di
                hi = (lows[i] - x0[i]) / di
            if lo > lo_a:
                lo_a = lo
            if hi < hi_a:
                hi_a = hi
        if lo_a > hi_a:
            return f0, x0[:], False

        # bound by step0 too (keep local)
        A = step0
        if A <= 0.0:
            A = 0.0
        lo_a = max(lo_a, -A)
        hi_a = min(hi_a, A)
        if hi_a - lo_a <= 1e-18:
            return f0, x0[:], False

        def x_of(alpha):
            return clamp([x0[i] + alpha * d[i] for i in range(dim)])

        # Initial coarse samples (5 points)
        # (avoid too many evals)
        alphas = [lo_a, lo_a * 0.5, 0.0, hi_a * 0.5, hi_a]
        best_f = f0
        best_xl = x0[:]
        best_a = 0.0

        vals = []
        for a in alphas:
            if now() >= time_end:
                break
            xx = x_of(a)
            ff = evalf(xx)
            vals.append((a, ff, xx))
            if ff < best_f:
                best_f, best_xl, best_a = ff, xx, a

        improved = (best_f < f0)

        # Parabolic refinement around best (if we have neighbors)
        # take three points around best_a
        if now() < time_end and len(vals) >= 3:
            vals.sort(key=lambda t: t[0])
            # find nearest triple containing best_a
            k = 0
            for i in range(len(vals)):
                if vals[i][0] == best_a:
                    k = i
                    break
            i0 = max(0, k - 1)
            i2 = min(len(vals) - 1, k + 1)
            i1 = k
            # ensure 3 distinct indices
            if i0 != i1 and i1 != i2 and i0 != i2:
                a0, f0p, _ = vals[i0]
                a1, f1p, _ = vals[i1]
                a2, f2p, _ = vals[i2]
                # fit parabola: alpha* = a1 - 0.5 * ((a1-a0)^2*(f1-f2)-(a1-a2)^2*(f1-f0)) / denom
                denom = (a1 - a0) * (f1p - f2p) - (a1 - a2) * (f1p - f0p)
                if abs(denom) > 1e-18:
                    num = (a1 - a0) ** 2 * (f1p - f2p) - (a1 - a2) ** 2 * (f1p - f0p)
                    astar = a1 - 0.5 * (num / denom)
                    if astar < lo_a:
                        astar = lo_a
                    elif astar > hi_a:
                        astar = hi_a
                    if now() < time_end:
                        xx = x_of(astar)
                        ff = evalf(xx)
                        if ff < best_f:
                            best_f, best_xl, best_a = ff, xx, astar
                            improved = True

        # A couple of interval shrinks around best_a
        # (very cheap local polish)
        if now() < time_end:
            width = (hi_a - lo_a)
            for _ in range(2):
                if now() >= time_end:
                    break
                width *= 0.35
                lo2 = max(lo_a, best_a - width)
                hi2 = min(hi_a, best_a + width)
                if hi2 - lo2 <= 1e-18:
                    break
                for a in (lo2, (lo2 + hi2) * 0.5, hi2):
                    if now() >= time_end:
                        break
                    xx = x_of(a)
                    ff = evalf(xx)
                    if ff < best_f:
                        best_f, best_xl, best_a = ff, xx, a
                        improved = True

        return best_f, best_xl, improved

    # ---------- Powell-like direction search (bounded, time-sliced) ----------
    def powell_refine(seed_x, seed_f, time_end):
        # Directions start as coordinate basis
        x = seed_x[:]
        fx = seed_f

        # initial step scale based on bounds and time fraction
        span_max = max(spans) if spans else 0.0
        floor = 1e-12 + 1e-10 * (span_max + 1.0)

        # Keep a compact set of directions to stay cheap
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = 1.0
            dirs.append(d)

        # local step amplitude: shrink with time
        frac = (now() - t0) / max(1e-12, float(max_time))
        baseA = (0.25 * (1.0 - 0.85 * frac) + 0.03)
        A = [max(floor, baseA * spans[i]) if not fixed[i] else 0.0 for i in range(dim)]
        Amax = max(A) if A else 0.0
        if Amax <= 0.0:
            return fx, x, False

        improved_any = False
        # 1-2 sweeps depending on time/dim
        sweeps = 1 if dim > 25 else 2
        for _ in range(sweeps):
            if now() >= time_end:
                break
            x_start = x[:]
            f_start = fx

            # sweep directions
            for k in range(len(dirs)):
                if now() >= time_end:
                    break
                d = dirs[k]
                # step amplitude along this direction: use average of A where d != 0
                step0 = 0.0
                cnt = 0
                for i in range(dim):
                    if d[i] != 0.0 and not fixed[i]:
                        step0 += A[i]
                        cnt += 1
                if cnt == 0:
                    continue
                step0 = step0 / cnt

                f2, x2, imp = line_search(x, fx, d, step0, time_end)
                if imp:
                    x, fx = x2, f2
                    improved_any = True
                    if fx < best:
                        # update globals quickly
                        pass

            if now() >= time_end:
                break

            # Add "delta" direction (Powell)
            delta = [x[i] - x_start[i] for i in range(dim)]
            # if delta is meaningful, replace worst direction with it
            norm = 0.0
            for i in range(dim):
                norm += delta[i] * delta[i]
            if norm > 1e-24:
                # try one line-search along delta
                step0 = 0.0
                cnt = 0
                for i in range(dim):
                    if not fixed[i]:
                        step0 += A[i]
                        cnt += 1
                step0 = (step0 / max(1, cnt)) if cnt else 0.0
                f3, x3, imp = line_search(x, fx, delta, step0, time_end)
                if imp:
                    x, fx = x3, f3
                    improved_any = True
                    # rotate directions: drop first, append delta (keeps size dim)
                    if dirs:
                        dirs.pop(0)
                    # normalize delta to reduce scaling issues
                    invn = 1.0 / math.sqrt(norm)
                    dirs.append([delta[i] * invn for i in range(dim)])

            # early stop if little progress
            if fx >= f_start and not improved_any:
                break

        return fx, x, improved_any

    # ---------- stochastic ES loop + periodic Powell intensification ----------
    span_max = max(spans) if spans else 0.0
    sigma_floor = 1e-12 + 1e-10 * (span_max + 1.0)

    sigma0 = [0.18 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
    sigma = sigma0[:]

    window = max(30, 12 + 2 * dim)
    succ = 0
    trials = 0
    no_improve = 0
    restart_after = 110 + 18 * dim

    x_cur = best_x[:]
    f_cur = best

    next_intensify = now()  # intensify immediately once
    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # periodic deterministic intensification (Powell)
        if t >= next_intensify:
            # short slice; scale with dim but keep small
            slice_len = min(deadline - t, 0.045 + 0.0035 * dim)
            time_end = t + slice_len

            # seed mostly from best, sometimes from other elite
            if elites and random.random() < 0.8:
                seed = elites[0][1][:]
            elif elites:
                seed = random.choice(elites)[1][:]
            else:
                seed = x_cur[:]

            fs = evalf(seed)
            fP, xP, _ = powell_refine(seed, fs, time_end)
            if fP < best:
                best, best_x = fP, xP[:]
                push_elite(best, best_x)
                x_cur, f_cur = xP[:], fP
                no_improve = 0

            # schedule: more often early, less often late
            next_intensify = now() + (0.06 + 0.012 * dim) * (0.45 + 0.90 * frac)

            if now() >= deadline:
                break

        # choose base
        if elites and random.random() < 0.22:
            x_base = random.choice(elites)[1][:]
        else:
            x_base = x_cur

        # mutation type
        r = random.random()
        if r < 0.75:
            # gaussian-ish
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                g = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random() - 3.0) / 3.0
                x_new[i] += g * s
        elif r < 0.91:
            # heavy tail
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                x_new[i] += 0.45 * s * cauchy()
        else:
            # global
            if random.random() < 0.7:
                x_new = halton(idx)
                idx += 1
            else:
                x_new = rand_point()

        x_new = clamp(x_new)
        f_new = evalf(x_new)

        trials += 1
        if f_new <= f_cur:
            x_cur, f_cur = x_new, f_new
            succ += 1

        if f_new < best:
            best, best_x = f_new, x_new[:]
            push_elite(best, best_x)
            no_improve = 0
        else:
            no_improve += 1

        # 1/5 success adaptation
        if trials >= window:
            rate = succ / float(trials)
            if rate > 0.22:
                factor = 1.22
            elif rate < 0.18:
                factor = 0.78
            else:
                factor = 1.0
            if factor != 1.0:
                for i in range(dim):
                    if fixed[i]:
                        continue
                    sigma[i] *= factor
                    if sigma[i] > 0.85 * spans[i]:
                        sigma[i] = 0.85 * spans[i]
                    if sigma[i] < sigma_floor:
                        sigma[i] = sigma_floor
            succ = 0
            trials = 0

        # stagnation restart
        if no_improve >= restart_after:
            if elites and random.random() < 0.9:
                x_cur = (elites[0][1] if random.random() < 0.7 else random.choice(elites)[1])[:]
            else:
                x_cur = halton(idx)
                idx += 1

            # time-aware jitter (smaller later)
            jitter_scale = (0.14 * (1.0 - 0.80 * frac) + 0.015)
            for i in range(dim):
                if fixed[i]:
                    continue
                x_cur[i] += (random.random() * 2.0 - 1.0) * jitter_scale * spans[i]
            x_cur = clamp(x_cur)
            f_cur = evalf(x_cur)
            if f_cur < best:
                best, best_x = f_cur, x_cur[:]
                push_elite(best, best_x)

            # reset sigmas (smaller later)
            shrink = 0.55 * (1.0 - 0.85 * frac) + 0.12
            for i in range(dim):
                sigma[i] = 0.0 if fixed[i] else max(sigma_floor, min(0.65 * spans[i], shrink * sigma0[i]))

            no_improve = 0

    return best
