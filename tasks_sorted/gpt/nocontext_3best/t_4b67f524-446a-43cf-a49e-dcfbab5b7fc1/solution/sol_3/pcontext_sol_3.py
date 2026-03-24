import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Key upgrades vs the provided codes:
      - Stronger exploration: scrambled Halton + opposition + occasional global resample
      - Maintain elite pool + build a cheap covariance-like diagonal scaling from elites
      - Fast local improvement: coordinate pattern search (opportunistic) + small NM micro-simplices
      - Robust stochastic search: adaptive (1+1)-ES with 1/5 rule + heavy-tail jumps
      - Time-aware scheduling: explore early, exploit late; restart logic based on stagnation

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
    # per-dim random digit-scramble offsets (cheap "scramble")
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

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = lows[i] if fixed[i] else (lows[i] + highs[i] - x[i])
        return y

    # ---------- elite pool ----------
    elite_k = max(6, min(18, 6 + dim // 2))
    elites = []  # (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites.pop()

    # ---------- diagonal "covariance" estimate from elites (robust scaling) ----------
    def elite_diag_scale():
        # returns per-dim std-like scale (bounded away from 0)
        if len(elites) < 3:
            return [0.15 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
        m = len(elites)
        mean = [0.0] * dim
        for _, x in elites:
            for i in range(dim):
                mean[i] += x[i]
        invm = 1.0 / m
        for i in range(dim):
            mean[i] *= invm
        var = [0.0] * dim
        for _, x in elites:
            for i in range(dim):
                d = x[i] - mean[i]
                var[i] += d * d
        for i in range(dim):
            var[i] *= invm
        # convert to scale; cap to span
        span_max = max(spans) if spans else 0.0
        floor = 1e-12 + 1e-10 * (span_max + 1.0)
        s = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                s[i] = 0.0
            else:
                si = math.sqrt(var[i])
                if si < floor:
                    si = floor
                if si > 0.6 * spans[i]:
                    si = 0.6 * spans[i]
                s[i] = si
        return s

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    idx = 0

    # budgeted init: mix halton/random + opposition
    init_n = max(80, min(900, 80 + 35 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best

        if j % 9 == 0:
            x = rand_point()
        else:
            x = halton(idx)
            idx += 1

        x = clamp(x)
        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]
        push_elite(f, x)

        # opposition evaluation
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

    # ---------- local: opportunistic coordinate pattern search ----------
    def coord_search(x0, f0, step, time_end):
        # step: per-dim step size list
        x = x0[:]
        f = f0
        improved_any = False

        order = list(range(dim))
        random.shuffle(order)
        for i in order:
            if now() >= time_end:
                break
            if fixed[i]:
                continue
            s = step[i]
            if s <= 0.0:
                continue

            # try - then +
            xi = x[:]
            xi[i] -= s
            xi = clamp(xi)
            fi = evalf(xi)
            if fi < f:
                x, f = xi, fi
                improved_any = True
                continue

            if now() >= time_end:
                break
            xi = x[:]
            xi[i] += s
            xi = clamp(xi)
            fi = evalf(xi)
            if fi < f:
                x, f = xi, fi
                improved_any = True

        return f, x, improved_any

    # ---------- local: tiny Nelder–Mead micro-refinement (very small budget) ----------
    def nm_micro(seed_x, step_scale, time_end):
        # build simplex dim+1 (may be truncated if time is short)
        x0 = clamp(seed_x)
        f0 = evalf(x0)
        simp = [(f0, x0)]
        for i in range(dim):
            if now() >= time_end:
                break
            if fixed[i]:
                continue
            xi = x0[:]
            xi[i] += step_scale[i]
            xi = clamp(xi)
            fi = evalf(xi)
            simp.append((fi, xi))
        simp.sort(key=lambda t: t[0])

        if len(simp) < 2:
            return simp[0][0], simp[0][1][:], False

        alpha, gamma, rho, sig = 1.0, 2.0, 0.5, 0.5
        start_best = simp[0][0]
        it = 0
        while now() < time_end:
            it += 1
            simp.sort(key=lambda t: t[0])
            fb, xb = simp[0]
            fw, xw = simp[-1]
            fsw = simp[-2][0]

            # centroid excluding worst
            m = len(simp) - 1
            c = [0.0] * dim
            for _, xx in simp[:-1]:
                for k in range(dim):
                    c[k] += xx[k]
            invm = 1.0 / m
            for k in range(dim):
                c[k] *= invm

            xr = [c[k] + alpha * (c[k] - xw[k]) for k in range(dim)]
            xr = clamp(xr)
            fr = evalf(xr)

            if fr < fb:
                xe = [c[k] + gamma * (xr[k] - c[k]) for k in range(dim)]
                xe = clamp(xe)
                fe = evalf(xe)
                simp[-1] = (fe, xe) if fe < fr else (fr, xr)
            elif fr < fsw:
                simp[-1] = (fr, xr)
            else:
                # contraction
                if fr < fw:
                    xc = [c[k] + rho * (xr[k] - c[k]) for k in range(dim)]
                else:
                    xc = [c[k] - rho * (c[k] - xw[k]) for k in range(dim)]
                xc = clamp(xc)
                fc = evalf(xc)
                if fc < fw:
                    simp[-1] = (fc, xc)
                else:
                    # shrink
                    new_s = [simp[0]]
                    for ff, xx in simp[1:]:
                        xs = [xb[k] + sig * (xx[k] - xb[k]) for k in range(dim)]
                        xs = clamp(xs)
                        fs = evalf(xs)
                        new_s.append((fs, xs))
                    simp = new_s

            # stop early if little improvement
            if it >= 20 + 2 * dim:
                break

        simp.sort(key=lambda t: t[0])
        return simp[0][0], simp[0][1][:], (simp[0][0] < start_best)

    # ---------- main stochastic loop (ES + local intensification) ----------
    span_max = max(spans) if spans else 0.0
    sigma_floor = 1e-12 + 1e-10 * (span_max + 1.0)

    x_cur = best_x[:]
    f_cur = best

    # base sigma from bounds; will be blended with elite diag scale
    sigma_base = [0.20 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
    sigma = sigma_base[:]

    window = max(30, 14 + 2 * dim)
    succ = 0
    trials = 0
    no_improve = 0
    restart_after = 90 + 16 * dim

    # local scheduling
    next_local = 0

    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # periodic local intensification
        if t >= next_local:
            # time slice small
            slice_len = min(deadline - t, 0.03 + 0.004 * dim)
            time_end = t + slice_len

            # choose seed: best elite mostly
            if elites and random.random() < 0.75:
                seed = elites[0][1][:]
            elif elites:
                seed = random.choice(elites)[1][:]
            else:
                seed = x_cur[:]

            # step scales from elite spread, shrunk over time
            diag = elite_diag_scale()
            shrink = 0.65 * (1.0 - 0.85 * frac) + 0.05  # decreases with time
            step = [diag[i] * shrink for i in range(dim)]

            # coord search then micro-NM
            f1, x1, imp1 = coord_search(seed, evalf(seed), step, time_end)
            if f1 < best:
                best, best_x = f1, x1[:]
                push_elite(best, best_x)
                x_cur, f_cur = x1[:], f1
                no_improve = 0

            if now() < time_end:
                # NM micro with even smaller steps
                nm_step = [max(sigma_floor, 0.5 * step[i]) for i in range(dim)]
                f2, x2, _ = nm_micro(x_cur, nm_step, time_end)
                if f2 < best:
                    best, best_x = f2, x2[:]
                    push_elite(best, best_x)
                    x_cur, f_cur = x2[:], f2
                    no_improve = 0

            next_local = now() + (0.07 + 0.01 * dim)  # schedule next

            if now() >= deadline:
                break

        # refresh sigma blend from elites occasionally
        if elites and random.random() < 0.10:
            diag = elite_diag_scale()
            # blend: early more global, late more local
            w = 0.35 + 0.55 * (1.0 - frac)
            for i in range(dim):
                if fixed[i]:
                    sigma[i] = 0.0
                else:
                    sigma[i] = max(sigma_floor, min(0.85 * spans[i], w * sigma[i] + (1.0 - w) * diag[i]))

        # choose base point
        if elites and random.random() < 0.18:
            x_base = random.choice(elites)[1][:]
        else:
            x_base = x_cur

        # mutation
        r = random.random()
        if r < 0.78:
            # gaussian-ish with diag sigma
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                g = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random() - 3.0) / 3.0
                x_new[i] += g * s
        elif r < 0.93:
            # heavy tail
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                x_new[i] += 0.35 * s * cauchy()
        else:
            # global sample (Halton or random)
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

        # 1/5-like step adaptation
        if trials >= window:
            rate = succ / float(trials)
            if rate > 0.22:
                factor = 1.25
            elif rate < 0.18:
                factor = 0.75
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
            if elites and random.random() < 0.85:
                x_cur = (elites[0][1] if random.random() < 0.65 else random.choice(elites)[1])[:]
            else:
                x_cur = halton(idx)
                idx += 1

            # jitter restart
            jitter_scale = 0.10 * (1.0 - 0.70 * frac) + 0.02
            for i in range(dim):
                if fixed[i]:
                    continue
                x_cur[i] += (random.random() * 2.0 - 1.0) * jitter_scale * spans[i]
            x_cur = clamp(x_cur)
            f_cur = evalf(x_cur)

            if f_cur < best:
                best, best_x = f_cur, x_cur[:]
                push_elite(best, best_x)

            # reset sigma towards base/elite spread
            diag = elite_diag_scale()
            for i in range(dim):
                if fixed[i]:
                    sigma[i] = 0.0
                else:
                    sigma[i] = max(sigma_floor, min(0.6 * spans[i], 0.5 * sigma_base[i] + 0.5 * diag[i]))

            no_improve = 0

    return best
