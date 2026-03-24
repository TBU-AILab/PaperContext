import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained; no external libs).

    Why this should improve over your ~20.17 hybrid:
      - Uses a *real* sep-CMA-ES (diagonal covariance) as the main optimizer:
        evolution paths + step-size control + rank-µ diagonal update.
      - Keeps a JADE current-to-pbest/1 DE burst as a *diversification / restart operator*
        when CMA stagnates or periodically (short, bounded).
      - Strong, low-overhead initialization: center + scrambled Halton + opposition + a few corners.
      - Final polish: coordinate pattern + 1D quadratic interpolation + small (1+1)-ES with adaptive sigma.
      - Evaluation cache: exact + quantized (reduces wasted calls without harmful over-collisions).

    Returns:
        best fitness (float) found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)
    if dim <= 0 or max_time <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    # ---------------- RNG helpers ----------------
    _bm_has = False
    _bm_val = 0.0

    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _bm_val = z1
        _bm_has = True
        return z0

    def cauchy0(scale=1.0):
        # centered Cauchy with scale
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # ---------------- bounds handling ----------------
    def mirror_wrap_clip_inplace(x):
        # mirror-wrap into [lo,hi] with period 2*span, then clip
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            s = spans[i]
            if s <= 0.0:
                x[i] = lo
                continue
            v = x[i]
            if v < lo or v > hi:
                t = (v - lo) % (2.0 * s)
                if t < 0.0:
                    t += 2.0 * s
                if t > s:
                    t = 2.0 * s - t
                v = lo + t
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            x[i] = v
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---------------- evaluation cache (exact + quantized) ----------------
    cache_q = {}
    cache_exact = {}
    # quant step: tiny fraction of span, but not too tiny
    qstep = [max(1e-12, (spans[i] if spans[i] > 0 else 1.0) * 1e-11) for i in range(dim)]

    def key_q(x):
        return tuple(int(round(x[i] / qstep[i])) for i in range(dim))

    def evaluate(x):
        ke = tuple(x)
        v = cache_exact.get(ke)
        if v is not None:
            return v
        kq = key_q(x)
        v = cache_q.get(kq)
        if v is None:
            v = float(func(list(x)))
            cache_q[kq] = v
        cache_exact[ke] = v
        return v

    # ---------------- scrambled Halton init ----------------
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

    def van_der_corput(k, base, scramble=0):
        v = 0.0
        denom = 1.0
        while k > 0:
            k, rem = divmod(k, base)
            rem = (rem + scramble) % base
            denom *= base
            v += rem / denom
        return v

    primes = first_primes(max(1, dim))
    scr = [random.randrange(primes[i]) for i in range(dim)]

    def halton_point(index):  # index >= 1
        return [van_der_corput(index, primes[j], scr[j]) for j in range(dim)]

    def from_unit(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    # ---------------- local polish: coord + quad + (1+1)-ES ----------------
    def local_polish(x0, f0, time_limit):
        x = x0[:]
        fx = f0

        steps = [0.06 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        sig = [0.05 for _ in range(dim)]
        min_step = 1e-14
        order = list(range(dim))

        def axis_quad(i, base_x, base_f, h):
            if h <= 0.0:
                return None, None
            xm = base_x[:]
            xp = base_x[:]
            xm[i] -= h
            xp[i] += h
            mirror_wrap_clip_inplace(xm)
            mirror_wrap_clip_inplace(xp)
            fm = evaluate(xm)
            fp = evaluate(xp)
            denom = (fm - 2.0 * base_f + fp)
            if denom == 0.0:
                return None, None
            xstar = h * (fm - fp) / (2.0 * denom)
            if abs(xstar) > 1.8 * h:
                return None, None
            xq = base_x[:]
            xq[i] += xstar
            mirror_wrap_clip_inplace(xq)
            fq = evaluate(xq)
            return xq, fq

        succ = 0
        trials = 0

        while time.time() < time_limit:
            trials += 1

            # (1+1)-ES
            y = x[:]
            for j in range(dim):
                if spans[j] > 0:
                    y[j] += randn() * (sig[j] * spans[j])
            mirror_wrap_clip_inplace(y)
            fy = evaluate(y)
            if fy < fx:
                x, fx = y, fy
                succ += 1
            else:
                random.shuffle(order)
                improved = False
                for i in order[:max(2, min(dim, 5))]:
                    h = steps[i]
                    if h <= min_step:
                        continue
                    xp = x[:]
                    xm = x[:]
                    xp[i] += h
                    xm[i] -= h
                    mirror_wrap_clip_inplace(xp)
                    mirror_wrap_clip_inplace(xm)
                    fp = evaluate(xp)
                    fm = evaluate(xm)

                    best_local_f = fx
                    best_local_x = None
                    if fp < best_local_f:
                        best_local_f, best_local_x = fp, xp
                    if fm < best_local_f:
                        best_local_f, best_local_x = fm, xm

                    xq, fq = axis_quad(i, x, fx, h)
                    if xq is not None and fq < best_local_f:
                        best_local_f, best_local_x = fq, xq

                    if best_local_x is not None:
                        x, fx = best_local_x, best_local_f
                        succ += 1
                        improved = True
                        break
                    else:
                        steps[i] *= 0.82

                if not improved:
                    for j in range(dim):
                        steps[j] *= 0.92
                        sig[j] *= 0.97

            # adapt sig
            if trials % 24 == 0:
                rate = succ / 24.0
                if rate > 0.25:
                    mul = 1.20
                elif rate < 0.15:
                    mul = 0.78
                else:
                    mul = 1.0
                for j in range(dim):
                    sig[j] = max(1e-6, min(0.30, sig[j] * mul))
                succ = 0

            if max(steps) <= min_step:
                break

        return x, fx

    # ---------------- JADE DE burst (short) ----------------
    def de_burst(pop, fit, time_limit):
        n = len(pop)
        if n < 4:
            bi = min(range(n), key=lambda i: fit[i])
            return fit[bi], pop[bi][:]

        archive = []
        archive_max = n
        mu_F, mu_CR = 0.55, 0.60
        c = 0.1

        def pick_from_union(exclude_ids):
            union = pop + archive
            for _ in range(20):
                cand = random.choice(union)
                if id(cand) not in exclude_ids:
                    return cand
            return random.choice(union)

        best_i = min(range(n), key=lambda i: fit[i])
        best = fit[best_i]
        best_x = pop[best_i][:]

        while time.time() < time_limit:
            ranked = sorted(range(n), key=lambda i: fit[i])
            pcount = max(2, int(0.12 * n))

            S_F, S_CR = [], []
            for i in range(n):
                if time.time() >= time_limit:
                    break

                xi = pop[i]
                CR = mu_CR + 0.1 * randn()
                CR = 0.0 if CR < 0.0 else (1.0 if CR > 1.0 else CR)

                F = mu_F + 0.1 * cauchy0(1.0)
                tries = 0
                while F <= 0.0 and tries < 10:
                    F = mu_F + 0.1 * cauchy0(1.0)
                    tries += 1
                if F <= 0.0:
                    F = 0.1
                if F > 1.0:
                    F = 1.0

                pbest = pop[ranked[random.randrange(pcount)]]
                exclude = {id(xi), id(pbest)}

                for _ in range(20):
                    r1 = random.randrange(n)
                    xr1 = pop[r1]
                    if r1 != i and id(xr1) not in exclude:
                        break
                else:
                    xr1 = pop[(i + 1) % n]
                exclude.add(id(xr1))

                xr2 = pick_from_union(exclude)

                v = [xi[j] + F * (pbest[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]
                u = xi[:]
                jrand = random.randrange(dim)
                for j in range(dim):
                    if random.random() < CR or j == jrand:
                        u[j] = v[j]
                mirror_wrap_clip_inplace(u)
                fu = evaluate(u)

                if fu <= fit[i]:
                    archive.append(xi)
                    if len(archive) > archive_max:
                        archive.pop(random.randrange(len(archive)))
                    pop[i] = u
                    fit[i] = fu
                    S_F.append(F)
                    S_CR.append(CR)
                    if fu < best:
                        best, best_x = fu, u[:]

            if S_F:
                sf = sum(S_F)
                if sf > 0.0:
                    mu_F = (1.0 - c) * mu_F + c * (sum(f * f for f in S_F) / sf)
                mu_CR = (1.0 - c) * mu_CR + c * (sum(S_CR) / len(S_CR))

        return best, best_x

    # ---------------- sep-CMA-ES episode ----------------
    def cma_episode(time_budget, seed_index):
        ep_deadline = min(deadline, time.time() + time_budget)

        lam = max(10, int(4 + 3 * math.log(dim + 1.0) + 2 * math.sqrt(dim)))
        lam = min(72, max(lam, 10))
        mu = max(2, lam // 2)

        w = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
        sw = sum(w) if sum(w) > 0 else 1.0
        w = [wi / sw for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        c_mu = min(1.0, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
        u = from_unit(halton_point(max(1, seed_index)))
        cand = [center, u, opposition(u), rand_vec()]
        fc = [evaluate(x) for x in cand]
        bi = min(range(len(cand)), key=lambda i: fc[i])
        mean = cand[bi][:]
        best = fc[bi]
        best_x = mean[:]

        avg_span = sum(spans) / max(1, dim)
        base = avg_span if avg_span > 0 else 1.0
        sigma = 0.22 * base

        D = [max(1e-12, 0.30 * spans[i] if spans[i] > 0 else 1e-12) for i in range(dim)]
        ps = [0.0] * dim
        pc = [0.0] * dim

        def sample():
            z = [randn() for _ in range(dim)]
            x = [mean[i] + sigma * D[i] * z[i] for i in range(dim)]
            mirror_wrap_clip_inplace(x)
            return x, z

        last_best = best
        stagn = 0
        next_de = time.time() + 0.35 * time_budget

        while time.time() < ep_deadline:
            xs, zs, fs = [], [], []
            for _ in range(lam):
                if time.time() >= ep_deadline:
                    break
                x, z = sample()
                f = evaluate(x)
                xs.append(x)
                zs.append(z)
                fs.append(f)
                if f < best:
                    best, best_x = f, x[:]

            if not fs:
                break

            idx = sorted(range(len(fs)), key=lambda i: fs[i])[:mu]

            zmean = [0.0] * dim
            xmean = [0.0] * dim
            for wi, ii in zip(w, idx):
                zi = zs[ii]
                xi = xs[ii]
                for j in range(dim):
                    zmean[j] += wi * zi[j]
                    xmean[j] += wi * xi[j]
            mean = xmean

            a = math.sqrt(c_sigma * (2.0 - c_sigma) * mueff)
            for j in range(dim):
                ps[j] = (1.0 - c_sigma) * ps[j] + a * zmean[j]
            ps_norm = math.sqrt(sum(p * p for p in ps))
            sigma *= math.exp((c_sigma / d_sigma) * (ps_norm / chiN - 1.0))
            sigma = max(1e-15, min(sigma, 0.9 * base))

            hsig = 1.0 if (ps_norm / chiN) < (1.6 + 2.0 / (dim + 1.0)) else 0.0
            b = math.sqrt(c_c * (2.0 - c_c) * mueff) * hsig
            for j in range(dim):
                pc[j] = (1.0 - c_c) * pc[j] + b * (D[j] * zmean[j])

            for j in range(dim):
                Dj2 = D[j] * D[j]
                s = 0.0
                for wi, ii in zip(w, idx):
                    zj = zs[ii][j]
                    s += wi * (D[j] * zj) * (D[j] * zj)
                Dj2 = (1.0 - c_mu) * Dj2 + c_mu * s + 0.06 * (pc[j] * pc[j])

                floor = max(1e-15, 7e-4 * spans[j] if spans[j] > 0 else 1e-15)
                cap = 0.95 * spans[j] if spans[j] > 0 else max(1e-12, math.sqrt(Dj2))
                D[j] = math.sqrt(max(floor * floor, min(Dj2, cap * cap)))

            if best < last_best - 1e-12:
                last_best = best
                stagn = 0
            else:
                stagn += 1

            now = time.time()
            if (stagn >= 4 or now >= next_de) and now + 1e-9 < ep_deadline:
                n = max(18, min(52, 9 * dim + 18))
                pop = [best_x[:], mean[:]]
                k0 = seed_index + 3
                while len(pop) < n:
                    pop.append(from_unit(halton_point(k0)))
                    k0 += 1
                    if len(pop) < n:
                        pop.append(rand_vec())
                for i in range(len(pop)):
                    if random.random() < 0.55:
                        x = pop[i][:]
                        for j in range(dim):
                            if spans[j] > 0:
                                x[j] += (random.random() - random.random()) * (0.05 * spans[j])
                        mirror_wrap_clip_inplace(x)
                        pop[i] = x
                fit = [evaluate(p) for p in pop]
                de_time = min(0.10 * time_budget, 0.25 * (ep_deadline - now))
                if de_time > 0.0:
                    b2, x2 = de_burst(pop, fit, now + de_time)
                    if b2 < best:
                        best, best_x = b2, x2[:]
                        mean = best_x[:]
                        sigma = min(0.9 * base, sigma * 1.35)
                stagn = 0
                next_de = time.time() + 0.33 * time_budget

        return best, best_x

    # ---------------- main scheduling ----------------
    total = float(max_time)
    polish_time = (0.16 * total) if total >= 1.0 else (0.10 * total)
    main_time = max(0.0, total - polish_time)

    episodes = 5 if main_time >= 1.5 else (4 if main_time >= 1.0 else (3 if main_time >= 0.6 else (2 if main_time >= 0.3 else 1)))
    per = main_time / max(1, episodes)

    best = float("inf")
    best_x = None

    # initial probing
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    best = evaluate(center)
    best_x = center[:]
    for k in range(1, min(10, 2 * dim + 4)):
        x = from_unit(halton_point(k))
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]
        xo = opposition(x)
        fo = evaluate(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    for e in range(episodes):
        if time.time() >= deadline:
            break
        jitter = 0.85 + 0.35 * random.random()
        budget = per * jitter
        budget = min(budget, max(0.0, deadline - time.time() - polish_time))
        if budget <= 0.0:
            break
        b, bx = cma_episode(budget, seed_index=17 + 61 * e)
        if b < best:
            best, best_x = b, bx[:]

    now = time.time()
    if best_x is not None and now < deadline:
        rem = deadline - now
        budget = min(polish_time, rem)
        if budget > 0.0:
            x2, f2 = local_polish(best_x, best, now + budget)
            if f2 < best:
                best = f2

    if best == float("inf"):
        for _ in range(12):
            best = min(best, evaluate(rand_vec()))

    return best
