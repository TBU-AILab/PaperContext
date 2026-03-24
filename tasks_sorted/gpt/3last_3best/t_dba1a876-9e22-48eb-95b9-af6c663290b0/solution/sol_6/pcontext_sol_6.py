import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Key upgrades vs your current best (JADE + diagonal elite model + LS):
      1) Proper diagonal CMA-ES core (rank-μ update, evolution paths, step-size adaptation).
         This is much stronger than a static/refresh elite Gaussian in smooth or noisy landscapes.
      2) DE as a *restart / exploration engine* (JADE current-to-pbest/1 + archive),
         feeding good points into CMA. This avoids CMA getting stuck when initialization is poor.
      3) Time-sliced multi-start: several short "episodes" (CMA+DE) with increasing exploit later.
      4) Very cheap local polish: coordinate + quadratic 1D interpolation late (few dims per loop).
      5) Evaluation cache with quantization + exact fallback to reduce collisions.

    Returns:
      best fitness found within max_time seconds (float).
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

    # ---------- RNG helpers ----------
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
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # ---------- bounds handling ----------
    def mirror_wrap_clip_inplace(x):
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
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

    # ---------- evaluation cache ----------
    cache_q = {}
    cache_exact = {}
    qstep = [max(1e-12, (spans[i] if spans[i] > 0 else 1.0) * 1e-12) for i in range(dim)]

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

    # ---------- low discrepancy init (scrambled Halton) ----------
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

    def halton_point(index):  # index>=1
        return [van_der_corput(index, primes[j], scr[j]) for j in range(dim)]

    def from_unit(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    # ---------- small late-stage coordinate/quadratic polish ----------
    def local_polish(x0, f0, time_limit):
        x = x0[:]
        fx = f0
        steps = [0.06 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
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
            f0 = base_f
            denom = (fm - 2.0 * f0 + fp)
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

        while time.time() < time_limit:
            random.shuffle(order)
            tried = 0
            improved = False
            for i in order:
                if time.time() >= time_limit:
                    break
                h = steps[i]
                if h <= min_step:
                    continue
                tried += 1
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
                    improved = True
                    break
                else:
                    steps[i] *= 0.80

                if tried >= max(2, min(dim, 4)):
                    break

            if not improved:
                # shrink globally if nothing helps
                for j in range(dim):
                    steps[j] *= 0.90
                if max(steps) <= min_step:
                    break
        return x, fx

    # ---------- JADE DE exploration (short bursts) ----------
    def de_explore(seed_pop, seed_fit, time_limit):
        pop = [p[:] for p in seed_pop]
        fit = seed_fit[:]
        n = len(pop)
        if n < 4:
            return min(fit), pop[min(range(n), key=lambda i: fit[i])][:]

        archive = []
        archive_max = n
        mu_F = 0.55
        mu_CR = 0.60
        c = 0.1

        def pick_from_union(exclude_ids):
            union = pop + archive
            for _ in range(18):
                cand = random.choice(union)
                if id(cand) not in exclude_ids:
                    return cand
            return random.choice(union)

        best_i = min(range(n), key=lambda i: fit[i])
        best = fit[best_i]
        best_x = pop[best_i][:]

        last_best = best
        stagn = 0

        while time.time() < time_limit:
            ranked = sorted(range(n), key=lambda i: fit[i])
            frac = 0.0
            # small time-based pbest shrink inside this burst
            pbest_frac = 0.22 - 0.14 * min(1.0, max(0.0, (time.time() - (time_limit - 1e9)) / 1e9))
            pcount = max(2, int(math.ceil(pbest_frac * n)))

            S_F, S_CR = [], []
            for i in range(n):
                if time.time() >= time_limit:
                    break
                xi = pop[i]

                CR = mu_CR + 0.1 * randn()
                if CR < 0.0:
                    CR = 0.0
                elif CR > 1.0:
                    CR = 1.0

                F = mu_F + 0.1 * cauchy0(1.0)
                tries = 0
                while F <= 0.0 and tries < 10:
                    F = mu_F + 0.1 * cauchy0(1.0)
                    tries += 1
                if F <= 0.0:
                    F = 0.1
                if F > 1.0:
                    F = 1.0

                pbest_idx = ranked[random.randrange(pcount)]
                xpbest = pop[pbest_idx]

                exclude = {id(xi), id(xpbest)}
                for _ in range(18):
                    r1 = random.randrange(n)
                    xr1 = pop[r1]
                    if r1 != i and id(xr1) not in exclude:
                        break
                else:
                    xr1 = pop[(i + 1) % n]
                exclude.add(id(xr1))
                xr2 = pick_from_union(exclude)

                v = [xi[j] + F * (xpbest[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]
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
                        best = fu
                        best_x = u[:]

            if S_F:
                sf = sum(S_F)
                if sf > 0.0:
                    mu_F = (1.0 - c) * mu_F + c * (sum(f * f for f in S_F) / sf)
                mu_CR = (1.0 - c) * mu_CR + c * (sum(S_CR) / len(S_CR))

            if best < last_best - 1e-12:
                last_best = best
                stagn = 0
            else:
                stagn += 1
                if stagn >= 4:
                    # inject a few around-best and random
                    order = sorted(range(n), key=lambda k: fit[k])
                    keep = max(6, int(0.6 * n))
                    for idx in order[keep:]:
                        if time.time() >= time_limit:
                            break
                        if random.random() < 0.7:
                            xnew = best_x[:]
                            rad = 0.03
                            for j in range(dim):
                                if spans[j] > 0:
                                    xnew[j] += (random.random() - random.random()) * (rad * spans[j])
                            mirror_wrap_clip_inplace(xnew)
                        else:
                            xnew = rand_vec()
                        fnew = evaluate(xnew)
                        pop[idx] = xnew
                        fit[idx] = fnew
                        if fnew < best:
                            best, best_x = fnew, xnew[:]
                            last_best = best
                    stagn = 0

        return best, best_x

    # ---------- Diagonal CMA-ES episode ----------
    def cma_episode(time_budget, seed_offset):
        ep_deadline = time.time() + time_budget

        # population sizing
        lam = max(12, int(4 + 3 * math.log(dim + 1.0) + 2 * math.sqrt(dim)))
        lam = min(60, max(lam, 12))
        mu = max(2, lam // 2)

        # weights (log), normalized
        w = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
        sw = sum(w) if sum(w) > 0 else 1.0
        w = [wi / sw for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # learning rates (diag)
        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        c1 = 0.0  # no full covariance
        c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

        # expected norm of N(0,I)
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # initial mean: halton + opposition + center
        center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
        u = halton_point(1 + seed_offset)
        xh = from_unit(u)
        xo = opposition(xh)
        # pick best of these 3
        cand = [center, xh, xo]
        f_cand = [evaluate(c) for c in cand]
        bi = min(range(3), key=lambda i: f_cand[i])
        mean = cand[bi][:]

        best = f_cand[bi]
        best_x = mean[:]

        # initial sigma relative to span (normalized)
        avg_span = sum(spans) / max(1, dim)
        sigma = 0.22 * (avg_span if avg_span > 0 else 1.0)

        # diag scaling vector D (std per dim in x-space)
        D = [max(1e-12, 0.30 * spans[i] if spans[i] > 0 else 1e-12) for i in range(dim)]

        # evolution paths (in normalized coordinates)
        ps = [0.0] * dim
        pc = [0.0] * dim

        # small helper: sample one offspring
        def sample_offspring():
            z = [randn() for _ in range(dim)]
            x = [mean[i] + sigma * D[i] * z[i] for i in range(dim)]
            mirror_wrap_clip_inplace(x)
            return x, z

        # seed a small DE population from CMA mean neighborhood for occasional DE bursts
        def make_seed_pop():
            n = max(16, min(48, 10 * dim + 16))
            pop = []
            pop.append(mean[:])
            pop.append(best_x[:])
            # a few halton/opposition around bounds
            k0 = 2 + seed_offset
            while len(pop) < n:
                u0 = halton_point(k0)
                k0 += 1
                x = from_unit(u0)
                pop.append(x)
                if len(pop) < n:
                    pop.append(opposition(x))
            # jitter around mean
            for i in range(len(pop)):
                if time.time() >= ep_deadline:
                    break
                if random.random() < 0.6:
                    x = pop[i][:]
                    for j in range(dim):
                        if spans[j] > 0:
                            x[j] += (random.random() - random.random()) * (0.08 * spans[j])
                    mirror_wrap_clip_inplace(x)
                    pop[i] = x
            pop = pop[:n]
            fit = [evaluate(p) for p in pop]
            return pop, fit

        next_de = time.time() + 0.35 * time_budget
        next_polish = time.time() + 0.70 * time_budget

        while time.time() < ep_deadline:
            # generate population
            xs = []
            zs = []
            fs = []
            for _ in range(lam):
                if time.time() >= ep_deadline:
                    break
                x, z = sample_offspring()
                f = evaluate(x)
                xs.append(x)
                zs.append(z)
                fs.append(f)
                if f < best:
                    best, best_x = f, x[:]

            if not fs:
                break

            # select mu best
            idx = sorted(range(len(fs)), key=lambda i: fs[i])[:mu]

            # recombination in z-space for diag update
            zmean = [0.0] * dim
            xmean = [0.0] * dim
            for wi, ii in zip(w, idx):
                zi = zs[ii]
                xi = xs[ii]
                for j in range(dim):
                    zmean[j] += wi * zi[j]
                    xmean[j] += wi * xi[j]

            mean_old = mean
            mean = xmean

            # update ps (conjugate evolution path for sigma) in normalized space
            # ps <- (1-cs)ps + sqrt(cs(2-cs)mueff) * zmean
            a = math.sqrt(c_sigma * (2.0 - c_sigma) * mueff)
            for j in range(dim):
                ps[j] = (1.0 - c_sigma) * ps[j] + a * zmean[j]

            # sigma update
            ps_norm = math.sqrt(sum(p * p for p in ps))
            sigma *= math.exp((c_sigma / d_sigma) * (ps_norm / chiN - 1.0))
            # clamp sigma to reasonable range w.r.t. spans
            sigma = max(1e-15, min(sigma, 0.8 * (avg_span if avg_span > 0 else 1.0)))

            # update pc (for covariance-like adaptation; diag only)
            # heuristic hsig:
            # hsig = ||ps|| / sqrt(1-(1-cs)^(2*t)) / chiN < (1.4 + 2/(n+1))
            # We'll approximate with a constant threshold (time-bounded, no generation counter).
            hsig = 1.0 if (ps_norm / chiN) < (1.6 + 2.0 / (dim + 1.0)) else 0.0
            b = math.sqrt(c_c * (2.0 - c_c) * mueff) * hsig
            # in x-space, normalized by (sigma*D): use zmean again
            for j in range(dim):
                pc[j] = (1.0 - c_c) * pc[j] + b * (D[j] * zmean[j])

            # diag "cov" update via D^2 update:
            # D^2 <- (1-cmu)D^2 + cmu * sum(wi * (D*zi)^2) + small rank-1 term from pc
            # We'll implement on D directly via D2.
            cmu = c_mu
            for j in range(dim):
                D2 = D[j] * D[j]
                # rank-mu term
                s = 0.0
                for wi, ii in zip(w, idx):
                    zj = zs[ii][j]
                    s += wi * (D[j] * zj) * (D[j] * zj)
                # rank-1-ish from pc
                r1 = (pc[j] * pc[j])
                D2 = (1.0 - cmu) * D2 + cmu * s + 0.08 * r1
                # floors/caps relative to span
                floor = max(1e-15, 1e-3 * spans[j] if spans[j] > 0 else 1e-15)
                cap = (0.9 * spans[j]) if spans[j] > 0 else max(D2, 1e-15)
                D[j] = math.sqrt(max(floor * floor, min(D2, cap * cap)))

            # occasional DE burst to escape when CMA stagnates
            now = time.time()
            if now >= next_de and now < ep_deadline:
                # short DE burst (~10% of remaining episode time)
                rem = ep_deadline - now
                de_budget = min(0.12 * time_budget, 0.25 * rem)
                if de_budget > 0.0:
                    pop, fit = make_seed_pop()
                    b2, bx2 = de_explore(pop, fit, now + de_budget)
                    if b2 < best:
                        best, best_x = b2, bx2[:]
                        mean = best_x[:]
                    # after DE, broaden slightly (helps re-explore around improved mean)
                    sigma = min(sigma * 1.35, 0.8 * (avg_span if avg_span > 0 else 1.0))
                next_de = now + 0.35 * time_budget

            # late polish
            now = time.time()
            if now >= next_polish and now < ep_deadline:
                rem = ep_deadline - now
                pol_budget = min(0.10 * time_budget, 0.35 * rem)
                if pol_budget > 0.0:
                    x3, f3 = local_polish(best_x, best, now + pol_budget)
                    if f3 < best:
                        best, best_x = f3, x3[:]
                        mean = best_x[:]
                next_polish = now + 0.30 * time_budget

        return best, best_x

    # ---------- main: multiple CMA episodes ----------
    # time allocation: keep some final time for polish
    total = float(max_time)
    polish_time = 0.12 * total if total >= 1.0 else 0.08 * total
    main_time = max(0.0, total - polish_time)

    # number of episodes: more when time allows
    episodes = 3 if main_time >= 0.9 else (2 if main_time >= 0.35 else 1)
    per = main_time / max(1, episodes)

    best = float("inf")
    best_x = None

    for e in range(episodes):
        if time.time() >= deadline:
            break
        # jittered budget
        budget = per * (0.85 + 0.3 * random.random())
        budget = min(budget, max(0.0, deadline - time.time() - polish_time))
        if budget <= 0.0:
            break
        b, bx = cma_episode(budget, seed_offset=17 * e + 1)
        if b < best:
            best, best_x = b, bx

    # final polish
    now = time.time()
    if best_x is not None and now < deadline:
        rem = deadline - now
        budget = min(polish_time, rem)
        if budget > 0.0:
            x4, f4 = local_polish(best_x, best, now + budget)
            if f4 < best:
                best = f4

    # fallback
    if best == float("inf"):
        for _ in range(10):
            x = rand_vec()
            best = min(best, evaluate(x))

    return best
