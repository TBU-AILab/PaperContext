import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Compared to your best previous (diag-CMA-like ~24.38), this version typically improves robustness by:
      - Using a *full-covariance* CMA-ES core (Cholesky factor) for rotations/ill-conditioning.
      - Adding *active covariance update* (uses bad solutions too) for faster adaptation.
      - Using *adaptive mirrored resampling* (not just mirror once) to reduce boundary bias.
      - Maintaining a *small trust-region local search* (coordinate + pattern steps) as a cheap finisher.
      - Using *BIPOP-like restarts*: alternates small/large populations with different sigmas.
      - Keeping an elite archive to seed restarts and stabilize progress.

    Returns:
      best objective value (float)
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
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    spans_nz = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    # -------- normal RNG (Box-Muller with spare) --------
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        t = 2.0 * math.pi * u2
        z0 = r * math.cos(t)
        z1 = r * math.sin(t)
        _spare = z1
        _has_spare = True
        return z0

    # -------- bounds handling: mirror + limited resample --------
    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def repair_mirror(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    def sample_in_bounds_from_gaussian(m, step_vec, sigma):
        # propose x = m + sigma*step_vec; if too far out, resample a few times; else mirror
        # (reduces pathological mirror “teleporting” when sigma is huge early)
        for _ in range(3):
            x = [m[i] + sigma * step_vec[i] for i in range(dim)]
            ok = True
            for i in range(dim):
                if x[i] < lows[i] or x[i] > highs[i]:
                    ok = False
                    break
            if ok:
                return x
            # resample step_vec in-place
            for i in range(dim):
                step_vec[i] = step_vec[i] * 0.35 + randn()  # keep some directionality
        x = [m[i] + sigma * step_vec[i] for i in range(dim)]
        return repair_mirror(x)

    # -------- Halton init --------
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

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # -------- elite archive --------
    elite_size = max(10, min(40, 12 + int(3.5 * math.sqrt(dim))))
    elites = []  # sorted list of (f,x)

    def push_elite(fx, x):
        nonlocal elites
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        lo, hi = 0, len(elites)
        while lo < hi:
            mid = (lo + hi) // 2
            if fx < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > elite_size:
            elites.pop()

    def get_best():
        if not elites:
            return float("inf"), None
        return elites[0][0], elites[0][1][:]

    # -------- small linear algebra (self-contained) --------
    def eye(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(dim))

    def mat_vec(L, v):
        # L lower-triangular used as general mat here when needed
        return [sum(L[i][j] * v[j] for j in range(dim)) for i in range(dim)]

    def outer(a, b):
        return [[a[i] * b[j] for j in range(dim)] for i in range(dim)]

    def symmetrize(A):
        for i in range(dim):
            for j in range(i + 1, dim):
                v = 0.5 * (A[i][j] + A[j][i])
                A[i][j] = v
                A[j][i] = v
        return A

    def cholesky_spd(A):
        # Cholesky with jitter escalation
        jitter = 1e-14
        for _ in range(7):
            L = [[0.0] * dim for _ in range(dim)]
            ok = True
            for i in range(dim):
                for j in range(i + 1):
                    s = A[i][j]
                    for k in range(j):
                        s -= L[i][k] * L[j][k]
                    if i == j:
                        if s <= 0.0:
                            ok = False
                            break
                        L[i][j] = math.sqrt(s)
                    else:
                        djj = L[j][j]
                        if djj == 0.0:
                            ok = False
                            break
                        L[i][j] = s / djj
                if not ok:
                    break
            if ok:
                return L
            for i in range(dim):
                A[i][i] += jitter
            jitter *= 100.0
        # fallback diagonal
        L = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            L[i][i] = math.sqrt(max(1e-24, A[i][i]))
        return L

    def forward_solve(L, b):
        # solve L x = b
        x = [0.0] * dim
        for i in range(dim):
            s = b[i]
            for j in range(i):
                s -= L[i][j] * x[j]
            di = L[i][i] if L[i][i] != 0.0 else 1e-18
            x[i] = s / di
        return x

    # -------- initialization --------
    best = float("inf")
    best_x = None

    init_n = max(40, min(320, 48 + 18 * int(math.sqrt(dim))))
    for _ in range(init_n):
        if now() >= deadline:
            return best
        if random.random() < 0.82:
            x = halton_point(hal_k)
            hal_k += 1
        else:
            x = rand_uniform_point()

        fx = evaluate(x)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        if now() >= deadline:
            return best

        # opposition
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        repair_mirror(xo)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]
        push_elite(best, best_x)

    # -------- cheap local polish (coordinate + pattern) --------
    def polish(x, fx, max_coord=12, rounds=2):
        if dim == 0:
            return fx, x
        idxs = list(range(dim))
        idxs.sort(key=lambda i: spans_nz[i], reverse=True)
        idxs = idxs[:max(1, min(dim, max_coord))]

        xbest = x[:]
        fbest = fx
        step_scale = 0.06
        for _ in range(rounds):
            improved = False
            for i in idxs:
                if now() >= deadline or spans[i] == 0.0:
                    continue
                delta = min(0.25 * spans_nz[i], max(1e-12, step_scale * spans_nz[i]))
                xp = xbest[:]
                xp[i] += delta
                repair_mirror(xp)
                fp = evaluate(xp)
                if fp < fbest:
                    # pattern step
                    step = xp[i] - xbest[i]
                    xbest, fbest = xp, fp
                    xpp = xbest[:]
                    xpp[i] += step
                    repair_mirror(xpp)
                    fpp = evaluate(xpp)
                    if fpp < fbest:
                        xbest, fbest = xpp, fpp
                    improved = True
                    continue

                xm = xbest[:]
                xm[i] -= delta
                repair_mirror(xm)
                fm = evaluate(xm)
                if fm < fbest:
                    step = xm[i] - xbest[i]
                    xbest, fbest = xm, fm
                    xmm = xbest[:]
                    xmm[i] += step
                    repair_mirror(xmm)
                    fmm = evaluate(xmm)
                    if fmm < fbest:
                        xbest, fbest = xmm, fmm
                    improved = True
            if not improved:
                step_scale *= 0.45
                if step_scale < 1e-12:
                    break
        return fbest, xbest

    # -------- CMA-ES with active covariance + BIPOP-like restarts --------
    def start_state(center, lam, sigma):
        mu = max(2, lam // 2)
        # log weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # active weights for worst individuals (negative)
        # keep total negative mass modest
        mu_neg = max(1, min(mu, lam - mu))
        wneg = [math.log(mu_neg + 0.5) - math.log(i + 1.0) for i in range(mu_neg)]
        wnegsum = sum(wneg)
        wneg = [wi / wnegsum for wi in wneg]
        neg_mass = 0.20  # fraction of positive mass
        wneg = [-neg_mass * wi for wi in wneg]

        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        m = center[:]
        C = eye(dim)
        pc = [0.0] * dim
        ps = [0.0] * dim
        return {
            "lam": lam, "mu": mu, "w": w, "mueff": mueff, "wneg": wneg, "mu_neg": mu_neg,
            "cc": cc, "cs": cs, "c1": c1, "cmu": cmu, "damps": damps, "chiN": chiN,
            "m": m, "C": C, "pc": pc, "ps": ps, "sigma": sigma,
            "gen": 0, "no_best": 0
        }

    # restart schedule
    base_lam = max(10, min(44, 12 + 4 * int(math.sqrt(dim))))
    big_lam = min(160, max(base_lam, 2 * base_lam))
    small_lam = base_lam
    use_big = False
    restart_id = 0

    # initial sigma tied to spans
    avg_span = sum(spans_nz) / float(dim)
    sigma_start = max(1e-12, 0.22 * avg_span)
    state = start_state(best_x, small_lam, sigma_start)

    L = cholesky_spd([row[:] for row in state["C"]])
    L_age = 0

    while now() < deadline:
        state["gen"] += 1
        gen = state["gen"]
        lam = state["lam"]
        mu = state["mu"]
        w = state["w"]
        wneg = state["wneg"]
        mu_neg = state["mu_neg"]
        mueff = state["mueff"]
        cc = state["cc"]
        cs = state["cs"]
        c1 = state["c1"]
        cmu = state["cmu"]
        damps = state["damps"]
        chiN = state["chiN"]
        m = state["m"]
        C = state["C"]
        pc = state["pc"]
        ps = state["ps"]
        sigma = state["sigma"]

        # refresh cholesky
        if L_age <= 0 or (gen % 6 == 0):
            L = cholesky_spd([row[:] for row in C])
            L_age = 6
        else:
            L_age -= 1

        # sample population
        pop = []  # (f, x, y)
        for _ in range(lam):
            if now() >= deadline:
                break
            z = [randn() for _ in range(dim)]
            y = mat_vec(L, z)  # y ~ N(0,C)
            x = sample_in_bounds_from_gaussian(m, y, sigma)
            fx = evaluate(x)
            pop.append((fx, x, y))

        if len(pop) < max(3, mu):
            break

        pop.sort(key=lambda t: t[0])

        # update elites/best
        for j in range(min(len(pop), max(3, lam // 3))):
            push_elite(pop[j][0], pop[j][1])
        b, bx = get_best()
        if b < best:
            best, best_x = b, bx
            state["no_best"] = 0
        else:
            state["no_best"] += 1

        # recombination
        old_m = m[:]
        m = [0.0] * dim
        y_w = [0.0] * dim
        for i in range(mu):
            wi = w[i]
            xi = pop[i][1]
            yi = pop[i][2]
            for k in range(dim):
                m[k] += wi * xi[k]
                y_w[k] += wi * yi[k]

        # compute z_w = L^{-1} y_w
        z_w = forward_solve(L, y_w)

        # evolution paths
        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * z_w[i]
        norm_ps = math.sqrt(sum(ps[i] * ps[i] for i in range(dim)))
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0
        for i in range(dim):
            pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

        # covariance update (active)
        # baseline decay
        Cnew = [[(1.0 - c1 - cmu) * C[i][j] for j in range(dim)] for i in range(dim)]
        # rank-1
        pcpc = outer(pc, pc)
        for i in range(dim):
            for j in range(dim):
                Cnew[i][j] += c1 * pcpc[i][j]
        # rank-mu positive
        for i_sel in range(mu):
            wi = w[i_sel]
            yi = pop[i_sel][2]
            yy = outer(yi, yi)
            for a in range(dim):
                row = Cnew[a]
                yya = yy[a]
                for b2 in range(dim):
                    row[b2] += cmu * wi * yya[b2]
        # active negative update from worst
        # helps eliminate bad directions; keep stable by limiting to diagonal if necessary
        start_worst = len(pop) - mu_neg
        if start_worst >= mu:
            for t in range(mu_neg):
                wi = wneg[t]
                yi = pop[start_worst + t][2]
                yy = outer(yi, yi)
                for a in range(dim):
                    row = Cnew[a]
                    yya = yy[a]
                    for b2 in range(dim):
                        row[b2] += cmu * wi * yya[b2]

        symmetrize(Cnew)
        # keep SPD via small diagonal tied to span
        eps = 1e-18
        for i in range(dim):
            Cnew[i][i] += eps

        C = Cnew

        # step-size adaptation (CSA)
        sigma *= math.exp((cs / damps) * (norm_ps / chiN - 1.0))
        sigma = max(1e-12, sigma)

        # occasional mean eval
        if gen % 6 == 0 and now() < deadline:
            me = m[:]
            repair_mirror(me)
            fme = evaluate(me)
            push_elite(fme, me)
            if fme < best:
                best, best_x = fme, me[:]
                state["no_best"] = 0

        # endgame polish
        time_left = deadline - now()
        if time_left > 0 and (time_left / float(max_time) < 0.18 or gen % (8 + int(math.sqrt(dim))) == 0):
            fp, xp = polish(best_x, best, max_coord=12, rounds=1)
            push_elite(fp, xp)
            if fp < best:
                best, best_x = fp, xp

        # restart on stagnation
        stagn_lim = 10 + int(2.2 * math.sqrt(dim))
        if state["no_best"] >= stagn_lim:
            restart_id += 1
            state["no_best"] = 0
            use_big = not use_big

            # choose population and sigma
            if use_big:
                lam_new = big_lam
                sigma_new = max(1e-12, (0.45 + 0.15 * random.random()) * avg_span)
            else:
                lam_new = small_lam
                sigma_new = max(1e-12, (0.18 + 0.12 * random.random()) * avg_span)

            # choose new center: from elites mostly; sometimes Halton
            if elites and random.random() < 0.80:
                idx = int((random.random() ** 2) * min(len(elites), 14))
                center = elites[idx][1][:]
            else:
                center = halton_point(hal_k)
                hal_k += 1

            state = start_state(center, lam_new, sigma_new)
            L = cholesky_spd([row[:] for row in state["C"]])
            L_age = 0
            continue

        # commit
        state["m"] = m
        state["C"] = C
        state["pc"] = pc
        state["ps"] = ps
        state["sigma"] = sigma

    return best
