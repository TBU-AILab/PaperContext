import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Hybrid, time-bounded minimizer (self-contained, no external libs).

    Improvements vs the provided set:
      - Uses a true *CMA-ES style* update (full covariance via Cholesky), not diagonal-only.
      - Adds *two-point step-size adaptation (TPA)* for stability under noisy/ill-scaled objectives.
      - Uses *ask/tell* with mirrored bounds + *repair-aware* sampling (reduces boundary bias).
      - Uses *IPOP-style restarts* (increasing population) + stagnation detection.
      - Keeps a small elite archive to seed restarts and do cheap endgame coordinate polish.

    Returns:
      best objective value found within max_time.
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

    # ---------- helpers ----------
    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    # Box-Muller N(0,1)
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        t = 2.0 * math.pi * u2
        z0 = r * math.cos(t)
        z1 = r * math.sin(t)
        _spare = z1
        _has_spare = True
        return z0

    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def mirror_point(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- Halton for initial exploration / restarts ----------
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

    # ---------- small elite archive ----------
    elite_size = max(8, min(28, 10 + int(2.5 * math.sqrt(dim))))
    elites = []  # (f, x)

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

    # ---------- linear algebra (self-contained) ----------
    def eye(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def mat_vec(A, v):
        return [sum(A[i][j] * v[j] for j in range(dim)) for i in range(dim)]

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(dim))

    def outer(a, b):
        return [[a[i] * b[j] for j in range(dim)] for i in range(dim)]

    def mat_add(A, B, alpha=1.0, beta=1.0):
        return [[alpha * A[i][j] + beta * B[i][j] for j in range(dim)] for i in range(dim)]

    def mat_scale(A, s):
        return [[A[i][j] * s for j in range(dim)] for i in range(dim)]

    def symmetrize(A):
        for i in range(dim):
            for j in range(i + 1, dim):
                v = 0.5 * (A[i][j] + A[j][i])
                A[i][j] = v
                A[j][i] = v
        return A

    def cholesky_spd(A):
        # Robust-ish Cholesky with small jitter escalation
        # Returns lower-triangular L such that L*L^T ~ A (SPD).
        jitter = 1e-14
        for _ in range(6):
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
                        if L[j][j] == 0.0:
                            ok = False
                            break
                        L[i][j] = s / L[j][j]
                if not ok:
                    break
            if ok:
                return L
            # add jitter to diagonal and retry
            for i in range(dim):
                A[i][i] += jitter
            jitter *= 100.0
        # fallback: diagonal
        L = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            L[i][i] = math.sqrt(max(1e-24, A[i][i]))
        return L

    # ---------- initialization ----------
    best = float("inf")
    best_x = None

    init_n = max(28, min(240, 36 + 14 * int(math.sqrt(dim))))
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

        # opposition probe (cheap early win sometimes)
        if now() >= deadline:
            return best
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        mirror_point(xo)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]
        push_elite(best, best_x)

    # ---------- CMA-ES core with IPOP restarts ----------
    # restart state
    restart_count = 0
    base_lam = max(8, min(40, 10 + 4 * int(math.sqrt(dim))))
    lam = base_lam
    no_best_gens = 0
    gen = 0

    # persistent endgame polish parameters
    polish_period = max(6, 2 + int(math.sqrt(dim)))
    polish_coords = max(1, min(dim, 12))

    def start_cma(center):
        nonlocal lam
        # strategy parameters (standard-ish)
        mu = lam // 2
        if mu < 2:
            mu = 2
        # log weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # learning rates
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

        # step-size
        sigma = 0.25 * (sum(spans_nz) / float(dim))
        sigma = max(1e-12, sigma)

        # dynamic state
        m = center[:]
        C = eye(dim)
        pc = [0.0] * dim
        ps = [0.0] * dim
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # for TPA
        tpa_delta = 0.25
        return (m, C, pc, ps, sigma, mu, w, mueff, cc, cs, c1, cmu, damps, chiN, tpa_delta)

    # seed center from best
    state = start_cma(best_x)

    # For periodic Cholesky recompute
    L = cholesky_spd([row[:] for row in state[1]])
    L_age = 0

    while now() < deadline:
        gen += 1

        (m, C, pc, ps, sigma, mu, w, mueff, cc, cs, c1, cmu, damps, chiN, tpa_delta) = state

        # recompute cholesky occasionally or when sigma changed a lot
        if L_age <= 0 or (gen % 6 == 0):
            L = cholesky_spd([row[:] for row in C])
            L_age = 6
        else:
            L_age -= 1

        # --- Two-Point Adaptation (TPA): probe along a random direction
        if now() >= deadline:
            break
        z_tpa = [randn() for _ in range(dim)]
        y_tpa = mat_vec(L, z_tpa)
        x_plus = [m[i] + sigma * tpa_delta * y_tpa[i] for i in range(dim)]
        x_minus = [m[i] - sigma * tpa_delta * y_tpa[i] for i in range(dim)]
        mirror_point(x_plus)
        mirror_point(x_minus)
        f_plus = evaluate(x_plus)
        if now() >= deadline:
            push_elite(f_plus, x_plus)
            if f_plus < best:
                best, best_x = f_plus, x_plus[:]
            break
        f_minus = evaluate(x_minus)

        # update elites/best from TPA probes
        push_elite(f_plus, x_plus)
        push_elite(f_minus, x_minus)
        if f_plus < best:
            best, best_x = f_plus, x_plus[:]
        if f_minus < best:
            best, best_x = f_minus, x_minus[:]

        # sigma tweak based on which side is better
        # mild to avoid instability
        if f_plus < f_minus:
            sigma *= math.exp(0.12 / damps)
        elif f_minus < f_plus:
            sigma *= math.exp(-0.12 / damps)
        sigma = max(1e-12, sigma)

        # --- sample population
        pop = []  # (f, x, y, z)
        # boundary mix: occasionally force some coords to bounds to help edge optima
        near_bounds = (random.random() < 0.06)

        for _ in range(lam):
            if now() >= deadline:
                break
            z = [randn() for _ in range(dim)]
            y = mat_vec(L, z)
            x = [m[i] + sigma * y[i] for i in range(dim)]
            if near_bounds:
                for i in range(dim):
                    if spans[i] > 0.0 and random.random() < 0.15:
                        x[i] = lows[i] if random.random() < 0.5 else highs[i]
            mirror_point(x)
            fx = evaluate(x)
            pop.append((fx, x, y, z))

        if not pop:
            break
        pop.sort(key=lambda t: t[0])

        # update elites and best
        for j in range(min(len(pop), max(3, lam // 3))):
            push_elite(pop[j][0], pop[j][1])
        bnew, xbnew = get_best()
        if bnew < best:
            best, best_x = bnew, xbnew
            no_best_gens = 0
        else:
            no_best_gens += 1

        # --- recombination
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

        # --- update evolution paths (using C^{-1/2} * (m-old_m)/sigma ~ z_w)
        # We approximate z_w by solving L * z = y_w => forward substitution since L is lower-tri
        # z_w = L^{-1} y_w
        z_w = [0.0] * dim
        for i in range(dim):
            s = y_w[i]
            for j in range(i):
                s -= L[i][j] * z_w[j]
            diag = L[i][i] if L[i][i] != 0.0 else 1e-18
            z_w[i] = s / diag

        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * z_w[i]

        norm_ps = math.sqrt(sum(ps[i] * ps[i] for i in range(dim)))
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0

        for i in range(dim):
            pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

        # --- covariance update (rank-1 + rank-mu), keep symmetric
        # C = (1 - c1 - cmu) C + c1 * (pc pc^T + (1-hsig)*cc*(2-cc)C) + cmu * sum w_i * y_i y_i^T
        # add correction term when hsig==0
        alpha = (1.0 - c1 - cmu)
        C = mat_scale(C, alpha)
        # rank-1
        pcpc = outer(pc, pc)
        C = mat_add(C, pcpc, alpha=1.0, beta=c1)

        if hsig < 0.5:
            corr = c1 * cc * (2.0 - cc)
            C = mat_add(C, C, alpha=1.0, beta=corr)  # C += corr*C  (using current C as approx)
        # rank-mu
        for i in range(mu):
            wi = w[i]
            yi = pop[i][2]
            yy = outer(yi, yi)
            C = mat_add(C, yy, alpha=1.0, beta=cmu * wi)

        symmetrize(C)

        # keep C well-conditioned: add tiny diagonal tied to span
        eps = 1e-18
        for i in range(dim):
            C[i][i] += eps

        # step-size adaptation (CSA)
        sigma *= math.exp((cs / damps) * (norm_ps / chiN - 1.0))
        sigma = max(1e-12, sigma)

        # occasional mean evaluation to anchor (cheap bookkeeping)
        if gen % 5 == 0 and now() < deadline:
            me = m[:]
            mirror_point(me)
            fme = evaluate(me)
            push_elite(fme, me)
            if fme < best:
                best, best_x = fme, me[:]
                no_best_gens = 0

        # --- restarts (IPOP + stagnation)
        stagnate_lim = 9 + int(2.3 * math.sqrt(dim))
        if no_best_gens >= stagnate_lim:
            restart_count += 1
            no_best_gens = 0

            # increase population (IPOP)
            lam = min(160, max(base_lam, int(base_lam * (2 ** min(4, restart_count)))))

            # pick restart center from elite or Halton
            if elites and random.random() < 0.75:
                # biased choice: better elites more likely
                idx = int((random.random() ** 2) * min(len(elites), 12))
                center = elites[idx][1][:]
            else:
                center = halton_point(hal_k)
                hal_k += 1

            state = start_cma(center)
            L = cholesky_spd([row[:] for row in state[1]])
            L_age = 0
            continue

        # --- endgame coordinate polish (very bounded cost)
        time_left = deadline - now()
        if time_left <= 0:
            break
        endgame = (time_left / float(max_time)) < 0.20 if max_time > 0 else True
        if endgame or (gen % polish_period == 0):
            idxs = list(range(dim))
            # prefer widest spans first (most impactful)
            idxs.sort(key=lambda i: spans_nz[i], reverse=True)
            idxs = idxs[:polish_coords]

            x0 = best_x[:]
            f0 = best
            # small step from sigma
            for i in idxs:
                if now() >= deadline or spans[i] == 0.0:
                    continue
                delta = max(1e-12, (0.35 if endgame else 0.6) * sigma)
                # scale delta to the variable span (sigma is in abs units from spans avg)
                # keep it reasonable relative to span
                delta = min(0.25 * spans_nz[i], delta)

                xp = x0[:]
                xp[i] += delta
                mirror_point(xp)
                fp = evaluate(xp)

                xm = x0[:]
                xm[i] -= delta
                mirror_point(xm)
                fm2 = evaluate(xm)

                if fp < f0 or fm2 < f0:
                    if fp <= fm2:
                        x0, f0 = xp, fp
                    else:
                        x0, f0 = xm, fm2

            if f0 < best:
                best, best_x = f0, x0[:]
                push_elite(f0, x0)

        # commit state
        state = (m, C, pc, ps, sigma, mu, w, mueff, cc, cs, c1, cmu, damps, chiN, tpa_delta)

    return best
