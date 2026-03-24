import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Upgrade over your best (#3, ES+SA+pattern+restarts):
      - Adds a *real* surrogate-assisted stage using a tiny Gaussian Process (GP)
        with RBF kernel on a small active set (<= ~30 points).
      - Chooses new points via Thompson-like sampling / LCB acquisition (cheap, effective),
        then evaluates only a few promising candidates.
      - Keeps your strong backbone: diversified init (LHS+Halton+corners), ES exploration,
        occasional SA heavy-tail jump, and coordinate pattern refinement.
      - Adds duplicate/near-duplicate rejection in normalized space.

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    eps_time = 1e-4

    # ---------------- basic guards ----------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0

    inv_spans = [1.0 / spans[i] for i in range(dim)]
    avg_span = sum(spans) / float(dim)

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def eval_x(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    def to_unit(x):
        # map to [0,1]
        return [(x[i] - lows[i]) * inv_spans[i] for i in range(dim)]

    def from_unit(u):
        return [clamp(lows[i] + u[i] * spans[i], i) for i in range(dim)]

    # ---------------- RNG helpers ----------------
    def randn():
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = random.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # ---------------- Halton + stratified init ----------------
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

    primes = first_primes(dim)

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        n = index
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(k):
        u = []
        for i in range(dim):
            u.append(van_der_corput(k, primes[i]))
        return from_unit(u)

    def stratified_points(n):
        strata = []
        for i in range(dim):
            vals = [((j + random.random()) / n) for j in range(n)]
            random.shuffle(vals)
            strata.append(vals)
        pts = []
        for j in range(n):
            pts.append(from_unit([strata[i][j] for i in range(dim)]))
        return pts

    # ---------------- "too close" rejection ----------------
    recent = []
    recent_cap = 120
    close_thr2 = (2e-5) ** 2  # in unit-space (normalized)

    def dist2_unit_u(u1, u2):
        s = 0.0
        for i in range(dim):
            d = u1[i] - u2[i]
            s += d * d
        return s

    def accept_new_point(x):
        u = to_unit(x)
        for yu in recent:
            if dist2_unit_u(u, yu) < close_thr2:
                return False
        recent.append(u)
        if len(recent) > recent_cap:
            del recent[0:len(recent) - recent_cap]
        return True

    # ---------------- coordinate pattern search ----------------
    def pattern_search(x0, f0, base_scale):
        x = x0[:]
        fx = f0
        step = [base_scale * spans[i] for i in range(dim)]
        for _round in range(2):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - eps_time:
                    return fx, x
                si = step[i]
                if si <= 0.0:
                    continue
                xi = x[i]

                cand = x[:]
                cand[i] = clamp(xi + si, i)
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        continue

                cand = x[:]
                cand[i] = clamp(xi - si, i)
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True

            step = [v * (0.55 if improved else 0.35) for v in step]
            if not improved:
                break
        return fx, x

    # ---------------- tiny GP surrogate (active set) ----------------
    # RBF kernel in unit space: k(u,v)=exp(-||u-v||^2/(2*ell^2))
    def solve_linear(A, b):
        # Gauss-Jordan with partial pivoting; A is nxn, b is n
        n = len(A)
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            piv = col
            best = abs(M[col][col])
            for r in range(col + 1, n):
                v = abs(M[r][col])
                if v > best:
                    best = v
                    piv = r
            if best < 1e-14:
                return None
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
            div = M[col][col]
            inv = 1.0 / div
            for j in range(col, n + 1):
                M[col][j] *= inv
            for r in range(n):
                if r == col:
                    continue
                factor = M[r][col]
                if factor == 0.0:
                    continue
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
        return [M[i][n] for i in range(n)]

    def build_gp(active_u, active_y):
        n = len(active_u)
        if n < 8:
            return None

        # length-scale from median pairwise distance (unit space)
        dists = []
        for i in range(n):
            ui = active_u[i]
            for j in range(i + 1, n):
                d2 = dist2_unit_u(ui, active_u[j])
                dists.append(d2)
        if not dists:
            return None
        dists.sort()
        med_d2 = dists[len(dists) // 2]
        ell2 = max(1e-6, med_d2)
        ell = math.sqrt(ell2)

        # noise term (nugget)
        nug = 1e-8

        # build K + nug I
        K = [[0.0] * n for _ in range(n)]
        inv2ell2 = 1.0 / (2.0 * ell2)
        for i in range(n):
            K[i][i] = 1.0 + nug
            ui = active_u[i]
            for j in range(i + 1, n):
                d2 = dist2_unit_u(ui, active_u[j])
                kij = math.exp(-d2 * inv2ell2)
                K[i][j] = kij
                K[j][i] = kij

        alpha = solve_linear(K, active_y)
        if alpha is None:
            return None

        def predict(u):
            # mean and cheap variance estimate: var = k(u,u) - k^T K^{-1} k
            # We approximate K^{-1}k by solving K * v = k (another solve).
            # To keep it cheap, we do it only for a handful of candidates.
            kvec = [0.0] * n
            for i in range(n):
                d2 = dist2_unit_u(u, active_u[i])
                kvec[i] = math.exp(-d2 * inv2ell2)
            mu = 0.0
            for i in range(n):
                mu += kvec[i] * alpha[i]

            v = solve_linear(K, kvec)
            if v is None:
                return mu, 1.0
            kv = 0.0
            for i in range(n):
                kv += kvec[i] * v[i]
            var = max(1e-12, 1.0 - kv)  # k(u,u)=1 for RBF
            return mu, var

        return predict

    def select_active(data_u, data_f, max_n):
        # take best half + diversify with farthest-first from a top pool
        n = len(data_u)
        idx = list(range(n))
        idx.sort(key=lambda k: data_f[k])
        top_pool = idx[:min(n, max(5 * max_n, max_n + 20))]

        active = []
        active_f = []
        # start with a few best
        seed = top_pool[:max(4, max_n // 3)]
        for k in seed:
            active.append(data_u[k])
            active_f.append(data_f[k])
            if len(active) >= max_n:
                return active, active_f

        # greedy farthest-first (in unit space)
        while len(active) < min(max_n, len(top_pool)):
            best_k = None
            best_d = -1.0
            for k in top_pool:
                u = data_u[k]
                # distance to nearest active
                md = 1e100
                for a in active:
                    d2 = dist2_unit_u(u, a)
                    if d2 < md:
                        md = d2
                if md > best_d:
                    best_d = md
                    best_k = k
            if best_k is None:
                break
            active.append(data_u[best_k])
            active_f.append(data_f[best_k])
        return active, active_f

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    data_u = []
    data_f = []

    n_lhs = max(18, min(90, 9 * dim))
    n_hal = max(18, min(160, 16 * dim))
    n_rnd = max(10, min(90, 7 * dim))

    init_pts = []
    init_pts.extend(stratified_points(n_lhs))
    init_pts.extend(halton_point(k) for k in range(1, n_hal + 1))
    init_pts.extend(rand_point() for _ in range(n_rnd))
    for _ in range(min(2 * dim, 24)):
        init_pts.append([highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)])

    # elites
    mu = max(6, min(26, 3 + dim // 2))
    elites = []  # list of (f,x)

    def push_elite(fx, x):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > mu:
            elites = elites[:mu]

    for x in init_pts:
        if time.time() >= deadline - eps_time:
            return best
        if not accept_new_point(x):
            continue
        fx = eval_x(x)
        u = to_unit(x)
        data_u.append(u)
        data_f.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(fx, x)

    if best_x is None:
        return best

    # ---------------- main search params ----------------
    lam = max(16, min(100, 8 + 4 * dim))
    sigma = 0.22 * avg_span
    sigma_min = 1e-14 * avg_span + 1e-18
    sigma_max = 1.8 * avg_span

    # SA temperature
    T = 1.0
    T_min = 1e-9
    cool = 0.995

    stall = 0
    restart_after = max(90, 18 * dim)
    last_best = best

    # GP refresh
    gp = None
    next_gp_time = t0 + 0.05

    # local search schedule
    next_local_time = t0 + 0.15
    local_period = 0.20

    while time.time() < deadline - eps_time:
        now = time.time()
        frac = (now - t0) / max(1e-9, max_time)

        # refresh GP occasionally
        if now >= next_gp_time and len(data_u) >= 14:
            max_active = max(12, min(30, 10 + 2 * dim))
            au, af = select_active(data_u, data_f, max_active)
            gp = build_gp(au, af)
            next_gp_time = now + 0.12  # small periodic refresh

        # parents
        elites.sort(key=lambda t: t[0])
        parents = [x for (_, x) in elites[:max(3, min(mu, len(elites)))]]
        parents.append(best_x[:])

        # --- ES batch ---
        for _ in range(lam):
            if time.time() >= deadline - eps_time:
                break
            p = parents[random.randrange(len(parents))]
            child = p[:]
            s = sigma * math.exp(0.25 * randn())
            if s < sigma_min:
                s = sigma_min
            elif s > sigma_max:
                s = sigma_max

            heavy = (random.random() < 0.18)
            for i in range(dim):
                z = cauchy() if heavy else randn()
                child[i] = clamp(child[i] + z * s, i)

            if not accept_new_point(child):
                continue
            fchild = eval_x(child)

            u = to_unit(child)
            data_u.append(u)
            data_f.append(fchild)
            push_elite(fchild, child)

            if fchild < best:
                best, best_x = fchild, child[:]
                stall = 0

        # --- GP-guided proposals (evaluate a few best LCB) ---
        if gp is not None and time.time() < deadline - eps_time:
            # sample candidates around best and around a random elite, rank by LCB
            kappa = 1.5 if frac < 0.6 else 1.1  # more exploit later
            cand_list = []

            def propose_around(base_x, scale, ntry):
                for _ in range(ntry):
                    x = base_x[:]
                    heavy = (random.random() < 0.25)
                    for i in range(dim):
                        z = cauchy() if heavy else randn()
                        x[i] = clamp(x[i] + z * scale, i)
                    u = to_unit(x)
                    mu_hat, var_hat = gp(u)
                    lcb = mu_hat - kappa * math.sqrt(var_hat)
                    cand_list.append((lcb, x, u))

            scale1 = max(sigma_min, min(sigma_max, 0.65 * sigma))
            scale2 = max(sigma_min, min(sigma_max, 1.10 * sigma))

            propose_around(best_x, scale1, 18)
            if elites:
                propose_around(elites[random.randrange(len(elites))][1], scale2, 12)

            cand_list.sort(key=lambda t: t[0])

            # evaluate top few not-too-close
            evals = 0
            for _, x, u in cand_list[:10]:
                if time.time() >= deadline - eps_time:
                    break
                if not accept_new_point(x):
                    continue
                fx = eval_x(x)
                data_u.append(u)
                data_f.append(fx)
                push_elite(fx, x)
                evals += 1
                if fx < best:
                    best, best_x = fx, x[:]
                    stall = 0
                if evals >= 2:
                    break

        # --- SA heavy-tail single move around best ---
        if time.time() < deadline - eps_time:
            x = best_x[:]
            step_scale = max(sigma_min, min(sigma_max, 0.85 * sigma))
            for i in range(dim):
                z = cauchy() if random.random() < 0.35 else randn()
                x[i] = clamp(x[i] + z * step_scale, i)
            if accept_new_point(x):
                fx = eval_x(x)
                u = to_unit(x)
                data_u.append(u)
                data_f.append(fx)
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x[:]
                    stall = 0
                else:
                    df = fx - best
                    if T > T_min and df < 700.0 * T and random.random() < math.exp(-df / max(T, T_min)):
                        pass

        # --- periodic local pattern refinement ---
        local_period = max(0.08, 0.20 * (1.0 - 0.65 * frac))
        if now >= next_local_time and time.time() < deadline - eps_time:
            f2, x2 = pattern_search(best_x, best, base_scale=0.06)
            if f2 < best:
                best, best_x = f2, x2[:]
                push_elite(best, best_x)
                stall = 0
                last_best = best
                sigma = max(sigma_min, sigma * 0.88)
            next_local_time = now + local_period

        # --- adapt sigma / stall ---
        if best < last_best:
            sigma = max(sigma_min, sigma * 0.92)
            last_best = best
        else:
            stall += 1
            if stall % max(10, 2 * dim) == 0:
                sigma = min(sigma_max, sigma * 1.07)

        T = max(T_min, T * cool)

        # --- restart on stagnation ---
        if stall >= restart_after and time.time() < deadline - eps_time:
            stall = 0
            T = 1.0
            sigma = max(sigma_min, min(sigma_max, 0.26 * avg_span))

            new_elites = [(best, best_x[:])]
            m = max(8, mu)
            for _ in range(m - 1):
                if time.time() >= deadline - eps_time:
                    break
                if random.random() < 0.55:
                    y = halton_point(random.randint(1, max(80, 24 * dim)))
                else:
                    y = rand_point()
                a = 0.25 + 0.70 * random.random()
                x = [clamp(a * best_x[i] + (1.0 - a) * y[i], i) for i in range(dim)]
                if not accept_new_point(x):
                    continue
                fx = eval_x(x)
                u = to_unit(x)
                data_u.append(u)
                data_f.append(fx)
                new_elites.append((fx, x[:]))
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
            new_elites.sort(key=lambda t: t[0])
            elites = new_elites[:mu]

    return best
