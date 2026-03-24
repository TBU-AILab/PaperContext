import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no numpy).

    Key ideas (aimed at better results under short budgets):
      1) Diversified seeding: Halton + stratified (LHS-like) + corners + random
      2) Fast "ask/tell" style loop with TWO local optimizers:
         - (mu,lambda)-ES with self-adapting sigma (global exploration)
         - Simulated-annealing acceptor with heavy-tailed steps (escape basins)
      3) Cheap deterministic coordinate pattern search around incumbent best
      4) Stagnation-triggered restarts with biased mixing toward best

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ---------------- bounds / helpers ----------------
    if dim <= 0:
        # Degenerate dimension: evaluate once
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0

    avg_span = sum(spans) / float(dim) if dim else 1.0

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

    # ---------------- Halton (cheap LDS) ----------------
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
        x = []
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x.append(lows[i] + u * (highs[i] - lows[i]))
        return x

    # ---------------- LHS-like stratified points ----------------
    def stratified_points(n):
        strata = []
        for i in range(dim):
            vals = [((j + random.random()) / n) for j in range(n)]
            random.shuffle(vals)
            strata.append([lows[i] + v * (highs[i] - lows[i]) for v in vals])
        pts = []
        for j in range(n):
            pts.append([strata[i][j] for i in range(dim)])
        return pts

    # ---------------- normal + heavy tail helpers ----------------
    def randn():
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        # tan(pi*(u-0.5)) heavy-tailed
        u = random.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # keep modest but richer than before
    n_lhs = max(16, min(80, 8 * dim))
    n_hal = max(16, min(140, 14 * dim))
    n_rnd = max(10, min(70, 6 * dim))

    init_pts = []
    init_pts.extend(stratified_points(n_lhs))
    init_pts.extend(halton_point(k) for k in range(1, n_hal + 1))
    init_pts.extend(rand_point() for _ in range(n_rnd))

    # corners-ish
    for _ in range(min(2 * dim, 20)):
        x = [highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)]
        init_pts.append(x)

    # evaluate init
    evaluated = []
    for x in init_pts:
        if time.time() >= deadline - eps_time:
            return best
        fx = eval_x(x)
        evaluated.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        return best

    evaluated.sort(key=lambda t: t[0])

    # ---------------- coordinate pattern search (cheap intensification) ----------------
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
                fc = eval_x(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
                    continue
                cand = x[:]
                cand[i] = clamp(xi - si, i)
                fc = eval_x(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
            step = [v * (0.55 if improved else 0.35) for v in step]
            if not improved:
                break
        return fx, x

    # ---------------- build initial elites for ES ----------------
    mu = max(5, min(24, 3 + dim // 2))
    lam = max(14, min(90, 8 + 4 * dim))
    elites = [(f, x[:]) for (f, x) in evaluated[:max(mu, 8)]]
    elites.sort(key=lambda t: t[0])
    elites = elites[:mu]

    # sigma for ES
    sigma = 0.22 * avg_span
    sigma_min = 1e-14 * avg_span + 1e-18
    sigma_max = 1.8 * avg_span

    # SA temperature: scaled to typical fitness differences (unknown -> heuristic)
    T = 1.0
    T_min = 1e-9
    cool = 0.995

    # stagnation / restart control
    stall = 0
    restart_after = max(80, 18 * dim)
    last_best = best

    # ---------------- main loop ----------------
    it = 0
    while time.time() < deadline - eps_time:
        it += 1

        # --- ES offspring generation (global-ish search) ---
        elites.sort(key=lambda t: t[0])
        parents = [x for (_, x) in elites[:max(3, mu)]]
        # always include best_x as potential parent
        parents.append(best_x[:])

        offspring = []
        for _ in range(lam):
            if time.time() >= deadline - eps_time:
                break
            p = parents[random.randrange(len(parents))]

            # mixture of Gaussian + occasional Cauchy for escapes
            child = p[:]
            # self-adaptive-ish per-child factor
            s = sigma * math.exp(0.25 * randn())
            if s < sigma_min:
                s = sigma_min
            if s > sigma_max:
                s = sigma_max

            heavy = (random.random() < 0.18)
            for i in range(dim):
                z = cauchy() if heavy else randn()
                child[i] = clamp(child[i] + z * s, i)

            fchild = eval_x(child)
            offspring.append((fchild, child))

            # SA-style acceptance into elites candidate pool (helps ridge crossing)
            if fchild < best:
                best, best_x = fchild, child[:]
                stall = 0

        if not offspring:
            break

        # combine and select elites
        pool = elites + offspring
        pool.sort(key=lambda t: t[0])
        elites = pool[:mu]

        # update best from elites
        if elites[0][0] < best:
            best, best_x = elites[0][0], elites[0][1][:]
            stall = 0

        # --- SA move around best (single candidate with Metropolis accept) ---
        if time.time() < deadline - eps_time:
            # propose from best with heavy-tailed step; scale linked to sigma
            x = best_x[:]
            step_scale = max(sigma_min, min(sigma_max, 0.8 * sigma))
            for i in range(dim):
                z = cauchy() if random.random() < 0.35 else randn()
                x[i] = clamp(x[i] + z * step_scale, i)
            fx = eval_x(x)

            accept = False
            if fx <= best:
                accept = True
            else:
                # Metropolis
                df = fx - best
                # protect against overflow/underflow
                if T > T_min and df < 700.0 * T:
                    if random.random() < math.exp(-df / max(T, T_min)):
                        accept = True

            if accept:
                # accept as a new incumbent candidate (not necessarily best)
                elites.append((fx, x[:]))
                elites.sort(key=lambda t: t[0])
                elites = elites[:mu]
                if fx < best:
                    best, best_x = fx, x[:]
                    stall = 0

        # --- sigma adaptation: if improving, slightly reduce for exploitation; else increase a bit ---
        if best < last_best:
            # improvement -> exploit a bit more
            sigma = max(sigma_min, sigma * 0.92)
            last_best = best
        else:
            stall += 1
            if stall % max(12, 2 * dim) == 0:
                sigma = min(sigma_max, sigma * 1.08)

        # cool SA temperature
        T = max(T_min, T * cool)

        # --- occasional intensification ---
        if stall % max(25, 4 * dim) == 0 and time.time() < deadline - eps_time:
            f2, x2 = pattern_search(best_x, best, base_scale=0.07)
            if f2 < best:
                best, best_x = f2, x2[:]
                elites.append((best, best_x[:]))
                elites.sort(key=lambda t: t[0])
                elites = elites[:mu]
                stall = 0
                last_best = best
                sigma = max(sigma_min, sigma * 0.85)

        # --- restart on stagnation ---
        if stall >= restart_after and time.time() < deadline - eps_time:
            stall = 0
            T = 1.0
            sigma = max(sigma_min, min(sigma_max, 0.25 * avg_span))

            # create a few new elites by mixing best with diversified samples
            new_elites = [(best, best_x[:])]
            m = max(6, mu)
            for j in range(m - 1):
                if time.time() >= deadline - eps_time:
                    break
                if random.random() < 0.55:
                    y = halton_point(random.randint(1, max(60, 20 * dim)))
                else:
                    y = rand_point()
                a = 0.25 + 0.70 * random.random()  # bias towards best
                x = [clamp(a * best_x[i] + (1.0 - a) * y[i], i) for i in range(dim)]
                fx = eval_x(x)
                new_elites.append((fx, x))
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
            new_elites.sort(key=lambda t: t[0])
            elites = new_elites[:mu]

    return best
