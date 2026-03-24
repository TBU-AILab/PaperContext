import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Main improvements vs provided code:
      - Adds a tiny surrogate-guided local search (RBF-like using nearest samples) to propose candidates.
      - Upgrades local refinement to a bounded Powell-style direction set search (robust, fast).
      - Stronger restarts: when stagnating, rebuild part of population around best with decreasing radius.
      - Evaluation cache for exact duplicate points (saves time on some functions).
      - Tighter time management (checks inside expensive loops).

    Returns:
      best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    if min(spans) <= 0.0:
        x = [lows[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # ----------------- utilities -----------------
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def reflect_into_bounds(v, lo, hi):
        if lo == hi:
            return lo
        # mirror reflection until inside
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            else:
                v = hi - (v - hi)
        return clamp(v, lo, hi)

    def ensure_bounds(x):
        return [clamp(x[i], lows[i], highs[i]) for i in range(dim)]

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def rand_gauss_vec_around(center, sigma_frac):
        # sigma_frac relative to span; uses reflection
        x = [0.0] * dim
        for i in range(dim):
            sigma = max(1e-15, sigma_frac * spans[i])
            x[i] = reflect_into_bounds(center[i] + random.gauss(0.0, sigma), lows[i], highs[i])
        return x

    def dist2(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    # ----------------- evaluation + cache -----------------
    best = float("inf")
    best_x = None

    # Cheap cache for exact duplicates (rounded)
    cache = {}
    cache_decimals = 12  # enough to catch exact repeats from algorithmic operations

    def key_of(x):
        return tuple(round(v, cache_decimals) for v in x)

    def eval_f(x):
        nonlocal best, best_x
        k = key_of(x)
        if k in cache:
            fx = cache[k]
        else:
            fx = float(func(x))
            cache[k] = fx
        if fx < best:
            best = fx
            best_x = x[:]  # copy
        return fx

    # LHS-ish init
    def init_population_lhs(NP):
        bins = []
        for j in range(dim):
            perm = list(range(NP))
            random.shuffle(perm)
            bins.append(perm)
        pop = []
        for i in range(NP):
            x = [0.0] * dim
            for j in range(dim):
                u = (bins[j][i] + random.random()) / NP
                x[j] = lows[j] + u * spans[j]
            pop.append(x)
        return pop

    # ----------------- parameters -----------------
    NP = int(max(28, min(20 * dim, 240)))

    p_best = 0.2
    A = []
    Amax = NP

    # SHADE memory
    H = 12
    M_CR = [0.5] * H
    M_F = [0.5] * H
    h_idx = 0

    # Stagnation / restart
    stagn = 0
    stagn_limit = 10
    inject_frac = 0.25
    restart_shrink = 0.75  # restart radius shrink factor
    restart_sigma = 0.25   # initial restart sigma (fraction of span)

    # Local refinement schedule
    refine_every = 6
    refine_iters = 1  # keep time-safe

    # Surrogate (sample memory)
    # Keep a moderate sized sample set of (x, f)
    Smax = int(max(200, min(2000, 30 * NP)))
    samples_x = []
    samples_f = []

    def add_sample(x, fx):
        samples_x.append(x[:])
        samples_f.append(float(fx))
        # light pruning: remove worst among random subset if too big
        if len(samples_x) > Smax:
            # remove one of the worst in a small random pool (cheap)
            m = min(20, len(samples_x))
            idxs = [random.randrange(len(samples_x)) for _ in range(m)]
            worst_i = idxs[0]
            worst_f = samples_f[worst_i]
            for ii in idxs[1:]:
                if samples_f[ii] > worst_f:
                    worst_f = samples_f[ii]
                    worst_i = ii
            samples_x.pop(worst_i)
            samples_f.pop(worst_i)

    # ----------------- Powell-like local search (bounded) -----------------
    def line_search_bounded(x, fx, d, alpha0):
        """1D search along direction d with bracketed backtracking + small forward tries."""
        if time.time() >= deadline:
            return x, fx

        # normalize direction loosely to avoid huge steps
        norm = 0.0
        for i in range(dim):
            norm += d[i] * d[i]
        if norm <= 1e-30:
            return x, fx
        inv = 1.0 / math.sqrt(norm)
        dd = [d[i] * inv for i in range(dim)]

        # start step based on alpha0 in absolute units
        alpha = alpha0
        bestx = x[:]
        bestf = fx

        # forward tries (a few)
        for _ in range(4):
            if time.time() >= deadline:
                break
            y = [0.0] * dim
            for i in range(dim):
                y[i] = reflect_into_bounds(x[i] + alpha * dd[i], lows[i], highs[i])
            fy = eval_f(y)
            add_sample(y, fy)
            if fy < bestf:
                bestf, bestx = fy, y
                alpha *= 1.8
            else:
                break

        # backtrack around best found so far
        alpha = alpha0
        for _ in range(8):
            if time.time() >= deadline:
                break
            y = [0.0] * dim
            for i in range(dim):
                y[i] = reflect_into_bounds(x[i] - alpha * dd[i], lows[i], highs[i])
            fy = eval_f(y)
            add_sample(y, fy)
            if fy < bestf:
                bestf, bestx = fy, y
            alpha *= 0.5
            if alpha <= 1e-14:
                break

        return bestx, bestf

    def local_refine_powell(x0, f0, iters):
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        # Initial direction set: coordinate axes
        dirs = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            dirs[i][i] = 1.0

        # initial step (absolute)
        base = 0.15 * (sum(spans) / float(dim))

        for _ in range(iters):
            if time.time() >= deadline:
                break
            x_start = x[:]
            f_start = fx

            # sweep directions
            biggest_drop = 0.0
            best_dir_idx = -1

            for k in range(dim):
                if time.time() >= deadline:
                    break
                x_new, f_new = line_search_bounded(x, fx, dirs[k], base)
                drop = fx - f_new
                if drop > biggest_drop:
                    biggest_drop = drop
                    best_dir_idx = k
                x, fx = x_new, f_new

            # extrapolated direction
            if time.time() >= deadline:
                break
            d_ex = [x[i] - x_start[i] for i in range(dim)]
            x_new, f_new = line_search_bounded(x, fx, d_ex, base)
            x, fx = x_new, f_new

            # update directions (Powell): replace best-improving direction
            if best_dir_idx >= 0:
                dirs[best_dir_idx] = d_ex[:] if any(abs(v) > 0.0 for v in d_ex) else dirs[best_dir_idx]

            # step adaptation
            if fx < f_start:
                base *= 1.05
            else:
                base *= 0.7
            base = max(base, 1e-12)

        return x, fx

    # ----------------- tiny surrogate proposer -----------------
    def surrogate_propose():
        """
        Build a very cheap RBF-like score using K nearest samples (by Euclidean distance),
        then sample candidates near the current best and pick the best predicted.
        """
        if best_x is None or len(samples_x) < max(10, 4 * dim):
            return None

        K = min(12, len(samples_x))
        tries = 18  # keep cheap
        sigma = max(0.03, 0.18 * (restart_sigma))  # relative to spans; tied to restart_sigma

        best_pred = float("inf")
        best_cand = None

        # Preselect a small pool of recent/best samples to speed nearest search a bit
        # (random subset + ensure best few)
        pool = []
        n = len(samples_x)
        m = min(120, n)
        for _ in range(m):
            pool.append(random.randrange(n))

        # ensure best few included
        best_idx = sorted(range(n), key=lambda i: samples_f[i])[:min(10, n)]
        for bi in best_idx:
            pool.append(bi)

        for _ in range(tries):
            if time.time() >= deadline:
                break
            cand = rand_gauss_vec_around(best_x, sigma)
            # find K nearest in pool
            dlist = []
            for ii in pool:
                d2 = dist2(cand, samples_x[ii])
                dlist.append((d2, samples_f[ii]))
            dlist.sort(key=lambda t: t[0])
            dlist = dlist[:K]

            # inverse-distance weighted average (add small epsilon)
            num = 0.0
            den = 0.0
            for d2, f in dlist:
                w = 1.0 / (1e-12 + d2)
                num += w * f
                den += w
            pred = num / den if den > 0.0 else float("inf")

            if pred < best_pred:
                best_pred = pred
                best_cand = cand

        return best_cand

    # ----------------- initialize -----------------
    pop = init_population_lhs(NP)
    fit = [0.0] * NP

    for i in range(NP):
        if time.time() >= deadline:
            return best
        fi = eval_f(pop[i])
        fit[i] = fi
        add_sample(pop[i], fi)

    last_best = best
    gen = 0

    # ----------------- main loop -----------------
    while time.time() < deadline:
        gen += 1

        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(p_best * NP))
        pset = idx_sorted[:pcount]

        # generation success sets for SHADE
        S_CR, S_F, S_df = [], [], []

        # Occasionally attempt surrogate proposal (one eval)
        if (gen % 3 == 0) and time.time() < deadline:
            cand = surrogate_propose()
            if cand is not None:
                fc = eval_f(cand)
                add_sample(cand, fc)
                # Replace worst if it helps (or just inject as diversity)
                worst_i = max(range(NP), key=lambda i: fit[i])
                if fc < fit[worst_i] or random.random() < 0.25:
                    pop[worst_i] = cand
                    fit[worst_i] = fc

        for i in range(NP):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)
            muCR = M_CR[r]
            muF = M_F[r]

            # CR ~ N(mu,0.1) truncated
            CRi = muCR + 0.1 * random.gauss(0.0, 1.0)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # F ~ Cauchy(mu,0.1) truncated to (0,1]
            Fi = -1.0
            for _ in range(20):
                u = random.random()
                Fi = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < Fi <= 1.0:
                    break
            if Fi <= 0.0:
                Fi = min(1.0, max(0.08, muF))

            pbest = random.choice(pset)

            def pick_pop_index(exclude_set):
                for _ in range(60):
                    a = random.randrange(NP)
                    if a not in exclude_set:
                        return a
                return random.randrange(NP)

            r1 = pick_pop_index({i, pbest})
            poolN = NP + len(A)

            r2 = None
            for _ in range(80):
                t = random.randrange(poolN)
                if t < NP:
                    if t == i or t == pbest or t == r1:
                        continue
                    r2 = ("P", t)
                else:
                    r2 = ("A", t - NP)
                break
            if r2 is None:
                r2 = ("P", pick_pop_index({i, pbest, r1}))

            xi = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]
            xr2 = pop[r2[1]] if r2[0] == "P" else A[r2[1]]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = reflect_into_bounds(vj, lows[j], highs[j])

            # crossover
            uvec = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    uvec[j] = v[j]
            uvec = ensure_bounds(uvec)

            fu = eval_f(uvec)
            add_sample(uvec, fu)

            if fu <= fit[i]:
                # archive
                A.append(xi)
                if len(A) > Amax:
                    A.pop(random.randrange(len(A)))

                df = fit[i] - fu
                pop[i] = uvec
                fit[i] = fu

                S_CR.append(CRi)
                S_F.append(Fi)
                S_df.append(max(1e-12, df))

        # SHADE update
        if S_F:
            wsum = sum(S_df)
            weights = [d / wsum for d in S_df] if wsum > 0.0 else [1.0 / len(S_df)] * len(S_df)

            mcr = 0.0
            for w, cr in zip(weights, S_CR):
                mcr += w * cr

            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            mf = (num / den) if den > 0.0 else 0.5

            M_CR[h_idx] = mcr
            M_F[h_idx] = mf
            h_idx = (h_idx + 1) % H

        # stagnation tracking
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # periodic local refinement on incumbent best
        if (gen % refine_every == 0) and (time.time() < deadline) and best_x is not None:
            bx, bf = local_refine_powell(best_x, best, refine_iters)
            best_x, best = bx, bf

        # restart / injection when stagnating
        if stagn >= stagn_limit and time.time() < deadline:
            stagn = 0

            # shrink restart radius over time to intensify
            restart_sigma *= restart_shrink
            restart_sigma = max(0.01, restart_sigma)

            k = max(2, int(inject_frac * NP))
            worst = sorted(range(NP), key=lambda ii: fit[ii], reverse=True)[:k]

            for idx in worst:
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.85:
                    xnew = rand_gauss_vec_around(best_x, restart_sigma)
                    # occasional long jump
                    if random.random() < 0.15:
                        xnew = rand_uniform_vec()
                else:
                    xnew = rand_uniform_vec()

                fnew = eval_f(xnew)
                add_sample(xnew, fnew)
                pop[idx] = xnew
                fit[idx] = fnew

    return best
