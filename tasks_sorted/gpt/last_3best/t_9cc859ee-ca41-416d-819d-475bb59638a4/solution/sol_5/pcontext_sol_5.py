import random
import math
import time
import heapq

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (self-contained; stdlib only).

    Improvements over the provided surrogate+JADE:
      - Keep an explicit evaluated history (not just cache) for stronger modeling
      - RBF-like surrogate via ridge regression on random Fourier features (cheap kernel model)
      - Thompson-style sampling from an ensemble of surrogates (robust proposals)
      - DE engine upgraded to a mixed strategy pool (current-to-pbest/1, rand/1, best/2)
      - Stronger restart logic + trust-region local search near the end
      - Better cache with adaptive quantization to reduce duplicate evaluations
    Returns: best fitness (float)
    """

    t_end = time.time() + float(max_time)

    # ---------- bounds / normalization ----------
    lo = [0.0] * dim
    hi = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
    span = [hi[i] - lo[i] for i in range(dim)]
    active = [span[i] > 0.0 for i in range(dim)]
    act_idx = [i for i in range(dim) if active[i]]
    adim = len(act_idx)

    # degenerate: all fixed
    if adim == 0:
        x = [lo[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    def clamp01(z):
        for i in range(dim):
            if z[i] < 0.0:
                z[i] = 0.0
            elif z[i] > 1.0:
                z[i] = 1.0
        return z

    def to_real(z):
        x = [0.0] * dim
        for i in range(dim):
            if active[i]:
                x[i] = lo[i] + z[i] * span[i]
            else:
                x[i] = lo[i]
        return x

    # ---------- RNG helpers ----------
    def rand01():
        return random.random()

    def gauss():
        # Box-Muller
        u1 = rand01()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = rand01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = rand01()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---------- evaluation cache + history ----------
    # adaptive quantization: coarser early, finer later
    cache = {}
    X_hist = []   # normalized points (list of lists)
    y_hist = []   # fitness (float)

    def qstep():
        # decreases with number of evaluations
        n = len(y_hist)
        if n < 200:
            return 2e-6
        if n < 1500:
            return 8e-7
        return 3e-7

    def z_key(z):
        q = qstep()
        inv = 1.0 / q
        return tuple(int(v * inv + 0.5) for v in z)

    def eval_z(z):
        clamp01(z)
        k = z_key(z)
        if k in cache:
            return cache[k]
        fx = float(func(to_real(z)))
        cache[k] = fx
        X_hist.append(z[:])
        y_hist.append(fx)
        return fx

    # ---------- initialization (stratified + opposition + random) ----------
    def lhs_like(n):
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            z = [0.0] * dim
            for j in range(dim):
                if not active[j]:
                    z[j] = 0.0
                else:
                    z[j] = (perms[j][i] + rand01()) / n
            pts.append(z)
        return pts

    def opposite(z):
        return [1.0 - v for v in z]

    # population size: moderate, but enough for mixed DE
    NP = max(22, min(120, 18 + 5 * dim))
    if dim <= 3:
        NP = min(90, max(30, NP))

    cand = []
    base = lhs_like(NP)
    for z in base:
        cand.append(z)
        cand.append(opposite(z))
    for _ in range(min(3 * NP, 60)):
        cand.append([rand01() for _ in range(dim)])

    scored = []
    for z in cand:
        if time.time() >= t_end:
            break
        fz = eval_z(z[:])
        scored.append((fz, z[:]))
    if not scored:
        return float("inf")

    scored.sort(key=lambda t: t[0])
    scored = scored[:NP]
    pop = [z for (f, z) in scored]
    fit = [float(f) for (f, z) in scored]

    bi = min(range(NP), key=lambda i: fit[i])
    best = fit[bi]
    best_z = pop[bi][:]

    # ---------- DE state (JADE-ish + strategy pool) ----------
    mu_F = 0.55
    mu_CR = 0.55
    c_adapt = 0.08
    p_best_rate = 0.18
    arc = []
    arc_max = NP

    stagn = 0
    last_best = best

    # ---------- distances in active subspace ----------
    def dist2(a, b):
        s = 0.0
        for j in act_idx:
            d = a[j] - b[j]
            s += d * d
        return s

    # ---------- local search (trust-region coordinate + gaussian) ----------
    tr = 0.18
    tr_min = 2e-8
    tr_max = 0.45

    def local_polish(budget):
        nonlocal best, best_z, tr, stagn, last_best
        z0 = best_z[:]
        f0 = best
        used = 0

        # gaussian micro-steps
        for _ in range(min(6, budget)):
            if time.time() >= t_end:
                return
            z = z0[:]
            sig = max(0.01, 0.30 * tr)
            for j in act_idx:
                z[j] += sig * gauss()
            clamp01(z)
            fz = eval_z(z)
            used += 1
            if fz < best:
                best, best_z = fz, z[:]
                z0, f0 = z[:], fz
                stagn = 0
                last_best = best
            if used >= budget:
                break

        # coordinate / pattern steps
        order = act_idx[:]
        random.shuffle(order)
        improved = False
        for j in order:
            if time.time() >= t_end or used >= budget:
                break
            s = tr
            if s <= tr_min:
                continue
            for delta in (-s, +s):
                z = z0[:]
                z[j] = z[j] + delta
                if z[j] < 0.0: z[j] = 0.0
                elif z[j] > 1.0: z[j] = 1.0
                if z[j] == z0[j]:
                    continue
                fz = eval_z(z)
                used += 1
                if fz < best:
                    best, best_z = fz, z[:]
                    z0, f0 = z[:], fz
                    improved = True
                    stagn = 0
                    last_best = best
                if used >= budget:
                    break

        # trust-region update
        if improved:
            tr = min(tr_max, tr * 1.18)
        else:
            tr = max(tr_min, tr * 0.72)

    # ---------- surrogate: random Fourier features ridge regression ensemble ----------
    # We build a few cheap surrogates on-the-fly and use them to propose candidates.
    # No external libs: implement small linear solver via Gaussian elimination.

    def solve_linear(A, b):
        """Solve Ax=b for small dense A using Gauss-Jordan; returns x or None."""
        n = len(A)
        # build augmented matrix
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            # pivot
            piv = col
            bestabs = abs(M[piv][col])
            for r in range(col + 1, n):
                v = abs(M[r][col])
                if v > bestabs:
                    bestabs = v
                    piv = r
            if bestabs < 1e-14:
                return None
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
            # normalize
            inv = 1.0 / M[col][col]
            row = M[col]
            for k in range(col, n + 1):
                row[k] *= inv
            # eliminate
            for r in range(n):
                if r == col:
                    continue
                factor = M[r][col]
                if abs(factor) < 1e-18:
                    continue
                rr = M[r]
                for k in range(col, n + 1):
                    rr[k] -= factor * row[k]
        return [M[i][n] for i in range(n)]

    def build_train(maxn):
        # take top elites + some diverse/random from history
        n = len(y_hist)
        if n == 0:
            return []
        maxn = min(maxn, n)

        # elite set indices
        elite_k = min(max(12, maxn // 2), n)
        elite_idx = heapq.nsmallest(elite_k, range(n), key=lambda i: y_hist[i])

        # add diversity: farthest from current best among random candidates
        rest = maxn - len(elite_idx)
        chosen = set(elite_idx)
        if rest > 0:
            trials = min(n, 6 * maxn)
            cand_idx = [random.randrange(n) for _ in range(trials)]
            # greedy farthest
            for _ in range(rest):
                best_i = None
                best_d = -1.0
                for i in cand_idx:
                    if i in chosen:
                        continue
                    d = dist2(X_hist[i], best_z)
                    if d > best_d:
                        best_d = d
                        best_i = i
                if best_i is None:
                    break
                chosen.add(best_i)
        idxs = list(chosen)
        return [(X_hist[i], y_hist[i]) for i in idxs]

    def make_rff_model(train, mfeat, gamma, ridge):
        # features: [cos(w^T x + b)] with w~N(0, 2*gamma I) in active subspace
        # returns (W, B, wlin, ymean)
        n = len(train)
        if n < 8:
            return None

        # prepare random frequencies
        W = [[0.0] * adim for _ in range(mfeat)]
        B = [0.0] * mfeat
        sdev = math.sqrt(2.0 * gamma)
        for k in range(mfeat):
            for t in range(adim):
                W[k][t] = sdev * gauss()
            B[k] = 2.0 * math.pi * rand01()

        # build normal equations: (Phi^T Phi + ridge I) w = Phi^T y
        # include bias term as extra feature
        p = mfeat + 1
        A = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p

        # center y
        ys = [f for (_, f) in train]
        ymean = sum(ys) / float(len(ys))
        for (z, f) in train:
            y = f - ymean
            phi = [0.0] * p
            phi[0] = 1.0
            # compute features
            for k in range(mfeat):
                s = B[k]
                for tt, j in enumerate(act_idx):
                    s += W[k][tt] * z[j]
                phi[k + 1] = math.cos(s)

            # accumulate
            for i in range(p):
                rhs[i] += phi[i] * y
                Ai = A[i]
                vi = phi[i]
                for j in range(i, p):
                    Ai[j] += vi * phi[j]

        # symmetrize + ridge
        for i in range(p):
            for j in range(i):
                A[i][j] = A[j][i]
            A[i][i] += ridge

        wlin = solve_linear(A, rhs)
        if wlin is None:
            return None
        return (W, B, wlin, ymean)

    def rff_predict(model, z):
        W, B, wlin, ymean = model
        mfeat = len(W)
        # bias
        y = wlin[0]
        for k in range(mfeat):
            s = B[k]
            wk = W[k]
            for tt, j in enumerate(act_idx):
                s += wk[tt] * z[j]
            y += wlin[k + 1] * math.cos(s)
        return y + ymean

    def propose_surrogate():
        # Build small ensemble; do Thompson-like pick and acquisition with novelty
        train = build_train(maxn=min(140, max(60, 10 + 4 * dim)))
        if len(train) < 12:
            return [rand01() for _ in range(dim)]

        # fitness scale for novelty weight
        ys = [f for (_, f) in train]
        fmin = min(ys)
        fmax = max(ys)
        fr = max(1e-12, fmax - fmin)
        beta = (0.04 + 0.20 * min(1.0, stagn / 30.0)) * fr  # novelty weight

        # make 2-3 models with different gammas
        models = []
        mfeat = min(36 + 2 * dim, 90)
        gammas = [1.0, 3.0, 0.3]
        ridge = 1e-3
        for g in gammas:
            mdl = make_rff_model(train, mfeat=mfeat, gamma=g, ridge=ridge)
            if mdl is not None:
                models.append(mdl)
        if not models:
            return [rand01() for _ in range(dim)]

        best_cand = None
        best_score = float("inf")

        # sample candidates; score = predicted + exploration penalty (prefer far)
        tries = 80 + 8 * min(dim, 25)
        for _ in range(tries):
            r = rand01()
            if r < 0.55:
                z = best_z[:]
                sig = max(0.015, 0.33 * tr)
                for j in act_idx:
                    z[j] += sig * gauss()
            elif r < 0.85:
                # pick a good training point and perturb
                z0, _ = train[random.randrange(len(train))]
                z = z0[:]
                sig = 0.06 + 0.22 * rand01()
                for j in act_idx:
                    z[j] += sig * (2.0 * rand01() - 1.0)
            else:
                z = [rand01() for _ in range(dim)]
            clamp01(z)

            # pick one model randomly (Thompson-ish)
            mdl = models[random.randrange(len(models))]
            pred = rff_predict(mdl, z)

            # novelty: distance to nearest train point
            md2 = float("inf")
            for (zt, _) in train:
                d2 = dist2(z, zt)
                if d2 < md2:
                    md2 = d2
            score = pred - beta * math.sqrt(md2)

            if score < best_score:
                best_score = score
                best_cand = z[:]

        if best_cand is None:
            best_cand = [rand01() for _ in range(dim)]
        return best_cand

    # ---------- restart / injection ----------
    def inject(count):
        nonlocal best, best_z, stagn, last_best
        for _ in range(count):
            if time.time() >= t_end:
                return
            if rand01() < 0.5:
                z = [rand01() for _ in range(dim)]
            else:
                z = best_z[:]
                sig = 0.20 + 0.25 * rand01()
                for j in act_idx:
                    z[j] += sig * gauss()
            clamp01(z)
            fz = eval_z(z)
            wi = max(range(NP), key=lambda i: fit[i])
            if fz < fit[wi]:
                pop[wi] = z[:]
                fit[wi] = fz
            if fz < best:
                best, best_z = fz, z[:]
                stagn = 0
                last_best = best

    # ---------- main loop ----------
    gen = 0
    while time.time() < t_end:
        gen += 1

        # best/stagnation update
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_z = pop[bi][:]
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        time_left = t_end - time.time()

        # surrogate proposals (a few per generation; more if stagnating)
        if gen % 2 == 0:
            ns = 1 + (1 if stagn > 10 else 0) + (1 if stagn > 25 else 0)
            ns = min(5, ns)
            for _ in range(ns):
                if time.time() >= t_end:
                    return best
                zc = propose_surrogate()
                fc = eval_z(zc)
                if fc < best:
                    best, best_z = fc, zc[:]
                    stagn = 0
                    last_best = best
                # insert if better than worst
                wi = max(range(NP), key=lambda i: fit[i])
                if fc < fit[wi]:
                    pop[wi] = zc[:]
                    fit[wi] = fc

        # late local search
        if time_left < 0.22 * max_time:
            local_polish(budget=6)
        elif time_left < 0.10 * max_time:
            local_polish(budget=10)

        # restart/injection if stagnating
        if stagn > 22 and gen % 4 == 0:
            inject(2)
        if stagn > 45 and gen % 7 == 0:
            inject(4)

        # DE generation
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(p_best_rate * NP))
        top_idx = idx_sorted[:pcount]

        sF = []
        sCR = []

        # strategy probabilities shift with stagnation
        # 0: current-to-pbest/1 (exploit), 1: rand/1 (explore), 2: best/2 (aggressive)
        if stagn < 12:
            p_strat = (0.62, 0.25, 0.13)
        elif stagn < 30:
            p_strat = (0.50, 0.35, 0.15)
        else:
            p_strat = (0.40, 0.45, 0.15)

        for i in range(NP):
            if time.time() >= t_end:
                return best

            # sample CR, F
            CRi = mu_CR + 0.10 * gauss()
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            Fi = mu_F + 0.10 * cauchy()
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = mu_F + 0.10 * cauchy()
                tries += 1
            if Fi <= 0.0: Fi = 0.12
            elif Fi > 1.0: Fi = 1.0

            xi = pop[i]

            # pick distinct indices
            def pick_idx(excl):
                r = excl
                while r == excl:
                    r = random.randrange(NP)
                return r

            r1 = pick_idx(i)
            r2 = pick_idx(i)
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)
            r3 = pick_idx(i)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(NP)
            r4 = pick_idx(i)
            while r4 == i or r4 == r1 or r4 == r2 or r4 == r3:
                r4 = random.randrange(NP)

            # choose from pop or archive for r2/r3 sometimes
            def pick_from_union(excl_i, excl_a, excl_b):
                if arc and rand01() < (len(arc) / float(len(arc) + NP)):
                    return arc[random.randrange(len(arc))]
                r = random.randrange(NP)
                while r == excl_i or r == excl_a or r == excl_b:
                    r = random.randrange(NP)
                return pop[r]

            # choose strategy
            u = rand01()
            if u < p_strat[0]:
                # current-to-pbest/1
                pbest = pop[random.choice(top_idx)]
                x_r1 = pop[r1]
                x_r2 = pick_from_union(i, r1, -1)
                v = [0.0] * dim
                for j in range(dim):
                    v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])
            elif u < p_strat[0] + p_strat[1]:
                # rand/1
                xa = pop[r1]
                xb = pop[r2]
                xc = pick_from_union(i, r1, r2)
                v = [0.0] * dim
                for j in range(dim):
                    v[j] = xa[j] + Fi * (xb[j] - xc[j])
            else:
                # best/2
                xbest = pop[idx_sorted[0]]
                xa = pop[r1]
                xb = pop[r2]
                xc = pop[r3]
                xd = pop[r4]
                v = [0.0] * dim
                for j in range(dim):
                    v[j] = xbest[j] + Fi * (xa[j] - xb[j]) + 0.6 * Fi * (xc[j] - xd[j])

            # crossover
            trial = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if (rand01() < CRi) or (j == jrand):
                    trial[j] = v[j]
            clamp01(trial)

            ftrial = eval_z(trial)
            if ftrial <= fit[i]:
                arc.append(xi[:])
                if len(arc) > arc_max:
                    del arc[random.randrange(len(arc))]
                pop[i] = trial
                fit[i] = ftrial
                sF.append(Fi)
                sCR.append(CRi)
                if ftrial < best:
                    best = ftrial
                    best_z = trial[:]
                    stagn = 0
                    last_best = best

        # adapt mu_F, mu_CR
        if sF:
            sumF = 0.0
            sumF2 = 0.0
            for v in sF:
                sumF += v
                sumF2 += v * v
            lehmerF = (sumF2 / sumF) if sumF > 0.0 else mu_F
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmerF
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(sCR) / float(len(sCR)))
        else:
            mu_F = min(0.95, mu_F * 1.06)
            mu_CR = min(1.0, mu_CR + 0.03)

        if mu_F < 0.05: mu_F = 0.05
        elif mu_F > 0.95: mu_F = 0.95
        if mu_CR < 0.0: mu_CR = 0.0
        elif mu_CR > 1.0: mu_CR = 1.0

    return best
