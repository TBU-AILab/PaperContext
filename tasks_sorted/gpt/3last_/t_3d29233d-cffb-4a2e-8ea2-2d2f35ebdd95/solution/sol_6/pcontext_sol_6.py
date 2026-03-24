import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    What’s improved vs your previous three variants:
      - Stronger global exploration + faster convergence: Differential Evolution with
        * current-to-pbest mutation (JADE-style) + archive (better diversity)
        * adaptive F and CR learned from successful steps (SHADE-style)
      - Budget-aware: steady-state DE (1 eval per individual per generation) and
        early-exit time checks.
      - Robust bound handling: reflection.
      - Lightweight local search on best: small coordinate/pattern search.
      - Adaptive restarts on stagnation with elite preservation.

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # -------------------- helpers --------------------
    def eval_f(x):
        return float(func(x))

    def clip(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        if t <= w:
            return lo + t
        return hi - (t - w)

    def widths():
        return [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    W = widths()

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def lhs_points(n):
        # LHS-like stratified sampling per dimension
        strata = []
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            strata.append(perm)
        pts = []
        for j in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    x[d] = lo
                else:
                    u = (strata[d][j] + random.random()) / n
                    x[d] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
            else:
                y[i] = reflect(lo + hi - x[i], lo, hi)
        return y

    def pattern_refine(x0, f0, step, sweeps=2):
        # cheap coordinate search + pattern move
        x = x0[:]
        fx = f0
        steps = step[:]
        for _ in range(sweeps):
            base = x[:]
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                si = steps[i]
                if si <= 0.0:
                    continue
                lo, hi = bounds[i]
                if hi <= lo:
                    x[i] = lo
                    continue

                xi = x[i]
                xp = x[:]
                xp[i] = clip(xi + si, lo, hi)
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[i] = clip(xi - si, lo, hi)
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if improved:
                d = [x[i] - base[i] for i in range(dim)]
                xt = [reflect(x[i] + d[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft
            else:
                for i in range(dim):
                    steps[i] *= 0.5
        return x, fx

    def mean(vals):
        return sum(vals) / max(1, len(vals))

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None

    # Population size: moderate (good under strict time)
    NP = max(22, min(90, 12 * dim))

    # Build initial points: LHS + opposite + a few corners-like + random
    pts = []
    n_lhs = max(12, NP // 2)
    pts += lhs_points(n_lhs)
    pts += [opposite(p) for p in pts[:max(6, n_lhs // 2)]]

    # a few axis/edge probes
    center = [0.0] * dim
    for i in range(dim):
        lo, hi = bounds[i]
        center[i] = lo if hi <= lo else 0.5 * (lo + hi)
    pts.append(center[:])
    for i in range(min(dim, 12)):
        lo, hi = bounds[i]
        if hi > lo:
            x = center[:]
            x[i] = lo
            pts.append(x)
            x = center[:]
            x[i] = hi
            pts.append(x)

    while len(pts) < NP:
        pts.append(rand_vec())

    pop = []
    fit = []
    for x in pts[:NP]:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    if best_x is None:
        best_x = rand_vec()
        best = eval_f(best_x)

    # -------------------- SHADE/JADE-style DE state --------------------
    # Historical memories for F and CR (SHADE)
    H = 12
    M_F = [0.5] * H
    M_CR = [0.5] * H
    k_hist = 0

    # p-best fraction (JADE current-to-pbest)
    pmin = 2.0 / NP
    pmax = 0.20

    # external archive for diversity (stores replaced individuals)
    archive = []
    arch_max = NP

    # Stagnation / restart controls
    no_global = 0
    restart_after = max(80, 25 * (1 + dim // 5))
    refine_every = 30

    # -------------------- main loop --------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # rank indices by fitness (for p-best selection)
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        SF = []
        SCR = []
        dF = []

        # linearly vary p a bit to balance exploration/exploitation
        p = pmin + (pmax - pmin) * (0.5 + 0.5 * math.sin(0.15 * gen))
        pcount = max(2, int(math.ceil(p * NP)))

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick memory index
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample CR ~ N(muCR, 0.1), clipped
            CRi = muCR + 0.1 * random.gauss(0.0, 1.0)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # sample F ~ Cauchy(muF, 0.1) until in (0,1]
            Fi = None
            for _ in range(12):
                u = random.random()
                Fi_try = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < Fi_try <= 1.0:
                    Fi = Fi_try
                    break
            if Fi is None:
                Fi = clip(muF, 1e-3, 1.0)

            # choose pbest from top pcount
            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            # mutation: current-to-pbest/1 with archive
            # v = x + F*(pbest - x) + F*(xr1 - xr2)
            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # choose r2 from union(pop, archive) but not i or r1
            union_size = NP + len(archive)
            if union_size <= 2:
                r2_is_arch = False
                r2 = random.randrange(NP)
            else:
                # pick an index in union
                pick = random.randrange(union_size)
                r2_is_arch = pick >= NP
                r2 = pick - NP if r2_is_arch else pick

            xr1 = pop[r1]
            xr2 = archive[r2] if r2_is_arch else pop[r2]

            v = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    v[d] = lo
                else:
                    vraw = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                    v[d] = reflect(vraw, lo, hi)

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    u[d] = lo
                else:
                    if random.random() < CRi or d == jrand:
                        u[d] = v[d]

            fu = eval_f(u)

            # selection + archive update
            if fu <= fi:
                # store defeated parent into archive
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = u
                fit[i] = fu

                # collect successful parameters for memory update
                SF.append(Fi)
                SCR.append(CRi)
                dF.append(max(0.0, fi - fu))

                if fu < best:
                    best = fu
                    best_x = u[:]
                    no_global = 0
            else:
                no_global += 1

        # update historical memories (SHADE): weighted by improvements
        if SF:
            wsum = sum(dF)
            if wsum <= 0.0:
                w = [1.0 / len(SF)] * len(SF)
            else:
                w = [df / wsum for df in dF]

            # Lehmer mean for F, arithmetic mean for CR (common in SHADE)
            num = sum(w[j] * (SF[j] ** 2) for j in range(len(SF)))
            den = sum(w[j] * (SF[j]) for j in range(len(SF)))
            MF_new = (num / den) if den > 0.0 else mean(SF)
            MCR_new = sum(w[j] * SCR[j] for j in range(len(SCR)))

            M_F[k_hist] = clip(MF_new, 1e-3, 1.0)
            M_CR[k_hist] = clip(MCR_new, 0.0, 1.0)
            k_hist = (k_hist + 1) % H

        # occasional local refinement
        if (gen % refine_every) == 0 and best_x is not None and time.time() < deadline:
            step = [max(W[d] * 0.02, 1e-12) for d in range(dim)]
            rx, rf = pattern_refine(best_x, best, step, sweeps=2)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject into population replacing worst
                worst = max(range(NP), key=lambda t: fit[t])
                pop[worst] = best_x[:]
                fit[worst] = best
                no_global = 0

        # restart if stagnating
        if no_global >= restart_after and time.time() < deadline:
            no_global = 0
            # keep elite few, re-seed others around best + random
            elite_k = max(2, NP // 7)
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            elite = set(idx_sorted[:elite_k])

            archive.clear()

            for i in range(NP):
                if i in elite:
                    continue
                if time.time() >= deadline:
                    return best

                x = best_x[:]
                for d in range(dim):
                    lo, hi = bounds[d]
                    if hi <= lo:
                        x[d] = lo
                    else:
                        if random.random() < 0.25:
                            x[d] = random.uniform(lo, hi)
                        else:
                            # gaussian around best with moderately large scale
                            sd = max(W[d] * 0.18, 1e-12)
                            x[d] = reflect(x[d] + random.gauss(0.0, sd), lo, hi)

                fx = eval_f(x)
                pop[i] = x
                fit[i] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

    return best
