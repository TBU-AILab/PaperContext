import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (self-contained, no external libs).

    Upgrades vs previous version (typical quality improvement):
      - Uses SHADE-like parameter adaptation (memory of successful F/CR) instead of per-individual jDE.
      - Uses pbest selection + archive (JADE/SHADE style), generally stronger on many black-box tasks.
      - Adds inexpensive "opposition / quasi-reflection" injection when stagnating.
      - Local refinement replaced by an adaptive coordinate pattern-search with per-dim step sizes
        (usually more reliable than random-direction TR for bounded problems).
      - Better time guarding: batches and frequent deadline checks.

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
        if v < lo: return lo
        if v > hi: return hi
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

    def quasi_reflect(x, center):
        # "quasi-opposition": x' = center + r*(center - x)
        r = 0.5 + 0.5 * random.random()
        y = [0.0] * dim
        for i in range(dim):
            y[i] = reflect_into_bounds(center[i] + r * (center[i] - x[i]), lows[i], highs[i])
        return y

    best = float("inf")
    best_x = None

    def eval_f(x):
        nonlocal best, best_x
        fx = float(func(x))
        if fx < best:
            best = fx
            best_x = x[:]  # copy
        return fx

    # LHS-ish init (good coverage)
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
    # Population size: keep moderate
    NP = int(max(24, min(18 * dim, 220)))

    p_best = 0.2  # top fraction
    A = []
    Amax = NP

    # SHADE memory size
    H = 12
    M_CR = [0.5] * H
    M_F = [0.5] * H
    h_idx = 0

    # Stagnation / injections
    stagn = 0
    stagn_limit = 10
    inject_frac = 0.20

    # Local pattern-search schedule
    refine_every = 5
    refine_max_iters = 2  # small, time-safe

    # ----------------- initialize -----------------
    pop = init_population_lhs(NP)
    fit = [0.0] * NP

    # Evaluate initial pop
    for i in range(NP):
        if time.time() >= deadline:
            return best
        fit[i] = eval_f(pop[i])

    last_best = best
    gen = 0

    # ----------------- local refine: coordinate pattern search -----------------
    def local_refine_pattern(x0, f0, iters):
        # per-dimension step sizes; adaptively reduced
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0
        steps = [0.20 * spans[i] for i in range(dim)]
        # cap steps a bit
        for i in range(dim):
            if steps[i] <= 0.0:
                steps[i] = 1.0

        for _ in range(iters):
            if time.time() >= deadline:
                break

            improved_any = False
            order = list(range(dim))
            random.shuffle(order)

            for j in order:
                if time.time() >= deadline:
                    break

                sj = steps[j]
                if sj <= 1e-14 * (spans[j] + 1.0):
                    continue

                # try + then -
                x1 = x[:]
                x1[j] = reflect_into_bounds(x1[j] + sj, lows[j], highs[j])
                f1 = eval_f(x1)
                if f1 < fx:
                    x, fx = x1, f1
                    improved_any = True
                    steps[j] *= 1.35
                    continue

                if time.time() >= deadline:
                    break

                x2 = x[:]
                x2[j] = reflect_into_bounds(x2[j] - sj, lows[j], highs[j])
                f2 = eval_f(x2)
                if f2 < fx:
                    x, fx = x2, f2
                    improved_any = True
                    steps[j] *= 1.35
                else:
                    steps[j] *= 0.5

            # if nothing improved, shrink globally
            if not improved_any:
                for j in range(dim):
                    steps[j] *= 0.6

        return x, fx

    # ----------------- main loop -----------------
    while time.time() < deadline:
        gen += 1

        # Sort indices by fitness
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(p_best * NP))
        pset = idx_sorted[:pcount]

        # For SHADE updates this generation
        S_CR = []
        S_F = []
        S_df = []  # fitness improvements weights

        # mean center for opposition-style injections
        # (use top half for robustness)
        topk = max(2, NP // 2)
        center = [0.0] * dim
        for k in range(topk):
            xk = pop[idx_sorted[k]]
            for j in range(dim):
                center[j] += xk[j]
        inv_topk = 1.0 / float(topk)
        for j in range(dim):
            center[j] *= inv_topk

        for i in range(NP):
            if time.time() >= deadline:
                return best

            # pick memory index
            r = random.randrange(H)
            muCR = M_CR[r]
            muF = M_F[r]

            # sample CR ~ N(muCR, 0.1), truncated to [0,1]
            # sample F via Cauchy(muF, 0.1) until in (0,1]
            CRi = muCR + 0.1 * random.gauss(0.0, 1.0)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            Fi = -1.0
            for _ in range(20):
                u = random.random()
                # Cauchy via tan(pi*(u-0.5))
                Fi = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                if Fi > 0.0 and Fi <= 1.0:
                    break
            if Fi <= 0.0:
                Fi = min(1.0, max(0.1, muF))  # fallback

            # choose pbest
            pbest = random.choice(pset)

            # r1 from pop, r2 from pop+archive
            # ensure distinct indices as much as possible
            def pick_pop_index(exclude):
                for _ in range(50):
                    a = random.randrange(NP)
                    if a not in exclude:
                        return a
                # fallback
                a = random.randrange(NP)
                return a

            r1 = pick_pop_index({i, pbest})
            poolN = NP + len(A)

            # pick r2 from combined pool
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

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = reflect_into_bounds(vj, lows[j], highs[j])

            # crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    u[j] = v[j]
            u = ensure_bounds(u)

            fu = eval_f(u)
            if fu <= fit[i]:
                # successful: archive replaced
                A.append(xi)
                if len(A) > Amax:
                    A.pop(random.randrange(len(A)))

                df = fit[i] - fu
                pop[i] = u
                fit[i] = fu

                S_CR.append(CRi)
                S_F.append(Fi)
                # weight by improvement (at least small epsilon)
                S_df.append(max(1e-12, df))

        # SHADE memory update
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = float(len(S_df))
                weights = [1.0 / wsum] * len(S_df)
            else:
                weights = [d / wsum for d in S_df]

            # weighted arithmetic mean for CR
            mcr = 0.0
            for w, cr in zip(weights, S_CR):
                mcr += w * cr

            # weighted Lehmer mean for F: sum(w*F^2) / sum(w*F)
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            mf = (num / den) if den > 0.0 else 0.5

            M_CR[h_idx] = mcr
            M_F[h_idx] = mf
            h_idx = (h_idx + 1) % H

        # stagnation
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # periodic local refinement
        if (gen % refine_every == 0) and (time.time() < deadline) and best_x is not None:
            bx, bf = local_refine_pattern(best_x, best, refine_max_iters)
            best_x, best = bx, bf

        # stagnation handling: inject opposition / random
        if stagn >= stagn_limit and time.time() < deadline:
            stagn = 0
            k = max(2, int(inject_frac * NP))
            worst = sorted(range(NP), key=lambda ii: fit[ii], reverse=True)[:k]

            for idx in worst:
                if time.time() >= deadline:
                    break

                if best_x is not None and random.random() < 0.75:
                    # quasi-reflection around current center, blended with best
                    xnew = quasi_reflect(pop[idx], center)
                    if random.random() < 0.5:
                        # blend toward best
                        a = 0.3 + 0.6 * random.random()
                        for j in range(dim):
                            xnew[j] = reflect_into_bounds(
                                a * xnew[j] + (1.0 - a) * best_x[j],
                                lows[j], highs[j]
                            )
                else:
                    xnew = rand_uniform_vec()

                fnew = eval_f(xnew)
                pop[idx] = xnew
                fit[idx] = fnew

    return best
