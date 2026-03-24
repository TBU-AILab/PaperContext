import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs the provided last algorithm:
      1) Uses a stronger DE variant: JADE-style "current-to-pbest/1" + external archive
         with *adaptive* (muF, muCR) learning from successful moves.
      2) Adds explicit *opposition-based* sampling on init and during restarts (often big win).
      3) Uses a robust, cheap local search: coordinate + adaptive step pattern search with
         opportunistic acceleration (faster than Powell here, more stable under bounds).
      4) Better restart logic: multi-scale radii + diversify half the population when stuck.
      5) Keeps small elite set + occasional "elite recombination" candidate.
      6) Tighter evaluation budget usage with frequent time checks and duplicate-cache.

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
    if any(s <= 0.0 for s in spans):
        x = [lows[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # ----------------- utils -----------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect until in range (handles big jumps)
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

    def rand_gauss_around(center, sigma_frac):
        # sigma_frac relative to span
        out = [0.0] * dim
        for i in range(dim):
            sigma = max(1e-15, sigma_frac * spans[i])
            out[i] = reflect(center[i] + random.gauss(0.0, sigma), lows[i], highs[i])
        return out

    def opposite(x):
        # opposition point wrt bounds center
        # x' = lo + hi - x
        return [clamp(lows[i] + highs[i] - x[i], lows[i], highs[i]) for i in range(dim)]

    # ----------------- eval + cache -----------------
    best = float("inf")
    best_x = None

    cache = {}
    cache_decimals = 12

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

    # ----------------- init population (LHS-ish + opposition) -----------------
    def init_population(NP):
        # LHS-ish
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

        # opposition-based selection: for each x consider {x, opposite(x)} keep better
        fit = [None] * NP
        for i in range(NP):
            if time.time() >= deadline:
                break
            x = pop[i]
            fx = eval_f(x)
            xo = opposite(x)
            fo = eval_f(xo) if time.time() < deadline else float("inf")
            if fo < fx:
                pop[i] = xo
                fit[i] = fo
            else:
                fit[i] = fx
        return pop, fit

    # ----------------- local search: adaptive pattern search -----------------
    def local_pattern_search(x0, f0, max_sweeps, step0_frac):
        """
        Coordinate pattern search with adaptive step, bounds reflection.
        Opportunistic: if improvement, keep direction; otherwise shrink step.
        """
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        # absolute step per dimension
        steps = [max(1e-15, step0_frac * spans[i]) for i in range(dim)]
        min_steps = [1e-12 * spans[i] + 1e-15 for i in range(dim)]

        for _ in range(max_sweeps):
            if time.time() >= deadline:
                break

            improved_any = False

            # try coordinates in random order (helps)
            order = list(range(dim))
            random.shuffle(order)

            for j in order:
                if time.time() >= deadline:
                    break

                sj = steps[j]
                if sj <= min_steps[j]:
                    continue

                xj = x[j]

                # try + step
                xp = x[:]
                xp[j] = reflect(xj + sj, lows[j], highs[j])
                fp = eval_f(xp)

                if fp < fx:
                    x, fx = xp, fp
                    improved_any = True
                    # opportunistic acceleration on this coordinate
                    steps[j] = min(steps[j] * 1.35, spans[j])
                    continue

                # try - step
                xm = x[:]
                xm[j] = reflect(xj - sj, lows[j], highs[j])
                fm = eval_f(xm)

                if fm < fx:
                    x, fx = xm, fm
                    improved_any = True
                    steps[j] = min(steps[j] * 1.35, spans[j])
                    continue

                # no improvement -> shrink this coordinate step
                steps[j] = sj * 0.6

            if not improved_any:
                # shrink all steps a bit; stop if tiny
                small = True
                for j in range(dim):
                    steps[j] *= 0.7
                    if steps[j] > min_steps[j]:
                        small = False
                if small:
                    break

        return x, fx

    # ----------------- JADE-like DE core -----------------
    # Population size (moderate; time-bounded)
    NP = int(max(30, min(18 * dim, 220)))
    pop, fit = init_population(NP)

    # If time ended during init
    if time.time() >= deadline:
        return best

    # Archive for r2 selection
    archive = []
    Amax = NP

    # JADE adaptation
    muF = 0.5
    muCR = 0.5
    c = 0.1  # learning rate

    p_best = 0.15  # fraction for pbest set

    # restart/stagnation
    last_best = best
    stagn = 0
    stagn_limit = 12

    # elites for occasional recombination
    elite_k = max(2, min(10, NP // 10))

    # restart radius schedule
    restart_sigma = 0.35

    gen = 0
    while time.time() < deadline:
        gen += 1

        # sort indices by fitness
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(p_best * NP))
        pset = idx_sorted[:pcount]
        elites = idx_sorted[:elite_k]

        # success memories
        S_F = []
        S_CR = []
        S_df = []

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # sample Fi from Cauchy(muF, 0.1), resample if <=0
            Fi = -1.0
            for _ in range(25):
                u = random.random()
                Fi = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < Fi <= 1.0:
                    break
            if Fi <= 0.0:
                Fi = min(1.0, max(0.05, muF))

            # sample CRi from Normal(muCR, 0.1) truncated
            CRi = muCR + 0.1 * random.gauss(0.0, 1.0)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # choose pbest
            pbest = random.choice(pset)
            xp = pop[pbest]

            # pick r1 != i, pbest
            def pick_from_pop(excl):
                for _ in range(60):
                    r = random.randrange(NP)
                    if r not in excl:
                        return r
                return (i + 1) % NP

            r1 = pick_from_pop({i, pbest})

            # pick r2 from pop U archive, distinct
            poolN = NP + len(archive)
            r2_source = 0
            r2_index = 0
            for _ in range(80):
                t = random.randrange(poolN)
                if t < NP:
                    if t == i or t == pbest or t == r1:
                        continue
                    r2_source, r2_index = 0, t
                else:
                    r2_source, r2_index = 1, (t - NP)
                break

            xr1 = pop[r1]
            xr2 = pop[r2_index] if r2_source == 0 else archive[r2_index]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = reflect(vj, lows[j], highs[j])

            # crossover (binomial)
            uvec = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    uvec[j] = v[j]
            uvec = ensure_bounds(uvec)

            # occasional elite recombination injection (cheap and often helpful)
            if (gen % 4 == 0) and random.random() < 0.08 and time.time() < deadline:
                e1 = pop[random.choice(elites)]
                e2 = pop[random.choice(elites)]
                lam = random.random()
                mix = [reflect(lam * e1[j] + (1.0 - lam) * e2[j], lows[j], highs[j]) for j in range(dim)]
                # blend with uvec
                beta = random.random() * 0.5
                uvec = [reflect((1.0 - beta) * uvec[j] + beta * mix[j], lows[j], highs[j]) for j in range(dim)]

            fu = eval_f(uvec)

            if fu <= fit[i]:
                # archive old xi
                archive.append(xi)
                if len(archive) > Amax:
                    archive.pop(random.randrange(len(archive)))

                df = fit[i] - fu
                pop[i] = uvec
                fit[i] = fu

                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(max(1e-12, df))

        # adapt muF, muCR (JADE)
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                w = [1.0 / len(S_df)] * len(S_df)
            else:
                w = [d / wsum for d in S_df]

            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, S_F):
                num += wi * fi * fi
                den += wi * fi
            F_lehmer = (num / den) if den > 1e-30 else muF

            # weighted mean for CR
            CR_mean = 0.0
            for wi, cri in zip(w, S_CR):
                CR_mean += wi * cri

            muF = (1.0 - c) * muF + c * F_lehmer
            muCR = (1.0 - c) * muCR + c * CR_mean

        # stagnation + local search
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # periodic local improvement on the incumbent best
        if (gen % 5 == 0) and best_x is not None and time.time() < deadline:
            # quick, time-safe refinement
            sweeps = 2 if dim <= 20 else 1
            step0 = 0.12 if stagn < 4 else 0.07
            bx, bf = local_pattern_search(best_x, best, max_sweeps=sweeps, step0_frac=step0)
            best_x, best = bx, bf

        # restart/diversification when stagnating
        if stagn >= stagn_limit and time.time() < deadline:
            stagn = 0

            restart_sigma *= 0.75
            if restart_sigma < 0.02:
                restart_sigma = 0.25  # cycle back to larger radius occasionally

            # replace worst half with mixture of opposition, around-best, and uniform
            worst_cnt = max(2, NP // 2)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:worst_cnt]

            for idx in worst:
                if time.time() >= deadline:
                    break
                r = random.random()
                if best_x is not None and r < 0.55:
                    xnew = rand_gauss_around(best_x, restart_sigma)
                    # with some chance, take opposite of around-best (adds big jump)
                    if random.random() < 0.35:
                        xnew = opposite(xnew)
                elif r < 0.80:
                    xnew = opposite(rand_uniform_vec())
                else:
                    xnew = rand_uniform_vec()

                fnew = eval_f(xnew)
                pop[idx] = xnew
                fit[idx] = fnew

            # clear archive to avoid dragging old diversity after restart
            archive = []

    return best
