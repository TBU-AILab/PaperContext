import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Budgeted minimizer (no external libs) with improved robustness vs the provided DE/SHADE hybrids.

    Key upgrades vs your best (9.20...) version:
      - True L-SHADE style: population-size reduction (linear) to shift exploration->exploitation
      - More reliable archive usage (distinct indices) + current-to-pbest/1 as main driver
      - Occasional "best/2" injection late (strong exploitation) without overusing it early
      - Eigen-like random subspace local search around best (block coordinate + adaptive steps)
      - Better restart trigger using both stall gens and step collapse; restart uses opposition + near-best mix
      - Time-aware evaluation throttling: local search only when it can pay back

    Returns:
      best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------------- helpers ----------------------
    def isfinite(x):
        return (x == x) and (x != float("inf")) and (x != float("-inf"))

    def safe_eval(x):
        try:
            v = func(x)
            if isinstance(v, (int, float)):
                v = float(v)
                return v if isfinite(v) else float("inf")
            return float("inf")
        except Exception:
            return float("inf")

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly (handles large steps)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def ensure_reflect(x):
        y = x[:]  # list
        for j in range(dim):
            lo, hi = bounds[j]
            y[j] = reflect(y[j], lo, hi)
        return y

    def rand_vec():
        return [random.uniform(bounds[j][0], bounds[j][1]) for j in range(dim)]

    def opposite(x):
        y = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            y[j] = lo + hi - x[j]
        return ensure_reflect(y)

    def gauss01():
        # approx N(0,1)
        return (sum(random.random() for _ in range(12)) - 6.0)

    def lhs_init(n):
        # simple LHS-like init
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for j in range(dim):
                lo, hi = bounds[j]
                u = (perms[j][i] + random.random()) / float(n)
                x[j] = lo + u * (hi - lo)
            pop.append(x)
        return pop

    # ---------------------- setup ----------------------
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[j] - lo[j] for j in range(dim)]
    for j in range(dim):
        if span[j] <= 0.0:
            span[j] = 1.0
    avg_span = sum(span) / float(max(1, dim))

    # Initial and minimum population (L-SHADE style)
    NP_init = max(28, min(120, 18 + 5 * dim))
    NP_min = max(8, min(26, 6 + 2 * dim))

    # SHADE memories
    H = 10
    MF = [0.55] * H
    MCR = [0.55] * H
    mem_ptr = 0

    # p-best range over time
    pbest_max = 0.28
    pbest_min = 0.08

    # Archive
    archive = []
    # arch max tracks current NP (updated as NP shrinks)

    # Init population
    NP = NP_init
    pop = lhs_init(NP // 2) + [rand_vec() for _ in range(NP - NP // 2)]
    pop = [ensure_reflect(x) for x in pop]

    fit = [float("inf")] * NP
    best = float("inf")
    best_x = None

    # Evaluate init
    for i in range(NP):
        if time.time() >= deadline:
            return best
        f = safe_eval(pop[i])
        fit[i] = f
        if f < best:
            best = f
            best_x = pop[i][:]

    # Opposition check for a small elite subset
    ranked = sorted(range(NP), key=lambda i: fit[i])
    elite = ranked[:max(2, NP // 6)]
    for i in elite:
        if time.time() >= deadline:
            return best
        xo = opposite(pop[i])
        fo = safe_eval(xo)
        if fo < fit[i]:
            pop[i], fit[i] = xo, fo
        if fo < best:
            best, best_x = fo, xo[:]

    # Local search step sizes
    step = [0.10 * span[j] for j in range(dim)]
    min_step = [1e-12 + 2e-5 * span[j] for j in range(dim)]
    max_step = [0.75 * span[j] for j in range(dim)]

    # Stagnation tracking
    last_best = best
    stall_gens = 0
    last_improve_time = time.time()

    # Scheduling for local search
    last_local = t0
    local_interval = 0.14

    # ---------------------- local search ----------------------
    def block_coord_search(x, fx, blocks=2, block_size=10):
        """Random-subspace coordinate search with adaptive per-dim steps."""
        if x is None:
            return x, fx, False

        improved = False
        x0 = x[:]
        f0 = fx

        # choose blocks of coordinates
        coords = list(range(dim))
        random.shuffle(coords)
        # number of tested dims
        m = min(dim, max(6, blocks * block_size))
        coords = coords[:m]

        for j in coords:
            if time.time() >= deadline:
                break
            sj = step[j]
            if sj <= min_step[j]:
                continue

            best_local_f = f0
            best_local_x = None

            # 2-point pattern
            for sgn in (1.0, -1.0):
                xt = x0[:]
                xt[j] = xt[j] + sgn * sj
                xt = ensure_reflect(xt)
                ft = safe_eval(xt)
                if ft < best_local_f:
                    best_local_f = ft
                    best_local_x = xt

            if best_local_x is not None:
                x0, f0 = best_local_x, best_local_f
                improved = True
                step[j] = min(max_step[j], step[j] * 1.25)
            else:
                step[j] = max(min_step[j], step[j] * 0.65)

        return x0, f0, improved

    def spsa_refine(x, fx):
        """Very conservative SPSA step (2-4 evals typical)."""
        if x is None:
            return x, fx

        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))
        frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)

        c = (0.035 * (0.72 ** frac)) * (avg_span / math.sqrt(max(1, dim)))
        a0 = (0.090 * (0.70 ** frac)) * (avg_span / math.sqrt(max(1, dim)))
        c = max(1e-12, c)
        a0 = max(1e-12, a0)

        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
        x_plus = x[:]
        x_minus = x[:]
        for j in range(dim):
            x_plus[j] += c * delta[j]
            x_minus[j] -= c * delta[j]
        x_plus = ensure_reflect(x_plus)
        x_minus = ensure_reflect(x_minus)

        f_plus = safe_eval(x_plus)
        if time.time() >= deadline:
            return x, fx
        f_minus = safe_eval(x_minus)

        if f_plus == float("inf") and f_minus == float("inf"):
            return x, fx

        diff = (f_plus - f_minus) / (2.0 * c)
        g = [diff * delta[j] for j in range(dim)]
        gn = math.sqrt(sum(v * v for v in g))
        if gn <= 1e-18:
            return x, fx

        best_local_x = x[:]
        best_local_f = fx

        step_len = a0
        for _ in range(2):
            if time.time() >= deadline:
                break
            xt = x[:]
            sc = step_len / gn
            for j in range(dim):
                xt[j] -= sc * g[j]
            xt = ensure_reflect(xt)
            ft = safe_eval(xt)
            if ft < best_local_f:
                best_local_x, best_local_f = xt, ft
                break
            step_len *= 0.35

        return best_local_x, best_local_f

    # ---------------------- main loop ----------------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))
        frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)

        # L-SHADE: linear population size reduction
        target_NP = int(round(NP_init - (NP_init - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min
        if target_NP < NP:
            # remove worst individuals to shrink quickly (keeps best)
            ranked = sorted(range(NP), key=lambda i: fit[i])
            keep = ranked[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            # also shrink archive
            if len(archive) > NP:
                random.shuffle(archive)
                archive = archive[:NP]

        arch_max = NP

        # pbest schedule: smaller later
        pbest_rate = pbest_max - (pbest_max - pbest_min) * (frac ** 1.1)
        pnum = max(2, int(math.ceil(pbest_rate * NP)))

        # rank for pbest
        ranked = sorted(range(NP), key=lambda i: fit[i])
        pbest_set = ranked[:pnum]

        # success sets
        S_F, S_CR, S_w = [], [], []

        improved_gen = False

        # helper to pick distinct indices
        def pick_excluding(excl_set):
            k = random.randrange(NP)
            while k in excl_set:
                k = random.randrange(NP)
            return k

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # sample from memory
            r = random.randrange(H)
            mF = MF[r]
            mCR = MCR[r]

            # CR ~ N(mCR,0.1)
            CR = mCR + 0.10 * gauss01()
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # F ~ cauchy-like around mF (ratio of gaussians)
            F = -1.0
            for _ in range(6):
                g1 = gauss01()
                g2 = gauss01()
                if abs(g2) < 1e-12:
                    continue
                F = mF + 0.09 * (g1 / g2)
                if F > 0.0:
                    break
            if F <= 0.0:
                F = mF
            if F > 1.0:
                F = 1.0
            if F < 0.05:
                F = 0.05

            # choose strategy: mostly current-to-pbest/1; inject best/2 late
            use_best2 = (frac > 0.55 and random.random() < (0.10 + 0.20 * (frac - 0.55)))
            use_rand1 = (frac < 0.35 and random.random() < 0.22)

            use_archive = (archive and random.random() < 0.50)

            if use_rand1:
                # rand/1
                r0 = pick_excluding({i})
                r1 = pick_excluding({i, r0})
                if use_archive:
                    xr2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding({i, r0, r1})
                    xr2 = pop[r2]
                xr0, xr1 = pop[r0], pop[r1]
                v = [xr0[j] + F * (xr1[j] - xr2[j]) for j in range(dim)]
            elif use_best2 and best_x is not None and NP >= 5:
                # best/2: v = best + F*(r1-r2) + F*(r3-r4)
                r1 = pick_excluding({i})
                r2 = pick_excluding({i, r1})
                r3 = pick_excluding({i, r1, r2})
                if use_archive:
                    xr4 = archive[random.randrange(len(archive))]
                else:
                    r4 = pick_excluding({i, r1, r2, r3})
                    xr4 = pop[r4]
                v = [best_x[j] + F * (pop[r1][j] - pop[r2][j]) + F * (pop[r3][j] - xr4[j]) for j in range(dim)]
            else:
                # current-to-pbest/1
                pbest = pop[random.choice(pbest_set)]
                r1 = pick_excluding({i})
                if use_archive:
                    xr2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding({i, r1})
                    xr2 = pop[r2]
                xr1 = pop[r1]
                v = [xi[j] + F * (pbest[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]
            u = ensure_reflect(u)

            fu = safe_eval(u)

            if fu <= fi:
                # archive parent
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = u
                fit[i] = fu

                # record success (weight by improvement)
                df = fi - fu
                w = df if isfinite(df) and df > 0.0 else 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

                improved_gen = True
                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_improve_time = time.time()

        # update memories (weighted)
        if S_F:
            wsum = sum(S_w)
            if wsum <= 1e-30:
                wsum = float(len(S_w))
                weights = [1.0 / wsum] * len(S_w)
            else:
                weights = [w / wsum for w in S_w]

            # CR: weighted arithmetic
            newCR = 0.0
            for k in range(len(S_CR)):
                newCR += weights[k] * S_CR[k]

            # F: weighted Lehmer mean
            num = 0.0
            den = 0.0
            for k in range(len(S_F)):
                fk = S_F[k]
                wk = weights[k]
                num += wk * fk * fk
                den += wk * fk
            newF = (num / den) if den > 1e-18 else 0.55

            MCR[mem_ptr] = newCR
            MF[mem_ptr] = newF
            mem_ptr = (mem_ptr + 1) % H

        # stagnation
        if best < last_best - 1e-12:
            last_best = best
            stall_gens = 0
        else:
            stall_gens += 1

        # restart if stalling or steps collapsed too much
        step_small = sum(1 for j in range(dim) if step[j] <= (min_step[j] * 1.5))
        steps_collapsed = (step_small >= max(2, dim // 2))

        if stall_gens >= 14 or (steps_collapsed and stall_gens >= 8):
            stall_gens = 0
            # replace worst 30% with mix: near-best, opposition, random
            worst = sorted(range(NP), key=lambda k: fit[k], reverse=True)
            krep = max(1, int(0.30 * NP))
            for idx in worst[:krep]:
                if time.time() >= deadline:
                    return best
                if best_x is not None and pop[idx] == best_x:
                    continue

                r = random.random()
                if best_x is not None and r < 0.60:
                    x = best_x[:]
                    rad = 0.22 + 0.25 * random.random()
                    for j in range(dim):
                        x[j] += random.uniform(-rad, rad) * span[j]
                    x = ensure_reflect(x)
                    if random.random() < 0.30:
                        x = opposite(x)
                elif best_x is not None and r < 0.78:
                    x = opposite(best_x)
                else:
                    x = rand_vec()
                    if random.random() < 0.20:
                        x = opposite(x)

                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()

            # soften steps after restart (avoid being too tiny)
            for j in range(dim):
                step[j] = max(step[j], 0.05 * span[j])

            # refresh archive
            if archive and random.random() < 0.65:
                random.shuffle(archive)
                archive = archive[:max(0, len(archive) // 2)]

        # periodic local search (only if we have some remaining time and are not too early)
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            # avoid spending too much time early
            if frac > 0.18 or (now - last_improve_time) > 0.35:
                last_local = now

                x2, f2, imp = block_coord_search(best_x, best, blocks=2, block_size=8)
                if f2 < best:
                    best, best_x = f2, x2[:]
                    last_best = best
                    last_improve_time = time.time()

                # occasional SPSA late
                if frac > 0.45 and time.time() < deadline and random.random() < 0.40:
                    x3, f3 = spsa_refine(best_x, best)
                    if f3 < best:
                        best, best_x = f3, x3[:]
                        last_best = best
                        last_improve_time = time.time()

                # adapt interval
                if imp:
                    local_interval = max(0.06, local_interval * 0.90)
                else:
                    local_interval = min(0.30, local_interval * 1.05)

    return best
