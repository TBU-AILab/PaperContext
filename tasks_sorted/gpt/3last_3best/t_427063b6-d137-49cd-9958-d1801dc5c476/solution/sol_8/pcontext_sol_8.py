import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained; no external libraries).

    Improved over the provided JADE+local variant by adding:
      - Multi-strategy DE ensemble (current-to-pbest/1 + rand/1 + best/1)
      - Per-individual parameter memories (SHADE-style) instead of single global mu_F/mu_CR
      - Opposition-based sampling (both at init and occasionally during restarts)
      - Lightweight trust-region local search around best:
          * adaptive coordinate search with per-dim step sizes
          * occasional SPSA step (kept, but less frequent and more conservative)
      - Time-aware scheduling (shift from exploration -> exploitation)

    Returns:
      best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ------------------------ helpers ------------------------
    def isfinite(x):
        return (x == x) and (x != float("inf")) and (x != float("-inf"))

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect until within bounds
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def ensure_reflect(x):
        y = x[:]
        for j in range(dim):
            lo, hi = bounds[j]
            y[j] = reflect(y[j], lo, hi)
        return y

    def safe_eval(x):
        try:
            v = func(x)
            if isinstance(v, (int, float)):
                v = float(v)
                return v if isfinite(v) else float("inf")
            return float("inf")
        except Exception:
            return float("inf")

    # approx N(0,1) via sum of uniforms
    def gauss01():
        return (sum(random.random() for _ in range(12)) - 6.0)

    def rand_vec():
        return [random.uniform(bounds[j][0], bounds[j][1]) for j in range(dim)]

    def opposite(x):
        # opposition point in box: x' = lo + hi - x
        y = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            y[j] = lo + hi - x[j]
        return ensure_reflect(y)

    def lhs_init(n):
        # Latin-hypercube-ish init for coverage
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

    # ------------------------ setup ------------------------
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[j] - lo[j] for j in range(dim)]
    for j in range(dim):
        if span[j] <= 0.0:
            span[j] = 1.0
    avg_span = sum(span) / float(max(1, dim))

    # population size: robust but time-friendly
    NP = max(20, min(80, 14 + 4 * dim))

    # SHADE-style memory size (small to reduce overhead)
    H = 8
    MCR = [0.6] * H
    MF  = [0.6] * H
    mem_idx = 0

    # DE parameters
    arch = []
    arch_max = NP

    # pbest fraction
    p_best_min = 0.10
    p_best_max = 0.25

    # init population + opposition sampling
    pop = lhs_init(NP // 2) + [rand_vec() for _ in range(NP - NP // 2)]
    pop = [ensure_reflect(x) for x in pop]

    fit = [float("inf")] * NP
    best = float("inf")
    best_x = None

    # Evaluate init; also consider opposite points for the best few
    for i in range(NP):
        if time.time() >= deadline:
            return best
        fx = safe_eval(pop[i])
        fit[i] = fx
        if fx < best:
            best = fx
            best_x = pop[i][:]

    # quick opposition refinement on a subset
    ranked0 = sorted(range(NP), key=lambda i: fit[i])
    for i in ranked0[:max(2, NP // 6)]:
        if time.time() >= deadline:
            return best
        xo = opposite(pop[i])
        fo = safe_eval(xo)
        if fo < fit[i]:
            pop[i], fit[i] = xo, fo
        if fo < best:
            best, best_x = fo, xo[:]

    # local search state (per-dimension steps)
    step = [0.12 * span[j] for j in range(dim)]
    min_step = [1e-12 + 1e-4 * span[j] for j in range(dim)]

    last_best = best
    stall_gens = 0

    last_local = t0
    local_interval = 0.12

    last_spsa = t0
    spsa_interval = 0.28  # less frequent than before

    idx_all = list(range(NP))

    # ------------------------ local search ------------------------
    def coord_trust_search(x, fx, max_coords=12):
        # adaptive coordinate search with per-dim step sizes and shrink on failure
        x0 = x[:]
        f0 = fx

        coords = list(range(dim))
        random.shuffle(coords)
        coords = coords[:min(dim, max_coords)]

        improved_any = False
        for j in coords:
            if time.time() >= deadline:
                break
            sj = step[j]
            if sj <= min_step[j]:
                continue

            best_j_x = None
            best_j_f = f0

            for sgn in (1.0, -1.0):
                xt = x0[:]
                xt[j] = xt[j] + sgn * sj
                xt = ensure_reflect(xt)
                ft = safe_eval(xt)
                if ft < best_j_f:
                    best_j_f = ft
                    best_j_x = xt

            if best_j_x is not None:
                x0, f0 = best_j_x, best_j_f
                improved_any = True
                # slightly expand step on success
                step[j] = min(span[j], step[j] * 1.20)
            else:
                # shrink step on failure
                step[j] = max(min_step[j], step[j] * 0.60)

        return x0, f0, improved_any

    def spsa_step(x, fx):
        # conservative SPSA step with backtracking (2-4 evals typical)
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))

        c = (0.05 * (0.70 ** frac)) * (avg_span / math.sqrt(max(1, dim)))
        a0 = (0.12 * (0.65 ** frac)) * (avg_span / math.sqrt(max(1, dim)))
        c = max(1e-12, c)
        a0 = max(1e-12, a0)

        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

        x_plus = x[:]
        x_minus = x[:]
        for j in range(dim):
            x_plus[j]  = x_plus[j]  + c * delta[j]
            x_minus[j] = x_minus[j] - c * delta[j]
        x_plus = ensure_reflect(x_plus)
        x_minus = ensure_reflect(x_minus)

        f_plus = safe_eval(x_plus)
        if time.time() >= deadline:
            return x, fx
        f_minus = safe_eval(x_minus)

        if (f_plus == float("inf")) and (f_minus == float("inf")):
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
            scale = step_len / gn
            for j in range(dim):
                xt[j] = xt[j] - scale * g[j]
            xt = ensure_reflect(xt)
            ft = safe_eval(xt)
            if ft < best_local_f:
                best_local_x, best_local_f = xt, ft
                break
            step_len *= 0.35

        return best_local_x, best_local_f

    # ------------------------ main loop ------------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # time fraction: gradually exploit more
        frac = (time.time() - t0) / max(1e-12, float(max_time))
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # adapt pbest rate over time (more exploration early, more exploitation later)
        p_best_rate = p_best_max - (p_best_max - p_best_min) * (frac ** 1.2)

        ranked = sorted(idx_all, key=lambda i: fit[i])
        pnum = max(2, int(math.ceil(p_best_rate * NP)))
        pbest_set = ranked[:pnum]

        # success lists for updating memory
        S_F = []
        S_CR = []
        S_dF = []  # |delta f| weights (optional)

        improved_gen = False

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick memory slot
            r = random.randrange(H)
            mcr = MCR[r]
            mf = MF[r]

            # sample CR ~ N(mcr, 0.1)
            CRi = mcr + 0.10 * gauss01()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # sample F with heavy tail (Cauchy-like) around mf
            Fi = -1.0
            for _ in range(6):
                g1 = gauss01()
                g2 = gauss01()
                if abs(g2) < 1e-9:
                    continue
                Fi = mf + 0.08 * (g1 / g2)
                if Fi > 0.0:
                    break
            if Fi <= 0.0:
                Fi = mf
            if Fi > 1.0:
                Fi = 1.0
            if Fi < 0.05:
                Fi = 0.05

            # choose strategy (ensemble)
            # early: more rand/1; late: more current-to-pbest and best/1
            u = random.random()
            if u < (0.50 - 0.30 * frac):
                strategy = 0  # rand/1
            elif u < (0.90 - 0.15 * frac):
                strategy = 1  # current-to-pbest/1
            else:
                strategy = 2  # best/1

            # pick indices
            def pick_pop_index(exclude):
                k = random.randrange(NP)
                while k in exclude:
                    k = random.randrange(NP)
                return k

            # select vectors possibly from archive
            use_archive = (arch and random.random() < 0.45)

            if strategy == 0:
                # rand/1: v = xr0 + F*(xr1-xr2)
                r0 = pick_pop_index({i})
                r1 = pick_pop_index({i, r0})
                if use_archive:
                    xr2 = arch[random.randrange(len(arch))]
                else:
                    r2 = pick_pop_index({i, r0, r1})
                    xr2 = pop[r2]
                xr0 = pop[r0]
                xr1 = pop[r1]

                v = [xr0[j] + Fi * (xr1[j] - xr2[j]) for j in range(dim)]

            elif strategy == 1:
                # current-to-pbest/1: v = xi + F*(pbest-xi) + F*(xr1-xr2)
                pbest = pop[random.choice(pbest_set)]
                r1 = pick_pop_index({i})
                if use_archive:
                    xr2 = arch[random.randrange(len(arch))]
                else:
                    r2 = pick_pop_index({i, r1})
                    xr2 = pop[r2]
                xr1 = pop[r1]

                v = [xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j]) for j in range(dim)]

            else:
                # best/1: v = best_x + F*(xr1-xr2)
                if best_x is None:
                    best_base = xi
                else:
                    best_base = best_x
                r1 = pick_pop_index({i})
                if use_archive:
                    xr2 = arch[random.randrange(len(arch))]
                else:
                    r2 = pick_pop_index({i, r1})
                    xr2 = pop[r2]
                xr1 = pop[r1]

                v = [best_base[j] + Fi * (xr1[j] - xr2[j]) for j in range(dim)]

            # crossover
            jrand = random.randrange(dim)
            trial = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]
                else:
                    trial[j] = xi[j]
            trial = ensure_reflect(trial)

            ft = safe_eval(trial)

            if ft <= fi:
                # archive update with parent
                if len(arch) < arch_max:
                    arch.append(xi[:])
                else:
                    arch[random.randrange(arch_max)] = xi[:]

                pop[i] = trial
                fit[i] = ft

                # store successful parameters
                S_F.append(Fi)
                S_CR.append(CRi)
                df = abs(fi - ft)
                S_dF.append(df if isfinite(df) and df > 0.0 else 1.0)

                improved_gen = True
                if ft < best:
                    best = ft
                    best_x = trial[:]

        # Update SHADE memory using weighted means (weights ~ improvement)
        if S_F:
            wsum = sum(S_dF)
            if wsum <= 1e-18:
                wsum = float(len(S_dF))
                weights = [1.0 / wsum] * len(S_dF)
            else:
                weights = [w / wsum for w in S_dF]

            # weighted arithmetic mean for CR
            new_mcr = 0.0
            for k in range(len(S_CR)):
                new_mcr += weights[k] * S_CR[k]

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for k in range(len(S_F)):
                fk = S_F[k]
                wk = weights[k]
                num += wk * fk * fk
                den += wk * fk
            new_mf = (num / den) if den > 1e-18 else 0.6

            MCR[mem_idx] = new_mcr
            MF[mem_idx] = new_mf
            mem_idx = (mem_idx + 1) % H

        # stagnation tracking + restart
        if best < last_best - 1e-12:
            last_best = best
            stall_gens = 0
        else:
            stall_gens += 1

        if stall_gens >= 12:
            stall_gens = 0
            # replace worst 25% using mixture of:
            #  - random
            #  - near-best jitter
            #  - opposition of near-best jitter
            worst = sorted(idx_all, key=lambda k: fit[k], reverse=True)
            krep = max(1, NP // 4)
            for idx in worst[:krep]:
                if time.time() >= deadline:
                    return best
                if best_x is not None and pop[idx] == best_x:
                    continue

                r = random.random()
                if best_x is not None and r < 0.55:
                    x = best_x[:]
                    rad = 0.30 + 0.25 * random.random()
                    for j in range(dim):
                        x[j] += random.uniform(-rad, rad) * span[j]
                    x = ensure_reflect(x)
                    if random.random() < 0.35:
                        x = opposite(x)
                else:
                    x = rand_vec()
                    if random.random() < 0.20:
                        x = opposite(x)

                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best

            # refresh archive partially
            if len(arch) > arch_max:
                arch = arch[-arch_max:]
            if arch and random.random() < 0.6:
                random.shuffle(arch)
                arch = arch[:max(0, len(arch)//2)]

        # periodic local search around best
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now
            x2, f2, imp = coord_trust_search(best_x, best, max_coords=12)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best

            # time-aware local schedule
            if imp:
                local_interval = max(0.05, local_interval * 0.92)
            else:
                local_interval = min(0.28, local_interval * 1.05)

        # occasional SPSA step late in the run (more exploitation)
        now = time.time()
        if best_x is not None and frac > 0.35 and (now - last_spsa) >= spsa_interval and now < deadline:
            last_spsa = now
            x2, f2 = spsa_step(best_x, best)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best
                # tighten trust region a bit after success
                for j in range(dim):
                    step[j] = max(min_step[j], step[j] * 0.90)
            else:
                # if SPSA fails, don't spam it
                spsa_interval = min(0.50, spsa_interval * 1.08)
            # if it succeeds, allow slightly more frequent SPSA
            if f2 < best + 1e-12:
                spsa_interval = max(0.18, spsa_interval * 0.95)

    return best
