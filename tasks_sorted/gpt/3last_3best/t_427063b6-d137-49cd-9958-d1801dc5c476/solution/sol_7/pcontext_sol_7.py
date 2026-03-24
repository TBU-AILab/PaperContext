import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained; no external libraries).

    Main idea: Budget-aware hybrid of
      - JADE-style Differential Evolution (current-to-pbest/1 + archive) for global+fast convergence
      - Success-based parameter adaptation (mu_F, mu_CR) + occasional "pulse" exploration
      - Intensification around best using:
          * SPSA-like two-sided gradient probe + backtracking step (cheap in high-dim)
          * small coordinate pattern search (very cheap, robust)
      - Stagnation handling: partial restart of worst individuals + archive refresh

    Returns:
      best (float): best fitness found within max_time seconds.
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

    # population size: moderate but time-friendly
    NP = max(18, min(70, 12 + 4 * dim))
    idx_all = list(range(NP))

    # JADE-ish parameters
    p_best_rate = 0.18
    c_adapt = 0.12
    mu_F = 0.6
    mu_CR = 0.55

    # archive
    archive = []
    arch_max = NP

    # init population (LHS + random)
    pop = lhs_init(NP // 2) + [rand_vec() for _ in range(NP - NP // 2)]
    pop = [ensure_reflect(x) for x in pop]

    fit = [float("inf")] * NP
    best = float("inf")
    best_x = None

    for i in range(NP):
        if time.time() >= deadline:
            return best
        fx = safe_eval(pop[i])
        fit[i] = fx
        if fx < best:
            best = fx
            best_x = pop[i][:]

    # scheduling
    last_local = t0
    local_interval = 0.11

    last_spsa = t0
    spsa_interval = 0.18

    last_best = best
    stall_gens = 0

    # ------------------------ local intensification ------------------------
    def pattern_polish(x, fx, tries=10):
        # coordinate pattern search on a small subset
        x0 = x[:]
        f0 = fx

        coords = list(range(dim))
        random.shuffle(coords)
        coords = coords[:min(dim, tries)]

        # step scale shrinks with time
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))
        step_rel = max(0.006, 0.10 * (0.55 ** frac))

        for j in coords:
            if time.time() >= deadline:
                break
            step = step_rel * span[j]
            if step <= 0:
                continue

            for s in (1.0, -1.0):
                xt = x0[:]
                xt[j] = xt[j] + s * step
                xt = ensure_reflect(xt)
                ft = safe_eval(xt)
                if ft < f0:
                    x0, f0 = xt, ft

            # small refine if improved
            if f0 < fx:
                step2 = 0.35 * step
                for s in (1.0, -1.0):
                    xt = x0[:]
                    xt[j] = xt[j] + s * step2
                    xt = ensure_reflect(xt)
                    ft = safe_eval(xt)
                    if ft < f0:
                        x0, f0 = xt, ft
        return x0, f0

    def spsa_step(x, fx):
        # Two-sided simultaneous perturbation gradient probe + backtracking.
        # Costs 2-6 evals; good when func is smooth/noisy-ish.
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))

        # perturb size and step shrink with time and dimension
        c = max(1e-12, (0.08 * (0.65 ** frac)) * (sum(span) / float(dim)) / math.sqrt(max(1, dim)))
        a0 = (0.22 * (0.60 ** frac)) * (sum(span) / float(dim)) / math.sqrt(max(1, dim))

        # Rademacher perturbation (+1/-1)
        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

        x_plus = x[:]
        x_minus = x[:]
        for j in range(dim):
            x_plus[j] = x_plus[j] + c * delta[j]
            x_minus[j] = x_minus[j] - c * delta[j]
        x_plus = ensure_reflect(x_plus)
        x_minus = ensure_reflect(x_minus)

        f_plus = safe_eval(x_plus)
        if time.time() >= deadline:
            return x, fx
        f_minus = safe_eval(x_minus)

        if not isfinite(f_plus): f_plus = float("inf")
        if not isfinite(f_minus): f_minus = float("inf")

        if f_plus == float("inf") and f_minus == float("inf"):
            return x, fx

        # gradient estimate
        g = [0.0] * dim
        denom = 2.0 * c
        if denom <= 0:
            return x, fx
        diff = (f_plus - f_minus) / denom
        for j in range(dim):
            g[j] = diff * delta[j]

        # normalize gradient (avoid huge steps)
        gn = math.sqrt(sum(v * v for v in g))
        if gn <= 1e-18:
            return x, fx

        # backtracking line-search
        step = a0
        best_local_x = x[:]
        best_local_f = fx

        for _ in range(3):
            if time.time() >= deadline:
                break
            xt = x[:]
            scale = step / gn
            for j in range(dim):
                xt[j] = xt[j] - scale * g[j]
            xt = ensure_reflect(xt)
            ft = safe_eval(xt)
            if ft < best_local_f:
                best_local_x, best_local_f = xt, ft
                break
            step *= 0.35

        return best_local_x, best_local_f

    # ------------------------ main loop ------------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # rank for pbest
        ranked = sorted(idx_all, key=lambda i: fit[i])
        pnum = max(2, int(math.ceil(p_best_rate * NP)))
        pbest_set = ranked[:pnum]

        SF = []
        SCR = []
        improved_gen = False

        # "pulse" exploration sometimes when stalling: widen F distribution temporarily
        pulse = (stall_gens >= 6)

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # CR ~ N(mu_CR, 0.1)
            CRi = mu_CR + 0.10 * gauss01()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # F heavy-tail around mu_F
            Fi = -1.0
            scale = 0.10 if pulse else 0.075
            for _ in range(6):
                g1 = gauss01()
                g2 = gauss01()
                if abs(g2) < 1e-9:
                    continue
                Fi = mu_F + scale * (g1 / g2)
                if Fi > 0.0:
                    break
            if Fi <= 0.0:
                Fi = mu_F
            if Fi > 1.0:
                Fi = 1.0
            if Fi < 0.05:
                Fi = 0.05

            pbest = pop[random.choice(pbest_set)]

            r1 = random.randrange(NP)
            while r1 == i:
                r1 = random.randrange(NP)

            use_archive = (archive and random.random() < 0.5)
            if use_archive:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = random.randrange(NP)
                while r2 == i or r2 == r1:
                    r2 = random.randrange(NP)
                xr2 = pop[r2]
            xr1 = pop[r1]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]

            u = ensure_reflect(u)
            fu = safe_eval(u)

            if fu <= fit[i]:
                # archive update
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = u
                fit[i] = fu
                SF.append(Fi)
                SCR.append(CRi)
                improved_gen = True

                if fu < best:
                    best = fu
                    best_x = u[:]

        # adapt mu_F and mu_CR
        if SF:
            num = sum(f * f for f in SF)
            den = sum(SF)
            if den > 1e-12:
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * (num / den)
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(SCR) / float(len(SCR)))

        if mu_F < 0.05: mu_F = 0.05
        if mu_F > 0.95: mu_F = 0.95
        if mu_CR < 0.0: mu_CR = 0.0
        if mu_CR > 1.0: mu_CR = 1.0

        # stagnation
        if best < last_best - 1e-12:
            last_best = best
            stall_gens = 0
        else:
            stall_gens += 1

        # partial restart if stalling
        if stall_gens >= 12:
            stall_gens = 0
            # replace worst 25%, mostly near-best re-seeds
            worst = sorted(idx_all, key=lambda k: fit[k], reverse=True)
            krep = max(1, NP // 4)
            for idx in worst[:krep]:
                if time.time() >= deadline:
                    return best
                if best_x is not None and pop[idx] == best_x:
                    continue
                if best_x is not None and random.random() < 0.75:
                    x = best_x[:]
                    rad = 0.35
                    for j in range(dim):
                        x[j] = x[j] + random.uniform(-rad, rad) * span[j]
                    x = ensure_reflect(x)
                else:
                    x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best

            # refresh archive a bit
            if len(archive) > arch_max:
                archive = archive[-arch_max:]
            if len(archive) > 0 and random.random() < 0.5:
                # drop some archive items to reduce stale bias
                keep = max(0, len(archive) // 2)
                random.shuffle(archive)
                archive = archive[:keep]

        # periodic local improvements around best
        now = time.time()
        if best_x is not None and (now - last_spsa) >= spsa_interval and now < deadline:
            last_spsa = now
            x2, f2 = spsa_step(best_x, best)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best

        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now
            x2, f2 = pattern_polish(best_x, best, tries=10)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best

            # adapt local interval based on whether DE improved recently
            if improved_gen:
                local_interval = max(0.055, local_interval * 0.93)
                spsa_interval = max(0.10, spsa_interval * 0.95)
            else:
                local_interval = min(0.26, local_interval * 1.04)
                spsa_interval = min(0.35, spsa_interval * 1.04)

    return best
