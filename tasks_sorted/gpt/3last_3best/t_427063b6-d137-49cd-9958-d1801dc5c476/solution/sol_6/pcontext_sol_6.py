import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained; no external libraries).

    Algorithm: Memetic DE (JADE-like) + occasional diagonal-CMA style sampling + cheap local polish
      - DE/current-to-pbest/1 with an external archive (JADE idea) for strong convergence
      - Per-generation adaptation of F and CR around learned means (mu_F, mu_CR)
      - Reflection bound handling (more stable than clip)
      - Small periodic coordinate pattern-search around incumbent best
      - Rare "CMA-lite" diagonal Gaussian samples around best to exploit smooth basins
      - Light restart: repopulate some worst when stagnating

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
        # reflect until in range (handles big steps)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        # numeric safety
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

    # approx N(0,1) using sum of uniforms (fast, no libs)
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
    span = [bounds[j][1] - bounds[j][0] for j in range(dim)]
    for j in range(dim):
        if span[j] <= 0.0:
            span[j] = 1.0

    # population sizing tuned for limited time
    NP = max(14, min(60, 10 + 4 * dim))

    # JADE-ish parameters
    p_best_rate = 0.15  # top p% used as pbest
    c_adapt = 0.10      # learning rate for mu_F, mu_CR

    mu_F = 0.6
    mu_CR = 0.5

    # archive for JADE (stores replaced parents)
    archive = []
    arch_max = NP

    # init
    pop = lhs_init(NP // 2) + [rand_vec() for _ in range(NP - NP // 2)]
    pop = [ensure_reflect(x) for x in pop]
    fit = []
    best = float("inf")
    best_x = None

    for x in pop:
        if time.time() >= deadline:
            return best
        fx = safe_eval(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # local search scheduling
    last_local = time.time()
    local_interval = 0.12
    local_base_rel = 0.10  # relative to span (decays mildly)

    # CMA-lite scheduling
    last_cma = time.time()
    cma_interval = 0.20

    # stagnation handling
    last_best = best
    stall_gens = 0

    # pre-allocated index list to reduce overhead
    idx_all = list(range(NP))

    # ------------------------ main loop ------------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # rank indices for pbest selection
        ranked = sorted(idx_all, key=lambda i: fit[i])
        pnum = max(2, int(math.ceil(p_best_rate * NP)))
        pbest_set = ranked[:pnum]

        # success histories for adaptation
        SF = []
        SCR = []

        # for light restart if stuck
        improved_gen = False

        # main generation
        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # sample CR ~ N(mu_CR, 0.1), clamp
            CRi = mu_CR + 0.1 * gauss01()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # sample F using "Cauchy-like" via ratio of gaussians; resample if <=0
            # (simple, no libs): F = mu_F + 0.1 * tan(pi*(u-0.5)) approx
            # We'll use a robust heavy-tail by dividing gaussians with cap.
            Fi = -1.0
            for _ in range(6):
                g1 = gauss01()
                g2 = gauss01()
                if abs(g2) < 1e-9:
                    continue
                heavy = g1 / g2  # heavy tail
                Fi = mu_F + 0.08 * heavy
                if Fi > 0.0:
                    break
            if Fi <= 0.0:
                Fi = mu_F
            if Fi > 1.0:
                Fi = 1.0
            if Fi < 0.05:
                Fi = 0.05

            # pick pbest
            pbest = pop[random.choice(pbest_set)]

            # choose r1 from population excluding i
            r1 = random.randrange(NP)
            while r1 == i:
                r1 = random.randrange(NP)

            # choose r2 from population U archive, excluding i and r1
            use_archive = (len(archive) > 0 and random.random() < 0.5)
            if use_archive:
                # choose from archive
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = random.randrange(NP)
                while r2 == i or r2 == r1:
                    r2 = random.randrange(NP)
                xr2 = pop[r2]

            xr1 = pop[r1]

            # mutation: current-to-pbest/1 + (r1 - r2)
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]

            u = ensure_reflect(u)
            fu = safe_eval(u)

            # selection
            if fu <= fit[i]:
                # add parent to archive (JADE)
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
            # else keep parent

        # adapt mu_F and mu_CR from successes
        if SF:
            # Lehmer mean for F (JADE)
            num = sum(f*f for f in SF)
            den = sum(SF)
            if den > 1e-12:
                lehmer = num / den
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmer

            mu_cr_s = sum(SCR) / float(len(SCR))
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * mu_cr_s

        # bound learned params
        if mu_F < 0.05: mu_F = 0.05
        if mu_F > 0.95: mu_F = 0.95
        if mu_CR < 0.0: mu_CR = 0.0
        if mu_CR > 1.0: mu_CR = 1.0

        # stagnation accounting
        if best < last_best - 1e-12:
            last_best = best
            stall_gens = 0
        else:
            stall_gens += 1

        # small restart of worst if stalling
        if stall_gens >= 10:
            stall_gens = 0
            # replace worst 20% (keep best individual untouched)
            ranked = sorted(idx_all, key=lambda k: fit[k], reverse=True)
            krep = max(1, NP // 5)
            for idx in ranked[:krep]:
                if time.time() >= deadline:
                    return best
                # don't overwrite global best if it sits here
                if best_x is not None and pop[idx] == best_x:
                    continue
                if best_x is not None and random.random() < 0.65:
                    # re-seed near best
                    x = best_x[:]
                    rad = 0.25
                    for j in range(dim):
                        x[j] = x[j] + random.uniform(-rad, rad) * span[j]
                    x = ensure_reflect(x)
                else:
                    x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best

            # trim archive a bit to avoid stale bias
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # periodic local polish (cheap coordinate pattern search)
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now

            x0 = best_x[:]
            f0 = best

            # try a subset of coordinates
            coords = list(range(dim))
            random.shuffle(coords)
            mcoords = min(dim, 10)
            coords = coords[:mcoords]

            # mild decay of step size over time
            time_frac = (now - t0) / max(1e-12, float(max_time))
            step_rel = max(0.008, local_base_rel * (0.6 ** time_frac))

            for j in coords:
                if time.time() >= deadline:
                    return best
                step = step_rel * span[j]
                if step <= 0.0:
                    continue

                for s in (1.0, -1.0):
                    xt = x0[:]
                    xt[j] = xt[j] + s * step
                    xt = ensure_reflect(xt)
                    ft = safe_eval(xt)
                    if ft < f0:
                        x0, f0 = xt, ft

                # small refine if improved
                if f0 < best:
                    step2 = 0.35 * step
                    for s in (1.0, -1.0):
                        xt = x0[:]
                        xt[j] = xt[j] + s * step2
                        xt = ensure_reflect(xt)
                        ft = safe_eval(xt)
                        if ft < f0:
                            x0, f0 = xt, ft

            if f0 < best:
                best, best_x = f0, x0[:]

            # adapt local frequency a bit
            if improved_gen:
                local_interval = max(0.06, local_interval * 0.95)
            else:
                local_interval = min(0.25, local_interval * 1.03)

        # periodic CMA-lite sampling around best (diagonal)
        # This can help on smooth problems where DE stalls.
        now = time.time()
        if best_x is not None and (now - last_cma) >= cma_interval and now < deadline:
            last_cma = now

            # estimate diagonal scale from current population spread (cheap)
            # use mean absolute deviation around best on a few dimensions to reduce cost
            sample_dims = list(range(dim))
            random.shuffle(sample_dims)
            sample_dims = sample_dims[:min(dim, 12)]

            # compute per-dim scale
            scales = [0.0] * dim
            for j in sample_dims:
                s = 0.0
                bx = best_x[j]
                for i in range(NP):
                    s += abs(pop[i][j] - bx)
                mad = s / float(NP)
                # fallback if collapsed
                scales[j] = max(1e-12, 0.35 * mad + 0.02 * span[j])

            # generate a few gaussian samples
            ns = max(4, min(12, 2 + dim // 4))
            for _ in range(ns):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                for j in sample_dims:
                    x[j] = x[j] + scales[j] * gauss01()
                x = ensure_reflect(x)
                fx = safe_eval(x)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best

            # adapt interval a bit
            cma_interval = 0.16 if improved_gen else min(0.35, cma_interval * 1.05)

    return best
