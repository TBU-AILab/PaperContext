import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs):
      - Sobol-ish / LHS-like seeding + opposition points
      - Differential Evolution (DE/rand/1/bin) as global driver
      - Success-based parameter adaptation (jDE-style)
      - Periodic local refinement around the best (SPSA-like + coord tries)
      - Restart diversification when stalled

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0 else 1.0 for s in span]

    # ---------------- helpers ----------------
    def clamp_vec(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def opposition_point(x):
        # opposite within bounds: lo+hi-x
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def eval_point(x):
        v = func(x)
        try:
            fv = float(v)
        except Exception:
            return float("inf")
        if math.isnan(fv) or math.isinf(fv):
            return float("inf")
        return fv

    def lhs_points(n):
        # LHS-like stratified sampling (fast and decent)
        perms = []
        for _ in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        invn = 1.0 / n
        for k in range(n):
            x = []
            for i in range(dim):
                a = (perms[i][k] + random.random()) * invn
                x.append(lo[i] + a * span_safe[i])
            pts.append(x)
        return pts

    def coord_refine(x, fx, step_frac, tries_per_dim=1):
        # quick coordinate +/- tries
        step = [step_frac * span_safe[i] for i in range(dim)]
        idx = list(range(dim))
        random.shuffle(idx)
        for i in idx:
            si = step[i]
            if si <= 0:
                continue
            best_local_x = None
            best_local_f = fx
            for _ in range(tries_per_dim):
                for d in (-1.0, 1.0):
                    y = x[:]
                    y[i] = y[i] + d * si
                    clamp_vec(y)
                    fy = eval_point(y)
                    if fy < best_local_f:
                        best_local_f, best_local_x = fy, y
            if best_local_x is not None:
                x, fx = best_local_x, best_local_f
        return x, fx

    def spsa_refine(x, fx, step_frac, a_scale=0.2, c_scale=0.1, iters=6):
        # SPSA-like refinement (2 evaluations per iteration)
        # Uses simultaneous perturbation to estimate a pseudo-gradient.
        a0 = a_scale * step_frac
        c0 = c_scale * step_frac
        for k in range(1, iters + 1):
            if time.time() >= deadline:
                break
            ak = a0 / (k ** 0.602)
            ck = c0 / (k ** 0.101)

            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            xp = x[:]
            xm = x[:]
            for i in range(dim):
                perturb = ck * span_safe[i] * delta[i]
                xp[i] += perturb
                xm[i] -= perturb
            clamp_vec(xp)
            clamp_vec(xm)

            fp = eval_point(xp)
            if time.time() >= deadline:
                break
            fm = eval_point(xm)

            denom = (fp - fm)
            # If denom is tiny, skip update
            if not (math.isfinite(denom)) or abs(denom) < 1e-18:
                continue

            # Move opposite estimated gradient: g_i ~ (fp-fm)/(2*ck*delta_i)
            # So update x_i <- x_i - ak * g_i
            y = x[:]
            for i in range(dim):
                g_i = denom / (2.0 * ck * span_safe[i] * delta[i])
                y[i] = y[i] - ak * g_i * span_safe[i]
            clamp_vec(y)
            fy = eval_point(y)
            if fy < fx:
                x, fx = y, fy
        return x, fx

    # ---------------- initialization ----------------
    # Population size: moderate for time-bounded evaluation
    pop_size = max(12, min(60, 10 + 4 * dim))
    # Seed more points early (but bounded)
    n_seed = max(pop_size, min(3 * pop_size, 120))

    best = float("inf")
    best_x = None

    pop = []
    fit = []

    # Seeding: LHS + opposition (keeps better half)
    seeds = lhs_points(n_seed // 2)
    seeds2 = []
    for s in seeds:
        seeds2.append(s)
        seeds2.append(opposition_point(s))
    # If odd, top off
    while len(seeds2) < n_seed:
        seeds2.append(rand_point())

    # Evaluate seeds, keep best pop_size
    scored = []
    for x in seeds2:
        if time.time() >= deadline:
            return best
        fx = eval_point(x)
        scored.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    scored = scored[:pop_size]
    for fx, x in scored:
        pop.append(x[:])
        fit.append(fx)

    # Assign per-individual DE params (jDE-style)
    F = [0.5 for _ in range(pop_size)]
    CR = [0.9 for _ in range(pop_size)]

    # Stagnation tracking / restarts
    last_improve_time = time.time()
    stall_seconds = max(0.2, 0.18 * max_time)

    # ---------------- main loop: DE + refinement ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # Occasionally refine best (cheap local improvements)
        # Do it early and then periodically.
        if best_x is not None and (gen <= 3 or gen % 12 == 0):
            if time.time() >= deadline:
                break
            # small coordinate refine then SPSA
            bx, bf = coord_refine(best_x[:], best, step_frac=0.05, tries_per_dim=1)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve_time = time.time()
            if time.time() >= deadline:
                break
            bx, bf = spsa_refine(best_x[:], best, step_frac=0.08, iters=5)
            if bf < best:
                best, best_x = bf, bx[:]
                last_improve_time = time.time()

        # DE generation
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # jDE parameter self-adaptation
            if random.random() < 0.1:
                F[i] = 0.1 + 0.9 * random.random()
            if random.random() < 0.1:
                CR[i] = random.random()

            # choose r1,r2,r3 distinct and != i
            idxs = list(range(pop_size))
            # fast manual pick
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)
            r3 = i
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(pop_size)

            x1 = pop[r1]
            x2 = pop[r2]
            x3 = pop[r3]

            # mutation: v = x1 + F*(x2-x3)
            Fi = F[i]
            v = [x1[d] + Fi * (x2[d] - x3[d]) for d in range(dim)]

            # binomial crossover with at least one dimension from v
            jrand = random.randrange(dim) if dim > 0 else 0
            ui = []
            CRi = CR[i]
            xi = pop[i]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    ui.append(v[d])
                else:
                    ui.append(xi[d])

            clamp_vec(ui)

            # selection
            fui = eval_point(ui)
            if fui <= fit[i]:
                pop[i] = ui
                fit[i] = fui

                if fui < best:
                    best, best_x = fui, ui[:]
                    last_improve_time = time.time()

        # Mild elitism injection: replace worst with best (keeps progress)
        worst_idx = 0
        worst_val = fit[0]
        for k in range(1, pop_size):
            if fit[k] > worst_val:
                worst_val = fit[k]
                worst_idx = k
        if best_x is not None and best < worst_val:
            pop[worst_idx] = best_x[:]
            fit[worst_idx] = best

        # Diversifying restart if stalled
        if time.time() - last_improve_time > stall_seconds and time.time() < deadline:
            # Reinitialize a fraction around best + some randoms
            frac = 0.35
            m = max(2, int(frac * pop_size))
            # sort indices by fitness descending and replace worst m
            order = list(range(pop_size))
            order.sort(key=lambda k: fit[k], reverse=True)
            for t in range(m):
                k = order[t]
                if random.random() < 0.7 and best_x is not None:
                    # jitter around best with decaying radius
                    rad = 0.15 * (0.5 + 0.5 * random.random())
                    y = best_x[:]
                    for d in range(dim):
                        y[d] += (random.random() * 2.0 - 1.0) * rad * span_safe[d]
                    clamp_vec(y)
                else:
                    y = rand_point()
                fy = eval_point(y)
                pop[k] = y
                fit[k] = fy
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve_time = time.time()
            # reset stall timer even if no improvement to avoid frequent restarts
            last_improve_time = time.time()

    return best
