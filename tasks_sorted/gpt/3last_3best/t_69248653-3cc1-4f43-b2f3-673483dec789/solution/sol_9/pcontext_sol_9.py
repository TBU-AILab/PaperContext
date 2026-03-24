import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (stdlib-only, self-contained).

    Key upgrades vs your best (~3.80 DE + diag-ES):
      - Better global search early: DE with *two* mutation modes (current-to-pbest/1 and rand-to-pbest/2),
        plus jDE-style self-adaptation per individual (robust across problems).
      - Better local search late: a bounded *full Nelder–Mead* polish (simplex) + a short coordinate+random
        pattern micro-polish.
      - Better restart logic: stagnation-triggered partial restarts using (a) global Halton, (b) opposition,
        (c) jitter around best at multiple radii.
      - Tighter time scheduling: switches to local refinement earlier if DE stalls.
      - Robust bounds: reflection for local steps; bounce-back for DE.
      - Safe evaluation wrapper (NaN/inf/exception -> inf).

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]
    avg_scale = sum(scale) / max(1, dim)

    # ---------------- utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_repair(x):
        # mirror reflection into bounds (smooth for local search / ES-like steps)
        for i in range(dim):
            L, U = lo[i], hi[i]
            if U <= L:
                x[i] = L
                continue
            v = x[i]
            if v < L or v > U:
                w = U - L
                y = (v - L) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                x[i] = L + y
        return x

    def safe_eval(x):
        try:
            v = func(x)
            if v is None or isinstance(v, complex):
                return float("inf")
            v = float(v)
            if v != v or v == float("inf") or v == -float("inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def de_repair(trial, base):
        # bounce back toward base, then clip
        for j in range(dim):
            if trial[j] < lo[j]:
                r = random.random()
                trial[j] = lo[j] + r * (base[j] - lo[j])
            elif trial[j] > hi[j]:
                r = random.random()
                trial[j] = hi[j] - r * (hi[j] - base[j])
        return clip_inplace(trial)

    # ---------------- scrambled Halton seeding ----------------
    def _primes_upto(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        r = int(n ** 0.5)
        for p in range(2, r + 1):
            if sieve[p]:
                start = p * p
                step = p
                sieve[start:n + 1:step] = [False] * (((n - start) // step) + 1)
        return [i for i, ok in enumerate(sieve) if ok]

    def _first_n_primes(n):
        if n <= 0:
            return []
        ub = max(50, int(n * (math.log(max(3, n)) + math.log(math.log(max(3, n))) + 3)))
        primes = _primes_upto(ub)
        while len(primes) < n:
            ub = int(ub * 1.7) + 10
            primes = _primes_upto(ub)
        return primes[:n]

    primes = _first_n_primes(dim)
    scramble = [random.random() for _ in range(dim)]

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = (halton_value(k, primes[i]) + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # ---------------- local search: bounded Nelder–Mead ----------------
    def nelder_mead_bounded(x0, f0, feval_budget, time_budget):
        if feval_budget <= 0 or time_budget <= 0.0:
            return x0, f0, 0

        start = time.time()
        n = dim

        # initial simplex
        simplex = [list(x0)]
        fx = [f0]
        used = 0

        base_step = 0.08 * avg_scale
        for i in range(n):
            xi = list(x0)
            step = base_step * (scale[i] / (avg_scale if avg_scale > 0 else 1.0))
            if step <= 0.0:
                step = base_step
            xi[i] += step
            reflect_repair(xi)
            fi = safe_eval(xi)
            used += 1
            simplex.append(xi)
            fx.append(fi)

            if used >= feval_budget or (time.time() - start) >= time_budget or time.time() >= deadline:
                bi = min(range(len(fx)), key=lambda k: fx[k])
                return simplex[bi], fx[bi], used

        # coefficients
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        def centroid_excl_last():
            c = [0.0] * n
            for k in range(n):
                xk = simplex[k]
                for i in range(n):
                    c[i] += xk[i]
            inv = 1.0 / float(n)
            for i in range(n):
                c[i] *= inv
            return c

        while used < feval_budget and (time.time() - start) < time_budget and time.time() < deadline:
            order = sorted(range(n + 1), key=lambda k: fx[k])
            simplex = [simplex[k] for k in order]
            fx = [fx[k] for k in order]

            bestx, bestf = simplex[0], fx[0]
            worstx, worstf = simplex[-1], fx[-1]
            second_worstf = fx[-2]

            # stop if simplex small (scaled)
            size = 0.0
            for k in range(1, n + 1):
                d = 0.0
                xk = simplex[k]
                for i in range(n):
                    t = (xk[i] - bestx[i]) / (scale[i] if scale[i] > 0 else 1.0)
                    d += t * t
                if d > size:
                    size = d
            if size < 1e-18:
                break

            c = centroid_excl_last()

            # reflection
            xr = [c[i] + alpha * (c[i] - worstx[i]) for i in range(n)]
            reflect_repair(xr)
            fr = safe_eval(xr)
            used += 1

            if fr < bestf:
                # expansion
                xe = [c[i] + gamma * (xr[i] - c[i]) for i in range(n)]
                reflect_repair(xe)
                fe = safe_eval(xe)
                used += 1
                if fe < fr:
                    simplex[-1], fx[-1] = xe, fe
                else:
                    simplex[-1], fx[-1] = xr, fr
            elif fr < second_worstf:
                simplex[-1], fx[-1] = xr, fr
            else:
                # contraction
                if fr < worstf:
                    xc = [c[i] + rho * (xr[i] - c[i]) for i in range(n)]
                else:
                    xc = [c[i] + rho * (worstx[i] - c[i]) for i in range(n)]
                reflect_repair(xc)
                fc = safe_eval(xc)
                used += 1

                if fc < worstf:
                    simplex[-1], fx[-1] = xc, fc
                else:
                    # shrink
                    b = simplex[0]
                    for k in range(1, n + 1):
                        xs = [b[i] + sigma * (simplex[k][i] - b[i]) for i in range(n)]
                        reflect_repair(xs)
                        fs = safe_eval(xs)
                        used += 1
                        simplex[k], fx[k] = xs, fs
                        if used >= feval_budget or (time.time() - start) >= time_budget or time.time() >= deadline:
                            break

        bi = min(range(len(fx)), key=lambda k: fx[k])
        return simplex[bi], fx[bi], used

    # ---------------- micro polish: coord + random directions ----------------
    def micro_polish(x0, f0, eval_budget):
        x = list(x0)
        fx = f0
        used = 0

        # start step small-ish
        step = 0.02 * avg_scale
        step_min = 1e-14 * avg_scale
        coord_pass = max(1, min(2, dim // 10))

        while used < eval_budget and time.time() < deadline and step > step_min:
            improved = False

            # coordinate tweaks
            for _ in range(coord_pass):
                for j in range(dim):
                    if used >= eval_budget or time.time() >= deadline:
                        break
                    for sgn in (-1.0, 1.0):
                        xn = list(x)
                        xn[j] += sgn * step
                        reflect_repair(xn)
                        fn = safe_eval(xn)
                        used += 1
                        if fn < fx:
                            x, fx = xn, fn
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # random direction tweaks
            if not improved:
                tries = max(3, min(10, dim))
                for _ in range(tries):
                    if used >= eval_budget or time.time() >= deadline:
                        break
                    d = [random.gauss(0.0, 1.0) for _ in range(dim)]
                    dn = math.sqrt(sum(v * v for v in d))
                    if dn <= 0.0:
                        continue
                    inv = 1.0 / dn
                    for i in range(dim):
                        d[i] *= inv
                    for sgn in (-1.0, 1.0):
                        if used >= eval_budget or time.time() >= deadline:
                            break
                        xn = [x[i] + sgn * step * d[i] for i in range(dim)]
                        reflect_repair(xn)
                        fn = safe_eval(xn)
                        used += 1
                        if fn < fx:
                            x, fx = xn, fn
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                step *= 0.55

        return x, fx, used

    # ---------------- initialization ----------------
    # keep moderate sizes to leave time for local refinement
    NP0 = max(20, min(80, 14 + 5 * dim))
    NPmin = max(8, min(34, 10 + 2 * dim))
    NP = NP0

    pop, fit = [], []
    F_i = []   # jDE per-individual F
    CR_i = []  # jDE per-individual CR

    k = 1
    while len(pop) < NP and time.time() < deadline:
        if len(pop) % 4 == 0:
            x = rand_point()
        else:
            x = halton_point(k)
            k += 1

        fx = safe_eval(x)

        # opposition-based try
        if random.random() < 0.65:
            xo = opposite_point(x)
            fo = safe_eval(xo)
            if fo < fx:
                x, fx = xo, fo

        pop.append(list(x))
        fit.append(fx)

        # init jDE controls
        F_i.append(random.uniform(0.4, 0.9))
        CR_i.append(random.uniform(0.1, 0.9))

    if not pop:
        return float("inf")

    best_idx = min(range(NP), key=lambda i: fit[i])
    best_x = list(pop[best_idx])
    best = fit[best_idx]

    # archive for diversity
    archive = []
    Amax = 2 * NP0

    # stagnation control
    no_best = 0
    patience = max(60, 18 * dim)

    # for DE pbest
    def p_fraction(frac):
        # more exploitation as time passes
        p = 0.30 - 0.20 * frac
        pmin = 2.0 / max(2, NP)
        if p < pmin:
            p = pmin
        if p > 0.35:
            p = 0.35
        return p

    last_local = 0.0

    # ---------------- main loop ----------------
    it = 0
    while time.time() < deadline:
        it += 1
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))
        time_left = deadline - now

        # shrink population over time
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            F_i = [F_i[i] for i in keep]
            CR_i = [CR_i[i] for i in keep]
            NP = target_NP
            if len(archive) > 2 * NP:
                random.shuffle(archive)
                archive = archive[:2 * NP]

        # --- local refinement scheduling ---
        # do local work late, or earlier if stagnating
        if time_left > 0.05 and (frac > 0.65 or no_best > patience):
            if (now - last_local) > max(0.12, 0.03 * max_time):
                last_local = now
                # small NM slice
                nm_fe = max(12, min(90, 10 + 6 * dim))
                nm_time = min(0.10 * max_time, 0.30 * time_left)
                bx, bf, _ = nelder_mead_bounded(best_x, best, nm_fe, nm_time)
                if bf < best:
                    best, best_x = bf, list(bx)
                    no_best = 0

        # --- stagnation-triggered partial restart/injection ---
        if no_best > patience and time_left > 0.03:
            nrep = max(1, NP // 3)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for wi in worst:
                if time.time() >= deadline:
                    break
                r = random.random()
                if r < 0.34:
                    x = halton_point(k); k += 1
                elif r < 0.67:
                    x = opposite_point(best_x)
                    # jitter opposite a bit
                    for j in range(dim):
                        x[j] += random.gauss(0.0, 0.10 * scale[j])
                    reflect_repair(x)
                else:
                    # multi-radius jitter around best
                    rad = (0.15 if random.random() < 0.5 else 0.50)
                    x = [best_x[j] + random.gauss(0.0, rad * scale[j]) for j in range(dim)]
                    reflect_repair(x)
                fx = safe_eval(x)
                pop[wi] = list(x)
                fit[wi] = fx
                F_i[wi] = random.uniform(0.4, 0.9)
                CR_i[wi] = random.uniform(0.1, 0.9)
                if fx < best:
                    best, best_x = fx, list(x)
                    no_best = 0
            no_best = patience // 2

        # --- DE generation (jDE + pbest + archive) ---
        order = sorted(range(NP), key=lambda i: fit[i])
        p = p_fraction(frac)
        p_count = max(2, int(math.ceil(p * NP)))

        union = pop + archive
        union_n = len(union)

        def pick_from_union(exclude_pop_set):
            while True:
                r = random.randrange(union_n)
                if r < NP and r in exclude_pop_set:
                    continue
                return r

        improved = False

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            # jDE self-adaptation
            if random.random() < 0.1:
                F = random.uniform(0.1, 0.9)
            else:
                F = F_i[i]
            if random.random() < 0.1:
                CR = random.random()
            else:
                CR = CR_i[i]

            # choose pbest among top
            pbest = order[random.randrange(p_count)]
            xp = pop[pbest]

            # two mutation strategies mixed
            use_rand2 = (random.random() < (0.20 + 0.25 * (1.0 - frac)))  # a bit more early

            if not use_rand2:
                # current-to-pbest/1 + archive
                excl = {i, pbest}
                r1 = pick_from_union(excl)
                r2 = pick_from_union(excl)
                while r2 == r1:
                    r2 = pick_from_union(excl)
                xr1 = union[r1]
                xr2 = union[r2]
                v = [xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]
            else:
                # rand-to-pbest/2 (stronger exploration)
                excl = {i, pbest}
                r0 = random.randrange(NP)
                while r0 in excl:
                    r0 = random.randrange(NP)
                x0 = pop[r0]

                r1 = pick_from_union(excl | {r0})
                r2 = pick_from_union(excl | {r0})
                while r2 == r1:
                    r2 = pick_from_union(excl | {r0})

                r3 = pick_from_union(excl | {r0})
                r4 = pick_from_union(excl | {r0})
                while r4 == r3:
                    r4 = pick_from_union(excl | {r0})

                xr1 = union[r1]
                xr2 = union[r2]
                xr3 = union[r3]
                xr4 = union[r4]

                v = [x0[j] + F * (xp[j] - x0[j]) + F * (xr1[j] - xr2[j]) + 0.5 * F * (xr3[j] - xr4[j])
                     for j in range(dim)]

            de_repair(v, xi)

            # binomial crossover
            jrand = random.randrange(dim)
            u = [v[j] if (j == jrand or random.random() < CR) else xi[j] for j in range(dim)]
            de_repair(u, xi)

            fu = safe_eval(u)

            if fu <= fi:
                # accept + update personal params
                if random.random() < 0.2:
                    # occasional "keep moving" nudge
                    F_i[i] = min(0.95, max(0.05, 0.8 * F_i[i] + 0.2 * F))
                    CR_i[i] = min(1.0, max(0.0, 0.8 * CR_i[i] + 0.2 * CR))
                else:
                    F_i[i] = F
                    CR_i[i] = CR

                # archive parent
                if len(archive) < Amax:
                    archive.append(list(xi))
                else:
                    archive[random.randrange(Amax)] = list(xi)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, list(u)
                    improved = True

        if improved:
            no_best = 0
        else:
            no_best += 1

        # archive limit
        Amax = max(2 * NP, 2 * NP0)
        if len(archive) > Amax:
            random.shuffle(archive)
            archive = archive[:Amax]

        # if close to deadline, stop DE and do a quick polish
        if deadline - time.time() < 0.03:
            break

    # ---------------- final polish ----------------
    # quick micro-polish uses very few evals but can help a lot when already near optimum
    if time.time() < deadline:
        budget = min(60, max(12, 3 * dim))
        bx, bf, _ = micro_polish(best_x, best, budget)
        if bf < best:
            best, best_x = bf, bx

    # last tiny gaussian probes
    for _ in range(min(20, 3 + dim)):
        if time.time() >= deadline:
            break
        x = [best_x[i] + random.gauss(0.0, 0.01 * scale[i]) for i in range(dim)]
        reflect_repair(x)
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x

    return best
