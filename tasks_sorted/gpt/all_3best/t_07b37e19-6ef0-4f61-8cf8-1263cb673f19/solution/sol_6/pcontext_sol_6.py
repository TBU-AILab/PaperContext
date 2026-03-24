import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Hybrid time-bounded minimizer (self-contained, no external libs).

    Improvements over your current best (L-SHADE-ish + ES/coord polish):
      1) Better local exploitation: adds a robust Powell-like pattern search with
         adaptive step + opportunistic line-search (derivative-free).
      2) Better global driver: keeps L-SHADE current-to-pbest/1 + archive, but adds:
         - "either-or" mutation (current-to-pbest OR rand/1) to reduce premature convergence.
         - mild "eigen/coordinate" mix: occasional random subspace crossover (cheap).
      3) Stronger restart logic: budgeted micro-restarts around multiple elites (not only best).
      4) More time-aware: dynamically scales local-search intensity using observed eval speed.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- RNG helpers ----------------
    _bm_has = False
    _bm_val = 0.0

    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_val = z1
        _bm_has = True
        return z0

    def cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------------- helpers ----------------
    def bounce_repair(x):
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                xi = a + y
            if xi < a:
                xi = a
            elif xi > b:
                xi = b
            x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def eval_point(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def opposition_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------------- low discrepancy seeding (scrambled Halton) ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            ok = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def scrambled_halton_points(n):
        bases = first_primes(max(1, dim))
        shifts = [random.random() for _ in range(dim)]
        pts = []
        for k in range(1, n + 1):
            x = []
            for d in range(dim):
                u = (halton_value(k, bases[d]) + shifts[d]) % 1.0
                x.append(lo[d] + u * span_safe[d])
            pts.append(x)
        return pts

    # ---------------- eval time estimate ----------------
    def estimate_eval_time():
        k = 3
        ts = []
        for _ in range(k):
            if time.time() >= deadline:
                break
            x = rand_point()
            t1 = time.time()
            _ = eval_point(x)
            t2 = time.time()
            ts.append(max(1e-6, t2 - t1))
        if not ts:
            return 1e-3
        ts.sort()
        return ts[len(ts) // 2]

    eval_dt = estimate_eval_time()

    # ---------------- Local search: Powell-like pattern search ----------------
    def pattern_search(x0, f0, step_frac, max_evals):
        """
        Derivative-free local improvement:
          - exploratory coordinate moves +/- step
          - then a pattern move (extrapolation)
          - occasional 1D line-search along successful direction
        """
        if x0 is None:
            return x0, f0

        x = x0[:]
        fx = f0
        step = [max(1e-16, step_frac) * span_safe[i] for i in range(dim)]
        min_step = [1e-12 * span_safe[i] for i in range(dim)]

        evals = 0
        no_improve_sweeps = 0

        while evals < max_evals and time.time() < deadline:
            improved = False
            x_base = x[:]
            f_base = fx

            # exploratory moves
            idx = list(range(dim))
            random.shuffle(idx)
            for i in idx:
                if evals >= max_evals or time.time() >= deadline:
                    break
                si = step[i]
                if si <= min_step[i]:
                    continue

                best_local = fx
                best_vec = None

                # try + and -
                for sgn in (1.0, -1.0):
                    y = x[:]
                    y[i] += sgn * si
                    bounce_repair(y)
                    fy = eval_point(y)
                    evals += 1
                    if fy < best_local:
                        best_local = fy
                        best_vec = y

                    if evals >= max_evals or time.time() >= deadline:
                        break

                if best_vec is not None:
                    x, fx = best_vec, best_local
                    improved = True

            # pattern move (extrapolate)
            if improved and evals < max_evals and time.time() < deadline:
                d = [x[i] - x_base[i] for i in range(dim)]
                # try a few scaled steps along d
                for alpha in (1.0, 1.6):
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    y = [x[i] + alpha * d[i] for i in range(dim)]
                    bounce_repair(y)
                    fy = eval_point(y)
                    evals += 1
                    if fy < fx:
                        x, fx = y, fy

            # step adaptation
            if fx < f_base:
                # little expansion
                for i in range(dim):
                    step[i] = min(0.5 * span_safe[i], step[i] * 1.15)
                no_improve_sweeps = 0
            else:
                # shrink
                for i in range(dim):
                    step[i] *= 0.5
                no_improve_sweeps += 1

            # stop if steps are tiny or repeated failure
            if no_improve_sweeps >= 2:
                all_tiny = True
                for i in range(dim):
                    if step[i] > min_step[i]:
                        all_tiny = False
                        break
                if all_tiny:
                    break

        return x, fx

    # ---------------- initialization ----------------
    remaining = max(0.0, deadline - time.time())
    eval_budget = max(30, int(0.80 * remaining / max(eval_dt, 1e-9)))

    NP0 = int(18 + 4.5 * dim)
    NP0 = max(20, min(90, NP0))
    if eval_budget < 250:
        NP0 = max(12, min(NP0, 28))
    elif eval_budget < 600:
        NP0 = max(16, min(NP0, 45))

    NPmin = max(8, min(24, 6 + 2 * dim))

    n_seed = min(max(NP0, 3 * NP0), max(60, min(260, eval_budget // 3)))
    n_halton = max(2, int(0.70 * n_seed))
    n_rand = n_seed - n_halton

    seeds = scrambled_halton_points(n_halton)
    for _ in range(n_rand):
        seeds.append(rand_point())

    seeds2 = []
    for x in seeds:
        seeds2.append(x)
        seeds2.append(opposition_point(x))

    boundary_k = max(6, min(40, 2 * dim + 8))
    for _ in range(boundary_k):
        x = []
        for d in range(dim):
            r = random.random()
            if r < 0.34:
                u = (random.random() ** 2) * 0.02
                x.append(lo[d] + u * span_safe[d])
            elif r < 0.68:
                u = (random.random() ** 2) * 0.02
                x.append(hi[d] - u * span_safe[d])
            else:
                x.append(lo[d] + random.random() * span_safe[d])
        seeds2.append(x)

    best = float("inf")
    best_x = None
    scored = []

    for x in seeds2:
        if time.time() >= deadline:
            return best
        bounce_repair(x)
        fx = eval_point(x)
        scored.append((fx, x[:]))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    scored = scored[:NP0]
    pop = [x for (fx, x) in scored]
    fit = [fx for (fx, x) in scored]

    # keep an elite list for multi-start local search/restarts
    elite_max = max(3, min(10, 2 + dim))
    elite = []
    for fx, x in scored[:elite_max]:
        elite.append((fx, x[:]))

    # ---------------- SHADE memory ----------------
    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    archive = []
    archive_max = NP0

    pmin = 2.0 / max(2, NP0)
    pmax = 0.30

    last_improve = time.time()
    stall_seconds = max(0.20, 0.16 * max_time)

    # ---------------- main loop ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = time.time() - t0
        frac = min(1.0, max(0.0, elapsed / max(1e-9, max_time)))

        # linear population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac))
        if target_NP < NPmin:
            target_NP = NPmin
        if len(pop) > target_NP:
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            archive_max = max(target_NP, 8)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        if NP < 4:
            return best

        # update elite pool from current pop occasionally
        if gen % 5 == 0:
            order = sorted(range(NP), key=lambda i: fit[i])
            elite = [(fit[order[k]], pop[order[k]][:]) for k in range(min(elite_max, NP))]

        # periodic stronger local search (multi-start on elites)
        # allocate a tiny budget (in evals) scaled by remaining time
        if gen <= 2 or gen % 9 == 0 or (time.time() - last_improve) > stall_seconds:
            remaining = max(0.0, deadline - time.time())
            local_budget = int(min(0.12, 0.05 + 0.10 * frac) * (remaining / max(eval_dt, 1e-9)))
            local_budget = max(0, min(local_budget, 6 * (dim + 1) * elite_max))

            if local_budget > 0 and elite:
                # split budget across a few elites (best-biased)
                k_use = min(len(elite), 1 + (1 if frac < 0.4 else 2))
                for k in range(k_use):
                    if time.time() >= deadline or local_budget <= 0:
                        break
                    fx0, x0 = elite[k]
                    # smaller steps later
                    step_frac = 0.10 * (1.0 - 0.75 * frac)
                    step_frac = max(1e-6, step_frac)
                    # spend a chunk
                    chunk = max(4 * (dim + 1), local_budget // (k_use - k))
                    x1, f1 = pattern_search(x0, fx0, step_frac=step_frac, max_evals=chunk)
                    local_budget -= chunk
                    if f1 < fx0:
                        elite[k] = (f1, x1[:])
                    if f1 < best:
                        best, best_x = f1, x1[:]
                        last_improve = time.time()

        # pbest ordering
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = pmin + (pmax - pmin) * random.random()
        pcount = max(2, int(math.ceil(p * NP)))

        S_CR, S_F, S_df = [], [], []

        # DE evolve with either-or mutation
        for i in range(NP):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)
            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            Fi = cauchy(MF[r], 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            x_i = pop[i]

            # indices for mutation
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)

            pool_size = NP + len(archive)
            if pool_size <= 2:
                r3 = random.randrange(NP)
            else:
                r3 = i
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = random.randrange(pool_size)

            x_r1 = pop[r1]
            x_r2 = pop[r2]
            x_r3 = archive[r3 - NP] if r3 >= NP else pop[r3]

            # either-or: with prob use current-to-pbest/1 else rand/1
            if random.random() < (0.70 - 0.25 * frac):
                pbest_idx = order[random.randrange(pcount)]
                x_pbest = pop[pbest_idx]
                v = [x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r3[d]) for d in range(dim)]
            else:
                # rand/1
                v = [x_r1[d] + Fi * (x_r2[d] - x_r3[d]) for d in range(dim)]

            # crossover: sometimes do a small random subspace (reduces disruption in high-d)
            u = x_i[:]
            if dim > 0:
                if dim >= 8 and random.random() < 0.25:
                    # subspace mask
                    m = max(2, int(0.35 * dim))
                    idxs = random.sample(range(dim), m)
                    for d in idxs:
                        if random.random() < CRi:
                            u[d] = v[d]
                    # force at least one
                    u[idxs[0]] = v[idxs[0]]
                else:
                    jrand = random.randrange(dim)
                    for d in range(dim):
                        if d == jrand or random.random() < CRi:
                            u[d] = v[d]

            bounce_repair(u)
            fu = eval_point(u)

            if fu <= fit[i]:
                # archive
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                df = fit[i] - fu
                if df > 0.0:
                    S_CR.append(CRi)
                    S_F.append(Fi)
                    S_df.append(df)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = time.time()

        # update SHADE memories
        if S_df:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                wsum = 1.0

            cr_new = 0.0
            for k in range(len(S_df)):
                cr_new += (S_df[k] / wsum) * S_CR[k]

            num = 0.0
            den = 0.0
            for k in range(len(S_df)):
                wk = S_df[k] / wsum
                fk = S_F[k]
                num += wk * fk * fk
                den += wk * fk
            f_new = (num / den) if den > 1e-18 else MF[mem_idx]

            MCR[mem_idx] = cr_new
            MF[mem_idx] = f_new
            mem_idx = (mem_idx + 1) % H

        # stall handling: micro-restarts around multiple elites + some randoms
        if (time.time() - last_improve) > stall_seconds and time.time() < deadline:
            order_desc = sorted(range(NP), key=lambda i: fit[i], reverse=True)
            m = max(2, int(0.35 * NP))

            for t in range(m):
                if time.time() >= deadline:
                    return best
                k = order_desc[t]

                # choose center: best or another elite
                if elite and random.random() < 0.65:
                    center = elite[random.randrange(min(len(elite), elite_max))][1]
                elif best_x is not None:
                    center = best_x
                else:
                    center = None

                if center is not None and random.random() < 0.85:
                    y = center[:]
                    # heavy-tail-ish radius, but decrease late
                    rad = (0.10 + 0.25 * abs(cauchy(0.0, 1.0))) * (1.0 - 0.55 * frac)
                    if rad > 0.90:
                        rad = 0.90
                    for d in range(dim):
                        y[d] += (random.random() * 2.0 - 1.0) * rad * span_safe[d]
                    bounce_repair(y)
                else:
                    y = rand_point()

                fy = eval_point(y)
                pop[k] = y
                fit[k] = fy
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve = time.time()

            # after restart, attempt a tiny pattern-search on best
            if best_x is not None and time.time() < deadline:
                chunk = max(6, 3 * (dim + 1))
                x2, f2 = pattern_search(best_x, best, step_frac=max(1e-6, 0.08 * (1.0 - 0.6 * frac)), max_evals=chunk)
                if f2 < best:
                    best, best_x = f2, x2[:]
                    last_improve = time.time()
            else:
                last_improve = time.time()

    return best
