import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-budgeted box-constrained minimizer (stdlib-only).

    Improvements vs your last DE variants:
      - Uses a *portfolio* approach: several short DE “epochs” with different behaviors
        (explore / balanced / exploit) + hard time-splitting.
      - Adds an on-the-fly *coordinate-rotation* (cheap orthonormal basis via random
        Gram–Schmidt) and runs local search in that rotated space to handle rotated valleys.
      - Uses *Rosenbrock-style adaptive coordinate search* (very effective late-stage).
      - Better restart logic: if stalled, re-seed around best + opposition + random.
      - Safer/cheaper bound handling and fewer python overhead hotspots.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]
    avg_span = (sum(span_safe) / dim) if dim > 0 else 1.0

    # ---------- helpers ----------
    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    def reflect_inplace(x):
        # reflect into [lo,hi] with period 2w, then clamp
        for i in range(dim):
            a = lo[i]; b = hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                xi = (a + y) if (y <= w) else (b - (y - w))
            # numeric clamp
            if xi < a: xi = a
            elif xi > b: xi = b
            x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(random.random(), 1e-300)
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare = z1
        _has_spare = True
        return z0

    def rand_cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    # ---------- best bookkeeping ----------
    best = float("inf")
    best_x = None

    def try_update(x):
        nonlocal best, best_x
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    if now() >= deadline:
        return best

    # ---------- seeding (cheap but diverse) ----------
    # center
    x = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    reflect_inplace(x)
    try_update(x)

    # a few corners
    corner_bits = min(dim, 6)
    max_corners = min(16, 1 << corner_bits)
    for mask in range(max_corners):
        if now() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if i < corner_bits:
                x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
            else:
                x[i] = 0.5 * (lo[i] + hi[i])
        try_update(x)

    # random seeds
    for _ in range(14 + 6 * dim):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------- rotated basis for local search ----------
    # Build a random orthonormal basis Q (dim x dim) occasionally.
    # We'll represent Q as list of basis vectors q[k] (each length dim).
    def random_orthonormal_basis():
        if dim <= 0:
            return []
        q = []
        for k in range(dim):
            v = [randn() for _ in range(dim)]
            # Gram-Schmidt
            for u in q:
                proj = dot(v, u)
                if proj != 0.0:
                    for i in range(dim):
                        v[i] -= proj * u[i]
            nv = norm(v)
            if nv < 1e-12:
                # fallback to axis
                v = [0.0] * dim
                v[k] = 1.0
                q.append(v)
            else:
                inv = 1.0 / nv
                for i in range(dim):
                    v[i] *= inv
                q.append(v)
        return q

    # ---------- Rosenbrock-style adaptive local search ----------
    def rosenbrock_local(time_limit):
        nonlocal best, best_x
        if best_x is None or dim <= 0:
            return

        x = best_x[:]
        fx = best

        # directions: mix axes and rotated basis
        # rebuild rotated basis sometimes for cheap coordinate-free help
        Q = random_orthonormal_basis() if dim <= 40 else None  # avoid heavy cost for huge dim

        dirs = []
        # axes subset (all if small, else random subset)
        if dim <= 50:
            for j in range(dim):
                v = [0.0] * dim
                v[j] = 1.0
                dirs.append(v)
        else:
            k = max(12, int(2.5 * math.sqrt(dim)))
            idxs = random.sample(range(dim), k)
            for j in idxs:
                v = [0.0] * dim
                v[j] = 1.0
                dirs.append(v)

        if Q is not None:
            # add a handful of rotated directions
            kq = min(dim, 12)
            pick = random.sample(range(dim), kq) if dim >= kq else list(range(dim))
            for idx in pick:
                dirs.append(Q[idx])

        # adaptive step per direction
        base = 0.06 * avg_span
        steps = [base for _ in range(len(dirs))]
        min_step = 1e-14 * avg_span

        # limited sweeps
        sweeps = 2
        for _ in range(sweeps):
            if now() >= time_limit:
                break
            improved_any = False
            order = list(range(len(dirs)))
            random.shuffle(order)
            for ii in order:
                if now() >= time_limit:
                    break
                a = steps[ii]
                if a < min_step:
                    continue
                d = dirs[ii]

                # try +a
                y = x[:]
                for j in range(dim):
                    y[j] += a * d[j]
                reflect_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved_any = True
                    steps[ii] *= 1.35
                    continue

                # try -a
                y = x[:]
                for j in range(dim):
                    y[j] -= a * d[j]
                reflect_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved_any = True
                    steps[ii] *= 1.35
                else:
                    steps[ii] *= 0.55

            # if no improvement, shrink globally a bit
            if not improved_any:
                for i in range(len(steps)):
                    steps[i] *= 0.8

        if fx < best:
            best, best_x = fx, x[:]

    # ---------- DE epoch (SHADE-like current-to-pbest + archive) ----------
    def de_epoch(time_limit, mode):
        """
        mode in {"explore","balanced","exploit"} changes p, NP, and F/CR tendencies.
        """
        nonlocal best, best_x

        if dim <= 0:
            return

        # population size by mode (small exploitation pop late, larger early)
        if mode == "explore":
            NP0 = max(34, min(160, 24 + 5 * dim))
            p_hi, p_lo = 0.40, 0.12
            Fcap_hi, Fcap_lo = 1.0, 0.70
            cr_mu = 0.75
        elif mode == "exploit":
            NP0 = max(18, min(90, 16 + 3 * dim))
            p_hi, p_lo = 0.18, 0.05
            Fcap_hi, Fcap_lo = 0.85, 0.55
            cr_mu = 0.90
        else:
            NP0 = max(24, min(120, 18 + 4 * dim))
            p_hi, p_lo = 0.30, 0.07
            Fcap_hi, Fcap_lo = 0.95, 0.60
            cr_mu = 0.82

        NP_min = max(8, 6 + dim // 6)

        # init pop (inject best and jittered best)
        pop = []
        fit = []
        if best_x is not None:
            pop.append(best_x[:]); fit.append(best)
            jit = min(10, NP0 - 1)
            for _ in range(jit):
                if now() >= time_limit:
                    return
                x = best_x[:]
                # heavier tails in explore
                scale = 0.10 if mode == "explore" else (0.06 if mode == "balanced" else 0.04)
                for j in range(dim):
                    x[j] += scale * span_safe[j] * (rand_cauchy() if mode == "explore" and random.random() < 0.25 else randn())
                reflect_inplace(x)
                fx = try_update(x)
                pop.append(x); fit.append(fx)

        while len(pop) < NP0 and now() < time_limit:
            x = rand_point()
            fx = try_update(x)
            pop.append(x); fit.append(fx)

        if len(pop) < 4:
            return

        archive = []
        arch_max = len(pop)

        H = 12
        MF = [0.55] * H
        MCR = [cr_mu] * H
        k_mem = 0

        last_best_local = best
        stall = 0

        t_epoch0 = now()
        while now() < time_limit:
            frac = (now() - t_epoch0) / max(1e-12, (time_limit - t_epoch0))

            NP = len(pop)
            NP_target = int(round(NP_min + (NP0 - NP_min) * (1.0 - frac)))
            NP_target = max(NP_min, min(NP0, NP_target))

            # ranking for pbest
            idx_sorted = list(range(NP))
            idx_sorted.sort(key=lambda i: fit[i])

            p = p_hi - (p_hi - p_lo) * frac
            p = max(p_lo, min(p_hi, p))
            pcount = max(2, int(math.ceil(p * NP)))
            pbest_pool = idx_sorted[:pcount]

            SF = []
            SCR = []
            dImp = []

            unionN = NP + len(archive)

            def pick_union(exclude_pop):
                # exclude_pop: set of pop indices
                tot = unionN
                for _ in range(25):
                    r = random.randrange(tot)
                    if r < NP:
                        if r in exclude_pop:
                            continue
                        return pop[r]
                    else:
                        return archive[r - NP]
                for r in range(NP):
                    if r not in exclude_pop:
                        return pop[r]
                return pop[0]

            next_pop = [None] * NP
            next_fit = [None] * NP

            for i in range(NP):
                if now() >= time_limit:
                    break

                xi = pop[i]
                fi = fit[i]

                r = random.randrange(H)

                # F
                Fi = MF[r] + 0.12 * rand_cauchy()
                tries = 0
                while Fi <= 0.0 and tries < 6:
                    Fi = MF[r] + 0.12 * rand_cauchy()
                    tries += 1
                if Fi <= 0.0:
                    Fi = 0.1

                Fcap = Fcap_hi - (Fcap_hi - Fcap_lo) * frac
                if Fcap < 0.5:
                    Fcap = 0.5
                Fi = min(Fcap, max(0.05, Fi))

                # CR
                CRi = MCR[r] + 0.10 * randn()
                if CRi < 0.0: CRi = 0.0
                elif CRi > 1.0: CRi = 1.0

                pb = pbest_pool[random.randrange(len(pbest_pool))]
                xpbest = pop[pb]

                # indices selection (fast, correct)
                r1 = random.randrange(NP)
                tries = 0
                while (r1 == i or r1 == pb) and tries < 30:
                    r1 = random.randrange(NP)
                    tries += 1
                xr1 = pop[r1]

                exclude = {i, pb, r1}
                xr2 = pick_union(exclude)

                # weighted pull in exploit
                Fw = 0.85 if mode == "exploit" else (0.95 if mode == "balanced" else 1.05)

                v = [0.0] * dim
                for j in range(dim):
                    v[j] = xi[j] + (Fi * Fw) * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                reflect_inplace(v)

                # crossover
                jrand = random.randrange(dim)
                u = [0.0] * dim
                for j in range(dim):
                    if j == jrand or random.random() < CRi:
                        u[j] = v[j]
                    else:
                        u[j] = xi[j]
                reflect_inplace(u)

                fu = evaluate(u)

                if fu <= fi:
                    next_pop[i] = u
                    next_fit[i] = fu

                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        archive[random.randrange(len(archive))] = archive[-1]
                        archive.pop()

                    imp = fi - fu
                    if imp > 0.0:
                        SF.append(Fi)
                        SCR.append(CRi)
                        dImp.append(imp)

                    if fu < best:
                        best = fu
                        best_x = u[:]
                else:
                    next_pop[i] = xi
                    next_fit[i] = fi

            pop, fit = next_pop, next_fit

            # update memories
            if SF:
                wsum = sum(dImp)
                if wsum <= 0.0:
                    weights = [1.0 / len(SF)] * len(SF)
                else:
                    inv = 1.0 / wsum
                    weights = [imp * inv for imp in dImp]

                num = 0.0
                den = 0.0
                for w, fval in zip(weights, SF):
                    num += w * (fval * fval)
                    den += w * fval
                if den > 1e-30:
                    MF[k_mem] = num / den

                cr = 0.0
                for w, cval in zip(weights, SCR):
                    cr += w * cval
                MCR[k_mem] = cr

                k_mem = (k_mem + 1) % H

            # stall handling inside epoch
            if best < last_best_local - 1e-15 * (1.0 + abs(last_best_local)):
                last_best_local = best
                stall = 0
            else:
                stall += 1

            if stall > (10 + dim // 2):
                stall = 0
                # replace some worst
                NP = len(pop)
                idx_sorted = list(range(NP))
                idx_sorted.sort(key=lambda i: fit[i])
                worst_count = max(1, NP // 5)
                for k in range(worst_count):
                    if now() >= time_limit:
                        break
                    idx = idx_sorted[-1 - k]
                    r = random.random()
                    if best_x is None or r < 0.25:
                        xnew = rand_point()
                    elif r < 0.60:
                        xnew = best_x[:]
                        # sparse heavy-tailed jitter
                        kcount = max(1, int(math.sqrt(dim)))
                        if dim > 0:
                            coords = random.sample(range(dim), kcount) if dim >= kcount else list(range(dim))
                            sc = 0.18 if mode == "explore" else 0.12
                            for j in coords:
                                xnew[j] += sc * span_safe[j] * rand_cauchy()
                    else:
                        # opposition around box center
                        xnew = [(lo[j] + hi[j] - best_x[j]) for j in range(dim)]
                        for j in range(dim):
                            xnew[j] += 0.02 * span_safe[j] * randn()
                    reflect_inplace(xnew)
                    fnew = evaluate(xnew)
                    pop[idx] = xnew
                    fit[idx] = fnew
                    if fnew < best:
                        best, best_x = fnew, xnew[:]

            # reduce population
            if len(pop) > NP_target:
                idx_sorted = list(range(len(pop)))
                idx_sorted.sort(key=lambda i: fit[i])
                keep = idx_sorted[:NP_target]
                pop = [pop[i] for i in keep]
                fit = [fit[i] for i in keep]
                arch_max = len(pop)
                while len(archive) > arch_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

    # ---------- Time portfolio schedule ----------
    # Split time: explore -> balanced -> exploit, with local search sprinkled.
    T = float(max_time)
    if T <= 0.0:
        return best

    # Reserve a little tail for pure local search
    tail_local = min(0.22 * T, 0.8)
    main_deadline = deadline - tail_local

    # Run 2-3 DE epochs (shorter ones reduce risk of unlucky settings)
    epochs = [
        ("explore", 0.38),
        ("balanced", 0.37),
        ("exploit", 0.25),
    ]

    tcur = now()
    for mode, frac in epochs:
        if tcur >= main_deadline:
            break
        tlim = tcur + frac * (main_deadline - tcur)
        # also cap each epoch so it can't starve later stages
        tlim = min(tlim, main_deadline)
        de_epoch(tlim, mode)

        # quick local squeeze after each epoch
        if now() < main_deadline and best_x is not None:
            tl = min(main_deadline, now() + 0.08 * (main_deadline - now()))
            rosenbrock_local(tl)

        tcur = now()

    # Final local search tail
    while now() < deadline:
        # short repeated local bursts to respect deadline tightly
        tl = min(deadline, now() + 0.06)
        rosenbrock_local(tl)
        # if very little time remains, exit
        if deadline - now() < 0.01:
            break

    return best
