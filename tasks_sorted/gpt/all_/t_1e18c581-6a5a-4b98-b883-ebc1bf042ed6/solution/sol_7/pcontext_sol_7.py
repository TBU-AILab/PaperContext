import random, time, math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Main improvements vs the listed variants:
      - Uses a compact DE (Differential Evolution) core: strong global performance on rugged landscapes.
      - Maintains an elite archive and periodically "current-to-best" exploitation.
      - Adds a cheap but powerful local refinement: bounded coordinate pattern search with adaptive steps.
      - Uses reflection boundary handling (less corner sticking than clamping).
      - Time-aware scheduling: exploration early, intensification later, plus stall-triggered restarts.

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def proj(x):
        return [reflect(x[i], lows[i], highs[i]) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------- Halton (for fast early coverage) ----------
    def nth_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    bases = nth_primes(max(1, dim))
    h_index = 1
    h_shift = [random.random() for _ in range(dim)]

    def radical_inverse(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    def halton_vec():
        nonlocal h_index
        k = h_index
        h_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (radical_inverse(k, bases[i]) + h_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite archive ----------
    elite_cap = max(12, 6 + int(4.5 * math.sqrt(dim + 1.0)))
    elites = []  # (f, x) sorted

    def push_elite(f, x):
        nonlocal elites
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ---------- local search (coordinate pattern search) ----------
    def coord_polish(x0, f0, step0, passes=2):
        x = x0[:]
        f = f0
        step = step0[:]
        min_step = [1e-15 * spans_safe[i] for i in range(dim)]
        for _ in range(passes):
            if time.time() >= deadline:
                break
            improved_any = False
            for j in range(dim):
                if time.time() >= deadline:
                    break
                sj = step[j]
                # Try a few halvings if needed
                for _try in range(4):
                    if sj <= min_step[j]:
                        break
                    xp = x[:]
                    xm = x[:]
                    xp[j] = reflect(xp[j] + sj, lows[j], highs[j])
                    xm[j] = reflect(xm[j] - sj, lows[j], highs[j])

                    fp = evaluate(xp)
                    if fp < f:
                        x, f = xp, fp
                        improved_any = True
                        step[j] = min(step[j] * 1.20, spans_safe[j])
                        break

                    if time.time() >= deadline:
                        break
                    fm = evaluate(xm)
                    if fm < f:
                        x, f = xm, fm
                        improved_any = True
                        step[j] = min(step[j] * 1.20, spans_safe[j])
                        break

                    sj *= 0.5
                    step[j] = sj
            if not improved_any:
                break
        return x, f, step

    # ---------- initialize population ----------
    if dim <= 0:
        return float(func([]))

    pop = max(18, 10 + int(6.0 * math.log(dim + 1.0)))
    pop = min(pop, 60)  # keep evaluations predictable

    X = []
    F = []

    best = float("inf")
    best_x = None

    # initial points: mostly Halton + opposition, some random
    init_points = max(pop, 12 * dim)
    for _ in range(init_points):
        if time.time() >= deadline:
            return best
        x = halton_vec() if random.random() < 0.85 else rand_vec()
        fx = evaluate(x)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo

    # build starting pop from elites + a few fresh points for diversity
    seed = [e[1] for e in elites[:min(len(elites), pop)]]
    while len(seed) < pop:
        seed.append(halton_vec() if random.random() < 0.7 else rand_vec())

    for x in seed[:pop]:
        if time.time() >= deadline:
            return best
        fx = evaluate(x)
        X.append(x)
        F.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x

    # quick polish on best
    base_step = [0.12 * spans_safe[i] for i in range(dim)]
    best_x, best, base_step = coord_polish(best_x, best, base_step, passes=1)
    push_elite(best, best_x)

    # ---------- DE parameters (self-adaptive-ish scheduling) ----------
    # We'll jitter F/CR per trial; and add a "current-to-best/1" branch later in time.
    it = 0
    stall = 0
    last_best = best
    patience = max(80, 30 * dim)

    polish_every = max(20, 6 * dim)
    inject_every = max(35, 9 * dim)

    # ---------- main loop ----------
    while time.time() < deadline:
        it += 1
        # progress ratio in [0,1]
        tr = (time.time() - t0) / max(1e-9, max_time)
        if tr < 0.0:
            tr = 0.0
        if tr > 1.0:
            tr = 1.0

        # occasional injection to fight premature convergence
        if it % inject_every == 0:
            n_inj = max(2, pop // 6)
            for _ in range(n_inj):
                if time.time() >= deadline:
                    return best
                xg = halton_vec() if random.random() < 0.85 else rand_vec()
                fg = evaluate(xg)
                push_elite(fg, xg)
                if fg < best:
                    best, best_x = fg, xg
                    stall = 0
                # replace a random individual if injection is good or population is stale
                j = random.randrange(pop)
                if fg < F[j] or random.random() < 0.15:
                    X[j] = xg
                    F[j] = fg

        # mutation strategy blend:
        # early: rand/1/bin; late: current-to-best/1/bin more often.
        p_best = 0.15 + 0.70 * tr  # increase exploitation over time

        for i in range(pop):
            if time.time() >= deadline:
                return best

            # choose distinct indices
            idxs = list(range(pop))
            idxs.remove(i)
            r1 = random.choice(idxs); idxs.remove(r1)
            r2 = random.choice(idxs); idxs.remove(r2)
            r3 = random.choice(idxs)

            xi = X[i]
            xr1 = X[r1]
            xr2 = X[r2]
            xr3 = X[r3]

            # pick a "p-best" target from elite archive (or current best_x)
            if elites and random.random() < 0.85:
                kmax = max(1, int(p_best * len(elites)))
                kmax = min(len(elites), max(2, kmax))
                pb = elites[random.randrange(kmax)][1]
            else:
                pb = best_x

            # jitter parameters per-trial
            # F: moderate early, slightly smaller late for fine search
            Fm = 0.85 - 0.35 * tr
            FF = min(1.2, max(0.15, random.gauss(Fm, 0.12)))
            # CR: increase late for coordinate-wise refinement
            CRm = 0.55 + 0.35 * tr
            CR = min(0.98, max(0.05, random.gauss(CRm, 0.12)))

            use_ctb = (random.random() < (0.20 + 0.65 * tr))
            if use_ctb:
                # current-to-best/1
                v = [xi[d] + FF * (pb[d] - xi[d]) + FF * (xr1[d] - xr2[d]) for d in range(dim)]
            else:
                # rand/1
                v = [xr3[d] + FF * (xr1[d] - xr2[d]) for d in range(dim)]

            v = proj(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            fu = evaluate(u)
            push_elite(fu, u)

            if fu < F[i]:
                X[i] = u
                F[i] = fu
                if fu < best:
                    best, best_x = fu, u
                    stall = 0

        # periodic local polishing (more frequent late)
        if it % polish_every == 0 and time.time() < deadline:
            # steps shrink with time (trust-region-like)
            scale = 0.18 * (1.0 - 0.75 * tr)
            steps = [max(1e-15 * spans_safe[i], scale * spans_safe[i]) for i in range(dim)]
            best_x, best, _ = coord_polish(best_x, best, steps, passes=2)
            push_elite(best, best_x)

        # stall / restart: re-seed part of population from elites + global points
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= patience and time.time() < deadline:
            stall = 0
            # reinitialize worst fraction
            frac = 0.30 if tr < 0.7 else 0.20
            k = max(2, int(frac * pop))

            # indices of worst individuals
            order = sorted(range(pop), key=lambda idx: F[idx], reverse=True)
            worst = order[:k]

            for wi in worst:
                if time.time() >= deadline:
                    return best
                if elites and random.random() < 0.75:
                    # jitter an elite
                    eidx = int((random.random() ** 2.2) * min(len(elites), max(4, elite_cap)))
                    base = elites[eidx][1]
                    # jitter size tied to box
                    xnew = [base[d] + random.gauss(0.0, 0.20 * spans_safe[d]) for d in range(dim)]
                    xnew = proj(xnew)
                else:
                    xnew = halton_vec() if random.random() < 0.85 else rand_vec()

                fnew = evaluate(xnew)
                push_elite(fnew, xnew)
                X[wi] = xnew
                F[wi] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew

            # also try opposition of best (sometimes huge win)
            xop = opposite(best_x)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop

    return best
