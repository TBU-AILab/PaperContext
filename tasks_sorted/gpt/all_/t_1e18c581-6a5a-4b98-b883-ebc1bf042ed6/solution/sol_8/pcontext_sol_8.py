import random, time, math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libraries).

    What is improved vs your listed variants:
      - Uses a robust hybrid core: adaptive Differential Evolution (JADE-like) +
        periodic local coordinate search.
      - Maintains an elite archive and uses "current-to-pbest" mutation for faster
        convergence, while keeping diversity via external archive.
      - Stronger, cheaper initialization: scrambled Halton + opposition + small LHS batches.
      - Time-aware scheduling and stall-triggered partial restarts.
      - Reflection boundary handling (less corner sticking than clamping).

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        return float(func([]))

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
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

    # ---------------- Halton (scrambled by shift) ----------------
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

    # ---------------- small LHS batch ----------------
    def lhs_batch(n):
        perms = []
        for i in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        batch = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                u = (perms[i][k] + random.random()) / float(n)
                x[i] = lows[i] + u * spans[i]
            batch.append(x)
        return batch

    # ---------------- elite archive (for pbest + restart seeding) ----------------
    elite_cap = max(12, 6 + int(4.5 * math.sqrt(dim + 1.0)))
    elites = []  # (f, x) sorted

    def push_elite(f, x):
        nonlocal elites
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ---------------- local search: bounded coordinate search ----------------
    def coord_search(x0, f0, steps, max_passes=2):
        x = x0[:]
        f = f0
        step = steps[:]
        min_step = [1e-15 * spans_safe[i] for i in range(dim)]
        for _ in range(max_passes):
            if time.time() >= deadline:
                break
            improved = False
            for j in range(dim):
                if time.time() >= deadline:
                    break
                sj = step[j]
                # attempt a few halvings
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
                        improved = True
                        step[j] = min(spans_safe[j], step[j] * 1.20)
                        break

                    if time.time() >= deadline:
                        break
                    fm = evaluate(xm)
                    if fm < f:
                        x, f = xm, fm
                        improved = True
                        step[j] = min(spans_safe[j], step[j] * 1.20)
                        break

                    sj *= 0.5
                    step[j] = sj
            if not improved:
                break
        return x, f, step

    # ---------------- DE state (JADE-like) ----------------
    pop = max(20, 10 + int(8.0 * math.log(dim + 1.0)))
    pop = min(pop, 70)

    # External archive A for mutation diversity
    A = []
    A_cap = pop

    def archive_add(x):
        nonlocal A
        if A_cap <= 0:
            return
        if len(A) < A_cap:
            A.append(x[:])
        else:
            A[random.randrange(A_cap)] = x[:]

    # Running parameter means (JADE)
    mu_F = 0.6
    mu_CR = 0.5
    c_adapt = 0.1  # learning rate

    def sample_F():
        # Cauchy-like heavy tail but bounded.
        nonlocal mu_F
        for _ in range(8):
            f = mu_F + 0.1 * math.tan(math.pi * (random.random() - 0.5))
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return max(0.05, min(1.0, mu_F))

    def sample_CR():
        nonlocal mu_CR
        cr = random.gauss(mu_CR, 0.1)
        if cr < 0.0: cr = 0.0
        if cr > 1.0: cr = 1.0
        return cr

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # stronger initial sampling, time permitting
    init_target = max(pop, 10 * dim)
    # include a small LHS batch early
    if dim <= 60:
        for x in lhs_batch(max(6, min(16, 2 + dim // 2))):
            if time.time() >= deadline:
                return best
            f = evaluate(x)
            push_elite(f, x)
            if f < best:
                best, best_x = f, x

    for _ in range(init_target):
        if time.time() >= deadline:
            return best
        x = halton_vec() if random.random() < 0.85 else rand_vec()
        f = evaluate(x)
        push_elite(f, x)
        if f < best:
            best, best_x = f, x

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo

    # Build population from elites + some fresh diversity
    X, F = [], []
    seed = [e[1] for e in elites[:min(len(elites), pop)]]
    while len(seed) < pop:
        seed.append(halton_vec() if random.random() < 0.75 else rand_vec())

    for x in seed[:pop]:
        if time.time() >= deadline:
            return best
        fx = evaluate(x)
        X.append(x)
        F.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x

    # quick polish
    base_steps = [0.10 * spans_safe[i] for i in range(dim)]
    best_x, best, base_steps = coord_search(best_x, best, base_steps, max_passes=1)
    push_elite(best, best_x)

    # ---------------- main loop ----------------
    it = 0
    stall = 0
    last_best = best
    patience = max(80, 26 * dim)

    polish_every = max(18, 5 * dim)
    inject_every = max(40, 9 * dim)

    while time.time() < deadline:
        it += 1
        tr = (time.time() - t0) / (max_time if max_time > 0 else 1.0)
        if tr < 0.0: tr = 0.0
        if tr > 1.0: tr = 1.0

        # occasional injection (global points)
        if it % inject_every == 0:
            ninj = max(2, pop // 7)
            for _ in range(ninj):
                if time.time() >= deadline:
                    return best
                xg = halton_vec() if random.random() < 0.85 else rand_vec()
                fg = evaluate(xg)
                push_elite(fg, xg)
                if fg < best:
                    best, best_x = fg, xg
                    stall = 0
                j = random.randrange(pop)
                if fg < F[j] or random.random() < 0.10:
                    archive_add(X[j])
                    X[j] = xg
                    F[j] = fg

        # determine p-best fraction (more exploit over time)
        p = 0.12 + 0.65 * tr
        pcount = max(2, int(p * pop))

        # generation success memories for JADE updates
        SF, SCR = [], []

        # pre-sort indices for pbest selection
        order = sorted(range(pop), key=lambda i: F[i])
        pbest_pool = order[:pcount]

        for i in range(pop):
            if time.time() >= deadline:
                return best

            Fi = sample_F()
            CRi = sample_CR()

            # choose pbest
            pbest = X[random.choice(pbest_pool)]

            # choose r1 from population excluding i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop)

            # choose r2 from population U archive excluding i and r1
            use_archive = (A and random.random() < 0.5)
            if use_archive:
                r2_vec = A[random.randrange(len(A))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop)
                r2_vec = X[r2]

            xi = X[i]
            xr1 = X[r1]

            # current-to-pbest/1
            v = [xi[d] + Fi * (pbest[d] - xi[d]) + Fi * (xr1[d] - r2_vec[d]) for d in range(dim)]
            v = proj(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]

            fu = evaluate(u)
            push_elite(fu, u)

            if fu < F[i]:
                archive_add(X[i])
                X[i] = u
                F[i] = fu
                SF.append(Fi)
                SCR.append(CRi)
                if fu < best:
                    best, best_x = fu, u
                    stall = 0

        # JADE parameter mean updates
        if SF:
            # Lehmer mean for F
            num = sum(f * f for f in SF)
            den = sum(f for f in SF) + 1e-30
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * (num / den)
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(SCR) / float(len(SCR)))

        # periodic local polish
        if it % polish_every == 0 and time.time() < deadline:
            # shrink steps over time
            scale = 0.16 * (1.0 - 0.80 * tr)
            steps = [max(1e-15 * spans_safe[i], scale * spans_safe[i]) for i in range(dim)]
            best_x, best, _ = coord_search(best_x, best, steps, max_passes=2)
            push_elite(best, best_x)

        # stall logic + partial restart
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= patience and time.time() < deadline:
            stall = 0
            # reinit worst fraction from elites + global points
            frac = 0.35 if tr < 0.6 else 0.25
            k = max(2, int(frac * pop))
            worst = sorted(range(pop), key=lambda i: F[i], reverse=True)[:k]

            for wi in worst:
                if time.time() >= deadline:
                    return best
                if elites and random.random() < 0.80:
                    # jitter an elite
                    eidx = int((random.random() ** 2.0) * min(len(elites), elite_cap))
                    base = elites[eidx][1]
                    xnew = [base[d] + random.gauss(0.0, 0.18 * spans_safe[d]) for d in range(dim)]
                    xnew = proj(xnew)
                else:
                    xnew = halton_vec() if random.random() < 0.85 else rand_vec()

                fnew = evaluate(xnew)
                push_elite(fnew, xnew)
                archive_add(X[wi])
                X[wi] = xnew
                F[wi] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew

            # opposition attempt of incumbent best
            xop = opposite(best_x)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop

    return best
