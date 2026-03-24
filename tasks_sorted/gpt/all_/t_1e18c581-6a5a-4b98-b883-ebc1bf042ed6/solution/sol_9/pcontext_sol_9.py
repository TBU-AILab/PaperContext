import random, time, math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libraries).

    Core idea: L-SHADE-like Differential Evolution (success-history parameter adaptation)
    + p-best current-to-pbest mutation + external archive + occasional coordinate-polish
    + low-discrepancy (Halton) init + stall-triggered partial restarts.

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

    # --------------- Halton init ---------------
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

    # ------------- elite archive (best solutions) -------------
    elite_cap = max(14, 6 + int(5.0 * math.sqrt(dim + 1.0)))
    elites = []  # (f, x) sorted

    def push_elite(f, x):
        nonlocal elites
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ------------- local polish: coordinate search -------------
    def coord_polish(x0, f0, steps, passes=2):
        x = x0[:]
        f = f0
        step = steps[:]
        min_step = [1e-15 * spans_safe[i] for i in range(dim)]
        for _ in range(passes):
            if time.time() >= deadline:
                break
            improved_any = False
            for j in range(dim):
                if time.time() >= deadline:
                    break
                sj = step[j]
                for _try in range(4):
                    if sj <= min_step[j]:
                        break
                    xp = x[:]; xm = x[:]
                    xp[j] = reflect(xp[j] + sj, lows[j], highs[j])
                    xm[j] = reflect(xm[j] - sj, lows[j], highs[j])

                    fp = evaluate(xp)
                    if fp < f:
                        x, f = xp, fp
                        improved_any = True
                        step[j] = min(spans_safe[j], step[j] * 1.25)
                        break

                    if time.time() >= deadline:
                        break

                    fm = evaluate(xm)
                    if fm < f:
                        x, f = xm, fm
                        improved_any = True
                        step[j] = min(spans_safe[j], step[j] * 1.25)
                        break

                    sj *= 0.5
                    step[j] = sj
            if not improved_any:
                break
        return x, f, step

    # ---------------- L-SHADE style memories ----------------
    H = 6 + int(2.0 * math.log(dim + 1.0))
    if H < 6:
        H = 6
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    def sample_F(mu):
        # Cauchy around mu, resample if <=0
        for _ in range(12):
            f = mu + 0.1 * math.tan(math.pi * (random.random() - 0.5))
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return max(0.05, min(1.0, mu))

    def sample_CR(mu):
        cr = random.gauss(mu, 0.1)
        if cr < 0.0: cr = 0.0
        if cr > 1.0: cr = 1.0
        return cr

    # ---------------- external archive for diversity ----------------
    A = []
    # will set A_cap after pop known
    A_cap = 0

    def archive_add(x):
        nonlocal A
        if A_cap <= 0:
            return
        if len(A) < A_cap:
            A.append(x[:])
        else:
            A[random.randrange(A_cap)] = x[:]

    # ---------------- choose sizes ----------------
    pop = max(24, 12 + int(10.0 * math.log(dim + 1.0)))
    pop = min(pop, 80)
    A_cap = pop

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # Spend a controlled init budget; Halton + opposition + a little random
    init_n = max(pop, 12 * dim)
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_vec() if random.random() < 0.86 else rand_vec()
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

    # build population from best found + diverse points
    X, F = [], []
    # seed from elites
    for _, ex in elites[:min(len(elites), pop)]:
        X.append(ex[:])
    while len(X) < pop:
        X.append(halton_vec() if random.random() < 0.75 else rand_vec())

    for i in range(pop):
        if time.time() >= deadline:
            return best
        fi = evaluate(X[i])
        F.append(fi)
        push_elite(fi, X[i])
        if fi < best:
            best, best_x = fi, X[i][:]

    # quick initial polish
    steps0 = [0.10 * spans_safe[i] for i in range(dim)]
    best_x, best, _ = coord_polish(best_x, best, steps0, passes=1)
    push_elite(best, best_x)

    # ---------------- main loop ----------------
    it = 0
    stall = 0
    last_best = best
    patience = max(90, 28 * dim)

    polish_every = max(18, 5 * dim)
    inject_every = max(50, 10 * dim)

    while time.time() < deadline:
        it += 1
        tr = (time.time() - t0) / (max_time if max_time > 0 else 1.0)
        if tr < 0.0: tr = 0.0
        if tr > 1.0: tr = 1.0

        # occasional global injection
        if it % inject_every == 0:
            ninj = max(2, pop // 8)
            for _ in range(ninj):
                if time.time() >= deadline:
                    return best
                xg = halton_vec() if random.random() < 0.85 else rand_vec()
                fg = evaluate(xg)
                push_elite(fg, xg)
                if fg < best:
                    best, best_x = fg, xg
                    stall = 0
                j = max(range(pop), key=lambda k: F[k])  # replace worst deterministically
                if fg < F[j] or random.random() < 0.10:
                    archive_add(X[j])
                    X[j] = xg
                    F[j] = fg

        # sort for pbest pool
        order = sorted(range(pop), key=lambda i: F[i])
        # p decreases to emphasize exploitation late, but not too hard (keeps multimodality)
        pfrac = 0.25 - 0.15 * tr  # 0.25 -> 0.10
        if pfrac < 0.08:
            pfrac = 0.08
        pcount = max(2, int(pfrac * pop))
        pbest_pool = order[:pcount]

        SF, SCR, dF = [], [], []  # successful F, CR, and improvements

        for i in range(pop):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            xi = X[i]

            pbest = X[random.choice(pbest_pool)]

            r1 = i
            while r1 == i:
                r1 = random.randrange(pop)

            # r2 from union (pop U archive)
            if A and random.random() < 0.5:
                r2v = A[random.randrange(len(A))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop)
                r2v = X[r2]

            xr1 = X[r1]

            # current-to-pbest/1
            v = [xi[d] + Fi * (pbest[d] - xi[d]) + Fi * (xr1[d] - r2v[d]) for d in range(dim)]
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
                # selection
                archive_add(X[i])
                improvement = F[i] - fu
                X[i] = u
                F[i] = fu
                SF.append(Fi)
                SCR.append(CRi)
                dF.append(improvement)

                if fu < best:
                    best, best_x = fu, u[:]
                    stall = 0

        # update memories (success-history)
        if SF:
            # weighted Lehmer mean for F, weighted mean for CR (weights by improvement)
            wsum = sum(dF) + 1e-30
            # F (Lehmer): sum(w*F^2)/sum(w*F)
            numF = 0.0
            denF = 0.0
            numCR = 0.0
            for k in range(len(SF)):
                w = dF[k] / wsum
                f = SF[k]
                cr = SCR[k]
                numF += w * f * f
                denF += w * f
                numCR += w * cr

            MF_new = numF / (denF + 1e-30)
            MCR_new = numCR

            M_F[mem_idx] = MF_new
            M_CR[mem_idx] = MCR_new
            mem_idx = (mem_idx + 1) % H

        # periodic local polish (late slightly stronger)
        if it % polish_every == 0 and time.time() < deadline:
            scale = 0.14 * (1.0 - 0.75 * tr)  # shrink with time
            steps = [max(1e-15 * spans_safe[i], scale * spans_safe[i]) for i in range(dim)]
            best_x, best, _ = coord_polish(best_x, best, steps, passes=2)
            push_elite(best, best_x)

        # stall detection + partial restart
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= patience and time.time() < deadline:
            stall = 0
            # reinitialize worst fraction using elite jitter + some global points
            frac = 0.35 if tr < 0.6 else 0.25
            k = max(2, int(frac * pop))
            worst = sorted(range(pop), key=lambda i: F[i], reverse=True)[:k]

            for wi in worst:
                if time.time() >= deadline:
                    return best
                if elites and random.random() < 0.80:
                    # jitter good elite
                    emax = min(len(elites), elite_cap)
                    eidx = int((random.random() ** 2.0) * emax)
                    base = elites[eidx][1]
                    xnew = [base[d] + random.gauss(0.0, 0.20 * spans_safe[d]) for d in range(dim)]
                    xnew = proj(xnew)
                else:
                    xnew = halton_vec() if random.random() < 0.85 else rand_vec()

                fnew = evaluate(xnew)
                push_elite(fnew, xnew)
                archive_add(X[wi])
                X[wi] = xnew
                F[wi] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # opposition of incumbent best sometimes yields a jump
            xop = opposite(best_x)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop[:]

    return best
