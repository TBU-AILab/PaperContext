import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvement strategy (vs the provided DE variants):
      1) Keep a strong L-SHADE/JADE-like DE core (current-to-pbest/1 + archive + success memories).
      2) Make p-best selection truly cheap (no sorting): use a tiny tournament biased to good solutions.
      3) Add a *real* intensification stage that DE often lacks: a small-budget, best-centered
         SPSA-like gradient estimate + backtracking line-search (works well under noise and high-dim).
      4) Add periodic short trust-region coordinate/projection probes (very cheap exploitation).
      5) Better stagnation handling: adaptive restart injection when progress stalls.

    Returns:
        best (float): best function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    if max_time is None or max_time <= 0 or dim <= 0:
        return float("inf")

    # ---------------- bounds ----------------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            m = 0.5 * (lows[i] + highs[i])
            lows[i] = highs[i] = m

    # ---------------- fast RNG ----------------
    rng_state = random.getrandbits(64) ^ (int(time.time() * 1e9) & ((1 << 64) - 1))

    def u01():
        nonlocal rng_state
        rng_state = (6364136223846793005 * rng_state + 1442695040888963407) & ((1 << 64) - 1)
        return ((rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def randint(n):
        if n <= 1:
            return 0
        x = int(u01() * n)
        return x if x < n else (n - 1)

    def randn():
        a = max(1e-300, u01())
        b = u01()
        return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)

    def reflect(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect into range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        return v

    def eval_f(x):
        return float(func(list(x)))

    # ---------------- Halton init ----------------
    def first_primes(k):
        ps = []
        n = 2
        while len(ps) < k:
            ok = True
            r = int(math.sqrt(n))
            for p in ps:
                if p > r:
                    break
                if n % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(n)
            n += 1
        return ps

    primes = first_primes(min(32, max(1, dim)))

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def make_point(idx):
        x = [0.0] * dim
        shift = u01()
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
            else:
                base = primes[d % len(primes)]
                h = (halton(idx + 1, base) + 0.37 * shift + 0.13 * u01()) % 1.0
                x[d] = lows[d] + h * spans[d]
        return x

    def opposite(x):
        y = x[:]
        for d in range(dim):
            if spans[d] == 0.0:
                y[d] = lows[d]
            else:
                y[d] = reflect(lows[d] + highs[d] - x[d], d)
        return y

    # ---------------- DE parameters ----------------
    NP0 = max(28, min(160, 18 + 8 * int(math.sqrt(dim)) + dim // 2))
    NPmin = max(10, min(30, 6 + dim // 8))
    NP = NP0

    H = 10
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    pmin, pmax = 0.06, 0.25

    archive = []
    arch_max = NP0

    # ---------------- init population ----------------
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    second_best = float("inf")
    second_x = None

    # keep some budget for the main loop: cap init if time is tiny
    for i in range(NP0):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        x = make_point(i)
        fx = eval_f(x)
        xo = opposite(x)
        fxo = eval_f(xo)
        if fxo < fx:
            x, fx = xo, fxo
        pop.append(x)
        fit.append(fx)
        if fx < best:
            second_best, second_x = best, (best_x[:] if best_x is not None else None)
            best, best_x = fx, x[:]
        elif fx < second_best:
            second_best, second_x = fx, x[:]

    # ---------------- sampling F/CR ----------------
    def sample_F(mu):
        for _ in range(12):
            F = mu + 0.1 * math.tan(math.pi * (u01() - 0.5))
            if F > 0.0:
                return 1.0 if F > 1.0 else F
        return max(1e-3, min(1.0, mu))

    def sample_CR(mu):
        CR = mu + 0.1 * randn()
        if CR < 0.0:
            return 0.0
        if CR > 1.0:
            return 1.0
        return CR

    # ---------------- cheap pbest selection (tournament) ----------------
    def pbest_index(pfrac):
        # choose the best among m random candidates; m scales with pfrac
        k = max(2, int(math.ceil(pfrac * NP)))
        m = min(NP, max(10, 2 * k + 4))
        bi = randint(NP)
        bf = fit[bi]
        for _ in range(m - 1):
            j = randint(NP)
            fj = fit[j]
            if fj < bf:
                bf = fj
                bi = j
        return bi

    # ---------------- intensification: SPSA-like gradient + line search ----------------
    # (very few evaluations, but often gives a strong late-stage boost)
    spsa_alpha = 0.2  # step multiplier (will be adapted)
    spsa_base = 0.08  # perturbation fraction of span

    def spsa_refine(time_frac):
        nonlocal best, best_x, second_best, second_x, spsa_alpha, spsa_base

        if best_x is None or time.time() >= deadline:
            return

        # do more SPSA late; also keep tiny budgets
        rounds = 1 if time_frac < 0.45 else (2 if time_frac < 0.80 else 3)

        # anchor: mostly best, sometimes second-best
        anchor = best_x if (second_x is None or u01() < 0.85) else second_x
        f_anchor = best if anchor is best_x else second_best

        # per-round: 2 evals for gradient + a small backtracking line-search (<=3 evals)
        for _ in range(rounds):
            if time.time() >= deadline:
                return

            # build Rademacher direction (+1/-1); optionally sparse for large dim
            delta = [0] * dim
            if dim <= 60:
                for d in range(dim):
                    delta[d] = 1 if u01() < 0.5 else -1
            else:
                # sparse: only flip k coords, rest 0
                k = max(12, dim // 8)
                idxs = set()
                while len(idxs) < k:
                    idxs.add(randint(dim))
                for d in idxs:
                    delta[d] = 1 if u01() < 0.5 else -1

            # perturbation sizes
            c = spsa_base * (0.65 + 0.7 * (1.0 - time_frac))  # slightly smaller late
            xp = anchor[:]
            xm = anchor[:]
            for d in range(dim):
                if spans[d] == 0.0 or delta[d] == 0:
                    continue
                h = c * spans[d]
                xp[d] = reflect(xp[d] + h * delta[d], d)
                xm[d] = reflect(xm[d] - h * delta[d], d)

            fp = eval_f(xp)
            if time.time() >= deadline:
                return
            fm = eval_f(xm)

            # estimate gradient along delta: g_i ~ (fp - fm)/(2*h_i*delta_i)
            # then take a step: x_new = anchor - a * g
            # We approximate per-dim with scaling by span and avoid dividing by tiny.
            a = spsa_alpha * (0.30 + 0.85 * time_frac)  # slightly larger late (trust-region effect)
            x_try = anchor[:]
            denom = (fp - fm)  # scalar diff
            for d in range(dim):
                if spans[d] == 0.0 or delta[d] == 0:
                    continue
                h = max(1e-18, c * spans[d])
                gi = denom / (2.0 * h * float(delta[d]))
                # scale step by span to make it unitless-ish and stable across bounds
                step = a * gi
                # clip step (trust region)
                step = max(-0.35 * spans[d], min(0.35 * spans[d], step))
                x_try[d] = reflect(x_try[d] - step, d)

            f_try = eval_f(x_try)

            # simple backtracking if no improvement
            if f_try < f_anchor:
                anchor, f_anchor = x_try, f_try
                # modestly increase step sizes if improving
                spsa_alpha = min(0.7, spsa_alpha * 1.08)
                spsa_base = min(0.25, spsa_base * 1.03)
            else:
                # try smaller step (1-2 attempts)
                improved = False
                x_bt = x_try
                f_bt = f_try
                bt_a = a
                for _bt in range(2):
                    if time.time() >= deadline:
                        break
                    bt_a *= 0.35
                    x_bt = anchor[:]
                    for d in range(dim):
                        if spans[d] == 0.0 or delta[d] == 0:
                            continue
                        h = max(1e-18, c * spans[d])
                        gi = denom / (2.0 * h * float(delta[d]))
                        step = bt_a * gi
                        step = max(-0.25 * spans[d], min(0.25 * spans[d], step))
                        x_bt[d] = reflect(x_bt[d] - step, d)
                    f_bt = eval_f(x_bt)
                    if f_bt < f_anchor:
                        anchor, f_anchor = x_bt, f_bt
                        improved = True
                        break
                if improved:
                    spsa_alpha = min(0.7, spsa_alpha * 1.03)
                else:
                    # shrink if not helping
                    spsa_alpha = max(0.03, spsa_alpha * 0.78)
                    spsa_base = max(0.02, spsa_base * 0.90)

            # bookkeeping
            if f_anchor < best:
                second_best, second_x = best, (best_x[:] if best_x is not None else None)
                best, best_x = f_anchor, anchor[:]
            elif f_anchor < second_best and (best_x is None or anchor != best_x):
                second_best, second_x = f_anchor, anchor[:]

    # ---------------- micro local probes around best (coordinate + 2D) ----------------
    diag = [0.16] * dim
    diag_min, diag_max = 1e-14, 0.55

    def micro_probes(time_frac):
        nonlocal best, best_x, second_best, second_x
        if best_x is None or time.time() >= deadline:
            return
        # 1-3 cheap attempts
        L = 1 if time_frac < 0.55 else (2 if time_frac < 0.85 else 3)

        for _ in range(L):
            if time.time() >= deadline:
                return
            anchor = best_x[:]
            # either coordinate subset or 2D rotation
            y = anchor[:]
            if dim >= 2 and u01() < 0.25:
                i = randint(dim)
                j = randint(dim - 1)
                if j >= i:
                    j += 1
                a = 2.0 * math.pi * u01()
                c, s = math.cos(a), math.sin(a)
                si = (diag[i] * spans[i]) * randn() if spans[i] != 0.0 else 0.0
                sj = (diag[j] * spans[j]) * randn() if spans[j] != 0.0 else 0.0
                di = c * si - s * sj
                dj = s * si + c * sj
                if spans[i] != 0.0:
                    y[i] = reflect(y[i] + di, i)
                if spans[j] != 0.0:
                    y[j] = reflect(y[j] + dj, j)
            else:
                # sparse coordinate gaussian
                if dim <= 30:
                    coords = range(dim)
                else:
                    k = max(10, dim // 6)
                    coords = [randint(dim) for _ in range(k)]
                for d in coords:
                    if spans[d] == 0.0:
                        continue
                    y[d] = reflect(y[d] + (diag[d] * spans[d]) * randn(), d)

            fy = eval_f(y)
            if fy < best:
                second_best, second_x = best, (best_x[:] if best_x is not None else None)
                best, best_x = fy, y[:]
                # increase steps slightly on success
                for d in range(dim):
                    diag[d] = min(diag_max, diag[d] * 1.05)
            else:
                # shrink slightly on failure
                for d in range(dim):
                    diag[d] = max(diag_min, diag[d] * 0.98)

    # ---------------- main loop ----------------
    gen = 0
    last_best = best
    stall = 0

    while time.time() < deadline:
        now = time.time()
        time_frac = (now - t0) / max(1e-12, (deadline - t0))
        if time_frac >= 1.0:
            break

        # intensification schedule:
        # micro probes are cheap; SPSA is stronger but costs more evals.
        if u01() < (0.10 + 0.30 * time_frac):
            micro_probes(time_frac)
        if u01() < (0.04 + 0.22 * time_frac):
            spsa_refine(time_frac)

        # DE p schedule
        pfrac = pmin + (pmax - pmin) * (0.75 - 0.60 * time_frac)
        if pfrac < pmin:
            pfrac = pmin
        if pfrac > pmax:
            pfrac = pmax

        # shuffled indices
        idxs = list(range(NP))
        for i in range(NP - 1, 0, -1):
            j = randint(i + 1)
            idxs[i], idxs[j] = idxs[j], idxs[i]

        S_F, S_CR, S_w = [], [], []

        union = pop + archive
        unionN = len(union)

        for ii in range(NP):
            if time.time() >= deadline:
                return best
            i = idxs[ii]
            xi = pop[i]
            fi = fit[i]

            r = randint(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            pb = pbest_index(pfrac)
            xpb = pop[pb]

            r1 = i
            while r1 == i:
                r1 = randint(NP)
            xr1 = pop[r1]

            r2 = -1
            for _ in range(25):
                rr = randint(unionN)
                if rr < NP:
                    if rr != i and rr != r1:
                        r2 = rr
                        break
                else:
                    r2 = rr
                    break
            if r2 < 0:
                r2 = (r1 + 1) % NP
            xr2 = union[r2]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover with slight best mixing sometimes
            jrand = randint(dim)
            ui = xi[:]
            mix_best = (best_x is not None and dim >= 6 and u01() < 0.18)
            for d in range(dim):
                if d == jrand or u01() < CRi:
                    val = v[d]
                    if mix_best:
                        val = 0.92 * val + 0.08 * best_x[d]
                    ui[d] = reflect(val, d)

            fui = eval_f(ui)

            if fui < fi:
                # archive
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[randint(arch_max)] = xi[:]

                pop[i] = ui
                fit[i] = fui

                if fui < best:
                    second_best, second_x = best, (best_x[:] if best_x is not None else None)
                    best, best_x = fui, ui[:]
                elif fui < second_best and (best_x is None or ui != best_x):
                    second_best, second_x = fui, ui[:]

                w = fi - fui
                if w < 1e-12:
                    w = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                S_w.append(w)

        # update memories
        if S_w:
            wsum = sum(S_w)
            mcr = 0.0
            for w, cr in zip(S_w, S_CR):
                mcr += w * cr
            mcr /= wsum

            num = 0.0
            den = 0.0
            for w, Fv in zip(S_w, S_F):
                num += w * Fv * Fv
                den += w * Fv
            mf = (num / den) if den > 0.0 else 0.5

            M_F[mem_idx] = mf
            M_CR[mem_idx] = mcr
            mem_idx = (mem_idx + 1) % H

        # archive size control
        arch_max = max(NP, NP0)
        if len(archive) > arch_max:
            for _ in range(len(archive) - arch_max):
                archive.pop(randint(len(archive)))

        # population reduction with time
        gen += 1
        target_NP = int(round(NPmin + (NP0 - NPmin) * (1.0 - time_frac)))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            remove = NP - target_NP
            for _ in range(remove):
                worst = 0
                wf = fit[0]
                for j in range(1, NP):
                    if fit[j] > wf:
                        wf = fit[j]
                        worst = j
                pop.pop(worst)
                fit.pop(worst)
                NP -= 1
                if NP <= NPmin:
                    break

        # stagnation tracking + adaptive injection/restart
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        if (time_frac > 0.35 and stall >= 5) or (time_frac > 0.70 and stall >= 3):
            stall = 0
            # replace a few worst with (best-jitter) and (Halton) points
            inj = 1 if NP <= 20 else 2
            for _ in range(inj):
                if time.time() >= deadline:
                    break
                worst = 0
                wf = fit[0]
                for j in range(1, NP):
                    if fit[j] > wf:
                        wf = fit[j]
                        worst = j

                if best_x is not None and u01() < 0.75:
                    y = best_x[:]
                    # jitter radius depends on time; not too tiny to escape
                    jit = 0.10 * (1.0 - time_frac) + 0.015
                    for d in range(dim):
                        if spans[d] == 0.0:
                            continue
                        y[d] = reflect(y[d] + (u01() * 2.0 - 1.0) * jit * spans[d], d)
                else:
                    y = make_point(int(1e6 * u01()) + 13 * gen + worst)

                fy = eval_f(y)
                pop[worst] = y
                fit[worst] = fy
                if fy < best:
                    second_best, second_x = best, (best_x[:] if best_x is not None else None)
                    best, best_x = fy, y[:]
                elif fy < second_best and (best_x is None or y != best_x):
                    second_best, second_x = fy, y[:]

    return best
