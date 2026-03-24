import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over the provided JADE/DE variants:
      1) Keep a strong L-SHADE/JADE core: current-to-pbest/1 + external archive +
         success-history memories for F and CR (multiple memory slots, not single mu_F/mu_CR).
      2) Eliminate expensive full sorting each generation:
         - maintain an "elite set" (top-k indices) updated incrementally
         - pbest selection sampled uniformly from this elite set.
      3) Add an exploitation engine that is consistently effective:
         - "Bobyqa-lite" style trust-region coordinate search around best with adaptive radius
           and opportunistic 2D rotation steps (still very cheap).
      4) Add occasional SPSA refinement only late (small, controlled budget).
      5) Better stagnation response: targeted partial restart (replace worst) using
         best-centered heavy-tailed jitter + low-discrepancy points.

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
            spans[i] = 0.0

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

    def cauchy():
        # standard Cauchy via tan(pi*(u-0.5))
        return math.tan(math.pi * (u01() - 0.5))

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

    # ---------------- Halton init (cheap low-discrepancy) ----------------
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
                h = (halton(idx + 1, base) + 0.31 * shift + 0.17 * u01()) % 1.0
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

    # ---------------- DE / LSHADE-ish parameters ----------------
    NP0 = max(28, min(140, 16 + 7 * int(math.sqrt(dim)) + dim // 2))
    NPmin = max(10, min(32, 6 + dim // 8))
    NP = NP0

    H = 12  # memory size
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_pos = 0

    pmin, pmax = 0.06, 0.28

    archive = []
    arch_max = NP0

    # ---------------- init population ----------------
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    # keep a small elite-set for cheap pbest selection
    elite_k = max(2, min(NP0, int(0.20 * NP0) + 2))
    elite_idx = []  # indices of current elite (top-k)

    def rebuild_elite():
        nonlocal elite_idx, elite_k
        # find top elite_k by partial selection (O(NP*elite_k), but elite_k small)
        # Start with empty and insert.
        elite = []
        for i in range(NP):
            fi = fit[i]
            if len(elite) < elite_k:
                elite.append(i)
            else:
                # find worst in elite
                wi = elite[0]
                wf = fit[wi]
                for j in elite[1:]:
                    fj = fit[j]
                    if fj > wf:
                        wf = fj
                        wi = j
                if fi < wf:
                    elite.remove(wi)
                    elite.append(i)
        # sort elite for slightly better stability (still tiny)
        elite.sort(key=lambda idx: fit[idx])
        elite_idx = elite

    # init with halton + opposition
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
            best = fx
            best_x = x[:]

    rebuild_elite()

    # ---------------- sampling F/CR from memories ----------------
    def sample_F(mu):
        # Cauchy around mu, resample if <=0
        for _ in range(12):
            F = mu + 0.10 * cauchy()
            if F > 0.0:
                if F > 1.0:
                    F = 1.0
                return F
        return max(1e-3, min(1.0, mu))

    def sample_CR(mu):
        CR = mu + 0.10 * randn()
        if CR < 0.0:
            return 0.0
        if CR > 1.0:
            return 1.0
        return CR

    # ---------------- pbest selection from elite set ----------------
    def pbest_index(time_frac):
        # adapt elite size slightly with time: smaller later -> stronger exploitation
        if not elite_idx:
            return randint(NP)
        # choose random from top fraction of elite
        top = max(2, int(len(elite_idx) * (0.80 - 0.55 * time_frac)))
        if top < 2:
            top = 2
        return elite_idx[randint(top)]

    # ---------------- local trust-region coordinate/2D search ----------------
    # radius fractions per-dimension (like a diagonal trust region)
    tr = [0.18] * dim
    tr_min = 1e-14
    tr_max = 0.65

    def local_trust_search(time_frac):
        nonlocal best, best_x, tr
        if best_x is None:
            return
        # number of mini-iterations: small, more late
        iters = 1 if time_frac < 0.45 else (2 if time_frac < 0.80 else 3)
        # choose coordinates subset
        if dim <= 28:
            coords = list(range(dim))
        else:
            k = max(10, dim // 5)
            coords = [randint(dim) for _ in range(k)]
        # shuffle
        for i in range(len(coords) - 1, 0, -1):
            j = randint(i + 1)
            coords[i], coords[j] = coords[j], coords[i]

        x = best_x[:]
        fx = best

        for _ in range(iters):
            if time.time() >= deadline:
                break

            improved = False

            # coordinate moves
            for d in coords:
                if time.time() >= deadline:
                    break
                if spans[d] == 0.0:
                    continue
                step = tr[d] * spans[d]
                if step <= 0.0:
                    continue

                xp = x[:]
                xm = x[:]
                xp[d] = reflect(xp[d] + step, d)
                xm[d] = reflect(xm[d] - step, d)
                fp = eval_f(xp)
                if time.time() >= deadline:
                    return
                fm = eval_f(xm)

                if fp < fx or fm < fx:
                    if fp <= fm:
                        x, fx = xp, fp
                    else:
                        x, fx = xm, fm
                    improved = True

            # occasional 2D rotation move near end (captures ridges)
            if dim >= 2 and u01() < (0.10 + 0.20 * time_frac) and time.time() < deadline:
                i = randint(dim)
                j = randint(dim - 1)
                if j >= i:
                    j += 1
                if spans[i] != 0.0 or spans[j] != 0.0:
                    ang = 2.0 * math.pi * u01()
                    c, s = math.cos(ang), math.sin(ang)
                    di = (tr[i] * spans[i]) * randn() if spans[i] != 0.0 else 0.0
                    dj = (tr[j] * spans[j]) * randn() if spans[j] != 0.0 else 0.0
                    y = x[:]
                    if spans[i] != 0.0:
                        y[i] = reflect(y[i] + c * di - s * dj, i)
                    if spans[j] != 0.0:
                        y[j] = reflect(y[j] + s * di + c * dj, j)
                    fy = eval_f(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True

            # adapt trust region
            if improved:
                for d in coords[:min(len(coords), 24)]:
                    tr[d] = min(tr_max, tr[d] * 1.08)
            else:
                for d in coords[:min(len(coords), 24)]:
                    tr[d] = max(tr_min, tr[d] * 0.80)

        if fx < best:
            best = fx
            best_x = x

    # ---------------- SPSA refine (late, very small budget) ----------------
    spsa_alpha = 0.18
    spsa_c = 0.07

    def spsa_refine(time_frac):
        nonlocal best, best_x, spsa_alpha, spsa_c
        if best_x is None:
            return
        # only late and only sometimes
        if time_frac < 0.55:
            return
        rounds = 1 if time_frac < 0.78 else 2
        x0 = best_x[:]
        f0 = best

        for _ in range(rounds):
            if time.time() >= deadline:
                return
            # sparse rademacher direction for large dim
            delta = [0] * dim
            if dim <= 60:
                for d in range(dim):
                    delta[d] = 1 if u01() < 0.5 else -1
            else:
                k = max(12, dim // 8)
                chosen = set()
                while len(chosen) < k:
                    chosen.add(randint(dim))
                for d in chosen:
                    delta[d] = 1 if u01() < 0.5 else -1

            cfrac = spsa_c * (0.75 + 0.35 * (1.0 - time_frac))
            xp = x0[:]
            xm = x0[:]
            for d in range(dim):
                if spans[d] == 0.0 or delta[d] == 0:
                    continue
                h = cfrac * spans[d]
                xp[d] = reflect(xp[d] + h * delta[d], d)
                xm[d] = reflect(xm[d] - h * delta[d], d)

            fp = eval_f(xp)
            if time.time() >= deadline:
                return
            fm = eval_f(xm)

            # gradient estimate along delta
            diff = fp - fm
            a = spsa_alpha * (0.45 + 0.65 * time_frac)
            x1 = x0[:]
            for d in range(dim):
                if spans[d] == 0.0 or delta[d] == 0:
                    continue
                h = max(1e-18, cfrac * spans[d])
                gi = diff / (2.0 * h * float(delta[d]))
                step = max(-0.30 * spans[d], min(0.30 * spans[d], a * gi))
                x1[d] = reflect(x1[d] - step, d)

            f1 = eval_f(x1)
            if f1 < f0:
                x0, f0 = x1, f1
                spsa_alpha = min(0.65, spsa_alpha * 1.05)
                spsa_c = min(0.20, spsa_c * 1.02)
            else:
                spsa_alpha = max(0.03, spsa_alpha * 0.80)
                spsa_c = max(0.02, spsa_c * 0.92)

        if f0 < best:
            best = f0
            best_x = x0

    # ---------------- main loop ----------------
    last_best = best
    stall = 0
    gen = 0

    while time.time() < deadline:
        now = time.time()
        time_frac = (now - t0) / max(1e-12, (deadline - t0))
        if time_frac >= 1.0:
            break

        # exploitation schedule (cheap and effective)
        if u01() < (0.12 + 0.33 * time_frac):
            local_trust_search(time_frac)
        if u01() < (0.03 + 0.20 * time_frac):
            spsa_refine(time_frac)

        # p schedule (LSHADE style-ish)
        pfrac = pmin + (pmax - pmin) * (0.75 - 0.55 * time_frac)
        if pfrac < pmin:
            pfrac = pmin
        if pfrac > pmax:
            pfrac = pmax

        # shuffle indices
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

            pb = pbest_index(time_frac)
            xpb = pop[pb]

            r1 = i
            while r1 == i:
                r1 = randint(NP)

            # pick r2 from union, avoid i and r1 if from pop
            r2 = -1
            for _ in range(30):
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

            xr1 = pop[r1]
            xr2 = union[r2]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover
            jrand = randint(dim)
            ui = xi[:]
            # slight attraction to global best later
            mix = (best_x is not None and u01() < (0.06 + 0.18 * time_frac))
            for d in range(dim):
                if d == jrand or u01() < CRi:
                    val = v[d]
                    if mix:
                        val = 0.90 * val + 0.10 * best_x[d]
                    ui[d] = reflect(val, d)

            fui = eval_f(ui)

            if fui < fi:
                # archive replaced
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[randint(arch_max)] = xi[:]

                pop[i] = ui
                fit[i] = fui

                if fui < best:
                    best = fui
                    best_x = ui[:]

                w = fi - fui
                if w < 1e-12:
                    w = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                S_w.append(w)

        # update memories (success-history)
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

            M_F[mem_pos] = mf
            M_CR[mem_pos] = mcr
            mem_pos = (mem_pos + 1) % H

        # archive size control
        arch_max = max(NP, NP0)
        if len(archive) > arch_max:
            for _ in range(len(archive) - arch_max):
                archive.pop(randint(len(archive)))

        # population reduction over time
        gen += 1
        target_NP = int(round(NPmin + (NP0 - NPmin) * (1.0 - time_frac)))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            # remove worst individuals
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

        # update elite after population changes / evolution
        elite_k = max(2, min(NP, int((0.22 - 0.10 * time_frac) * NP) + 2))
        rebuild_elite()

        # stagnation handling / partial restart injection
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        if (time_frac > 0.35 and stall >= 6) or (time_frac > 0.70 and stall >= 3):
            stall = 0
            # replace a couple of worst with heavy-tailed best-jitter or halton
            inj = 1 if NP <= 22 else 2
            for _ in range(inj):
                if time.time() >= deadline:
                    break
                # find worst
                worst = 0
                wf = fit[0]
                for j in range(1, NP):
                    if fit[j] > wf:
                        wf = fit[j]
                        worst = j

                if best_x is not None and u01() < 0.80:
                    y = best_x[:]
                    # heavy-tailed jitter: Student-t-ish via Cauchy scaled and clipped
                    base = 0.10 * (1.0 - time_frac) + 0.012
                    for d in range(dim):
                        if spans[d] == 0.0:
                            continue
                        jit = base * spans[d] * max(-3.0, min(3.0, cauchy()))
                        y[d] = reflect(y[d] + jit, d)
                else:
                    y = make_point(int(1e6 * u01()) + 97 * gen + worst)

                fy = eval_f(y)
                pop[worst] = y
                fit[worst] = fy
                if fy < best:
                    best = fy
                    best_x = y[:]

            rebuild_elite()

    return best
