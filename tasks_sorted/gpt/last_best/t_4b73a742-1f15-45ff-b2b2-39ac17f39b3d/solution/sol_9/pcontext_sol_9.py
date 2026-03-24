import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained; no external libs).

    Main improvements over the provided versions:
      1) Better global exploration: scrambled Halton init + opposition + periodic restarts.
      2) Stronger DE core: L-SHADE-like "current-to-pbest/1" with:
         - success-history memories for F and CR (H memories)
         - external archive
         - linear population size reduction
      3) Much better exploitation: budgeted local refinement around the best using:
         - adaptive coordinate search with step-halving
         - occasional random 2D rotation steps
      4) Stagnation control: trigger (partial) restarts of worst individuals with a mix of
         - best-centered heavy-tailed perturbations
         - fresh quasi-random points

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0 or dim <= 0:
        return float("inf")
    deadline = t0 + float(max_time)

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

    # ---------------- fast RNG (LCG) ----------------
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
        return math.tan(math.pi * (u01() - 0.5))

    def reflect(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect repeatedly to handle large overshoots
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

    # ---------------- Halton init (scrambled) ----------------
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

    primes = first_primes(min(64, max(1, dim)))

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
                # light scrambling; helps avoid strong correlations
                h = (halton(idx + 1, base) + 0.47 * shift + 0.19 * u01()) % 1.0
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

    # ---------------- DE / L-SHADE-ish settings ----------------
    NP0 = max(36, min(180, 22 + 8 * int(math.sqrt(dim)) + dim // 2))
    NPmin = max(12, min(40, 8 + dim // 12))
    NP = NP0

    H = 16
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_pos = 0

    archive = []
    arch_max = NP0

    # selection pressure schedule
    pmin, pmax = 0.05, 0.35

    # ---------------- init population ----------------
    pop, fit = [], []
    best = float("inf")
    best_x = None

    # seed with Halton+opposition, plus a few pure randoms for robustness
    for i in range(NP0):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")

        if i < int(0.85 * NP0):
            x = make_point(i)
        else:
            x = [lows[d] + u01() * spans[d] if spans[d] != 0.0 else lows[d] for d in range(dim)]

        fx = eval_f(x)
        xo = opposite(x)
        fxo = eval_f(xo)
        if fxo < fx:
            x, fx = xo, fxo

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # ---------------- elite set for pbest ----------------
    elite_idx = []
    elite_k = max(3, min(NP, int(0.20 * NP) + 2))

    def rebuild_elite():
        nonlocal elite_idx, elite_k
        elite = []
        for i in range(NP):
            if len(elite) < elite_k:
                elite.append(i)
            else:
                wi = elite[0]
                wf = fit[wi]
                for j in elite[1:]:
                    fj = fit[j]
                    if fj > wf:
                        wf, wi = fj, j
                if fit[i] < wf:
                    elite.remove(wi)
                    elite.append(i)
        elite.sort(key=lambda k: fit[k])
        elite_idx = elite

    rebuild_elite()

    def pbest_index(time_frac):
        if not elite_idx:
            return randint(NP)
        # later -> smaller band -> more exploitation
        band = 0.95 - 0.70 * time_frac
        top = max(2, int(len(elite_idx) * band))
        return elite_idx[randint(top)]

    # ---------------- parameter sampling ----------------
    def sample_F(mu):
        # Cauchy around mu, resample if nonpositive
        for _ in range(16):
            F = mu + 0.12 * cauchy()
            if F > 0.0:
                return 1.0 if F > 1.0 else F
        return max(1e-3, min(1.0, mu))

    def sample_CR(mu):
        CR = mu + 0.10 * randn()
        if CR < 0.0:
            return 0.0
        if CR > 1.0:
            return 1.0
        return CR

    # ---------------- local refinement around best ----------------
    tr = [0.25] * dim
    tr_min = 1e-14
    tr_max = 0.75

    def local_search(time_frac):
        nonlocal best, best_x, tr
        if best_x is None:
            return

        # budgeted rounds
        rounds = 1 if time_frac < 0.35 else (2 if time_frac < 0.75 else 3)

        # choose coordinate subset if high-d
        if dim <= 30:
            coords = list(range(dim))
        else:
            k = max(14, dim // 5)
            coords = [randint(dim) for _ in range(k)]

        # shuffle
        for i in range(len(coords) - 1, 0, -1):
            j = randint(i + 1)
            coords[i], coords[j] = coords[j], coords[i]

        x = best_x[:]
        fx = best

        for _ in range(rounds):
            if time.time() >= deadline:
                break

            improved = False

            # coordinate moves with step halving
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
                xp[d] = reflect(x[d] + step, d)
                xm[d] = reflect(x[d] - step, d)

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
                else:
                    step2 = 0.5 * step
                    if step2 > 0.0:
                        xp[d] = reflect(x[d] + step2, d)
                        xm[d] = reflect(x[d] - step2, d)
                        fp2 = eval_f(xp)
                        if time.time() >= deadline:
                            return
                        fm2 = eval_f(xm)
                        if fp2 < fx or fm2 < fx:
                            if fp2 <= fm2:
                                x, fx = xp, fp2
                            else:
                                x, fx = xm, fm2
                            improved = True

            # occasional 2D rotation step (helps on rotated valleys)
            if dim >= 2 and u01() < (0.10 + 0.30 * time_frac) and time.time() < deadline:
                i = randint(dim)
                j = randint(dim - 1)
                if j >= i:
                    j += 1
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

            # adapt trust radii
            if improved:
                for d in coords[:min(32, len(coords))]:
                    tr[d] = min(tr_max, tr[d] * 1.12)
            else:
                for d in coords[:min(32, len(coords))]:
                    tr[d] = max(tr_min, tr[d] * 0.75)

        if fx < best:
            best, best_x = fx, x[:]

    # ---------------- best-centered heavy-tail jump ----------------
    def best_jump(time_frac):
        nonlocal best, best_x
        if best_x is None:
            return
        y = best_x[:]
        base = 0.20 * (1.0 - time_frac) + 0.008  # never fully off
        for d in range(dim):
            if spans[d] == 0.0:
                continue
            jit = base * spans[d] * max(-6.0, min(6.0, cauchy()))
            y[d] = reflect(y[d] + jit, d)
        fy = eval_f(y)
        if fy < best:
            best, best_x = fy, y

    # ---------------- restart/injection ----------------
    halton_counter = NP0 + 10

    def inject_worst(nrep, time_frac):
        nonlocal halton_counter, best, best_x
        for _ in range(nrep):
            if time.time() >= deadline or NP <= 0:
                return
            # find worst
            worst = 0
            wf = fit[0]
            for j in range(1, NP):
                if fit[j] > wf:
                    wf, worst = fit[j], j

            if best_x is not None and u01() < 0.80:
                y = best_x[:]
                base = 0.16 * (1.0 - time_frac) + 0.012
                for d in range(dim):
                    if spans[d] == 0.0:
                        continue
                    y[d] = reflect(y[d] + base * spans[d] * max(-5.0, min(5.0, cauchy())), d)
            else:
                y = make_point(halton_counter)
                halton_counter += 1

            fy = eval_f(y)
            pop[worst] = y
            fit[worst] = fy
            if fy < best:
                best, best_x = fy, y[:]

    # ---------------- main loop ----------------
    last_best = best
    stall = 0
    gen = 0

    while time.time() < deadline:
        now = time.time()
        time_frac = (now - t0) / max(1e-12, (deadline - t0))
        if time_frac >= 1.0:
            break

        # exploitation / exploration schedule
        if u01() < (0.16 + 0.45 * time_frac):
            local_search(time_frac)
        if u01() < (0.06 + 0.10 * (1.0 - time_frac)):
            best_jump(time_frac)

        # adapt elite size and rebuild occasionally
        elite_k = max(3, min(NP, int((0.24 - 0.10 * time_frac) * NP) + 2))
        if gen % 3 == 0:
            rebuild_elite()

        # p schedule
        pfrac = pmin + (pmax - pmin) * (0.90 - 0.70 * time_frac)
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
            xi, fi = pop[i], fit[i]

            rmem = randint(H)
            Fi = sample_F(M_F[rmem])
            CRi = sample_CR(M_CR[rmem])

            pb = pbest_index(time_frac)
            xpb = pop[pb]

            r1 = i
            while r1 == i:
                r1 = randint(NP)

            # choose r2 from union, avoid i and r1 when from pop
            r2 = -1
            for _ in range(32):
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

            # crossover with slight late pull to global best (stabilizes convergence)
            ui = xi[:]
            jrand = randint(dim)
            pull_best = (best_x is not None and u01() < (0.06 + 0.22 * time_frac))
            for d in range(dim):
                if d == jrand or u01() < CRi:
                    val = v[d]
                    if pull_best:
                        val = 0.90 * val + 0.10 * best_x[d]
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
                    best, best_x = fui, ui[:]

                w = fi - fui
                if w < 1e-12:
                    w = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                S_w.append(w)

        # update memories (weighted)
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

        # linear population reduction
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
                        wf, worst = fit[j], j
                pop.pop(worst)
                fit.pop(worst)
                NP -= 1
                if NP <= NPmin:
                    break
            if NP <= 0:
                return best
            rebuild_elite()

        # stagnation logic -> partial restart
        if best < last_best - 1e-12:
            last_best = best
            stall = 0
        else:
            stall += 1

        if (time_frac > 0.30 and stall >= 7) or (time_frac > 0.70 and stall >= 3):
            stall = 0
            nrep = 1 if NP <= 26 else 2
            if time_frac > 0.85 and NP > 18:
                nrep = 3
            inject_worst(nrep, time_frac)
            rebuild_elite()

    return best
