import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-limited black-box minimizer (stdlib-only, self-contained).

    Strategy (robust under tight time):
      - Normalized search space [0,1]^dim with reflection at bounds
      - Fast initial design: LHS-like + opposition + a few randoms
      - Main optimizer: JADE-style DE (current-to-pbest/1 + archive + parameter adaptation)
      - Extra acceleration: small trust-region local search around current best
        using adaptive step sizes + occasional random subspace
      - Evaluation cache (quantized) to avoid wasting calls on duplicates
      - Stagnation control: partial restart / immigrants + wider F exploration

    Returns:
      best (float): best fitness found within max_time seconds
    """

    t_end = time.time() + float(max_time)

    # ---------------- bounds / normalization ----------------
    lo = [0.0] * dim
    hi = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
    span = [hi[i] - lo[i] for i in range(dim)]
    active = [span[i] > 0.0 for i in range(dim)]
    act_idx = [i for i in range(dim) if active[i]]
    adim = len(act_idx)

    if dim <= 0:
        return float("inf")
    if adim == 0:
        x0 = [lo[i] for i in range(dim)]
        try:
            return float(func(x0))
        except Exception:
            return float("inf")

    def to_real(z):
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] + z[i] * span[i] if active[i] else lo[i]
        return x

    # ---------------- RNG helpers ----------------
    def rand01():
        return random.random()

    def gauss():
        # Box-Muller
        u1 = rand01()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = rand01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = rand01()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---------------- bound handling: reflection in [0,1] ----------------
    def reflect01_inplace(z):
        # fold with period 2 and reflect into [0,1]
        for i in range(dim):
            if not active[i]:
                z[i] = 0.0
                continue
            v = z[i]
            if v < 0.0 or v > 1.0:
                v = v % 2.0
                if v > 1.0:
                    v = 2.0 - v
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            z[i] = v
        return z

    # ---------------- evaluation cache ----------------
    cache = {}
    n_eval = 0

    def q_for_key():
        # coarser early (more reuse), finer later
        if n_eval < 250:
            return 2e-6
        if n_eval < 2500:
            return 8e-7
        return 3e-7

    def z_key(z):
        q = q_for_key()
        inv = 1.0 / q
        # only active dims matter; include inactive as 0
        return tuple(int(z[i] * inv + 0.5) if active[i] else 0 for i in range(dim))

    def eval_z(z):
        nonlocal n_eval
        reflect01_inplace(z)
        k = z_key(z)
        if k in cache:
            return cache[k]
        fx = float(func(to_real(z)))
        cache[k] = fx
        n_eval += 1
        return fx

    # ---------------- initialization: LHS-like + opposition ----------------
    def lhs_like(n):
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            z = [0.0] * dim
            for j in range(dim):
                if not active[j]:
                    z[j] = 0.0
                else:
                    z[j] = (perms[j][i] + rand01()) / n
            pts.append(z)
        return pts

    def opposite(z):
        return [1.0 - z[i] if active[i] else 0.0 for i in range(dim)]

    # Population size: moderate, time-safe; scales with dim but capped
    NP = max(18, min(120, 14 + 5 * dim))
    if dim <= 4:
        NP = min(NP, 90)

    # Candidate pool larger than NP to start well
    cand_n = max(NP, min(3 * NP, 40 + 8 * dim))
    base = lhs_like(cand_n)
    cand = base + [opposite(z) for z in base]
    for _ in range(min(30, 2 * dim + 10)):
        cand.append([rand01() if active[i] else 0.0 for i in range(dim)])

    scored = []
    for z in cand:
        if time.time() >= t_end:
            break
        fz = eval_z(z[:])
        scored.append((fz, z[:]))

    if not scored:
        return float("inf")
    scored.sort(key=lambda t: t[0])
    scored = scored[:NP]

    pop = [z for (f, z) in scored]
    fit = [float(f) for (f, z) in scored]
    best_i = min(range(NP), key=lambda i: fit[i])
    best = fit[best_i]
    best_z = pop[best_i][:]

    # ---------------- JADE state ----------------
    mu_F = 0.55
    mu_CR = 0.50
    c_adapt = 0.10
    p_best_rate = 0.20
    arc = []
    arc_max = NP

    # stagnation / restart controls
    last_best = best
    stagn = 0
    gen = 0

    # ---------------- local search (trust-region-ish) ----------------
    # step sizes in normalized space; per-dimension (active dims) to adapt
    step = [0.18] * dim
    for i in range(dim):
        if not active[i]:
            step[i] = 0.0
    step_min = 1e-8
    step_max = 0.45

    def polish(budget=12):
        nonlocal best, best_z, stagn
        if budget <= 0:
            return
        z0 = best_z[:]
        f0 = best

        # try mix of coordinate and gaussian subspace
        evals = 0

        # a few gaussian trials (subspace)
        for _ in range(min(5, budget)):
            if time.time() >= t_end:
                return
            z = z0[:]
            # random subspace size
            k = 1 + int(rand01() * min(adim, 6))
            sub = random.sample(act_idx, k)
            for j in sub:
                z[j] += (0.65 * step[j]) * gauss()
            reflect01_inplace(z)
            fz = eval_z(z)
            evals += 1
            if fz < best:
                best = fz
                best_z = z[:]
                z0, f0 = z, fz
                stagn = 0
            if evals >= budget:
                return

        # coordinate pattern search (best-first among +/-)
        order = act_idx[:]
        random.shuffle(order)
        improved_any = False
        for j in order:
            if time.time() >= t_end or evals >= budget:
                return
            s = step[j]
            if s <= step_min:
                continue

            # try +/- and also a half-step (helps fine tuning)
            tried = 0
            best_local = (f0, None)
            for dj in (-s, +s, -0.5 * s, +0.5 * s):
                z = z0[:]
                z[j] += dj
                reflect01_inplace(z)
                if z_key(z) == z_key(z0):
                    continue
                fz = eval_z(z)
                evals += 1
                tried += 1
                if fz < best_local[0]:
                    best_local = (fz, z[:])
                if fz < best:
                    best = fz
                    best_z = z[:]
                    stagn = 0
                if evals >= budget or time.time() >= t_end:
                    break

            if tried and best_local[1] is not None and best_local[0] < f0:
                z0, f0 = best_local[1], best_local[0]
                improved_any = True
                step[j] = min(step_max, step[j] * 1.20)
            else:
                step[j] = max(step_min, step[j] * 0.72)

        if improved_any:
            # small global bump to encourage progress
            for j in act_idx:
                step[j] = min(step_max, step[j] * 1.03)

    # ---------------- helpers ----------------
    def idx_sorted_by_fit():
        idx = list(range(NP))
        idx.sort(key=lambda i: fit[i])
        return idx

    # ---------------- main loop ----------------
    while time.time() < t_end:
        gen += 1

        # update best / stagnation
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_z = pop[bi][:]
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # time-aware polishing (more aggressive late and when stuck)
        time_left = t_end - time.time()
        if time_left < 0.22 * max_time:
            if gen % 2 == 0:
                polish(budget=10)
        elif stagn > 18 and gen % 3 == 0:
            polish(budget=8)

        idx_sorted = idx_sorted_by_fit()
        pcount = max(2, int(p_best_rate * NP))
        if stagn > 30:
            pcount = max(2, int(0.35 * NP))
        top_idx = idx_sorted[:pcount]

        sF = []
        sCR = []

        # if stagnating, allow wider F distribution
        F_scale = 0.10 if stagn < 15 else (0.18 if stagn < 40 else 0.25)
        CR_scale = 0.10 if stagn < 20 else 0.18

        for i in range(NP):
            if time.time() >= t_end:
                return best

            CRi = mu_CR + CR_scale * gauss()
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            Fi = mu_F + F_scale * cauchy()
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = mu_F + F_scale * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.08
            elif Fi > 1.0:
                Fi = 1.0

            xi = pop[i]
            pbest = pop[random.choice(top_idx)]

            # r1 from population != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            x_r1 = pop[r1]

            # r2 from pop U archive, distinct-ish
            use_arc = (len(arc) > 0)
            x_r2 = None
            for _ in range(10):
                if use_arc and rand01() < (len(arc) / float(len(arc) + NP)):
                    x_r2 = arc[random.randrange(len(arc))]
                    if x_r2 is not None:
                        break
                else:
                    r2 = random.randrange(NP)
                    if r2 != i and r2 != r1:
                        x_r2 = pop[r2]
                        break
            if x_r2 is None:
                r2 = random.randrange(NP)
                while r2 == i or r2 == r1:
                    r2 = random.randrange(NP)
                x_r2 = pop[r2]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])

            # crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if rand01() < CRi or j == jrand:
                    u[j] = v[j]

            reflect01_inplace(u)
            fu = eval_z(u)

            if fu <= fit[i]:
                # archive parent
                arc.append(xi[:])
                if len(arc) > arc_max:
                    del arc[random.randrange(len(arc))]

                pop[i] = u
                fit[i] = fu
                sF.append(Fi)
                sCR.append(CRi)

                if fu < best:
                    best = fu
                    best_z = u[:]
                    stagn = 0

        # adapt parameters
        if sF:
            sumF = 0.0
            sumF2 = 0.0
            for v in sF:
                sumF += v
                sumF2 += v * v
            lehmerF = (sumF2 / sumF) if sumF > 0.0 else mu_F
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmerF

            meanCR = sum(sCR) / float(len(sCR))
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * meanCR
        else:
            # encourage exploration if nothing worked
            mu_F = min(0.95, mu_F * 1.08 + 0.02)
            mu_CR = min(1.0, mu_CR + 0.04)

        if mu_F < 0.05: mu_F = 0.05
        elif mu_F > 0.95: mu_F = 0.95
        if mu_CR < 0.0: mu_CR = 0.0
        elif mu_CR > 1.0: mu_CR = 1.0

        # immigrants / partial restart if very stuck
        if stagn > 35 and (gen % 4 == 0) and time.time() < t_end:
            # replace a few worst with (best + noise) or random
            k_rep = 2 if dim < 15 else 3
            for _ in range(k_rep):
                if time.time() >= t_end:
                    break
                wi = max(range(NP), key=lambda k: fit[k])
                z = best_z[:]
                if rand01() < 0.40:
                    z = [rand01() if active[i] else 0.0 for i in range(dim)]
                else:
                    # widen step when stuck
                    for j in act_idx:
                        z[j] += max(0.06, 1.4 * step[j]) * gauss()
                reflect01_inplace(z)
                fz = eval_z(z)
                if fz < fit[wi]:
                    pop[wi] = z[:]
                    fit[wi] = fz
                    if fz < best:
                        best = fz
                        best_z = z[:]
                        stagn = 0
            # shrink steps slightly to stabilize after shake-up
            for j in act_idx:
                step[j] = max(step_min, min(step_max, step[j] * 0.92))

    return best
