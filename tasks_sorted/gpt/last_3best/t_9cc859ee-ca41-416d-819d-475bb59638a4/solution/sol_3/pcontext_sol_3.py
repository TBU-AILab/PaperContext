import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained, no external libs).

    Key upgrades vs the provided DE+jDE+ES:
      - Proper p-best selection using ranks (sorted indices once per generation)
      - JADE-style parameter adaptation (mu_F, mu_CR) with Lehmer mean for F
      - Current-to-pbest/1 mutation + external archive (prevents premature convergence)
      - Explicit normalized-space search (stable across wildly different bound scales)
      - Cheap, time-aware local "polish" (coordinate + gaussian) near the end
      - Robust initialization: LHS-like + opposition + a few randoms

    Returns:
      best fitness (float) found within max_time.
    """

    t_end = time.time() + float(max_time)

    # -------- bounds / normalization --------
    lo = [0.0] * dim
    hi = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
    span = [hi[i] - lo[i] for i in range(dim)]

    # handle degenerate dims
    active = [span[i] > 0.0 for i in range(dim)]

    def to_real(z):
        x = [0.0] * dim
        for i in range(dim):
            if active[i]:
                x[i] = lo[i] + z[i] * span[i]
            else:
                x[i] = lo[i]
        return x

    def clamp01_inplace(z):
        for i in range(dim):
            if z[i] < 0.0:
                z[i] = 0.0
            elif z[i] > 1.0:
                z[i] = 1.0
        return z

    def eval_z(z):
        # z is normalized in [0,1]^dim
        return float(func(to_real(z)))

    # -------- random helpers --------
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

    # -------- initialization (LHS-like + opposition) --------
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
                    u = (perms[j][i] + rand01()) / n
                    z[j] = u
            pts.append(z)
        return pts

    def opposite(z):
        return [1.0 - v for v in z]

    # Population sizing: keep it moderate but sufficient
    NP = max(14, min(90, 12 + 4 * dim))
    if dim <= 2:
        NP = min(60, max(20, NP))

    # Build candidate pool
    cand = []
    base = lhs_like(NP)
    for z in base:
        cand.append(z)
        cand.append(opposite(z))
    # few random injections
    for _ in range(min(12, NP)):
        cand.append([rand01() for _ in range(dim)])

    pop = []
    fit = []

    # Evaluate candidates; keep best NP
    scored = []
    for z in cand:
        if time.time() >= t_end:
            break
        clamp01_inplace(z)
        fz = eval_z(z)
        scored.append((fz, z[:]))
    if not scored:
        return float("inf")
    scored.sort(key=lambda t: t[0])
    scored = scored[:NP]
    for fz, z in scored:
        pop.append(z)
        fit.append(float(fz))

    # Best
    best_idx = min(range(NP), key=lambda i: fit[i])
    best = fit[best_idx]
    best_z = pop[best_idx][:]

    # -------- JADE parameters + archive --------
    mu_F = 0.5
    mu_CR = 0.5
    c_adapt = 0.1         # learning rate for mu_F/mu_CR
    p_best_rate = 0.2     # p in JADE (top p% are eligible as pbest)
    arc = []              # archive of replaced parents in normalized space
    arc_max = NP          # typical choice

    # -------- final-stage local polish parameters --------
    # coordinate step in normalized space
    coord_step = 0.15
    coord_min = 1e-6

    # Small polish budget: becomes more frequent later
    def polish_once():
        nonlocal best, best_z, coord_step
        if time.time() >= t_end:
            return
        z0 = best_z[:]
        f0 = best

        # try a couple of gaussian perturbations
        for _ in range(2):
            if time.time() >= t_end:
                return
            z = z0[:]
            for j in range(dim):
                if active[j]:
                    z[j] += 0.35 * coord_step * gauss()
            clamp01_inplace(z)
            fz = eval_z(z)
            if fz < best:
                best = fz
                best_z = z
                z0 = z
                f0 = fz

        # quick coordinate +/- step on shuffled dims
        order = list(range(dim))
        random.shuffle(order)
        improved = False
        for j in order:
            if time.time() >= t_end:
                return
            if not active[j]:
                continue
            s = coord_step
            if s <= coord_min:
                continue
            for d in (-s, +s):
                z = z0[:]
                z[j] += d
                if z[j] < 0.0: z[j] = 0.0
                elif z[j] > 1.0: z[j] = 1.0
                if z[j] == z0[j]:
                    continue
                fz = eval_z(z)
                if fz < best:
                    best = fz
                    best_z = z
                    z0 = z
                    f0 = fz
                    improved = True

        # adapt polish step
        if improved:
            coord_step = min(0.35, coord_step * 1.10)
        else:
            coord_step = max(coord_min, coord_step * 0.70)

    # -------- main loop --------
    gen = 0
    while time.time() < t_end:
        gen += 1

        # sort indices once per generation (enables true p-best selection)
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        best_idx = idx_sorted[0]
        if fit[best_idx] < best:
            best = fit[best_idx]
            best_z = pop[best_idx][:]

        pcount = max(2, int(p_best_rate * NP))
        top_idx = idx_sorted[:pcount]

        # successful parameters for JADE adaptation
        sF = []
        sCR = []

        # late-stage polishing (more often as time runs out)
        time_left = t_end - time.time()
        if time_left < 0.25 * max_time and (gen % 3 == 0):
            polish_once()

        for i in range(NP):
            if time.time() >= t_end:
                return best

            # sample CR ~ N(mu_CR, 0.1), clipped
            CRi = mu_CR + 0.1 * gauss()
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            # sample F ~ Cauchy(mu_F, 0.1), resample until in (0,1]
            Fi = mu_F + 0.1 * cauchy()
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 8:
                Fi = mu_F + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            elif Fi > 1.0:
                Fi = 1.0

            # choose pbest from top p%
            pbest = pop[random.choice(top_idx)]

            # choose r1 from population != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            x_r1 = pop[r1]

            # choose r2 from population U archive != i, r1
            # build a virtual pool: indices [0..NP-1] + archive entries
            use_arc = (len(arc) > 0)
            # attempt to pick r2 distinct; if too hard, fallback to random in pop
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
            else:
                # fallback
                r2 = random.randrange(NP)
                while r2 == i or r2 == r1:
                    r2 = random.randrange(NP)
                x_r2 = pop[r2]

            xi = pop[i]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])

            # crossover (binomial)
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if (rand01() < CRi) or (j == jrand):
                    u[j] = v[j]

            clamp01_inplace(u)

            fu = eval_z(u)

            # selection
            if fu <= fit[i]:
                # add parent to archive
                arc.append(xi[:])
                if len(arc) > arc_max:
                    # random removal
                    del arc[random.randrange(len(arc))]

                pop[i] = u
                fit[i] = fu

                sF.append(Fi)
                sCR.append(CRi)

                if fu < best:
                    best = fu
                    best_z = u[:]

        # JADE adaptation of mu_F and mu_CR
        if sF:
            # Lehmer mean for F: sum(F^2)/sum(F)
            sumF = 0.0
            sumF2 = 0.0
            for v in sF:
                sumF += v
                sumF2 += v * v
            lehmerF = (sumF2 / sumF) if sumF > 0.0 else mu_F
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmerF

            # arithmetic mean for CR
            meanCR = sum(sCR) / float(len(sCR))
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * meanCR

            # keep sane
            if mu_F < 0.05: mu_F = 0.05
            elif mu_F > 0.95: mu_F = 0.95
            if mu_CR < 0.0: mu_CR = 0.0
            elif mu_CR > 1.0: mu_CR = 1.0
        else:
            # If no success, mildly increase exploration
            mu_F = min(0.95, mu_F * 1.05)
            mu_CR = min(1.0, mu_CR + 0.02)

        # occasional random immigrant if stagnating / to keep diversity
        if gen % 12 == 0 and time.time() < t_end:
            worst = max(range(NP), key=lambda k: fit[k])
            z = [rand01() for _ in range(dim)]
            fz = eval_z(z)
            if fz < fit[worst]:
                pop[worst] = z
                fit[worst] = fz
                if fz < best:
                    best = fz
                    best_z = z[:]

    return best
