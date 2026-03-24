import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (self-contained, no external libs).

    Main improvements vs your last JADE-like DE:
      1) Surrogate-assisted search (very lightweight RBF/IDW model in normalized space)
         - Uses evaluated points to propose new candidates cheaply
         - Balances exploitation (low predicted) + exploration (far from known points)
      2) Dual-engine: DE (JADE/current-to-pbest/1 + archive) + surrogate proposals
         - Time-sliced: DE keeps global robustness; surrogate accelerates convergence
      3) Better polishing near the end: adaptive coordinate + small Gaussian steps
      4) Caching + duplicate avoidance (quantized keys) to not waste evaluations
      5) Restarts/injections when stagnating

    Returns:
      best fitness (float) found within max_time.
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

    def to_real(z):
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] + z[i] * span[i] if active[i] else lo[i]
        return x

    def clamp01(z):
        for i in range(dim):
            if z[i] < 0.0:
                z[i] = 0.0
            elif z[i] > 1.0:
                z[i] = 1.0
        return z

    # ---------------- RNG helpers ----------------
    def rand01():
        return random.random()

    def gauss():
        u1 = rand01()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = rand01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = rand01()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---------------- evaluation with cache ----------------
    # Quantized cache key in normalized space to avoid re-evaluating near-identical points.
    cache = {}
    def z_key(z, q=1e-6):
        # deterministic quantization; tuple of ints
        inv = 1.0 / q
        return tuple(int(v * inv + 0.5) for v in z)

    def eval_z(z):
        clamp01(z)
        k = z_key(z)
        if k in cache:
            return cache[k]
        fx = float(func(to_real(z)))
        cache[k] = fx
        return fx

    # ---------------- init: LHS-like + opposition ----------------
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
        return [1.0 - v for v in z]

    # ---------------- population sizing ----------------
    # Slightly larger than before (helps DE + gives surrogate more data), but still bounded.
    NP = max(18, min(110, 16 + 5 * dim))
    if dim <= 3:
        NP = min(80, max(26, NP))

    # Candidate pool
    cand = []
    base = lhs_like(NP)
    for z in base:
        cand.append(z)
        cand.append(opposite(z))
    for _ in range(min(20, 2 * NP)):
        cand.append([rand01() for _ in range(dim)])

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

    pop = [z for (fz, z) in scored]
    fit = [float(fz) for (fz, z) in scored]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best = fit[best_i]
    best_z = pop[best_i][:]

    # ---------------- DE (JADE-style) state ----------------
    mu_F = 0.5
    mu_CR = 0.5
    c_adapt = 0.1
    p_best_rate = 0.15
    arc = []
    arc_max = NP

    # stagnation tracking
    last_best = best
    stagn = 0

    # ---------------- local polish ----------------
    coord_step = 0.18
    coord_min = 1e-7

    def polish_once(budget_evals=10):
        nonlocal best, best_z, coord_step
        # Hybrid of: gaussian micro-steps + coordinate steps (normalized)
        z0 = best_z[:]
        f0 = best
        evals = 0

        # a few gaussian trials
        for _ in range(4):
            if time.time() >= t_end or evals >= budget_evals:
                return
            z = z0[:]
            # slightly anisotropic: only active dims
            for j in act_idx:
                z[j] += 0.30 * coord_step * gauss()
            clamp01(z)
            fz = eval_z(z)
            evals += 1
            if fz < best:
                best = fz
                best_z = z[:]
                z0, f0 = z, fz

        # coordinate tries (best-first among +/-)
        order = act_idx[:]
        random.shuffle(order)
        improved = False
        for j in order:
            if time.time() >= t_end or evals >= budget_evals:
                break
            s = coord_step
            if s <= coord_min:
                continue

            z_minus = z0[:]
            z_minus[j] = max(0.0, z_minus[j] - s)
            z_plus = z0[:]
            z_plus[j] = min(1.0, z_plus[j] + s)

            # evaluate the better-looking first by simple heuristic:
            # closer to center tends to be safer if unknown; but we don't know.
            # Just evaluate both if distinct.
            tried = 0
            for z in (z_minus, z_plus):
                if evals >= budget_evals or time.time() >= t_end:
                    break
                if z[j] == z0[j]:
                    continue
                fz = eval_z(z)
                evals += 1
                tried += 1
                if fz < best:
                    best = fz
                    best_z = z[:]
                    z0, f0 = z, fz
                    improved = True
            if tried == 0:
                continue

        # adapt step
        if improved:
            coord_step = min(0.40, coord_step * 1.12)
        else:
            coord_step = max(coord_min, coord_step * 0.72)

    # ---------------- surrogate model (IDW/RBF-like) ----------------
    # We build a simple inverse-distance weighted predictor using a subset of best+diverse points.
    # This is very cheap and works surprisingly well for guiding proposals.
    def dist2(a, b):
        s = 0.0
        for j in act_idx:
            d = a[j] - b[j]
            s += d * d
        return s

    # Keep a history of evaluated points (from cache) is hard without storing;
    # We'll maintain an explicit list of elites + some random samples from pop+archive.
    def build_trainset(maxn=80):
        # Take best K from population + some random from pop + some from archive
        idx = sorted(range(NP), key=lambda i: fit[i])
        K = min(max(10, maxn // 2), NP)
        train = [(pop[i][:], fit[i]) for i in idx[:K]]

        # add random from population
        extra = maxn - len(train)
        if extra > 0:
            for _ in range(min(extra, NP)):
                i = random.randrange(NP)
                train.append((pop[i][:], fit[i]))

        # add some archive points (need fitness; we don't have it) -> skip archive for training
        # to keep model consistent (fitness unknown). Use only evaluated pop points.

        # de-duplicate by key
        seen = set()
        out = []
        for z, fz in train:
            k = z_key(z, q=2e-6)
            if k in seen:
                continue
            seen.add(k)
            out.append((z, fz))
            if len(out) >= maxn:
                break
        return out

    def surrogate_predict(z, train):
        # IDW with exponent p; add tiny epsilon
        # returns (pred, mindist2)
        eps = 1e-12
        p = 2.0
        num = 0.0
        den = 0.0
        md2 = float("inf")
        for zi, fi in train:
            d2 = dist2(z, zi)
            if d2 < md2:
                md2 = d2
            w = 1.0 / ((d2 + eps) ** (p * 0.5))
            num += w * fi
            den += w
        pred = num / den if den > 0.0 else float("inf")
        return pred, md2

    def propose_by_surrogate(train, n_tries=60):
        # Sample around best + some global samples; score by acquisition:
        #   score = pred - alpha * sqrt(mindist2)  (encourage novelty)
        # Lower is better.
        if not train:
            return [rand01() for _ in range(dim)]

        # estimate scale of fitness for alpha
        fs = [f for (_, f) in train]
        fmin = min(fs)
        fmax = max(fs)
        frange = max(1e-12, fmax - fmin)

        # alpha ramps with stagnation (more exploration if stuck)
        alpha = (0.05 + 0.20 * min(1.0, stagn / 25.0)) * frange

        best_cand = None
        best_score = float("inf")

        for _ in range(n_tries):
            # mixture sampling
            r = rand01()
            if r < 0.55:
                # around best (gaussian)
                z = best_z[:]
                sig = max(0.02, 0.25 * coord_step)
                for j in act_idx:
                    z[j] += sig * gauss()
            elif r < 0.85:
                # around a random good point
                z0, _ = train[random.randrange(len(train))]
                z = z0[:]
                sig = 0.10 + 0.20 * rand01()
                for j in act_idx:
                    z[j] += sig * (rand01() * 2.0 - 1.0)
            else:
                # global
                z = [rand01() for _ in range(dim)]

            clamp01(z)

            pred, md2 = surrogate_predict(z, train)
            score = pred - alpha * math.sqrt(md2)
            if score < best_score:
                best_score = score
                best_cand = z[:]

        if best_cand is None:
            best_cand = [rand01() for _ in range(dim)]
        return best_cand

    # ---------------- main loop ----------------
    gen = 0
    while time.time() < t_end:
        gen += 1

        # update best
        bi = min(range(NP), key=lambda i: fit[i])
        if fit[bi] < best:
            best = fit[bi]
            best_z = pop[bi][:]
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # time-left actions
        time_left = t_end - time.time()
        if time_left < 0.18 * max_time and (gen % 2 == 0):
            polish_once(budget_evals=8)

        # build trainset and do a few surrogate-guided evaluations each generation
        # (but not too many; DE still does most work)
        if gen % 2 == 0 and time.time() < t_end:
            train = build_trainset(maxn=min(90, max(40, 10 + 3 * dim)))
            # number of surrogate proposals increases with stagnation, but bounded
            n_sur = 1 + (1 if stagn > 10 else 0) + (1 if stagn > 25 else 0)
            n_sur = min(4, n_sur)
            for _ in range(n_sur):
                if time.time() >= t_end:
                    return best
                zc = propose_by_surrogate(train, n_tries=40 + 5 * min(dim, 20))
                fc = eval_z(zc)
                if fc < best:
                    best = fc
                    best_z = zc[:]
                    stagn = 0
                # insert into population by replacing current worst if improves it
                wi = max(range(NP), key=lambda i: fit[i])
                if fc < fit[wi]:
                    pop[wi] = zc[:]
                    fit[wi] = fc

        # DE generation
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(p_best_rate * NP))
        top_idx = idx_sorted[:pcount]

        sF = []
        sCR = []

        # if badly stagnating, increase p-best pressure a bit
        if stagn > 20:
            pcount = max(2, int(0.30 * NP))
            top_idx = idx_sorted[:pcount]

        for i in range(NP):
            if time.time() >= t_end:
                return best

            # sample CR, F
            CRi = mu_CR + 0.1 * gauss()
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            Fi = mu_F + 0.1 * cauchy()
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = mu_F + 0.1 * cauchy()
                tries += 1
            if Fi <= 0.0:
                Fi = 0.12
            elif Fi > 1.0:
                Fi = 1.0

            xi = pop[i]

            pbest = pop[random.choice(top_idx)]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            x_r1 = pop[r1]

            # choose r2 from pop U archive
            use_arc = (len(arc) > 0)
            x_r2 = None
            for _ in range(12):
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

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])

            # crossover
            u = xi[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for j in range(dim):
                if (rand01() < CRi) or (j == jrand):
                    u[j] = v[j]
            clamp01(u)

            fu = eval_z(u)

            # selection
            if fu <= fit[i]:
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

        # adapt mu_F, mu_CR
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
            # no success => encourage exploration
            mu_F = min(0.95, mu_F * 1.07)
            mu_CR = min(1.0, mu_CR + 0.03)

        # clamp
        if mu_F < 0.05: mu_F = 0.05
        elif mu_F > 0.95: mu_F = 0.95
        if mu_CR < 0.0: mu_CR = 0.0
        elif mu_CR > 1.0: mu_CR = 1.0

        # diversity injection if stagnating
        if stagn > 18 and (gen % 4 == 0) and time.time() < t_end:
            # replace a couple of worst with random / around-best samples
            for _ in range(2):
                if time.time() >= t_end:
                    break
                wi = max(range(NP), key=lambda k: fit[k])
                if rand01() < 0.5:
                    z = [rand01() for _ in range(dim)]
                else:
                    z = best_z[:]
                    for j in act_idx:
                        z[j] += 0.25 * gauss()
                clamp01(z)
                fz = eval_z(z)
                if fz < fit[wi]:
                    pop[wi] = z
                    fit[wi] = fz
                    if fz < best:
                        best = fz
                        best_z = z[:]
                        stagn = 0

    return best
