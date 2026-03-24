import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (self-contained, no external libs).

    Improvements over the provided code:
      - Budget-aware evaluation counting with a safe time guard.
      - Better global exploration: LHS-like initialization + occasional global samples.
      - Stronger DE core: jDE-style self-adaptation per individual (Fi, CRi),
        plus current-to-pbest/1, archive, and periodic partial population restart.
      - Better local exploitation: lightweight trust-region random-direction + coordinate
        search around the incumbent, with adaptive radius and very low overhead.
      - Robust bound handling: reflection (mirror) with final clamp.

    Returns:
      best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    max_span = max(spans) if spans else 0.0
    if max_span <= 0.0:
        x = [lows[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # ----------------- utilities -----------------
    def time_left():
        return deadline - time.time()

    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect_into_bounds(v, lo, hi):
        # mirror reflection; handles overshoots reasonably for DE
        if lo == hi:
            return lo
        w = hi - lo
        # bring into [lo, hi] by reflecting across boundaries
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
        # tiny numerical clamp
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def ensure_bounds(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = clamp(x[i], lows[i], highs[i])
        return y

    evals = 0
    best = float("inf")
    best_x = None

    def eval_f(x):
        nonlocal evals, best, best_x
        evals += 1
        fx = float(func(x))
        if fx < best:
            best = fx
            best_x = x[:]  # copy
        return fx

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # LHS-like initialization: for each dimension, permute bins
    def init_population_lhs(NP):
        bins = []
        for j in range(dim):
            perm = list(range(NP))
            random.shuffle(perm)
            bins.append(perm)
        pop = []
        for i in range(NP):
            x = [0.0] * dim
            for j in range(dim):
                # sample within bin
                u = (bins[j][i] + random.random()) / NP
                x[j] = lows[j] + u * spans[j]
            pop.append(x)
        return pop

    # ----------------- algorithm parameters -----------------
    # Population sizing: moderate but not huge (time bounded)
    NP = int(max(24, min(12 * dim, 160)))

    # DE settings
    p_best = 0.2  # top fraction for p-best selection

    # jDE adaptation params
    tau1 = 0.1  # prob to resample F
    tau2 = 0.1  # prob to resample CR
    Fl = 0.1
    Fu = 0.9

    # Archive
    A = []
    Amax = NP

    # Restart/Injection control
    stagn_limit = 12
    inject_frac = 0.15

    # Local search schedule
    refine_every = 6  # generations
    refine_evals_cap = 40  # keep cheap
    min_radius = 1e-12

    # ----------------- initialize -----------------
    pop = init_population_lhs(NP)
    fit = [None] * NP

    # per-individual self-adaptation values
    F = [0.5 + 0.3 * (random.random() - 0.5) for _ in range(NP)]
    CR = [0.7 + 0.3 * (random.random() - 0.5) for _ in range(NP)]
    for i in range(NP):
        if F[i] < Fl: F[i] = Fl
        if F[i] > Fu: F[i] = Fu
        if CR[i] < 0.0: CR[i] = 0.0
        if CR[i] > 1.0: CR[i] = 1.0

    for i in range(NP):
        if time.time() >= deadline:
            return best
        fit[i] = eval_f(pop[i])

    last_best = best
    stagn = 0
    gen = 0

    # ----------------- local refinement -----------------
    def local_refine(x0, f0, max_evals):
        """Very cheap trust-region-ish refinement around incumbent."""
        if x0 is None:
            return x0, f0

        x = x0[:]
        fx = f0
        # initial radius relative to scale; shrink on failure
        base = sum(spans) / float(dim)
        radius = 0.15 * base
        if radius <= 0:
            radius = 0.1 * max_span

        used = 0

        # mix of random-direction steps and coordinate pokes
        while used < max_evals and time.time() < deadline and radius > min_radius * max_span:
            improved = False

            # 1) random-direction (one eval)
            # build random unit-ish direction
            d = [random.gauss(0.0, 1.0) for _ in range(dim)]
            # normalize cheaply
            norm = math.sqrt(sum(v * v for v in d))
            if norm == 0.0:
                norm = 1.0
            inv = 1.0 / norm

            # try both + and - (up to 2 evals)
            for sgn in (1.0, -1.0):
                if used >= max_evals or time.time() >= deadline:
                    break
                xn = [0.0] * dim
                for j in range(dim):
                    step = sgn * radius * d[j] * inv
                    xn[j] = reflect_into_bounds(x[j] + step, lows[j], highs[j])
                fn = eval_f(xn)
                used += 1
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    break

            # 2) a few coordinate tweaks (up to dim but capped)
            if used < max_evals and time.time() < deadline:
                # cap to avoid spending too much in high-d
                ccap = min(dim, 8)
                coords = list(range(dim))
                random.shuffle(coords)
                coords = coords[:ccap]
                for j in coords:
                    if used >= max_evals or time.time() >= deadline:
                        break
                    s = radius
                    # + then -
                    xn = x[:]
                    xn[j] = reflect_into_bounds(xn[j] + s, lows[j], highs[j])
                    fn = eval_f(xn); used += 1
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        continue
                    if used >= max_evals or time.time() >= deadline:
                        break
                    xn = x[:]
                    xn[j] = reflect_into_bounds(xn[j] - s, lows[j], highs[j])
                    fn = eval_f(xn); used += 1
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True

            if improved:
                radius *= 1.25
                # prevent too-large radius
                if radius > 0.5 * base:
                    radius = 0.5 * base
            else:
                radius *= 0.5

        return x, fx

    # ----------------- main loop -----------------
    while time.time() < deadline:
        gen += 1

        # p-best set
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(p_best * NP))
        pset = idx_sorted[:pcount]

        for i in range(NP):
            if time.time() >= deadline:
                return best

            # jDE: adapt Fi, CRi
            Fi = F[i]
            CRi = CR[i]
            if random.random() < tau1:
                Fi = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CRi = random.random()

            # choose pbest
            pbest = random.choice(pset)

            # choose r1 != i
            r1 = random.randrange(NP - 1)
            if r1 >= i:
                r1 += 1

            # choose r2 from pop+archive distinct from i, r1 if possible
            poolN = NP + len(A)
            # if archive empty, r2 from pop distinct
            for _ in range(50):
                r2 = random.randrange(poolN)
                if r2 == i or r2 == r1:
                    continue
                break
            # map r2 to vector
            if r2 < NP:
                xr2 = pop[r2]
            else:
                xr2 = A[r2 - NP]

            xi = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = reflect_into_bounds(vj, lows[j], highs[j])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    u[j] = v[j]
            # final clamp
            u = ensure_bounds(u)

            fu = eval_f(u)
            if fu <= fit[i]:
                # success: archive the replaced
                A.append(xi)
                if len(A) > Amax:
                    A.pop(random.randrange(len(A)))
                pop[i] = u
                fit[i] = fu
                # keep successful adapted params
                F[i] = Fi
                CR[i] = CRi

        # occasional global sampling (helps in deceptive landscapes)
        if time_left() > 0 and gen % 5 == 0:
            # 1-2 samples
            tries = 1 if dim > 20 else 2
            for _ in range(tries):
                if time.time() >= deadline:
                    break
                xg = rand_uniform_vec()
                eval_f(xg)

        # stagnation tracking
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # local refine
        if best_x is not None and gen % refine_every == 0 and time.time() < deadline:
            bx, bf = local_refine(best_x, best, refine_evals_cap)
            # eval_f already updates best, but keep consistent
            best_x, best = bx, bf

        # partial restart / injection on stagnation
        if stagn >= stagn_limit and time.time() < deadline:
            stagn = 0
            k = max(2, int(inject_frac * NP))
            worst = sorted(range(NP), key=lambda ii: fit[ii], reverse=True)[:k]
            for idx in worst:
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.7:
                    # sample around best with heavy tails
                    xnew = best_x[:]
                    for j in range(dim):
                        sigma = (0.10 + 0.25 * random.random()) * spans[j]
                        xnew[j] = reflect_into_bounds(xnew[j] + random.gauss(0.0, sigma), lows[j], highs[j])
                else:
                    xnew = rand_uniform_vec()
                fnew = eval_f(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                F[idx] = Fl + random.random() * (Fu - Fl)
                CR[idx] = random.random()

    return best
