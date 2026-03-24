import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Hybrid algorithm:
      1) Differential Evolution, "current-to-pbest/1" (JADE-like) with:
         - external archive for diversity
         - parameter adaptation (mu_F, mu_CR) using successful trials
         - p-best selection pressure (exploitation) while retaining exploration
      2) Periodic local improvement on the global best:
         - SPSA-style simultaneous perturbation (2 evals/step, good in high-d)
         - followed by short coordinate/pattern refinement with shrinking steps

    Returns:
      best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")

    # --- bounds ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp_val(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def ensure_bounds(x):
        return [clamp_val(x[i], i) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # --- population size ---
    # Keep moderate; JADE/DE works well with ~8..12*dim, but cap for time.
    NP = max(24, min(10 * dim, 140))

    # --- JADE-ish settings ---
    p = 0.15               # top-p fraction for pbest
    c = 0.1                # learning rate for mu_F, mu_CR
    mu_F = 0.55
    mu_CR = 0.6

    # archive for replaced individuals (diversity)
    A = []
    Amax = NP

    # --- init population ---
    pop = [rand_vec() for _ in range(NP)]
    fit = [None] * NP

    best = float("inf")
    best_x = None

    for i in range(NP):
        if time.time() >= deadline:
            return best
        fi = eval_f(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # --- sampling helpers for JADE parameters ---
    def rand_cauchy(loc, scale):
        # Cauchy via tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def rand_normal(mu, sigma):
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def sample_F():
        # Cauchy around mu_F, resample until >0, then cap at 1
        for _ in range(20):
            f = rand_cauchy(mu_F, 0.1)
            if f > 0:
                return 1.0 if f > 1.0 else f
        return 0.5

    def sample_CR():
        cr = rand_normal(mu_CR, 0.1)
        if cr < 0.0: cr = 0.0
        if cr > 1.0: cr = 1.0
        return cr

    # --- selection helpers ---
    def pick_index_excluding(excl, upper):
        # pick integer in [0, upper) excluding excl
        r = random.randrange(upper - 1)
        return r if r < excl else r + 1

    def pick_r1_r2(i, NP, A):
        # r1 from pop excluding i
        r1 = pick_index_excluding(i, NP)
        # r2 from pop+archive excluding i and r1 if possible
        pool_size = NP + len(A)
        # draw r2 until distinct in index space
        for _ in range(50):
            r2 = random.randrange(pool_size)
            # map r2 to actual vector later
            if r2 != i and r2 != r1:
                return r1, r2
        # fallback
        r2 = (r1 + 1) % NP
        if r2 == i:
            r2 = (r2 + 1) % NP
        return r1, r2

    # --- local refinement: SPSA + coordinate pattern search ---
    def local_refine(x0, f0, spsa_steps=18, coord_steps=18):
        x = x0[:]
        fx = f0

        # SPSA parameters relative to span
        # Good default scaling: a decreases slowly; c small fraction of span.
        base_span = sum(spans) / max(1, dim)
        if base_span <= 0:
            base_span = 1.0

        a0 = 0.15 * base_span
        c0 = 0.05 * base_span

        # --- SPSA (2 evaluations per step) ---
        for k in range(1, spsa_steps + 1):
            if time.time() >= deadline:
                return x, fx

            ak = a0 / (k ** 0.6)
            ck = c0 / (k ** 0.101)

            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            xp = x[:]
            xm = x[:]
            for j in range(dim):
                if spans[j] > 0:
                    xp[j] = clamp_val(xp[j] + ck * delta[j], j)
                    xm[j] = clamp_val(xm[j] - ck * delta[j], j)

            fp = eval_f(xp)
            if time.time() >= deadline:
                return x, fx
            fm = eval_f(xm)

            # gradient estimate and step
            # g_j ≈ (fp - fm) / (2*ck*delta_j)
            # update: x - ak * g
            step = (fp - fm) / (2.0 * ck)
            xn = x[:]
            for j in range(dim):
                if spans[j] > 0:
                    gj = step / delta[j]
                    xn[j] = clamp_val(xn[j] - ak * gj, j)

            fn = eval_f(xn)
            if fn < fx:
                x, fx = xn, fn
            else:
                # small "acceptance" of best of {xp,xm} if improves
                if fp < fx:
                    x, fx = xp, fp
                elif fm < fx:
                    x, fx = xm, fm

        # --- coordinate / pattern refinement with shrinking steps ---
        step = [0.10 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        for _ in range(coord_steps):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if time.time() >= deadline:
                    break
                s = step[j]
                if s <= 0:
                    continue

                # try +s then -s
                xp = x[:]
                xp[j] = clamp_val(xp[j] + s, j)
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[j] = clamp_val(xm[j] - s, j)
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if not improved:
                # shrink steps
                for j in range(dim):
                    step[j] *= 0.5
                if max(step) < 1e-12 * (max(spans) if max(spans) > 0 else 1.0):
                    break
            else:
                # mild expansion to keep moving
                for j in range(dim):
                    if spans[j] > 0:
                        step[j] = min(step[j] * 1.05, 0.35 * spans[j])

        return x, fx

    gen = 0
    refine_every = 7
    last_best = best
    stagn = 0

    while time.time() < deadline:
        gen += 1

        # sort indices by fitness for pbest selection
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p * NP)))
        pbest_set = idx_sorted[:pcount]

        SF = []
        SCR = []

        # main loop over individuals
        for i in range(NP):
            if time.time() >= deadline:
                return best

            Fi = sample_F()
            CRi = sample_CR()

            # choose pbest
            pbest = random.choice(pbest_set)

            # pick r1 and r2
            r1, r2 = pick_r1_r2(i, NP, A)

            xi = pop[i]
            xpbest = pop[pbest]
            xr1 = pop[r1]

            # r2 from pop or archive
            if r2 < NP:
                xr2 = pop[r2]
            else:
                xr2 = A[r2 - NP]

            # mutation: current-to-pbest/1
            # v = xi + Fi*(xpbest - xi) + Fi*(xr1 - xr2)
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                # "bounce" bound handling tends to work better than hard clamp
                if vj < lows[j]:
                    vj = lows[j] + (lows[j] - vj)
                    if vj > highs[j]:
                        vj = lows[j]
                elif vj > highs[j]:
                    vj = highs[j] - (vj - highs[j])
                    if vj < lows[j]:
                        vj = highs[j]
                v[j] = vj

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    u[j] = v[j]
            u = ensure_bounds(u)

            fu = eval_f(u)

            if fu <= fit[i]:
                # success: push old into archive, replace
                A.append(pop[i])
                if len(A) > Amax:
                    # remove random element to keep archive bounded
                    A.pop(random.randrange(len(A)))

                pop[i] = u
                fit[i] = fu

                SF.append(Fi)
                SCR.append(CRi)

                if fu < best:
                    best = fu
                    best_x = u[:]

        # update mu_F and mu_CR from successes (JADE)
        if SF:
            # Lehmer mean for F (favors larger successful F)
            num = sum(f * f for f in SF)
            den = sum(SF)
            if den > 0:
                lehmer = num / den
                mu_F = (1.0 - c) * mu_F + c * lehmer

            mu_CR = (1.0 - c) * mu_CR + c * (sum(SCR) / len(SCR))

        # stagnation tracking
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # periodic local refinement on best
        if best_x is not None and gen % refine_every == 0 and time.time() < deadline:
            bx, bf = local_refine(best_x, best, spsa_steps=10, coord_steps=10)
            if bf < best:
                best, best_x = bf, bx

        # adaptive restart / injection if stagnating
        if stagn >= 10 and time.time() < deadline:
            stagn = 0
            # replace a fraction of worst with random + near-best perturbations
            k = max(2, NP // 10)
            worst = sorted(range(NP), key=lambda ii: fit[ii], reverse=True)[:k]
            for idx in worst:
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.6:
                    # near-best perturbation
                    xnew = best_x[:]
                    for j in range(dim):
                        if spans[j] > 0:
                            # heavier tails occasionally
                            if random.random() < 0.2:
                                xnew[j] = clamp_val(xnew[j] + random.gauss(0.0, 0.20 * spans[j]), j)
                            else:
                                xnew[j] = clamp_val(xnew[j] + random.gauss(0.0, 0.08 * spans[j]), j)
                else:
                    xnew = rand_vec()
                fnew = eval_f(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best = fnew
                    best_x = xnew[:]

    return best
