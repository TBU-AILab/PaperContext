import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved minimizer: Multi-regime optimizer with
      - LHS-like seeding + opposite points + corner-biased points
      - Robust evaluation cache
      - Two simultaneous search processes:
          (A) sep-CMA-ES style diagonal adaptation (fast local convergence)
          (B) SHADE-like Differential Evolution (self-adaptive F/CR, archive-based)
      - "Stochastic local search" (1+1) around best with mixed steps (coord/subspace)
      - Time-aware scheduling + stagnation-triggered partial restarts

    Self-contained, no external libraries.
    Returns best fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]
    active = [i for i in range(dim) if not fixed[i]]
    adim = len(active)

    # ----------- helpers -----------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def reflect_to_bounds(x):
        # reflection keeps continuity near borders
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
                continue
            a, b = lo[i], hi[i]
            w = b - a
            if w <= 0.0:
                x[i] = a
                continue
            y = x[i] - a
            y = y % (2.0 * w)
            if y > w:
                y = 2.0 * w - y
            x[i] = a + y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] if fixed[i] else (lo[i] + random.random() * span[i])
        return x

    def pack_active(xfull):
        return [xfull[i] for i in active]

    def unpack_active(xact, template_full):
        x = template_full[:]
        for k, i in enumerate(active):
            x[i] = xact[k]
        return x

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    def norm2(v):
        return sum(a*a for a in v)

    def vec_norm(v):
        return math.sqrt(max(0.0, norm2(v)))

    # degenerate
    if adim == 0:
        x0 = [lo[i] for i in range(dim)]
        return safe_eval(x0)

    # scale reference
    avg_span = sum(span[i] for i in active) / max(1, adim)
    base = avg_span if avg_span > 0.0 else 1.0

    # ----------- cache -----------
    # scale-aware key; reduces redundant evals
    def key_of(x):
        k = []
        for i in range(dim):
            if fixed[i]:
                k.append(0)
            else:
                s = span[i] if span[i] != 0.0 else 1.0
                k.append(int(round((x[i] - lo[i]) / s * 1e12)))
        return tuple(k)

    cache = {}
    def eval_cached(x):
        reflect_to_bounds(x)
        k = key_of(x)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    # ----------- global archive / population store -----------
    best = float("inf")
    best_x = rand_uniform()

    # We'll maintain a "pool" of evaluated points for DE selection.
    # store as list of (f, xfull)
    POOL_MAX = max(120, min(900, 180 + 50 * adim))
    pool = []

    def pool_add(x, f):
        nonlocal best, best_x
        pool.append((f, x[:]))
        if f < best:
            best = f
            best_x = x[:]

    def pool_trim():
        if len(pool) <= POOL_MAX:
            return
        pool.sort(key=lambda t: t[0])
        keep_best = max(50, POOL_MAX // 2)
        kept = pool[:keep_best]
        rest = pool[keep_best:]
        # keep random remainder for diversity
        while len(kept) < POOL_MAX and rest:
            kept.append(rest.pop(random.randrange(len(rest))))
        pool[:] = kept

    # ----------- seeding: LHS-like + opposites + corner-biased -----------
    seed_n = min(45 + 7 * adim, 260)
    strata = []
    for i in range(dim):
        if fixed[i]:
            strata.append([lo[i]] * seed_n)
        else:
            perm = list(range(seed_n))
            random.shuffle(perm)
            vals = []
            for j in range(seed_n):
                u = (perm[j] + random.random()) / seed_n
                vals.append(lo[i] + u * span[i])
            strata.append(vals)

    for j in range(seed_n):
        if time.time() >= deadline:
            return best
        x = [strata[i][j] for i in range(dim)]
        if j & 1:
            for i in active:
                x[i] = lo[i] + (hi[i] - x[i])
        f = eval_cached(x)
        pool_add(x, f)

    # corner-biased points near bounds (useful for constrained opt)
    extra = min(10 + 2 * adim, 60)
    for _ in range(extra):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for i in active:
            if random.random() < 0.5:
                x[i] = lo[i] + (random.random() ** 2) * span[i] * 0.03
            else:
                x[i] = hi[i] - (random.random() ** 2) * span[i] * 0.03
        f = eval_cached(x)
        pool_add(x, f)

    pool_trim()

    # =============== Process A: sep-CMA-ish state (active only) ===============
    m_full = best_x[:]
    m = pack_active(m_full)
    Cdiag = [1.0] * adim
    ps = [0.0] * adim
    pc = [0.0] * adim

    lam0 = max(14, min(70, 10 + 2 * adim + int(3 * math.log(adim + 1.0))))
    lam = lam0
    sigma = 0.32 * base

    def make_weights(mu):
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        w = [wi / s for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)
        return w, mueff

    def sample_heavy():
        # mostly Gaussian; rare Cauchy tails
        if random.random() < 0.08:
            u = random.random()
            return math.tan(math.pi * (u - 0.5))
        return random.gauss(0.0, 1.0)

    # =============== Process B: SHADE-like DE state (self-adaptive F/CR) ===============
    # We'll keep a small DE population built from pool best+diverse.
    NP0 = max(18, min(90, 12 + 5 * int(math.sqrt(adim + 1)) + 2 * adim))
    NP = NP0

    # parameter memories
    H = max(6, min(30, 6 + adim // 2))
    MF = [0.5] * H
    MCR = [0.8] * H
    mptr = 0

    # external archive for DE (like JADE/SHADE) to increase diversity
    Aext = []  # list of xfull
    AEXT_MAX = max(40, min(400, 40 + 12 * adim))

    # build initial DE population from pool (best + random)
    def init_de_population():
        # choose some best points and some random points
        if not pool:
            return [rand_uniform() for _ in range(NP)]
        pool.sort(key=lambda t: t[0])
        popx = []
        take_best = min(len(pool), max(4, NP // 3))
        for i in range(take_best):
            popx.append(pool[i][1][:])
        while len(popx) < NP and len(pool) > 0:
            popx.append(pool[random.randrange(len(pool))][1][:])
        while len(popx) < NP:
            popx.append(rand_uniform())
        return popx[:NP]

    de_pop = init_de_population()
    de_fit = [eval_cached(x) for x in de_pop]
    for x, f in zip(de_pop, de_fit):
        pool_add(x, f)
    pool_trim()

    # utilities for DE
    def cauchy(loc, scale):
        # simple Cauchy sample
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def truncnorm(mean, sd):
        # cheap truncated normal to [0,1]
        for _ in range(12):
            v = random.gauss(mean, sd)
            if 0.0 <= v <= 1.0:
                return v
        return clamp(mean, 0.0, 1.0)

    def choose_pbest_index(sorted_idx, p=0.2):
        # choose from top p fraction
        n = len(sorted_idx)
        k = max(2, int(math.ceil(p * n)))
        return sorted_idx[random.randrange(k)]

    # ---------------- local refinement around incumbent ----------------
    ls_step = [0.06 * s for s in span]
    ls_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    def local_improve(tries):
        nonlocal m_full, m, ps, pc, sigma
        for _ in range(tries):
            if time.time() >= deadline:
                return
            x = best_x[:]
            r = random.random()
            if r < 0.62:
                # coordinate nudge
                d = random.randrange(dim)
                if fixed[d]:
                    continue
                sd = max(ls_min[d], ls_step[d])
                x[d] += sd if random.random() < 0.5 else -sd
            elif r < 0.88:
                # random subspace gaussian step (scaled)
                ksub = 1 if adim == 1 else min(adim, 1 + int(1 + math.sqrt(adim) * random.random()))
                idx = random.sample(active, ksub)
                g = [random.gauss(0.0, 1.0) for _ in range(ksub)]
                ng = math.sqrt(sum(v*v for v in g))
                if ng <= 0.0:
                    continue
                sc = (0.10 + 0.80 * random.random()) * (0.20 * sigma + 0.02 * base)
                for t, ii in enumerate(idx):
                    x[ii] += sc * (g[t] / ng)
            else:
                # very local "trust" step on all active
                g = [random.gauss(0.0, 1.0) for _ in range(adim)]
                ng = vec_norm(g)
                if ng <= 0.0:
                    continue
                rad = (0.02 + 0.18 * random.random()) * (0.35 * sigma + 0.03 * base)
                for t, ii in enumerate(active):
                    x[ii] += rad * (g[t] / ng)

            reflect_to_bounds(x)
            fx = eval_cached(x)
            pool_add(x, fx)
            if fx < best:
                # snap CMA mean to new best
                m_full = best_x[:]
                m = pack_active(m_full)
                ps = [0.0] * adim
                pc = [0.0] * adim
                sigma = max(sigma, 0.15 * base)
                return
            else:
                # gentle decay of coordinate step
                for d in active:
                    if ls_step[d] > ls_min[d]:
                        ls_step[d] *= 0.996

    # ---------------- time-scheduled controller ----------------
    gen = 0
    stall = 0
    last_best = best
    restarts = 0

    # relative allocation per outer loop iteration:
    # do some DE steps and some CMA steps; adapt toward whichever is improving.
    w_de = 0.55
    w_cma = 0.45

    def adapt_weights(improved_by_de, improved_by_cma):
        nonlocal w_de, w_cma
        # small, stable updates
        if improved_by_de and not improved_by_cma:
            w_de = min(0.85, w_de + 0.03)
            w_cma = 1.0 - w_de
        elif improved_by_cma and not improved_by_de:
            w_de = max(0.15, w_de - 0.03)
            w_cma = 1.0 - w_de
        # if neither or both improved -> keep

    # ======================= main loop =======================
    while time.time() < deadline:
        gen += 1
        pool_trim()

        # ---- DE generation chunk ----
        improved_de = False
        de_steps = max(1, int(round(w_de * 1.6 * NP / max(1, (lam0 // 2)))))
        de_steps = min(de_steps, 4)  # keep chunk small for responsiveness

        for _ in range(de_steps):
            if time.time() >= deadline:
                return best

            # sort indices by fitness
            idx_sorted = list(range(NP))
            idx_sorted.sort(key=lambda i: de_fit[i])

            # one SHADE generation over DE pop
            SF = []
            SCR = []
            Sdf = []  # fitness improvements

            new_pop = [None] * NP
            new_fit = [None] * NP

            # p-best fraction
            pbest = 0.10 + 0.20 * random.random()  # [0.10,0.30]

            for i in range(NP):
                if time.time() >= deadline:
                    return best

                xi = de_pop[i]
                fi = de_fit[i]

                r = random.randrange(H)
                # F from cauchy around MF[r], resample if <=0
                F = cauchy(MF[r], 0.1)
                tries = 0
                while F <= 0.0 and tries < 8:
                    F = cauchy(MF[r], 0.1)
                    tries += 1
                F = clamp(F, 0.05, 1.0)

                CR = truncnorm(MCR[r], 0.1)

                # choose pbest
                pbest_idx = choose_pbest_index(idx_sorted, p=pbest)
                xpbest = de_pop[pbest_idx]

                # choose r1 != i
                r1 = random.randrange(NP - 1)
                if r1 >= i:
                    r1 += 1
                xr1 = de_pop[r1]

                # choose xr2 from union(pop + Aext) distinct
                union = de_pop
                use_archive = (len(Aext) > 0) and (random.random() < 0.50)
                if use_archive:
                    # pick from archive
                    xr2 = Aext[random.randrange(len(Aext))]
                else:
                    r2 = random.randrange(NP - 1)
                    if r2 >= i:
                        r2 += 1
                    xr2 = de_pop[r2]

                # mutation: current-to-pbest/1
                v = xi[:]
                for d in active:
                    v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

                reflect_to_bounds(v)

                # binomial crossover
                u = xi[:]
                jrand = random.choice(active)
                for d in active:
                    if d == jrand or random.random() < CR:
                        u[d] = v[d]
                # jitter occasionally
                if random.random() < 0.15:
                    d = random.choice(active)
                    u[d] += random.gauss(0.0, 0.0025 * span[d] if span[d] > 0 else 0.0025)

                reflect_to_bounds(u)
                fu = eval_cached(u)
                pool_add(u, fu)

                # selection + archive update
                if fu <= fi:
                    new_pop[i] = u
                    new_fit[i] = fu
                    # archive old xi
                    if len(Aext) < AEXT_MAX:
                        Aext.append(xi[:])
                    else:
                        Aext[random.randrange(AEXT_MAX)] = xi[:]
                    df = fi - fu
                    if df > 0.0:
                        SF.append(F)
                        SCR.append(CR)
                        Sdf.append(df)
                else:
                    new_pop[i] = xi
                    new_fit[i] = fi

            de_pop = new_pop
            de_fit = new_fit

            # update memories (weighted by improvements)
            if Sdf:
                wsum = sum(Sdf)
                if wsum <= 0.0:
                    wsum = 1.0
                # lehmer mean for F, arithmetic for CR
                numF = 0.0
                denF = 0.0
                meanCR = 0.0
                for f, cr, df in zip(SF, SCR, Sdf):
                    w = df / wsum
                    numF += w * (f * f)
                    denF += w * f
                    meanCR += w * cr
                if denF > 0.0:
                    MF[mptr] = clamp(numF / denF, 0.05, 1.0)
                MCR[mptr] = clamp(meanCR, 0.0, 1.0)
                mptr = (mptr + 1) % H

            # check improvement
            cur_best_de = min(de_fit)
            if cur_best_de < last_best - 1e-15:
                improved_de = True

        # ---- CMA generation chunk ----
        improved_cma = False

        # CMA parameters
        mu = max(3, lam // 2)
        weights, mueff = make_weights(mu)
        cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
        cs = (mueff + 2.0) / (adim + mueff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

        # when stalled, enlarge sigma slightly and increase population a bit
        if stall > max(10, 2 * adim):
            sigma = min(3.0 * base, sigma * 1.03)
            lam = min(200, max(lam0, int(lam * 1.05)))
        else:
            lam = max(lam0, int(lam * 0.99))

        # build CMA population (mirrored)
        pop = []  # (f, xact)
        need = lam
        while need > 0:
            if time.time() >= deadline:
                return best
            z = [sample_heavy() for _ in range(adim)]
            for sign in (1.0, -1.0):
                if need <= 0:
                    break
                y = [math.sqrt(max(1e-30, Cdiag[k])) * (sign * z[k]) for k in range(adim)]
                # slight pull to global best to reduce drifting
                pull = 0.10 + 0.18 * random.random()
                bx = pack_active(best_x)
                xact = [m[k] + sigma * y[k] + pull * (bx[k] - m[k]) for k in range(adim)]
                xfull = unpack_active(xact, m_full)
                reflect_to_bounds(xfull)
                f = eval_cached(xfull)
                pool_add(xfull, f)
                pop.append((f, pack_active(xfull)))
                need -= 1

        pop.sort(key=lambda t: t[0])

        if pop and pop[0][0] < last_best - 1e-15:
            improved_cma = True

        # CMA update
        old_m = m[:]
        new_m = [0.0] * adim
        for i in range(mu):
            wi = weights[i]
            xi = pop[i][1]
            for k in range(adim):
                new_m[k] += wi * xi[k]
        m = new_m
        m_full = unpack_active(m, m_full)
        reflect_to_bounds(m_full)
        m = pack_active(m_full)

        invsigma = 1.0 / max(1e-30, sigma)
        y_w = [(m[k] - old_m[k]) * invsigma for k in range(adim)]
        invsqrtCy = [y_w[k] / math.sqrt(max(1e-30, Cdiag[k])) for k in range(adim)]

        coeff_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        for k in range(adim):
            ps[k] = (1.0 - cs) * ps[k] + coeff_ps * invsqrtCy[k]

        ps_norm = vec_norm(ps)
        left = ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen))
        hsig = 1.0 if left < (1.4 + 2.0 / (adim + 1.0)) * chiN else 0.0

        coeff_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for k in range(adim):
            pc[k] = (1.0 - cc) * pc[k] + hsig * coeff_pc * y_w[k]

        factor = 1.0 - c1 - cmu
        if factor < 0.0:
            factor = 0.0

        y_mu2 = [0.0] * adim
        for i in range(mu):
            xi = pop[i][1]
            wi = weights[i]
            for k in range(adim):
                yik = (xi[k] - old_m[k]) * invsigma
                y_mu2[k] += wi * (yik * yik)

        for k in range(adim):
            Ck = Cdiag[k] * factor
            Ck += c1 * (pc[k] * pc[k])
            if hsig == 0.0:
                Ck += c1 * cc * (2.0 - cc) * Cdiag[k] * 0.2
            Ck += cmu * y_mu2[k]
            Cdiag[k] = clamp(Ck, 1e-30, 1e30)

        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        sigma = clamp(sigma, 1e-16 * base, 3.2 * base + 1e-12)

        # ---- progress / stall / controller ----
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        adapt_weights(improved_de, improved_cma)

        # ---- occasional local improvement ----
        if stall > max(6, 2 * adim) and time.time() + 0.002 < deadline:
            local_improve(tries=min(18 + dim, 50))

        # ---- restart logic ----
        if stall > max(28, 6 * adim) and time.time() + 0.01 < deadline:
            restarts += 1
            stall = 0

            # keep best, partially reset both processes
            # CMA reset around best with broadened sigma
            m_full = best_x[:]
            m = pack_active(m_full)
            ps = [0.0] * adim
            pc = [0.0] * adim
            Cdiag = [1.0] * adim
            sigma = max(0.65 * base, sigma)

            # DE: rebuild population from pool/best + random, keep memories
            NP = min(140, max(NP0, int(NP * (1.0 + 0.18 * restarts))))
            de_pop = init_de_population()
            de_fit = [eval_cached(x) for x in de_pop]
            for x, f in zip(de_pop, de_fit):
                pool_add(x, f)

            # inject fresh uniform points
            inj = min(12 + 3 * adim, 55)
            for _ in range(inj):
                if time.time() >= deadline:
                    return best
                x = rand_uniform()
                f = eval_cached(x)
                pool_add(x, f)

            # diversify DE archive
            if len(Aext) > AEXT_MAX // 2:
                Aext[:] = random.sample(Aext, AEXT_MAX // 2)

    return best
