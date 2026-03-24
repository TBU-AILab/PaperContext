import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (self-contained, no external libs).

    What is improved vs your current best (L-SHADE-ish + ES + coord polish):
      1) Adds CMA-ES-style local exploitation (diagonal/separable CMA) on top of DE:
         - Very strong on smooth/ill-conditioned problems.
         - Uses only O(dim) memory, no matrices.
      2) Keeps your best global driver (current-to-pbest/1 with archive + SHADE memory),
         but makes it more time-aware:
         - estimates eval time and allocates a fixed fraction of remaining time to CMA.
      3) Better “late-stage finishing”:
         - switches to more CMA steps and less DE as time runs out.
      4) Robust constraint handling:
         - bounce repair as before (reflection), good for DE and CMA sampling.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- RNG helpers ----------------
    _bm_has = False
    _bm_val = 0.0

    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_val = z1
        _bm_has = True
        return z0

    def cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------------- helpers ----------------
    def bounce_repair(x):
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                xi = a + y
            if xi < a:
                xi = a
            elif xi > b:
                xi = b
            x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def eval_point(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def opposition_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------------- low discrepancy seeding (scrambled Halton) ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def scrambled_halton_points(n):
        bases = first_primes(max(1, dim))
        shifts = [random.random() for _ in range(dim)]  # CP rotation
        pts = []
        for k in range(1, n + 1):
            x = []
            for d in range(dim):
                u = (halton_value(k, bases[d]) + shifts[d]) % 1.0
                x.append(lo[d] + u * span_safe[d])
            pts.append(x)
        return pts

    # ---------------- quick eval-time estimation ----------------
    def estimate_eval_time():
        k = 3
        times = []
        for _ in range(k):
            if time.time() >= deadline:
                break
            x = rand_point()
            t1 = time.time()
            _ = eval_point(x)
            t2 = time.time()
            times.append(max(1e-6, t2 - t1))
        if not times:
            return 1e-3
        times.sort()
        return times[len(times) // 2]

    eval_dt = estimate_eval_time()
    remaining0 = max(0.0, deadline - time.time())
    eval_budget0 = max(30, int(0.90 * remaining0 / max(eval_dt, 1e-9)))

    # ---------------- initialization ----------------
    NP0 = int(18 + 4.5 * dim)
    NP0 = max(18, min(90, NP0))
    if eval_budget0 < 250:
        NP0 = max(12, min(NP0, 28))
    elif eval_budget0 < 600:
        NP0 = max(16, min(NP0, 45))

    NPmin = max(8, min(24, 6 + 2 * dim))

    # seed size depends on remaining budget
    n_seed = min(max(NP0, 3 * NP0), max(60, min(260, eval_budget0 // 3)))
    n_halton = max(2, int(0.70 * n_seed))
    n_rand = n_seed - n_halton

    seeds = scrambled_halton_points(n_halton)
    for _ in range(n_rand):
        seeds.append(rand_point())

    # opposition + boundary-biased
    seeds2 = []
    for x in seeds:
        seeds2.append(x)
        seeds2.append(opposition_point(x))

    boundary_k = max(6, min(40, 2 * dim + 8))
    for _ in range(boundary_k):
        x = []
        for d in range(dim):
            r = random.random()
            if r < 0.34:
                u = (random.random() ** 2) * 0.02
                x.append(lo[d] + u * span_safe[d])
            elif r < 0.68:
                u = (random.random() ** 2) * 0.02
                x.append(hi[d] - u * span_safe[d])
            else:
                x.append(lo[d] + random.random() * span_safe[d])
        seeds2.append(x)

    best = float("inf")
    best_x = None
    scored = []
    for x in seeds2:
        if time.time() >= deadline:
            return best
        bounce_repair(x)
        fx = eval_point(x)
        scored.append((fx, x[:]))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    scored = scored[:NP0]
    pop = [x for (fx, x) in scored]
    fit = [fx for (fx, x) in scored]

    # ---------------- SHADE memory ----------------
    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0
    archive = []
    archive_max = NP0

    pmin = 2.0 / max(2, NP0)
    pmax = 0.30

    last_improve = time.time()
    stall_seconds = max(0.20, 0.16 * max_time)

    # ---------------- Diagonal CMA-ES state (separable CMA) ----------------
    # Initialized around current best; adapted when it improves.
    cma_m = best_x[:] if best_x is not None else rand_point()
    # global step size as fraction of span
    cma_sigma = 0.18
    # diagonal covariance (per-dim std multipliers), start at 1
    cma_diag = [1.0] * dim
    # evolution path (diagonal)
    cma_ps = [0.0] * dim

    # a small default lambda; adjusted by time and dim
    def cma_lambda():
        lam = 4 + int(3 * math.log(dim + 1.0))
        if lam < 6:
            lam = 6
        if lam > 24:
            lam = 24
        return lam

    # weights for recombination (computed per lambda)
    def cma_weights(lam):
        mu = lam // 2
        w = [0.0] * mu
        s = 0.0
        for i in range(mu):
            w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
            s += w[i]
        invs = 1.0 / max(1e-18, s)
        for i in range(mu):
            w[i] *= invs
        # effective mu
        mueff = 0.0
        for i in range(mu):
            mueff += w[i] * w[i]
        mueff = 1.0 / max(1e-18, mueff)
        return mu, w, mueff

    def cma_step(iter_budget, aggressiveness):
        """
        Run a few CMA-ES generations (diagonal), return updated (best, best_x).
        aggressiveness in [0,1]: late stage -> higher, more local sampling.
        """
        nonlocal cma_m, cma_sigma, cma_diag, cma_ps, best, best_x, last_improve

        lam = cma_lambda()
        mu, w, mueff = cma_weights(lam)

        # learning rates (diagonal/separable heuristics)
        # slightly more aggressive for late phase
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = (2.0 / ((dim + 1.3) ** 2 + mueff)) * (0.6 + 0.8 * aggressiveness)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

        # expected norm of N(0,I)
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)) if dim > 0 else 1.0

        # evaluation budget in number of samples
        evals_left = iter_budget
        while evals_left >= lam and time.time() < deadline:
            # sample population
            cand = []
            z_store = []  # keep z for update
            for _ in range(lam):
                if time.time() >= deadline:
                    break
                z = [randn() for _ in range(dim)]
                x = cma_m[:]
                # x = m + sigma * diag * z
                sig = cma_sigma
                for i in range(dim):
                    x[i] += sig * cma_diag[i] * z[i] * span_safe[i]
                bounce_repair(x)
                fx = eval_point(x)
                cand.append((fx, x))
                z_store.append(z)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve = time.time()

            if len(cand) < lam:
                break

            # sort by fitness
            order = list(range(lam))
            order.sort(key=lambda i: cand[i][0])

            # old mean
            m_old = cma_m[:]

            # recombination: new mean
            m_new = [0.0] * dim
            for i in range(dim):
                m_new[i] = 0.0
            for k in range(mu):
                idx = order[k]
                xk = cand[idx][1]
                wk = w[k]
                for i in range(dim):
                    m_new[i] += wk * xk[i]
            cma_m = m_new

            # compute y = (m_new - m_old) / (sigma*diag*span)
            y = [0.0] * dim
            invsig = 1.0 / max(1e-18, cma_sigma)
            for i in range(dim):
                denom = max(1e-18, cma_diag[i] * span_safe[i])
                y[i] = (cma_m[i] - m_old[i]) * invsig / denom

            # update ps (for sigma control)
            c_s = math.sqrt(cs * (2.0 - cs) * mueff)
            one_minus = (1.0 - cs)
            for i in range(dim):
                cma_ps[i] = one_minus * cma_ps[i] + c_s * y[i]

            # sigma update (CMA step-size control)
            ps_norm = 0.0
            for i in range(dim):
                ps_norm += cma_ps[i] * cma_ps[i]
            ps_norm = math.sqrt(ps_norm)
            cma_sigma *= math.exp((cs / damps) * (ps_norm / max(1e-18, chiN) - 1.0))
            # keep sigma within reasonable range
            if cma_sigma < 1e-12:
                cma_sigma = 1e-12
            if cma_sigma > 0.9:
                cma_sigma = 0.9

            # update diagonal covariance using selected z
            # Approx: diag <- sqrt((1-c1-cmu)*diag^2 + c1*(y^2) + cmu*sum(w*z^2))
            # Use y^2 as rank-1 proxy (diagonal)
            for i in range(dim):
                diag2 = cma_diag[i] * cma_diag[i]
                # rank-mu part from z
                z2 = 0.0
                for k in range(mu):
                    idx = order[k]
                    wk = w[k]
                    zi = z_store[idx][i]
                    z2 += wk * (zi * zi)
                new_diag2 = (1.0 - c1 - cmu) * diag2 + c1 * (y[i] * y[i]) + cmu * z2
                if new_diag2 < 1e-18:
                    new_diag2 = 1e-18
                cma_diag[i] = math.sqrt(new_diag2)

            evals_left -= lam

        return

    # ---------------- main loop (DE + CMA intensification) ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = time.time() - t0
        frac = min(1.0, max(0.0, elapsed / max(1e-9, max_time)))

        # Allocate more to CMA near the end (and if stalling)
        remaining = max(0.0, deadline - time.time())
        eval_budget = max(10, int(0.90 * remaining / max(eval_dt, 1e-9)))
        stalled = (time.time() - last_improve) > stall_seconds

        # occasional CMA bursts; stronger late
        if best_x is not None and (gen <= 2 or gen % 7 == 0 or stalled):
            # re-center CMA mean on best sometimes
            if best_x is not None and (gen <= 2 or stalled or random.random() < 0.35):
                cma_m = best_x[:]

            # set sigma based on stage: smaller later
            # also prevent it from collapsing too early
            target_sig = 0.20 * (1.0 - 0.75 * frac)
            if stalled:
                target_sig = max(target_sig, 0.10)
            cma_sigma = 0.70 * cma_sigma + 0.30 * target_sig

            # budget: a fraction of remaining evals
            share = 0.18 + 0.22 * frac
            if stalled:
                share = max(share, 0.28)
            cma_evals = int(max(0, min(eval_budget, share * eval_budget)))
            # make it multiple of lambda so it does complete generations
            lam = cma_lambda()
            cma_evals = (cma_evals // lam) * lam
            if cma_evals >= lam:
                cma_step(cma_evals, aggressiveness=frac)

        # ---- linear population reduction (L-SHADE) ----
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac))
        if target_NP < NPmin:
            target_NP = NPmin

        if len(pop) > target_NP:
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            archive_max = max(target_NP, 8)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        if NP < 4:
            return best

        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = pmin + (pmax - pmin) * random.random()
        pcount = max(2, int(math.ceil(p * NP)))

        S_CR, S_F, S_df = [], [], []

        # ---- evolve (DE) ----
        for i in range(NP):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)

            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            Fi = cauchy(MF[r], 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_pbest = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            pool_size = NP + len(archive)
            if pool_size <= 2:
                r2 = random.randrange(NP)
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_size)

            x_r1 = pop[r1]
            x_r2 = archive[r2 - NP] if r2 >= NP else pop[r2]

            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            u = x_i[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
            bounce_repair(u)

            fu = eval_point(u)
            if fu <= fit[i]:
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                df = fit[i] - fu
                if df > 0.0:
                    S_CR.append(CRi)
                    S_F.append(Fi)
                    S_df.append(df)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = time.time()
                    # also re-center CMA immediately on improvements
                    cma_m = best_x[:]

        # ---- update memories (SHADE) ----
        if S_df:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                wsum = 1.0

            cr_new = 0.0
            for k in range(len(S_df)):
                cr_new += (S_df[k] / wsum) * S_CR[k]

            num = 0.0
            den = 0.0
            for k in range(len(S_df)):
                wk = S_df[k] / wsum
                fk = S_F[k]
                num += wk * fk * fk
                den += wk * fk
            f_new = (num / den) if den > 1e-18 else MF[mem_idx]

            MCR[mem_idx] = cr_new
            MF[mem_idx] = f_new
            mem_idx = (mem_idx + 1) % H

        # ---- stall handling ----
        if time.time() - last_improve > stall_seconds and time.time() < deadline:
            # diversify worst part, but less aggressively late (to preserve convergence)
            order_desc = sorted(range(NP), key=lambda i: fit[i], reverse=True)
            frac_rep = 0.40 * (1.0 - 0.55 * frac)
            m = max(2, int(frac_rep * NP))

            for t in range(m):
                if time.time() >= deadline:
                    return best
                k = order_desc[t]
                if best_x is not None and random.random() < 0.80:
                    y = best_x[:]
                    rad = 0.10 + 0.20 * abs(cauchy(0.0, 1.0))
                    if rad > 0.80:
                        rad = 0.80
                    for d in range(dim):
                        y[d] += (random.random() * 2.0 - 1.0) * rad * span_safe[d]
                    bounce_repair(y)
                else:
                    y = rand_point()
                fy = eval_point(y)
                pop[k] = y
                fit[k] = fy
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve = time.time()
                    cma_m = best_x[:]

            # give CMA a short immediate chance after diversification
            lam = cma_lambda()
            cma_evals = lam * (1 if frac < 0.6 else 2)
            if time.time() < deadline:
                cma_step(cma_evals, aggressiveness=min(1.0, frac + 0.15))
            last_improve = time.time()

    return best
