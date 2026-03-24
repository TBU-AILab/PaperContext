import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only anytime minimizer (bounded).

    Compared to your last version, this one is typically stronger because it:
      - Uses a more correct/robust diagonal-CMA-ES update (mean from top-mu samples,
        with evolution path + step-size control) instead of always snapping mean to best.
      - Interleaves global search (adaptive DE/current-to-pbest/1 + archive) with
        exploitation (diag-CMA sampling) using a time-aware schedule.
      - Adds a small, very cheap multi-scale coordinate pattern search near the end.
      - Includes restart-on-stagnation that *resets* the CMA distribution sensibly and
        partially refreshes the DE population (keeps elites).

    Returns:
        best (float): best fitness found within max_time seconds
    """

    # ------------------ basic helpers ------------------
    def safe_float(y):
        try:
            y = float(y)
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == -float("inf"):
            return float("inf")
        return y

    if dim <= 0:
        try:
            return safe_float(func([]))
        except TypeError:
            return safe_float(func())

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_ok = [span[i] > 0.0 for i in range(dim)]

    def clamp(v, a, b):
        if v < a:
            return a
        if v > b:
            return b
        return v

    def reflect_into_bounds(x):
        # reflection then clamp; modifies in-place
        for d in range(dim):
            if not span_ok[d]:
                x[d] = lo[d]
                continue
            a = lo[d]
            b = hi[d]
            v = x[d]
            if v < a:
                v = a + (a - v)
                if v > b:
                    v = a + random.random() * (b - a)
            elif v > b:
                v = b - (v - b)
                if v < a:
                    v = a + random.random() * (b - a)
            if v < a:
                v = a
            elif v > b:
                v = b
            x[d] = v
        return x

    # quantized cache key (cheap dedup)
    inv_quant = []
    for i in range(dim):
        if not span_ok[i]:
            inv_quant.append(0.0)
        else:
            q = span[i] * 1e-7
            inv_quant.append(0.0 if q <= 0.0 else (1.0 / q))
    cache = {}

    def eval_f(x):
        key = tuple(int((x[i] - lo[i]) * inv_quant[i]) if inv_quant[i] > 0.0 else 0 for i in range(dim))
        v = cache.get(key)
        if v is not None:
            return v
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        y = safe_float(y)
        cache[key] = y
        return y

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] if not span_ok[i] else random.uniform(lo[i], hi[i])
        return x

    def opposite_vec(x):
        ox = x[:]
        for i in range(dim):
            if span_ok[i]:
                ox[i] = (lo[i] + hi[i]) - x[i]
                ox[i] = clamp(ox[i], lo[i], hi[i])
            else:
                ox[i] = lo[i]
        return ox

    # quasi-latin hypercube init (no numpy)
    def qlhs(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    x[d] = lo[d]
                else:
                    u = (perms[d][i] + random.random()) / n
                    x[d] = lo[d] + u * span[d]
            pop.append(x)
        return pop

    def to_norm01(x):
        z = [0.0] * dim
        for i in range(dim):
            z[i] = 0.0 if not span_ok[i] else (x[i] - lo[i]) / span[i]
        return z

    def from_norm01(z):
        x = [0.0] * dim
        for i in range(dim):
            if not span_ok[i]:
                x[i] = lo[i]
            else:
                v = z[i]
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                x[i] = lo[i] + v * span[i]
        return x

    def gauss():
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy(scale):
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        if t > 30.0:
            t = 30.0
        elif t < -30.0:
            t = -30.0
        return scale * t

    # small coordinate/pattern search near the end
    def coord_search(bestx, bestf, max_evals):
        if max_evals <= 0:
            return bestx, bestf
        x = bestx[:]
        f = bestf
        # start step as fraction of span; shrink fast
        steps = [(0.04 * span[d] if span_ok[d] else 0.0) for d in range(dim)]
        evals = 0
        for _round in range(3):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for d in order:
                if evals >= max_evals:
                    return x, f
                s = steps[d]
                if s <= 0.0:
                    continue
                base = x[d]
                # try +/- step and a 2-step jump if it helps
                for mult in (1.0, 2.0):
                    for sgn in (-1.0, 1.0):
                        if evals >= max_evals:
                            return x, f
                        cand = x[:]
                        cand[d] = clamp(base + sgn * mult * s, lo[d], hi[d])
                        if cand[d] == base:
                            continue
                        fc = eval_f(cand)
                        evals += 1
                        if fc < f:
                            x, f = cand, fc
                            improved = True
                            base = x[d]
            if improved:
                for d in range(dim):
                    if span_ok[d]:
                        steps[d] = min(steps[d] * 1.2, 0.25 * span[d])
            else:
                for d in range(dim):
                    steps[d] *= 0.25
        return x, f

    # ------------------ time setup ------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ------------------ init population ------------------
    NP = max(18, min(96, 10 * dim + 10))
    init = qlhs(NP)
    init += [opposite_vec(x) for x in init]

    pop, fit = [], []
    for x in init:
        if time.time() >= deadline:
            break
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)

    if not fit:
        return float("inf")

    while len(pop) < NP and time.time() < deadline:
        x = rand_vec()
        pop.append(x)
        fit.append(eval_f(x))

    if len(pop) > NP:
        idx = sorted(range(len(pop)), key=lambda i: fit[i])[:NP]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    # ------------------ DE (L-SHADE-ish) state ------------------
    H = 6
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_pos = 0
    archive = []
    arch_max = NP

    # ------------------ Diagonal CMA-ES state (normalized space) ------------------
    # Use mean from best initially, but then update from top-mu samples
    m = to_norm01(bestx)  # mean in [0,1]
    # global sigma in normalized coordinates
    sigma = 0.22
    # diagonal covariance (positive)
    Cdiag = [0.20] * dim  # not 1.0: keep steps moderate in [0,1]
    pc = [0.0] * dim
    ps = [0.0] * dim

    lam = max(10, min(44, 4 + int(3.0 * math.sqrt(max(1, dim)))))
    mu = max(2, lam // 2)
    w_raw = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w_raw)
    w = [wi / wsum for wi in w_raw]
    mueff = 1.0 / sum(wi * wi for wi in w)

    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)) if dim > 1 else 1.0

    # ------------------ stagnation / restarts ------------------
    stall = 0
    last_best = best
    max_stall = max(12, 4 + dim)

    def cma_soft_reset_around_best():
        nonlocal m, sigma, Cdiag, pc, ps
        m = to_norm01(bestx)
        sigma = 0.26
        # widen a bit to escape
        for d in range(dim):
            Cdiag[d] = 0.30
            pc[d] = 0.0
            ps[d] = 0.0

    gen = 0
    while time.time() < deadline:
        gen += 1
        tl = deadline - time.time()
        frac_left = tl / max(1e-12, float(max_time))
        near_end = frac_left <= 0.25
        very_end = frac_left <= 0.08

        # =========================================================
        # A) DE generation (exploration / global)
        # =========================================================
        # More DE early, less late
        do_de = (not near_end) or (random.random() < 0.55)

        if do_de and time.time() < deadline:
            p = 0.18 if not near_end else 0.10
            pnum = max(2, int(p * NP))
            order = sorted(range(NP), key=lambda i: fit[i])

            SF, SCR, dF = [], [], []
            new_pop = [None] * NP
            new_fit = [None] * NP

            for i in range(NP):
                if time.time() >= deadline:
                    break

                xi = pop[i]
                fi = fit[i]

                r = random.randrange(H)
                mu_cr = MCR[r]
                mu_f = MF[r]

                CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() - 1.5)
                if CR < 0.0:
                    CR = 0.0
                elif CR > 1.0:
                    CR = 1.0

                F = -1.0
                for _ in range(6):
                    F = mu_f + cauchy(0.10)
                    if 0.0 < F <= 1.0:
                        break
                if not (0.0 < F <= 1.0):
                    F = 0.5

                pbest = order[random.randrange(pnum)]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(NP)

                total_pool = NP + len(archive)
                if total_pool <= 2:
                    r2 = (i + 1) % NP
                    x_r2 = pop[r2]
                else:
                    while True:
                        r2 = random.randrange(total_pool)
                        if r2 < NP:
                            if r2 != i and r2 != r1:
                                x_r2 = pop[r2]
                                break
                        else:
                            x_r2 = archive[r2 - NP]
                            break

                x_r1 = pop[r1]
                x_pb = pop[pbest]

                v = [0.0] * dim
                for d in range(dim):
                    if not span_ok[d]:
                        v[d] = lo[d]
                    else:
                        v[d] = xi[d] + F * (x_pb[d] - xi[d]) + F * (x_r1[d] - x_r2[d])

                u = xi[:]
                jrand = random.randrange(dim)
                for d in range(dim):
                    if (random.random() < CR) or (d == jrand):
                        u[d] = v[d]

                reflect_into_bounds(u)
                fu = eval_f(u)

                if fu <= fi:
                    new_pop[i] = u
                    new_fit[i] = fu

                    if fu < fi:
                        if arch_max > 0:
                            if len(archive) < arch_max:
                                archive.append(xi[:])
                            else:
                                archive[random.randrange(len(archive))] = xi[:]
                        SF.append(F)
                        SCR.append(CR)
                        dF.append(fi - fu)

                    if fu < best:
                        best = fu
                        bestx = u[:]
                else:
                    new_pop[i] = xi
                    new_fit[i] = fi

            if new_fit[0] is not None:
                pop, fit = new_pop, new_fit

            if dF:
                wsum_imp = sum(dF)
                ww = ([1.0 / len(dF)] * len(dF)) if wsum_imp <= 0.0 else [di / wsum_imp for di in dF]

                mcr = 0.0
                for wi, cri in zip(ww, SCR):
                    mcr += wi * cri

                num = 0.0
                den = 0.0
                for wi, Fi in zip(ww, SF):
                    num += wi * Fi * Fi
                    den += wi * Fi
                mf = (num / den) if den > 1e-12 else MF[mem_pos]

                MCR[mem_pos] = mcr
                MF[mem_pos] = mf
                mem_pos = (mem_pos + 1) % H

        # =========================================================
        # B) Diagonal CMA-ES batch (exploitation)
        # =========================================================
        if time.time() < deadline:
            # time-aware batch size
            lam_use = lam if not near_end else max(6, lam // 2)
            if very_end:
                lam_use = max(6, lam // 3)

            # sample lambda candidates around mean m (in normalized coords)
            samples = []
            for _ in range(lam_use):
                if time.time() >= deadline:
                    break
                z = [gauss() for _ in range(dim)]
                y = [0.0] * dim
                xz = [0.0] * dim
                for d in range(dim):
                    if not span_ok[d]:
                        y[d] = 0.0
                        xz[d] = 0.0
                        continue
                    sd = math.sqrt(max(1e-18, Cdiag[d]))
                    y[d] = sigma * sd * z[d]
                    v = m[d] + y[d]
                    if v < 0.0:
                        v = 0.0
                    elif v > 1.0:
                        v = 1.0
                    xz[d] = v

                x = from_norm01(xz)
                fx = eval_f(x)
                samples.append((fx, xz, y))
                if fx < best:
                    best = fx
                    bestx = x[:]

            if len(samples) >= mu:
                samples.sort(key=lambda t: t[0])

                # update mean to weighted recombination of best mu points (in normalized coords)
                m_old = m[:]
                m = [0.0] * dim
                for i in range(mu):
                    wi = w[i]
                    xzi = samples[i][1]
                    for d in range(dim):
                        m[d] += wi * xzi[d]

                # y_w = (m - m_old) in normalized coords
                y_w = [m[d] - m_old[d] for d in range(dim)]

                # update ps (conjugate evolution path, diag approx)
                c_fac = math.sqrt(cs * (2.0 - cs) * mueff)
                for d in range(dim):
                    if not span_ok[d]:
                        ps[d] = 0.0
                        continue
                    sd = math.sqrt(max(1e-18, Cdiag[d]))
                    ps[d] = (1.0 - cs) * ps[d] + c_fac * (y_w[d] / max(1e-18, sigma)) / sd

                # sigma update
                ps_norm = math.sqrt(sum(v * v for v in ps))
                sigma *= math.exp((cs / ds) * (ps_norm / max(1e-18, chiN) - 1.0))
                if sigma < 1e-7:
                    sigma = 1e-7
                if sigma > 0.8:
                    sigma = 0.8

                # update pc
                cc_fac = math.sqrt(cc * (2.0 - cc) * mueff)
                for d in range(dim):
                    pc[d] = (1.0 - cc) * pc[d] + cc_fac * (y_w[d] / max(1e-18, sigma))

                # update Cdiag: rank-one + rank-mu using selected steps in y-space
                for d in range(dim):
                    if not span_ok[d]:
                        Cdiag[d] = 0.20
                        continue
                    old = Cdiag[d]
                    rank_one = pc[d] * pc[d]
                    rank_mu = 0.0
                    invsig = 1.0 / max(1e-18, sigma)
                    for i in range(mu):
                        yi = samples[i][2][d] * invsig
                        rank_mu += w[i] * (yi * yi)
                    Cdiag[d] = (1.0 - c1 - cmu) * old + c1 * rank_one + cmu * rank_mu
                    if Cdiag[d] < 1e-12:
                        Cdiag[d] = 1e-12
                    elif Cdiag[d] > 1e6:
                        Cdiag[d] = 1e6

                # gentle pull of mean toward bestx to sync basins (without breaking CMA update)
                if random.random() < 0.25:
                    bz = to_norm01(bestx)
                    alpha = 0.15 if not near_end else 0.25
                    for d in range(dim):
                        m[d] = (1.0 - alpha) * m[d] + alpha * bz[d]

        # =========================================================
        # C) End-game coordinate search
        # =========================================================
        if very_end and time.time() < deadline:
            # keep it tiny and predictable
            bx, bf = coord_search(bestx, best, max_evals=6 + dim)
            if bf < best:
                best, bestx = bf, bx[:]
                # re-center CMA mean to improved best
                m = to_norm01(bestx)

        # =========================================================
        # D) stagnation handling: partial refresh + CMA reset
        # =========================================================
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0
            # refresh weakest part of DE population, keep elites
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(4, NP // 6)

            for idx in order[keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.65:
                    # sample around best (wider if early)
                    x = bestx[:]
                    scale = (0.22 if not near_end else 0.14)
                    for d in range(dim):
                        if span_ok[d]:
                            x[d] = clamp(x[d] + random.uniform(-1.0, 1.0) * scale * span[d], lo[d], hi[d])
                    pop[idx] = x
                else:
                    pop[idx] = rand_vec()
                fit[idx] = eval_f(pop[idx])

            # reset CMA distribution around best (escape local minima)
            cma_soft_reset_around_best()

            # trim archive
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif archive and random.random() < 0.35:
                for _ in range(max(1, len(archive) // 4)):
                    if archive:
                        archive.pop(random.randrange(len(archive)))

            bi = min(range(NP), key=lambda i: fit[i])
            if fit[bi] < best:
                best = fit[bi]
                bestx = pop[bi][:]

    return best
