import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger anytime bounded minimizer (stdlib-only).

    Key improvements vs your last version:
      - Adds a CMA-ES style "search distribution" (mean + covariance diagonal) to exploit
        while DE explores globally.
      - Alternates DE generations with CMA-like sampling batches (fast progress on smooth-ish problems).
      - Uses rank-based weights + evolution paths + step-size control (robust).
      - Keeps your good pieces: QLHS init, opposition, archive-based DE, cheap local refine.
      - Evaluation cache for exact duplicates (rare but cheap safeguard).

    Returns:
        best (float): best fitness found within max_time seconds
    """

    # ---------------------------
    # Helpers
    # ---------------------------
    def clamp(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    def safe_float(y):
        try:
            y = float(y)
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == -float("inf"):
            return float("inf")
        return y

    # allow func(x) or func(*x)
    def eval_f(x):
        # caching by rounded tuple (stability; avoids huge keys)
        key = tuple(int((x[i] - lo[i]) * inv_quant[i]) if inv_quant[i] > 0 else 0 for i in range(dim))
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

    if dim <= 0:
        try:
            return safe_float(func([]))
        except TypeError:
            return safe_float(func())

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_ok = [span[i] > 0.0 for i in range(dim)]
    # quantization for cache keys (relative to range; prevents giant dict)
    # about ~1e-6 of span, clamped
    inv_quant = []
    for i in range(dim):
        if not span_ok[i]:
            inv_quant.append(0.0)
        else:
            q = span[i] * 1e-6
            if q <= 0.0:
                inv_quant.append(0.0)
            else:
                inv_quant.append(1.0 / q)

    cache = {}

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                x[i] = random.uniform(lo[i], hi[i])
            else:
                x[i] = lo[i]
        return x

    def opposite_vec(x):
        ox = x[:]
        for i in range(dim):
            if span_ok[i]:
                ox[i] = (lo[i] + hi[i]) - x[i]
                if ox[i] < lo[i]: ox[i] = lo[i]
                elif ox[i] > hi[i]: ox[i] = hi[i]
            else:
                ox[i] = lo[i]
        return ox

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

    def cauchy(scale):
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        if t > 30.0: t = 30.0
        if t < -30.0: t = -30.0
        return scale * t

    def local_refine(x, fx, budget):
        if budget <= 0:
            return x, fx
        steps = [0.0] * dim
        for d in range(dim):
            steps[d] = (0.05 * span[d]) if span_ok[d] else 0.0

        bestx = x[:]
        bestf = fx
        used = 0

        for _ in range(3):
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for d in order:
                if used >= budget:
                    return bestx, bestf
                s = steps[d]
                if s <= 0.0:
                    continue
                base = bestx[d]
                for sgn in (-1.0, 1.0):
                    if used >= budget:
                        return bestx, bestf
                    cand = bestx[:]
                    cand[d] = clamp(base + sgn * s, lo[d], hi[d])
                    if cand[d] == base:
                        continue
                    fc = eval_f(cand)
                    used += 1
                    if fc < bestf:
                        bestx, bestf = cand, fc
                        improved = True
                        base = bestx[d]
            if improved:
                for d in range(dim):
                    if span_ok[d]:
                        steps[d] = min(steps[d] * 1.20, 0.30 * span[d])
            else:
                for d in range(dim):
                    steps[d] *= 0.25
        return bestx, bestf

    # ---------------------------
    # Time bookkeeping
    # ---------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------------------
    # Initialization
    # ---------------------------
    NP = max(18, min(96, 12 * dim))
    init = qlhs(NP)
    init += [opposite_vec(x) for x in init]

    pop, fit = [], []
    for x in init:
        if time.time() >= deadline: break
        pop.append(x)
        fit.append(eval_f(x))

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

    best_i = min(range(NP), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    # ---------------------------
    # DE: L-SHADE-ish with archive
    # ---------------------------
    H = 6
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_pos = 0
    archive = []
    arch_max = NP

    # ---------------------------
    # CMA-ES (diagonal) state
    # ---------------------------
    # mean starts at best; sigma as fraction of span
    m = bestx[:]
    sigma = 0.25  # global step factor in normalized coords
    # diagonal covariance in normalized space (start at 1)
    Cdiag = [1.0] * dim
    # evolution paths (normalized space)
    pc = [0.0] * dim
    ps = [0.0] * dim

    # settings derived from dim
    lam = max(10, min(40, 4 + int(3.0 * math.sqrt(dim))))
    mu = max(2, lam // 2)
    # log weights
    ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(ws)
    w = [wi / wsum for wi in ws]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # learning rates (diagonal)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    # expected norm of N(0,I)
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)) if dim > 0 else 1.0

    def to_norm(x):
        # normalized to [0,1] box (and then centered later)
        z = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                z[i] = (x[i] - lo[i]) / span[i]
            else:
                z[i] = 0.0
        return z

    def from_norm(z):
        x = [0.0] * dim
        for i in range(dim):
            if span_ok[i]:
                x[i] = lo[i] + z[i] * span[i]
            else:
                x[i] = lo[i]
        return x

    # ---------------------------
    # Main loop: alternate DE and CMA batches + occasional local refine
    # ---------------------------
    stall = 0
    last_best = best
    max_stall = max(10, 3 + dim)
    gen = 0

    while time.time() < deadline:
        gen += 1
        tl = deadline - time.time()
        near_end = tl <= 0.25 * max_time

        # ---------------------------
        # DE generation
        # ---------------------------
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

            # CR approx normal via sum of uniforms
            CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() - 1.5)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = -1.0
            for _ in range(6):
                F = mu_f + cauchy(0.1)
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

            # reflection bounds
            for d in range(dim):
                if not span_ok[d]:
                    u[d] = lo[d]
                    continue
                if u[d] < lo[d]:
                    u[d] = lo[d] + (lo[d] - u[d])
                    if u[d] > hi[d]:
                        u[d] = lo[d] + random.random() * span[d]
                elif u[d] > hi[d]:
                    u[d] = hi[d] - (u[d] - hi[d])
                    if u[d] < lo[d]:
                        u[d] = lo[d] + random.random() * span[d]
                if u[d] < lo[d]: u[d] = lo[d]
                elif u[d] > hi[d]: u[d] = hi[d]

            fu = eval_f(u)

            if fu <= fi:
                new_pop[i] = u
                new_fit[i] = fu
                if fu < fi:
                    # archive parent
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
            if wsum_imp <= 0.0:
                ww = [1.0 / len(dF)] * len(dF)
            else:
                ww = [di / wsum_imp for di in dF]

            mcr = 0.0
            for wi, cri in zip(ww, SCR):
                mcr += wi * cri

            num = 0.0
            den = 0.0
            for wi, fi in zip(ww, SF):
                num += wi * fi * fi
                den += wi * fi
            mf = (num / den) if den > 1e-12 else MF[mem_pos]

            MCR[mem_pos] = mcr
            MF[mem_pos] = mf
            mem_pos = (mem_pos + 1) % H

        # ---------------------------
        # CMA diagonal batch (exploitation + rotation-insensitive scaling)
        # ---------------------------
        # Use remaining time; keep batch small near end.
        if time.time() < deadline:
            # update mean toward current best (helps if DE found new basin)
            m = bestx[:]

            # sample in normalized space around mean
            mz = to_norm(m)
            # produce lambda samples
            samples = []
            for _ in range(lam if not near_end else max(6, lam // 2)):
                if time.time() >= deadline:
                    break
                # gaussian via Box-Muller per-dim (cheap enough)
                z = [0.0] * dim
                for d in range(dim):
                    if not span_ok[d]:
                        z[d] = 0.0
                        continue
                    u1 = random.random()
                    u2 = random.random()
                    g = math.sqrt(-2.0 * math.log(max(1e-12, u1))) * math.cos(2.0 * math.pi * u2)
                    z[d] = g

                # y = sigma * sqrt(Cdiag) * z
                yz = [0.0] * dim
                xz = [0.0] * dim
                for d in range(dim):
                    if not span_ok[d]:
                        yz[d] = 0.0
                        xz[d] = 0.0
                    else:
                        sd = math.sqrt(max(1e-18, Cdiag[d]))
                        yz[d] = sigma * sd * z[d]
                        # mean + step, then clamp to [0,1]
                        v = mz[d] + yz[d]
                        if v < 0.0: v = 0.0
                        elif v > 1.0: v = 1.0
                        xz[d] = v

                x = from_norm(xz)
                fx = eval_f(x)
                samples.append((fx, x, yz))

                if fx < best:
                    best = fx
                    bestx = x[:]
                    m = x[:]
                    mz = to_norm(m)

            if len(samples) >= mu:
                samples.sort(key=lambda t: t[0])
                # recombination in normalized space: mz_new = mz + sum w_i * y_i
                yz_mean = [0.0] * dim
                for i in range(mu):
                    wi = w[i]
                    yzi = samples[i][2]
                    for d in range(dim):
                        yz_mean[d] += wi * yzi[d]

                # update evolution path ps (in normalized space)
                # ps = (1-cs)*ps + sqrt(cs(2-cs)mueff) * (yz_mean / sigma) / sqrt(Cdiag)
                c_fac = math.sqrt(cs * (2.0 - cs) * mueff)
                for d in range(dim):
                    if not span_ok[d]:
                        ps[d] = 0.0
                        continue
                    sd = math.sqrt(max(1e-18, Cdiag[d]))
                    ps[d] = (1.0 - cs) * ps[d] + c_fac * (yz_mean[d] / max(1e-18, sigma)) / sd

                # sigma update
                ps_norm = math.sqrt(sum(v * v for v in ps))
                sigma *= math.exp((cs / ds) * (ps_norm / max(1e-18, chiN) - 1.0))
                if sigma < 1e-6: sigma = 1e-6
                if sigma > 0.8: sigma = 0.8

                # update pc
                cc_fac = math.sqrt(cc * (2.0 - cc) * mueff)
                for d in range(dim):
                    pc[d] = (1.0 - cc) * pc[d] + cc_fac * (yz_mean[d] / max(1e-18, sigma))

                # update Cdiag (rank-one + rank-mu)
                # C = (1-c1-cmu)*C + c1*pc^2 + cmu*sum w_i*(y_i/sigma)^2
                for d in range(dim):
                    if not span_ok[d]:
                        Cdiag[d] = 1.0
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
                    if Cdiag[d] > 1e6:
                        Cdiag[d] = 1e6

        # ---------------------------
        # Periodic local refine on best
        # ---------------------------
        if time.time() < deadline and (near_end or gen % 10 == 0):
            budget = 8 + 2 * dim
            bx, bf = local_refine(bestx, best, budget)
            if bf < best:
                best, bestx = bf, bx
                m = bx[:]

        # ---------------------------
        # Stagnation handling: diversify part of pop + reset sigma a bit
        # ---------------------------
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(4, NP // 6)

            for idx in order[keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.60:
                    # around best
                    x = bestx[:]
                    sc = 0.18
                    for d in range(dim):
                        if span_ok[d]:
                            x[d] = clamp(x[d] + random.uniform(-1.0, 1.0) * sc * span[d], lo[d], hi[d])
                    pop[idx] = x
                else:
                    pop[idx] = rand_vec()
                fit[idx] = eval_f(pop[idx])

            # small covariance / step refresh
            sigma = min(0.35, max(0.08, sigma * 1.25))
            for d in range(dim):
                Cdiag[d] = min(5.0, max(0.2, Cdiag[d]))

            # prune archive a bit
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif archive and random.random() < 0.3:
                for _ in range(max(1, len(archive) // 4)):
                    if archive:
                        archive.pop(random.randrange(len(archive)))

            best_i = min(range(NP), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                bestx = pop[best_i][:]

    return best
