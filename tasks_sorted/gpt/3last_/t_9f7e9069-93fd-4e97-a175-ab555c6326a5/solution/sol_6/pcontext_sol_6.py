import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Hybrid time-bounded minimizer (no external libs).

    Designed to improve over the provided algorithms by combining:
      1) multi-start seeding (LHS-like + opposites),
      2) fast diagonal-CMA-ES core (sep-CMA) for efficient local progress,
      3) DE/rand/1/bin style differential-evolution injections for rugged landscapes,
      4) bandit-like adaptation of operator rates based on recent success,
      5) very cheap local coordinate refinement around the incumbent.

    Returns: best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]
    active = [i for i in range(dim) if not fixed[i]]
    adim = len(active)

    # ---------- helpers ----------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def reflect_to_bounds(x):
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

    def clamp01(u):
        if u < 0.0:
            return 0.0
        if u > 1.0:
            return 1.0
        return u

    def vec_norm(v):
        return math.sqrt(max(0.0, sum(a*a for a in v)))

    def sample_heavy():
        # mixture: mostly Gaussian, sometimes Cauchy-like for long jumps
        if random.random() < 0.10:
            u = random.random()
            return math.tan(math.pi * (u - 0.5))  # Cauchy(0,1)
        return random.gauss(0.0, 1.0)

    if adim == 0:
        x0 = [lo[i] for i in range(dim)]
        return safe_eval(x0)

    avg_span = sum(span[i] for i in active) / max(1, adim)
    base = avg_span if avg_span > 0.0 else 1.0

    # ---------- archive (for DE + restarts) ----------
    # keep a small evaluated set: (fitness, xfull)
    ARC_MAX = max(80, min(500, 120 + 40 * adim))
    archive = []

    def arc_add(x, f):
        nonlocal best, best_x
        archive.append((f, x[:]))
        if f < best:
            best = f
            best_x = x[:]

    def arc_trim():
        # keep best fraction + random remainder for diversity
        if len(archive) <= ARC_MAX:
            return
        archive.sort(key=lambda t: t[0])
        keep_best = max(30, ARC_MAX // 2)
        kept = archive[:keep_best]
        # random pick from the rest
        rest = archive[keep_best:]
        while len(kept) < ARC_MAX and rest:
            j = random.randrange(len(rest))
            kept.append(rest.pop(j))
        archive[:] = kept

    # ---------- seeding (LHS-like over active dims) ----------
    best = float("inf")
    best_x = rand_uniform()

    seed_n = min(30 + 5 * adim, 160)
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
        reflect_to_bounds(x)
        f = safe_eval(x)
        arc_add(x, f)

    # ---------- sep-CMA-ES state (active space) ----------
    m_full = best_x[:]
    m = pack_active(m_full)

    # diagonal covariance variances
    Cdiag = [1.0] * adim
    ps = [0.0] * adim
    pc = [0.0] * adim

    # population size
    lam0 = max(10, min(50, 6 + 2 * adim + int(3 * math.log(adim + 1.0))))
    lam = lam0

    def make_weights(mu):
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        w = [wi / s for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)
        return w, mueff

    sigma = 0.30 * base
    gen = 0
    stall = 0
    last_best = best
    restarts = 0

    # local refine step sizes
    ls_step = [0.06 * s for s in span]
    ls_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # Operator rates (adapted)
    # p_de: probability to generate a DE candidate instead of CMA sample
    p_de = 0.12
    p_global = 0.03  # pure uniform injection

    # success stats for simple adaptation
    succ_de = 1.0
    succ_cma = 1.0
    succ_global = 0.5

    def update_rates():
        nonlocal p_de, p_global
        # normalize success -> probabilities (bounded)
        ssum = succ_de + succ_cma + succ_global
        if ssum <= 0.0:
            return
        t_de = succ_de / ssum
        t_gl = succ_global / ssum
        # smooth update
        p_de = 0.85 * p_de + 0.15 * (0.05 + 0.35 * t_de)
        p_global = 0.85 * p_global + 0.15 * (0.01 + 0.10 * t_gl)
        p_de = clamp01(p_de)
        p_global = clamp01(p_global)
        # keep some CMA always
        if p_de + p_global > 0.85:
            scale = 0.85 / (p_de + p_global)
            p_de *= scale
            p_global *= scale

    def de_candidate():
        # DE/rand/1/bin on active dims using archive points
        n = len(archive)
        if n < 6:
            return None
        # choose 3 distinct indices
        idxs = random.sample(range(n), 3)
        xa = archive[idxs[0]][1]
        xb = archive[idxs[1]][1]
        xc = archive[idxs[2]][1]

        F = 0.4 + 0.6 * random.random()  # [0.4,1.0]
        CR = 0.6 + 0.35 * random.random()  # [0.6,0.95]
        trial = best_x[:]  # start from best as base for binomial crossover stability

        jrand = random.randrange(adim)
        for kk, i in enumerate(active):
            if kk == jrand or random.random() < CR:
                v = xa[i] + F * (xb[i] - xc[i])
                trial[i] = v
        # small jitter to avoid duplicates
        if random.random() < 0.25:
            i = random.choice(active)
            trial[i] += random.gauss(0.0, 0.01 * span[i] if span[i] > 0 else 0.01)
        reflect_to_bounds(trial)
        return trial

    while time.time() < deadline:
        gen += 1

        # CMA parameters per generation (depends on lambda)
        mu = max(2, lam // 2)
        weights, mueff = make_weights(mu)

        cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
        cs = (mueff + 2.0) / (adim + mueff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

        # adapt operator rates based on recent history
        if gen % 6 == 0:
            update_rates()

        # build population
        pop = []  # (f, x_act, src) where src in {"cma","de","glob"}
        # more exploration when stalled
        if stall > max(10, 2 * adim):
            p_global_eff = min(0.12, p_global + 0.05)
            p_de_eff = min(0.45, p_de + 0.10)
        else:
            p_global_eff = p_global
            p_de_eff = p_de

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            r = random.random()
            if r < p_global_eff:
                xfull = rand_uniform()
                f = safe_eval(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull), "glob"))
                continue

            if r < p_global_eff + p_de_eff:
                xfull = de_candidate()
                if xfull is None:
                    xfull = rand_uniform()
                f = safe_eval(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull), "de"))
                continue

            # CMA sample
            z = [sample_heavy() for _ in range(adim)]
            y = [math.sqrt(max(1e-30, Cdiag[k])) * z[k] for k in range(adim)]
            xact = [m[k] + sigma * y[k] for k in range(adim)]
            xfull = unpack_active(xact, m_full)
            reflect_to_bounds(xfull)
            f = safe_eval(xfull)
            arc_add(xfull, f)
            pop.append((f, pack_active(xfull), "cma"))

        arc_trim()
        pop.sort(key=lambda t: t[0])

        # success accounting (improvement vs previous best at start of gen)
        if pop[0][0] < last_best - 1e-15:
            # reward the source of the best individual of this gen
            if pop[0][2] == "de":
                succ_de += 1.0
            elif pop[0][2] == "glob":
                succ_global += 1.0
            else:
                succ_cma += 1.0

        # update best/stall
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        # CMA recombination/update using top mu from population that are not global junk:
        # Still safe to use all; but use top mu regardless (works fine).
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

        # rank-mu diagonal update
        y_mu2 = [0.0] * adim
        for i in range(mu):
            xi = pop[i][1]
            wi = weights[i]
            for k in range(adim):
                yik = (xi[k] - old_m[k]) * invsigma
                y_mu2[k] += wi * (yik * yik)

        for k in range(adim):
            Ck = Cdiag[k]
            Ck *= factor
            Ck += c1 * (pc[k] * pc[k])
            if hsig == 0.0:
                Ck += c1 * cc * (2.0 - cc) * Cdiag[k] * 0.2
            Ck += cmu * y_mu2[k]
            if Ck < 1e-30:
                Ck = 1e-30
            elif Ck > 1e30:
                Ck = 1e30
            Cdiag[k] = Ck

        # sigma update
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        # clamp sigma
        if sigma < 1e-16 * base:
            sigma = 1e-16 * base
        if sigma > 3.0 * base + 1e-12:
            sigma = 3.0 * base + 1e-12

        # ----- cheap local refinement when stalled -----
        if stall > max(8, 2 * adim) and time.time() + 0.002 < deadline:
            trials = min(14 + dim, 34)
            improved = False
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                if random.random() < 0.80:
                    d = random.randrange(dim)
                    if fixed[d]:
                        continue
                    sd = max(ls_min[d], ls_step[d])
                    x[d] += sd if random.random() < 0.5 else -sd
                else:
                    # random direction move in active subspace
                    g = [random.gauss(0.0, 1.0) for _ in range(adim)]
                    ng = vec_norm(g)
                    if ng <= 0.0:
                        continue
                    scale = (0.25 + 0.75 * random.random()) * (0.05 * base)
                    for kk, i in enumerate(active):
                        x[i] += scale * (g[kk] / ng)
                reflect_to_bounds(x)
                fx = safe_eval(x)
                arc_add(x, fx)
                if fx < best:
                    # pull mean to new best and reset a bit
                    m_full = best_x[:]
                    m = pack_active(m_full)
                    ps = [0.0] * adim
                    pc = [0.0] * adim
                    improved = True
                    stall = 0
                    break
                else:
                    for d in range(dim):
                        if not fixed[d] and ls_step[d] > ls_min[d]:
                            ls_step[d] *= 0.996
            if improved:
                succ_cma += 0.3  # credit general local improvement

        # ----- restart if heavily stalled -----
        if stall > max(22, 5 * adim) and time.time() + 0.01 < deadline:
            stall = 0
            restarts += 1
            # modest IPOP-like growth
            lam = min(140, int(lam0 * (1.0 + 0.35 * restarts)))
            # reset CMA state around best
            m_full = best_x[:]
            m = pack_active(m_full)
            Cdiag = [1.0] * adim
            ps = [0.0] * adim
            pc = [0.0] * adim
            sigma = max(sigma, 0.55 * base)
            # increase exploration operators a bit after restart
            succ_de += 0.5
            succ_global += 0.2
            # refresh local steps
            for d in range(dim):
                if not fixed[d]:
                    ls_step[d] = max(ls_step[d], 0.05 * span[d])

    return best
