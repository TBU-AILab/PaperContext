import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib-only).

    Key improvements vs provided code:
      1) Adds a compact *CMA-ES style* optimizer (diagonal covariance, sep-CMA) as
         the main intensification engine. This is typically stronger than DE on
         smooth/medium-noise continuous problems and still robust on many others.
      2) Keeps lightweight *DE/current-to-pbest/1* as a diversification/back-up
         stream (helps on rugged/multimodal landscapes).
      3) Uses a better local search: *1+1 ES with 1/5th success rule* around the
         incumbent and occasional coordinate quadratic refinement.
      4) Strict time control + adaptive budgeting based on measured evaluation time.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---- basics ----
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def repair_reflect(x):
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                y[i] = lo
                continue
            v = y[i]
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            w = hi - lo
            if w > 0 and (v < lo or v > hi):
                v = lo + (v - lo) % w
            y[i] = clamp(v, lo, hi)
        return y

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            sp = spans[i]
            x[i] = lows[i] if sp <= 0.0 else (lows[i] + random.random() * sp)
        return x

    def center_vec():
        return [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    if time.time() >= deadline:
        return evaluate(center_vec())

    # ---- measure eval time ----
    probe_n = 3
    probe_best = float("inf")
    st = time.time()
    for _ in range(probe_n):
        if time.time() >= deadline:
            return probe_best
        f = evaluate(rand_vec())
        if f < probe_best:
            probe_best = f
    eval_time = max(1e-6, (time.time() - st) / float(probe_n))

    def remaining_evals():
        rem = deadline - time.time()
        if rem <= 0:
            return 0
        return max(0, int(rem / max(eval_time, 1e-12)))

    # ---- bounded history (for elites / DE) ----
    HIST_MAX = 900
    hist_x, hist_f = [], []

    def hist_add(x, f):
        hist_x.append(x[:])
        hist_f.append(float(f))
        n = len(hist_x)
        if n > HIST_MAX:
            # keep best 45% + newest 35%
            idx = list(range(n))
            idx.sort(key=lambda i: hist_f[i])
            keep_best = idx[:max(80, int(0.45 * HIST_MAX))]
            keep_new = list(range(max(0, n - int(0.35 * HIST_MAX)), n))
            keep = sorted(set(keep_best + keep_new))
            hist_x[:] = [hist_x[i] for i in keep]
            hist_f[:] = [hist_f[i] for i in keep]

    # ---- initialization (LHS-ish) ----
    rem0 = max(50, remaining_evals())
    NP0 = max(18, 8 + 4 * dim)
    NP0 = min(NP0, max(18, rem0 // 5 if rem0 > 0 else NP0))

    perms = []
    for d in range(dim):
        p = list(range(NP0))
        random.shuffle(p)
        perms.append(p)

    pop = []
    for i in range(NP0):
        x = [0.0] * dim
        for d in range(dim):
            sp = spans[d]
            if sp <= 0:
                x[d] = lows[d]
            else:
                u = (perms[d][i] + random.random()) / float(NP0)
                x[d] = lows[d] + sp * u
        pop.append(x)

    pop[0] = center_vec()
    for k in range(1, min(NP0, 1 + max(3, NP0 // 8))):
        pop[k] = rand_vec()

    fits = [float("inf")] * NP0
    best = float("inf")
    best_x = None

    for i in range(NP0):
        if time.time() >= deadline:
            return best if best < float("inf") else probe_best
        f = evaluate(pop[i])
        fits[i] = f
        hist_add(pop[i], f)
        if f < best:
            best, best_x = f, pop[i][:]

    # ========= sep-CMA-ES (diagonal covariance) =========
    # internal coordinates: z in R^dim (unbounded), mapped to x within bounds by affine+clamp
    # Here we operate directly in x-space with diagonal stddev per dimension.
    # This is not full CMA-ES but is very effective and cheap.

    # initial mean: best of initial pop
    m = best_x[:] if best_x is not None else center_vec()

    # initial sigma per-dimension
    sig = []
    for i in range(dim):
        sp = spans[i]
        sig.append(0.30 * sp if sp > 0 else 0.0)

    # learning rates (sep-CMA style)
    lam = max(8, 4 + int(3 * math.log(dim + 1.0)))
    mu = lam // 2

    # recombination weights (log)
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # step-size adaptation (CSA-like but simplified)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)

    # covariance adaptation (diagonal)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    ps = [0.0] * dim
    pc = [0.0] * dim
    # diagonal covariance in std form
    diagC = [1.0] * dim

    # ===== DE backup (light) =====
    archive = []
    arch_max = NP0

    def pick_excluding(n, exclude):
        for _ in range(32):
            r = random.randrange(n)
            if r not in exclude:
                return r
        for r in range(n):
            if r not in exclude:
                return r
        return 0

    def elite_base():
        n = len(hist_x)
        if n < max(20, 3 * dim):
            return best_x[:] if best_x is not None else rand_vec()
        idx = list(range(n))
        idx.sort(key=lambda i: hist_f[i])
        K = max(6, min(50, n // 5))
        return hist_x[random.choice(idx[:K])][:]

    # ---- quadratic helper (coordinate) ----
    def parabola_minimizer(a, fa, b, fb, c, fc):
        denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)
        if abs(denom) <= 1e-18:
            return None
        num = (b - a) * (b - a) * (fb - fc) - (b - c) * (b - c) * (fb - fa)
        return b - 0.5 * (num / denom)

    # ---- local 1+1 around incumbent ----
    def one_plus_one(x0, f0, budget, ratio):
        if budget <= 0:
            return x0, f0, 0
        x = x0[:]
        fx = f0
        evals = 0
        # start radius tied to remaining time
        base = (0.15 * (1.0 - ratio) + 0.01)
        step = [base * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        succ = 0
        tried = 0

        while evals < budget and time.time() < deadline:
            y = x[:]
            for i in range(dim):
                if spans[i] > 0 and step[i] > 0:
                    y[i] = clamp(y[i] + random.gauss(0.0, step[i]), lows[i], highs[i])
            fy = evaluate(y); evals += 1
            hist_add(y, fy)
            tried += 1
            if fy < fx:
                x, fx = y, fy
                succ += 1
                if fx < best:
                    pass
            # 1/5 success rule every ~10 tries
            if tried >= 10:
                rate = succ / float(tried)
                mult = 1.22 if rate > 0.2 else 0.82
                for i in range(dim):
                    step[i] = max(1e-12, step[i] * mult)
                succ = 0
                tried = 0

            # occasionally do a single coordinate quadratic refine
            if random.random() < 0.10 and time.time() < deadline:
                d = random.randrange(dim)
                if spans[d] > 0:
                    h = max(1e-12, 0.05 * spans[d] * (1.0 - ratio) + 1e-4 * spans[d])
                    b = x[d]
                    a = clamp(b - h, lows[d], highs[d])
                    c = clamp(b + h, lows[d], highs[d])
                    if abs(c - a) > 1e-15:
                        ya = x[:]; ya[d] = a
                        yc = x[:]; yc[d] = c
                        fa = evaluate(ya); evals += 1; hist_add(ya, fa)
                        fc = evaluate(yc); evals += 1; hist_add(yc, fc)
                        xm = parabola_minimizer(a, fa, b, fx, c, fc)
                        if xm is not None:
                            xm = clamp(xm, lows[d], highs[d])
                            if abs(xm - b) > 1e-15:
                                ym = x[:]; ym[d] = xm
                                fm = evaluate(ym); evals += 1; hist_add(ym, fm)
                                if fm < fx:
                                    x, fx = ym, fm
        return x, fx, evals

    # ---- main loop ----
    gen = 0
    last_best = best
    no_improve = 0

    # keep a working pop for DE sampling
    pop_x = [p[:] for p in pop]
    pop_f = fits[:]

    while time.time() < deadline:
        gen += 1
        elapsed = time.time() - t0
        T = max(1e-9, deadline - t0)
        ratio = elapsed / T
        if ratio < 0.0: ratio = 0.0
        if ratio > 1.0: ratio = 1.0

        # decide stream: CMA-ES majority, DE sometimes, and local search periodically
        # If very few evals left, just do local around best.
        rem = remaining_evals()
        if rem <= 0:
            break

        # periodic local improvement
        if best_x is not None and (gen % 5 == 0 or no_improve >= 7):
            budget = min(max(12, 2 * dim), max(12, rem // 8))
            x2, f2, _ = one_plus_one(best_x, best, budget, ratio)
            if f2 < best:
                best, best_x = f2, x2[:]
                m = best_x[:]
                no_improve = 0

        # ---- CMA-ES generation ----
        # if sigma collapsed too much, expand a bit (restart-ish)
        if best_x is not None:
            for i in range(dim):
                if spans[i] > 0 and sig[i] < 1e-12 + 1e-9 * spans[i]:
                    sig[i] = 0.02 * spans[i]

        # sample lambda candidates
        lam_now = lam
        if rem < lam_now + 5:
            lam_now = max(4, rem)

        # candidate list: (f, x, y) where y are standardized steps
        cand = []
        for _ in range(lam_now):
            if time.time() >= deadline:
                break

            # sample step ~ N(0, diagC)
            y = [0.0] * dim
            x = m[:]
            for i in range(dim):
                if spans[i] <= 0.0:
                    x[i] = lows[i]
                    y[i] = 0.0
                else:
                    zi = random.gauss(0.0, 1.0)
                    # sqrt(diagC) scaling
                    yi = zi * math.sqrt(max(1e-30, diagC[i]))
                    y[i] = yi
                    x[i] = x[i] + sig[i] * yi
            x = repair_reflect(x)

            f = evaluate(x)
            hist_add(x, f)
            cand.append((f, x, y))
            if f < best:
                best, best_x = f, x[:]

        if not cand:
            break

        cand.sort(key=lambda t: t[0])
        # update mean
        old_m = m[:]
        m = [0.0] * dim
        for i in range(mu if mu < len(cand) else len(cand)):
            _, x, _ = cand[i]
            wi = w[i] if i < len(w) else 0.0
            for d in range(dim):
                m[d] += wi * x[d]

        # evolution path for sigma (ps)
        # approximate inverse sqrt of C for diagonal: 1/sqrt(diagC)
        diff = [0.0] * dim
        for d in range(dim):
            diff[d] = (m[d] - old_m[d]) / max(1e-18, sig[d])

        for d in range(dim):
            ps[d] = (1.0 - cs) * ps[d] + math.sqrt(cs * (2.0 - cs) * mueff) * (diff[d] / math.sqrt(max(1e-30, diagC[d])))

        # sigma update (global factor applied per-dimension here by scaling sig)
        norm_ps = math.sqrt(sum(v * v for v in ps))
        # exp factor
        sigma_factor = math.exp((cs / ds) * (norm_ps / chiN - 1.0))
        # clamp to avoid blowups
        if sigma_factor < 0.6:
            sigma_factor = 0.6
        elif sigma_factor > 1.8:
            sigma_factor = 1.8
        for d in range(dim):
            sig[d] *= sigma_factor
            # keep within practical bounds
            if spans[d] > 0:
                sig[d] = clamp(sig[d], 1e-16 * spans[d], 0.7 * spans[d])

        # covariance path pc
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) < (1.4 + 2.0 / (dim + 1.0)) * chiN) else 0.0
        for d in range(dim):
            pc[d] = (1.0 - cc) * pc[d] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * diff[d]

        # diagC update using rank-one + rank-mu on standardized steps y
        # We need y_i from selected individuals in *step space* relative to old_m.
        # Use stored y but note mapping includes repair/clamp; still works reasonably in practice.
        for d in range(dim):
            diagC[d] = (1.0 - c1 - cmu) * diagC[d] + c1 * (pc[d] * pc[d])

        # rank-mu
        mcount = mu if mu < len(cand) else len(cand)
        for i in range(mcount):
            _, _, y = cand[i]
            wi = w[i] if i < len(w) else 0.0
            for d in range(dim):
                diagC[d] += cmu * wi * (y[d] * y[d])

        # numerical safety
        for d in range(dim):
            if diagC[d] < 1e-30:
                diagC[d] = 1e-30
            elif diagC[d] > 1e30:
                diagC[d] = 1e30

        # ---- occasional DE diversification/injection ----
        # Maintain a small population from history (best + random recent) for DE moves.
        if len(pop_x) < NP0:
            pop_x.append(rand_vec())
            pop_f.append(float("inf"))

        # refresh pop with some elites and recent points
        if gen % 4 == 1 and len(hist_x) >= 10:
            # take some best and some random recent
            idx = list(range(len(hist_x)))
            idx.sort(key=lambda i: hist_f[i])
            elites = idx[:min(len(idx), max(6, min(NP0 // 2, 18)))]
            recent = list(range(max(0, len(hist_x) - 80), len(hist_x)))
            random.shuffle(recent)
            recent = recent[:max(4, min(NP0 // 3, 12))]
            sel = elites + recent
            sel = sel[:NP0]
            pop_x = [hist_x[i][:] for i in sel]
            pop_f = [hist_f[i] for i in sel]

        if random.random() < (0.25 + 0.20 * (1.0 - ratio)) and rem > 6:
            # one small DE sweep on this pool
            NP = len(pop_x)
            order = list(range(NP))
            order.sort(key=lambda i: pop_f[i])
            p = 0.25 - 0.18 * ratio
            pbest_n = max(2, int(math.ceil(max(0.08, p) * NP)))
            top = order[:pbest_n]

            trials = max(2, min(NP, 6))
            for _ in range(trials):
                if time.time() >= deadline:
                    break
                i = random.randrange(NP)
                xi, fi = pop_x[i], pop_f[i]
                xp = pop_x[random.choice(top)]
                F = clamp(0.5 + 0.3 * random.random(), 0.15, 0.95)
                CR = clamp(0.7 + 0.2 * random.gauss(0.0, 1.0), 0.0, 1.0)

                excl = {i}
                r1 = pick_excluding(NP, excl); excl.add(r1)
                r2 = pick_excluding(NP, excl); excl.add(r2)
                xr1 = pop_x[r1]
                xr2 = pop_x[r2]

                vi = [xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
                ui = xi[:]
                jrand = random.randrange(dim)
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        ui[d] = vi[d]
                ui = repair_reflect(ui)

                fu = evaluate(ui)
                hist_add(ui, fu)
                if fu <= fi:
                    if len(archive) < arch_max:
                        archive.append(xi[:])
                    else:
                        archive[random.randrange(arch_max)] = xi[:]
                    pop_x[i], pop_f[i] = ui, fu
                    if fu < best:
                        best, best_x = fu, ui[:]
                        m = best_x[:]

        # ---- stagnation handling / restart mean near elite ----
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 10 and time.time() < deadline:
            no_improve = 0
            base = elite_base()
            m = base[:]
            # broaden search a bit
            for d in range(dim):
                if spans[d] > 0:
                    sig[d] = max(sig[d], (0.12 + 0.10 * random.random()) * spans[d])
                    diagC[d] = 1.0
            ps = [0.0] * dim
            pc = [0.0] * dim

    return best
