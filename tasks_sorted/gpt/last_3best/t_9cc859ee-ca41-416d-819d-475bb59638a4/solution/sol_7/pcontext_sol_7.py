import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (stdlib-only).

    Main improvements vs previous CMA-ES-only version:
      - Hybrid global+local: multi-start CMA-ES + "DE/rand/1/bin" bursts for global jumps
      - Much faster linear algebra: covariance is kept diagonal (sep-CMA-ES)
        (avoids expensive Jacobi eigen-decomp; enables many more evaluations/time)
      - Better restart logic: mixtures of (best-based) and (random/LHS-based) restarts
      - Smarter final polish: adaptive pattern search + small Gaussian steps
      - Evaluation cache with quantization (avoid wasting calls); reflection bounds
      - Robust to degenerate bounds (zero span dims)

    Returns:
      best fitness (float)
    """

    t_end = time.time() + float(max_time)
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

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
    adim = len(act_idx)

    if adim == 0:
        x = [lo[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    def to_real(z01):
        x = [0.0] * dim
        for i in range(dim):
            if active[i]:
                x[i] = lo[i] + z01[i] * span[i]
            else:
                x[i] = lo[i]
        return x

    # ---------------- RNG helpers ----------------
    def rand01():
        return random.random()

    def gauss():
        # Box-Muller
        u1 = rand01()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = rand01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---------------- bound handling (reflection in [0,1]) ----------------
    def reflect01_inplace(z):
        for i in range(dim):
            if not active[i]:
                z[i] = 0.0
                continue
            v = z[i]
            if v < 0.0 or v > 1.0:
                v = v % 2.0
                if v > 1.0:
                    v = 2.0 - v
            # numerical safety
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            z[i] = v
        return z

    # ---------------- evaluation cache + best tracking ----------------
    best = float("inf")
    best_z = [0.5] * dim

    # quantization step (dynamic: coarser early -> finer later)
    n_eval = 0
    cache = {}

    def qstep(ne):
        if ne < 150:
            return 2e-6
        if ne < 1200:
            return 8e-7
        return 3e-7

    def z_key(z):
        q = qstep(n_eval)
        inv = 1.0 / q
        return tuple(int(v * inv + 0.5) for v in z)

    def eval_z(z):
        nonlocal best, best_z, n_eval
        reflect01_inplace(z)
        k = z_key(z)
        if k in cache:
            return cache[k]
        fx = float(func(to_real(z)))
        cache[k] = fx
        n_eval += 1
        if fx < best:
            best = fx
            best_z = z[:]
        return fx

    # ---------------- initialization: LHS-like + opposition + random ----------------
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
        o = z[:]
        for i in range(dim):
            o[i] = 1.0 - o[i] if active[i] else 0.0
        return o

    seed_n = max(16, min(96, 12 + 5 * dim))
    seeds = lhs_like(seed_n)
    seeds += [opposite(z) for z in seeds]
    for _ in range(min(40, 2 * seed_n)):
        seeds.append([rand01() if active[i] else 0.0 for i in range(dim)])

    for z in seeds:
        if time.time() >= t_end:
            return best
        eval_z(z[:])

    # ---------------- helper: build a working population around best + some diversity ----------------
    def make_pop(NP, center=None, spread=0.25):
        pop = []
        if center is None:
            center = [rand01() if active[i] else 0.0 for i in range(dim)]
        # half around center, half random
        for _ in range(NP // 2):
            z = center[:]
            for j in act_idx:
                z[j] += spread * gauss()
            reflect01_inplace(z)
            pop.append(z)
        while len(pop) < NP:
            pop.append([rand01() if active[i] else 0.0 for i in range(dim)])
        return pop

    # ---------------- DE burst (cheap global jump mechanism) ----------------
    def de_burst(pop, fit, iters=1):
        # classic DE/rand/1/bin with reflection bounds
        NP = len(pop)
        if NP < 5:
            return pop, fit
        for _ in range(iters):
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            # slightly bias toward using better points as base
            for i in range(NP):
                if time.time() >= t_end:
                    return pop, fit
                # choose indices
                a = random.choice(idx_sorted[: max(2, NP // 2)])
                b = random.randrange(NP)
                while b == a:
                    b = random.randrange(NP)
                c = random.randrange(NP)
                while c == a or c == b:
                    c = random.randrange(NP)

                F = 0.5 + 0.3 * gauss()
                if F < 0.05:
                    F = 0.05
                if F > 0.95:
                    F = 0.95
                CR = 0.5 + 0.2 * gauss()
                if CR < 0.0:
                    CR = 0.0
                if CR > 1.0:
                    CR = 1.0

                xa, xb, xc = pop[a], pop[b], pop[c]
                v = pop[i][:]
                for j in act_idx:
                    v[j] = xa[j] + F * (xb[j] - xc[j])

                u = pop[i][:]
                jrand = random.randrange(dim)
                for j in range(dim):
                    if (not active[j]):
                        u[j] = 0.0
                    elif (rand01() < CR) or (j == jrand):
                        u[j] = v[j]
                reflect01_inplace(u)
                fu = eval_z(u)
                if fu <= fit[i]:
                    pop[i] = u
                    fit[i] = fu
        return pop, fit

    # ---------------- local polish: pattern search + gaussian micro-steps ----------------
    polish_step = 0.15  # in normalized units

    def polish(budget_evals=20):
        nonlocal polish_step, best, best_z
        step = polish_step
        used = 0

        # gaussian micro-steps
        for _ in range(min(6, budget_evals // 2)):
            if time.time() >= t_end or used >= budget_evals:
                break
            z = best_z[:]
            for j in act_idx:
                z[j] += 0.25 * step * gauss()
            reflect01_inplace(z)
            eval_z(z)
            used += 1

        # coordinate/pattern search with adaptive step
        improved_any = False
        order = act_idx[:]
        random.shuffle(order)
        for j in order:
            if time.time() >= t_end or used >= budget_evals:
                break
            z0 = best_z[:]
            f0 = best
            # try +/- step
            z_minus = z0[:]
            z_plus = z0[:]
            z_minus[j] -= step
            z_plus[j] += step
            reflect01_inplace(z_minus)
            reflect01_inplace(z_plus)

            f_minus = eval_z(z_minus); used += 1
            if time.time() >= t_end or used >= budget_evals:
                break
            f_plus = eval_z(z_plus); used += 1

            if best < f0 - 1e-15:
                improved_any = True

        # adapt step
        if improved_any:
            polish_step = min(0.35, polish_step * 1.10)
        else:
            polish_step = max(1e-8, polish_step * 0.65)

    # ---------------- sep-CMA-ES core (diagonal covariance) ----------------
    # This is much cheaper than full CMA-ES and typically allows more evaluations,
    # often improving time-bounded performance.
    restart = 0
    base_lambda = max(10, min(50 + 4 * adim, 8 + 3 * adim + int(3 * math.log(adim + 1.0)))))

    # initial mean from best seed
    m = [best_z[j] for j in act_idx]

    # diagonal standard deviations (in normalized space)
    # start moderately wide, then shrink with restarts
    def init_sigmas(rst):
        s0 = 0.25 * (0.85 ** rst)
        s0 = max(0.03, min(0.35, s0))
        return [s0] * adim

    # sep-CMA learning rates
    def sep_params(lam, mu_eff):
        # conservative defaults
        cs = (mu_eff + 2.0) / (adim + mu_eff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (adim + 1.0)) - 1.0) + cs
        cc = (4.0 + mu_eff / adim) / (adim + 4.0 + 2.0 * mu_eff / adim)
        # diagonal covariance update rates
        c1 = 2.0 / ((adim + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((adim + 2.0) ** 2 + mu_eff))
        return cs, damps, cc, c1, cmu

    # E||N(0,I)||
    chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

    # We'll maintain a small working population for DE bursts too
    work_NP = max(10, min(60, 10 + 3 * dim))
    work_pop = make_pop(work_NP, center=best_z, spread=0.22)
    work_fit = [eval_z(z[:]) for z in work_pop]

    while time.time() < t_end:
        lam = int(base_lambda * (2 ** restart))
        lam = max(8, min(lam, 80 + 5 * adim))
        mu = lam // 2

        # log weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w)

        cs, damps, cc, c1, cmu = sep_params(lam, mu_eff)

        # state
        sigma = max(1e-12, 0.30 * (0.80 ** restart))
        if restart == 0:
            sigma = max(sigma, 0.20)

        diagC = [1.0] * adim
        ps = [0.0] * adim
        pc = [0.0] * adim
        sigmas = init_sigmas(restart)  # per-dim initial scales

        # occasional restart mean: mix best and random
        if restart > 0:
            if rand01() < 0.65:
                m = [best_z[j] for j in act_idx]
            else:
                zr = [rand01() if active[i] else 0.0 for i in range(dim)]
                m = [zr[j] for j in act_idx]

        no_improve = 0
        best_at_start = best
        gen = 0

        while time.time() < t_end:
            gen += 1

            # late-stage: polish more
            time_left = t_end - time.time()
            if time_left < 0.15 * max_time and (gen % 3 == 0):
                polish(budget_evals=10)

            # occasional DE burst to escape local basins / improve diversity
            if (gen % 8 == 0) and (time_left > 0.05 * max_time):
                # refresh work_pop around current best
                if rand01() < 0.35:
                    work_pop = make_pop(work_NP, center=best_z, spread=0.28)
                    work_fit = [eval_z(z[:]) for z in work_pop]
                work_pop, work_fit = de_burst(work_pop, work_fit, iters=1)
                # inject best work_pop into global best
                bi = min(range(work_NP), key=lambda i: work_fit[i])
                if work_fit[bi] < best:
                    best = work_fit[bi]
                    best_z = work_pop[bi][:]
                    m = [best_z[j] for j in act_idx]
                    no_improve = 0

            # sample population (sep-CMA-ES)
            arx = []   # full-dim points in [0,1]
            arz = []   # standard normal
            ary = []   # y = sqrt(diagC)*z (diagonal)
            fit = []

            # precompute sqrtC
            sqrtC = [math.sqrt(diagC[i]) for i in range(adim)]

            for _ in range(lam):
                if time.time() >= t_end:
                    return best
                z = [gauss() for _ in range(adim)]
                y = [sqrtC[i] * z[i] for i in range(adim)]
                x = best_z[:]  # template
                for t, j in enumerate(act_idx):
                    x[j] = m[t] + sigma * sigmas[t] * y[t]
                reflect01_inplace(x)
                f = eval_z(x)
                arx.append(x)
                arz.append(z)
                ary.append(y)
                fit.append(f)

            # sort
            idx = list(range(lam))
            idx.sort(key=lambda i: fit[i])

            if fit[idx[0]] < best - 1e-15:
                best = fit[idx[0]]
                best_z = arx[idx[0]][:]
                no_improve = 0
            else:
                no_improve += 1

            # recombination
            m_old = m[:]
            y_w = [0.0] * adim
            z_w = [0.0] * adim
            for k in range(mu):
                i = idx[k]
                wi = w[k]
                yi = ary[i]
                zi = arz[i]
                for t in range(adim):
                    y_w[t] += wi * yi[t]
                    z_w[t] += wi * zi[t]
            for t in range(adim):
                m[t] = m_old[t] + sigma * sigmas[t] * y_w[t]

            # update ps (since B=I in diagonal CMA)
            cfac = math.sqrt(cs * (2.0 - cs) * mu_eff)
            for t in range(adim):
                ps[t] = (1.0 - cs) * ps[t] + cfac * z_w[t]

            ps_norm = math.sqrt(sum(v * v for v in ps))
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chiN) < (1.4 + 2.0 / (adim + 1.0)) else 0.0

            # update pc
            cfac2 = math.sqrt(cc * (2.0 - cc) * mu_eff)
            for t in range(adim):
                pc[t] = (1.0 - cc) * pc[t] + hsig * cfac2 * (sigmas[t] * y_w[t])

            # diagonal covariance update
            # diagC = (1-c1-cmu)*diagC + c1*(pc^2 + (1-hsig)*cc*(2-cc)*diagC) + cmu*sum(w_i * (sigmas*y_i)^2)
            old_diagC = diagC[:]
            for t in range(adim):
                diagC[t] = (1.0 - c1 - cmu) * old_diagC[t]

            # rank-one
            if hsig < 0.5:
                adj = cc * (2.0 - cc)
                for t in range(adim):
                    diagC[t] += c1 * (pc[t] * pc[t] + adj * old_diagC[t])
            else:
                for t in range(adim):
                    diagC[t] += c1 * (pc[t] * pc[t])

            # rank-mu
            for k in range(mu):
                i = idx[k]
                wi = w[k]
                yi = ary[i]
                for t in range(adim):
                    v = sigmas[t] * yi[t]
                    diagC[t] += cmu * wi * (v * v)

            # keep diagC sane
            for t in range(adim):
                if diagC[t] < 1e-18:
                    diagC[t] = 1e-18
                elif diagC[t] > 1e6:
                    diagC[t] = 1e6

            # step-size control
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 0.8:
                sigma = 0.8

            # restart / stop conditions for this run
            if no_improve > (16 + 2 * adim):
                break
            if sigma < 1e-10 and no_improve > (8 + adim):
                break

            # if we've improved a lot, keep anchoring mean to best occasionally
            if gen % 10 == 0 and rand01() < 0.25:
                m = [best_z[j] for j in act_idx]

        # if restart made no progress, diversify more next time
        if best >= best_at_start - 1e-15:
            # try a quick global shake by evaluating a few randoms
            shake = max(4, min(20, 2 + dim))
            for _ in range(shake):
                if time.time() >= t_end:
                    return best
                zr = [rand01() if active[i] else 0.0 for i in range(dim)]
                eval_z(zr)

        restart += 1

    return best
