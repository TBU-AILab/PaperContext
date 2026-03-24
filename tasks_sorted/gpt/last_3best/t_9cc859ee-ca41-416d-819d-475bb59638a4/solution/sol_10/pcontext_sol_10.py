import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-limited black-box minimizer (stdlib-only).

    Key improvements vs the provided DE-only attempt:
      - Hybrid portfolio: (1) CMA-ES style search in active subspace (fast local/global)
                          (2) JADE-style DE fallback/booster (robust global exploration)
      - Cheap surrogate-free "guided sampling": adaptive sampling around best + elites
      - Better time management: phases (init -> CMA/DE cycles -> aggressive polish late)
      - Stronger local polish: adaptive coordinate + small stochastic steps (1+1-ES-ish)
      - Evaluation cache with time-varying quantization to reduce duplicate calls
      - Handles fixed dimensions (zero span) and uses reflection to respect bounds
    Returns:
      best fitness (float)
    """

    t_end = time.time() + float(max_time)
    if dim <= 0:
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

    def cauchy():
        u = rand01()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---------------- bound handling: reflection in [0,1] ----------------
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
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            z[i] = v
        return z

    # ---------------- evaluation cache ----------------
    cache = {}
    n_eval = 0

    def q_for_key():
        # coarser early => fewer duplicates evaluated; finer later => more precision
        if n_eval < 300:
            return 3e-6
        if n_eval < 2500:
            return 1.2e-6
        return 5e-7

    def z_key(z):
        q = q_for_key()
        inv = 1.0 / q
        return tuple(int(z[i] * inv + 0.5) if active[i] else 0 for i in range(dim))

    best = float("inf")
    best_z = [0.5 if active[i] else 0.0 for i in range(dim)]

    def eval_z(z):
        nonlocal n_eval, best, best_z
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
                if active[j]:
                    z[j] = (perms[j][i] + rand01()) / n
                else:
                    z[j] = 0.0
            pts.append(z)
        return pts

    def opposite(z):
        return [1.0 - z[i] if active[i] else 0.0 for i in range(dim)]

    # keep init modest for time; still diverse
    init_n = max(18, min(120, 12 + 6 * dim))
    init_pts = lhs_like(init_n)
    init_pts += [opposite(z) for z in init_pts]
    for _ in range(min(40, 3 * dim + 18)):
        init_pts.append([rand01() if active[i] else 0.0 for i in range(dim)])

    for z in init_pts:
        if time.time() >= t_end:
            return best
        eval_z(z[:])

    # ---------------- utilities ----------------
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(dot(a, a))

    # ---------------- Local polish: adaptive coordinate + stochastic steps ----------------
    step = [0.18 if active[i] else 0.0 for i in range(dim)]
    step_min = 1e-10
    step_max = 0.45

    def polish(budget=12):
        nonlocal best, best_z, step
        if budget <= 0:
            return
        z0 = best_z[:]
        f0 = best
        used = 0

        # 1) small gaussian steps (subspace)
        for _ in range(min(5, budget)):
            if time.time() >= t_end:
                return
            z = z0[:]
            k = 1 + int(rand01() * min(len(act_idx), 6))
            sub = random.sample(act_idx, k)
            for j in sub:
                z[j] += 0.6 * step[j] * gauss()
            reflect01_inplace(z)
            fz = eval_z(z)
            used += 1
            if fz < best:
                best, best_z = fz, z[:]
                z0, f0 = z, fz
            if used >= budget:
                return

        # 2) coordinate pattern steps
        order = act_idx[:]
        random.shuffle(order)
        for j in order:
            if time.time() >= t_end or used >= budget:
                return
            s = step[j]
            if s <= step_min:
                continue

            improved = False
            best_local_f = f0
            best_local_z = None

            for dj in (-s, +s, -0.5 * s, +0.5 * s):
                if time.time() >= t_end or used >= budget:
                    break
                z = z0[:]
                z[j] += dj
                reflect01_inplace(z)
                if z_key(z) == z_key(z0):
                    continue
                fz = eval_z(z)
                used += 1
                if fz < best_local_f:
                    best_local_f = fz
                    best_local_z = z[:]
                if fz < best:
                    best, best_z = fz, z[:]

            if best_local_z is not None and best_local_f < f0:
                z0, f0 = best_local_z, best_local_f
                step[j] = min(step_max, step[j] * 1.25)
                improved = True
            if not improved:
                step[j] = max(step_min, step[j] * 0.70)

    # ---------------- CMA-ES (active dims) - lightweight diagonal + rank-one-ish ----------------
    # This is a pragmatic, time-friendly variant: diagonal covariance (sep-CMA).
    # Often very strong for moderate/high dim with tight budgets.
    def sep_cmaes_round(mu_state, budget_evals):
        """
        mu_state: dict carrying ('m','sigma','c','ps','pc','diagC','gen')
        budget_evals: max evaluations to spend in this round
        Updates mu_state in-place and uses eval_z to update global best.
        """
        if budget_evals <= 0:
            return

        # unpack
        m = mu_state["m"]           # active-dim mean (list length adim)
        sigma = mu_state["sigma"]
        ps = mu_state["ps"]
        pc = mu_state["pc"]
        diagC = mu_state["diagC"]
        gen0 = mu_state["gen"]

        # parameters
        lam = mu_state["lam"]
        mu = lam // 2
        # weights (log)
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w)

        cc = (4.0 + mu_eff / adim) / (adim + 4.0 + 2.0 * mu_eff / adim)
        cs = (mu_eff + 2.0) / (adim + mu_eff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((adim + 2.0) ** 2 + mu_eff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (adim + 1.0)) - 1.0) + cs

        chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))
        epsC = 1e-18

        used = 0
        while used < budget_evals and time.time() < t_end:
            gen0 += 1

            # sample
            pop = []
            fpop = []
            ypop = []  # in active coords
            for _ in range(lam):
                if used >= budget_evals or time.time() >= t_end:
                    break
                y = [0.0] * adim
                x01 = best_z[:]  # full dim template
                for t, j in enumerate(act_idx):
                    y[t] = math.sqrt(max(epsC, diagC[t])) * gauss()
                    x01[j] = m[t] + sigma * y[t]
                reflect01_inplace(x01)
                fy = eval_z(x01)
                used += 1
                pop.append(x01)
                fpop.append(fy)
                ypop.append(y)

            if not fpop:
                break

            # select best mu
            idx = list(range(len(fpop)))
            idx.sort(key=lambda i: fpop[i])
            # recombination in y-space
            y_w = [0.0] * adim
            for k in range(min(mu, len(idx))):
                i = idx[k]
                wi = w[k]
                yi = ypop[i]
                for t in range(adim):
                    y_w[t] += wi * yi[t]

            m_old = m[:]
            for t in range(adim):
                m[t] = m_old[t] + sigma * y_w[t]

            # update ps (in sep-CMA: use y_w / sqrt(diagC))
            c_fac = math.sqrt(cs * (2.0 - cs) * mu_eff)
            for t in range(adim):
                ps[t] = (1.0 - cs) * ps[t] + c_fac * (y_w[t] / math.sqrt(max(epsC, diagC[t])))

            psn = norm(ps)
            hsig = 1.0 if (psn / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen0)) / chiN) < (1.4 + 2.0 / (adim + 1.0)) else 0.0

            # update pc
            c_fac2 = math.sqrt(cc * (2.0 - cc) * mu_eff)
            for t in range(adim):
                pc[t] = (1.0 - cc) * pc[t] + hsig * c_fac2 * y_w[t]

            # update diagC
            # diagC = (1-c1-cmu)diagC + c1*(pc^2 + (1-hsig)*cc(2-cc)*diagC) + cmu*sum(w_i*y_i^2)
            for t in range(adim):
                diagC[t] *= (1.0 - c1 - cmu)

            # rank-one
            if hsig < 0.5:
                adj = cc * (2.0 - cc)
                for t in range(adim):
                    diagC[t] += c1 * (pc[t] * pc[t] + adj * diagC[t])
            else:
                for t in range(adim):
                    diagC[t] += c1 * (pc[t] * pc[t])

            # rank-mu
            for k in range(min(mu, len(idx))):
                i = idx[k]
                wi = w[k]
                yi = ypop[i]
                for t in range(adim):
                    diagC[t] += cmu * wi * (yi[t] * yi[t])

            # update sigma
            sigma *= math.exp((cs / damps) * (psn / chiN - 1.0))
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 0.8:
                sigma = 0.8

            # small safety clamp for mean
            for t in range(adim):
                if m[t] < -0.5:
                    m[t] = -0.5
                elif m[t] > 1.5:
                    m[t] = 1.5

            # occasional quick polish when very close to end
            if (t_end - time.time()) < 0.12 * max_time and (gen0 % 4 == 0):
                polish(budget=8)

        # pack back
        mu_state["m"] = m
        mu_state["sigma"] = sigma
        mu_state["ps"] = ps
        mu_state["pc"] = pc
        mu_state["diagC"] = diagC
        mu_state["gen"] = gen0

    # Initialize sep-CMA around current best
    m0 = [best_z[j] for j in act_idx]
    mu_state = {
        "m": m0[:],
        "sigma": 0.28,
        "ps": [0.0] * adim,
        "pc": [0.0] * adim,
        "diagC": [1.0] * adim,
        "gen": 0,
        "lam": max(10, min(48, 6 + 2 * adim + int(3.0 * math.log(adim + 1.0))))
    }

    # ---------------- JADE-style DE as a booster ----------------
    NP = max(18, min(90, 14 + 4 * dim))
    # build initial DE population from cached/best-centered samples
    pop = []
    fit = []

    # seed with best and perturbations
    pop.append(best_z[:])
    fit.append(best)
    while len(pop) < NP and time.time() < t_end:
        z = best_z[:]
        for j in act_idx:
            z[j] += 0.35 * gauss()
        reflect01_inplace(z)
        fz = eval_z(z)
        pop.append(z)
        fit.append(fz)

    # if still short, random fill
    while len(pop) < NP and time.time() < t_end:
        z = [rand01() if active[i] else 0.0 for i in range(dim)]
        fz = eval_z(z)
        pop.append(z)
        fit.append(fz)

    mu_F = 0.55
    mu_CR = 0.50
    c_adapt = 0.10
    p_best_rate = 0.20
    arc = []
    arc_max = NP

    def de_one_generation():
        nonlocal mu_F, mu_CR, best, best_z
        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])
        pcount = max(2, int(p_best_rate * NP))
        top_idx = idx_sorted[:pcount]

        sF = []
        sCR = []

        # slightly more exploration if stuck or early
        for i in range(NP):
            if time.time() >= t_end:
                return

            CRi = mu_CR + (0.12 * gauss())
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            Fi = mu_F + (0.18 * cauchy())
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = mu_F + (0.18 * cauchy())
                tries += 1
            if Fi <= 0.0: Fi = 0.10
            if Fi > 1.0: Fi = 1.0

            xi = pop[i]
            pbest = pop[random.choice(top_idx)]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            x_r1 = pop[r1]

            # r2 from pop U archive
            x_r2 = None
            if arc and rand01() < 0.35:
                x_r2 = arc[random.randrange(len(arc))]
            else:
                r2 = random.randrange(NP)
                while r2 == i or r2 == r1:
                    r2 = random.randrange(NP)
                x_r2 = pop[r2]

            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if rand01() < CRi or j == jrand:
                    u[j] = v[j]

            reflect01_inplace(u)
            fu = eval_z(u)

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

        # adapt
        if sF:
            sumF = 0.0
            sumF2 = 0.0
            for v in sF:
                sumF += v
                sumF2 += v * v
            lehmerF = (sumF2 / sumF) if sumF > 0.0 else mu_F
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmerF
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(sCR) / float(len(sCR)))
        else:
            mu_F = min(0.95, mu_F * 1.05 + 0.01)
            mu_CR = min(1.0, mu_CR + 0.03)

        if mu_F < 0.05: mu_F = 0.05
        if mu_F > 0.95: mu_F = 0.95
        if mu_CR < 0.0: mu_CR = 0.0
        if mu_CR > 1.0: mu_CR = 1.0

    # ---------------- main time-sliced loop ----------------
    last_best = best
    stagn = 0
    iter_no = 0

    while time.time() < t_end:
        iter_no += 1

        # update CMA mean to current best occasionally (helps with restarts implicitly)
        if iter_no % 4 == 0:
            mu_state["m"] = [best_z[j] for j in act_idx]

        # allocate a small evaluation budget to CMA-ES chunk
        time_left = t_end - time.time()
        if time_left <= 0:
            break

        # chunk sizes: earlier more global (DE), later more local (CMA + polish)
        if time_left > 0.55 * max_time:
            cma_budget = 8 + min(18, 2 * adim)
        elif time_left > 0.20 * max_time:
            cma_budget = 10 + min(26, 3 * adim)
        else:
            cma_budget = 12 + min(34, 4 * adim)

        sep_cmaes_round(mu_state, budget_evals=cma_budget)

        # DE booster every other iteration, unless very late
        if (t_end - time.time()) > 0.10 * max_time and (iter_no % 2 == 0):
            de_one_generation()

        # track stagnation and inject diversity if needed
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if stagn > 10 and time.time() < t_end:
            # quick shake: sample a few guided points around best with larger radius
            for _ in range(2):
                if time.time() >= t_end:
                    break
                z = best_z[:]
                rad = 0.18 + 0.25 * rand01()
                for j in act_idx:
                    z[j] += rad * gauss()
                reflect01_inplace(z)
                fz = eval_z(z)
                # replace worst DE member if it helps population quality
                wi = max(range(NP), key=lambda k: fit[k])
                if fz < fit[wi]:
                    pop[wi] = z[:]
                    fit[wi] = fz
                if fz < best:
                    best = fz
                    best_z = z[:]
                    stagn = 0

        # aggressive polish near end
        if (t_end - time.time()) < 0.18 * max_time:
            polish(budget=10)

    return best
