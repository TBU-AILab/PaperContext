import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only) with stronger performance than the
    provided CMA-ES-like baselines on many bounded continuous problems.

    Key improvements vs your best (#1):
      1) Two-level search:
         - Global exploration via low-discrepancy seeding (Halton) + random
         - Main engine: (a) DE/rand/1/bin style differential evolution (very robust globally)
           (b) Diagonal CMA-ES-like refinement started from the best DE elite
      2) Better bounded handling:
         - Reflection into bounds (avoids clamp bias)
      3) Targeted local improvement:
         - Lightweight coordinate pattern-search polish on the best-so-far
      4) Strong restart & diversity controls:
         - Jittered re-initialization of worst individuals, adaptive DE parameters,
           and periodic refinement restarts.

    Returns:
        best (float): best objective value found within max_time seconds.
    """

    # ---------------------------- guards ---------------------------- #
    if dim <= 0:
        return float("inf")
    if max_time is None or max_time <= 0:
        # still do one eval if possible
        max_time = 1e-6

    # ---------------------------- bounds ---------------------------- #
    bnds = []
    for lo, hi in bounds:
        lo = float(lo); hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        bnds.append((lo, hi))
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    ranges = [hi - lo for (lo, hi) in bnds]
    active = sum(1 for r in ranges if r > 0.0)
    mean_range = (sum(r for r in ranges if r > 0.0) / active) if active else 1.0
    if mean_range <= 0.0:
        mean_range = 1.0

    start = time.time()
    deadline = start + float(max_time)

    # ---------------------------- helpers ---------------------------- #
    def reflect_into(v, lo, hi):
        if hi <= lo:
            return lo
        r = hi - lo
        x = (v - lo) % (2.0 * r)
        if x > r:
            x = 2.0 * r - x
        return lo + x

    def repair(x):
        return [reflect_into(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]

    def safe_eval(x):
        xx = repair(x)
        try:
            y = func(xx)
            if y is None:
                return float("inf"), xx
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf"), xx
            return y, xx
        except Exception:
            return float("inf"), xx

    def randn():
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def norm(v):
        return math.sqrt(sum(t * t for t in v))

    # ---------------------------- Halton seeding ---------------------------- #
    def first_primes(k):
        primes = []
        n = 2
        while len(primes) < k:
            is_p = True
            r = int(n ** 0.5)
            for p in primes:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(n)
            n += 1
        return primes

    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index, bases):
        return [vdc(index, b) for b in bases]

    def from_unit(u01):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            if r <= 0.0:
                x[i] = lo
            else:
                x[i] = lo + u01[i] * r
        return x

    def random_point():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bnds[i]
            if hi <= lo:
                x[i] = lo
            else:
                x[i] = lo + random.random() * (hi - lo)
        return x

    # ---------------------------- local polish ---------------------------- #
    def local_polish(x0, f0, t_end):
        x = x0[:]
        f = f0
        step = [0.06 * (ranges[i] if ranges[i] > 0 else 1.0) for i in range(dim)]
        min_step = [max(1e-12, 1e-9 * (ranges[i] if ranges[i] > 0 else 1.0)) for i in range(dim)]
        budget = 18 + 6 * dim
        used = 0

        while used < budget and time.time() < t_end:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if used >= budget or time.time() >= t_end:
                    break
                if ranges[i] <= 0.0 or step[i] <= min_step[i]:
                    continue

                s = step[i]
                xp = x[:]; xp[i] += s
                fp, xp = safe_eval(xp); used += 1

                xm = x[:]; xm[i] -= s
                fm, xm = safe_eval(xm); used += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    improved = True

            if improved:
                for i in range(dim):
                    step[i] *= 1.25
            else:
                for i in range(dim):
                    step[i] *= 0.55
                if all(step[i] <= min_step[i] or ranges[i] <= 0.0 for i in range(dim)):
                    break
        return x, f

    # ---------------------------- DE population init ---------------------------- #
    best = float("inf")
    best_x = None

    bases = first_primes(dim)
    hal_idx = 1

    # Population size: moderate but time-safe
    NP = max(18, 10 + 4 * int(math.log(dim + 2.0)))
    # If dimension is large, cap a bit to keep eval count reasonable
    NP = min(NP, 60)

    pop = []
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if random.random() < 0.65:
            x = from_unit(halton_point(hal_idx, bases))
            hal_idx += 1
        else:
            x = random_point()
        f, x = safe_eval(x)
        pop.append([x, f])
        if f < best:
            best, best_x = f, x

    # ---------------------------- DE loop ---------------------------- #
    # Parameter ranges
    F_min, F_max = 0.45, 0.95
    CR_min, CR_max = 0.10, 0.95

    # Stagnation/restart controls
    last_best = best
    no_improve_gens = 0
    gen = 0

    # Refinement (CMA-like) state, created on demand
    refine_state = None

    def init_refine(center_x):
        # Diagonal CMA-ES-like refinement initialized near center_x
        n = dim
        lam = max(10, 4 + int(6 * math.log(n + 1.0)))
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w) if mu > 0 else 1.0
        w = [wi / wsum for wi in w]
        mu_eff = 1.0 / sum(wi * wi for wi in w) if mu > 0 else 1.0

        c_sigma = min(0.35, (mu_eff + 2.0) / (n + mu_eff + 5.0))
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1e-12)) - 1.0) + c_sigma
        c_c = min(0.45, (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n))
        c1 = 0.12 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(0.30, 0.18 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        if c_mu < 0.0:
            c_mu = 0.0

        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)) if n > 1 else 1.0

        state = {
            "m": center_x[:],
            "sigma": 0.20 * mean_range,
            "diag": [1.0] * n,
            "p_sigma": [0.0] * n,
            "p_c": [0.0] * n,
            "lam": lam,
            "mu": mu,
            "w": w,
            "mu_eff": mu_eff,
            "c_sigma": c_sigma,
            "d_sigma": d_sigma,
            "c_c": c_c,
            "c1": c1,
            "c_mu": c_mu,
            "chi_n": chi_n,
        }
        # keep sigma sane
        if state["sigma"] <= 1e-15:
            state["sigma"] = 1.0
        return state

    def refine_step(state):
        nonlocal best, best_x
        n = dim
        lam = state["lam"]
        mu = state["mu"]
        m = state["m"]
        sigma = state["sigma"]
        diag = state["diag"]
        w = state["w"]

        # mirrored sampling
        popr = []
        half = lam // 2
        for _ in range(half):
            if time.time() >= deadline:
                break
            z = [randn() for _j in range(n)]
            for sgn in (1.0, -1.0):
                y = [diag[j] * (sgn * z[j]) for j in range(n)]
                x = [m[j] + sigma * y[j] for j in range(n)]
                f, x = safe_eval(x)
                popr.append((f, x, y))
                if f < best:
                    best, best_x = f, x

        if len(popr) < max(2, mu):
            return state

        popr.sort(key=lambda t: t[0])

        # recompute weighted mean
        m_new = [0.0] * n
        for i in range(mu):
            wi = w[i]
            xi = popr[i][1]
            for j in range(n):
                m_new[j] += wi * xi[j]

        # y_w
        y_w = [0.0] * n
        for i in range(mu):
            wi = w[i]
            yi = popr[i][2]
            for j in range(n):
                y_w[j] += wi * yi[j]

        # p_sigma
        cs = state["c_sigma"]
        coeff = math.sqrt(cs * (2.0 - cs) * state["mu_eff"])
        for j in range(n):
            state["p_sigma"][j] = (1.0 - cs) * state["p_sigma"][j] + coeff * y_w[j]

        # sigma
        pnorm = norm(state["p_sigma"])
        sigma *= math.exp((cs / state["d_sigma"]) * (pnorm / (state["chi_n"] + 1e-12) - 1.0))

        # p_c
        cc = state["c_c"]
        h_sigma = 1.0 if pnorm < (1.9 + 0.35 * n) * state["chi_n"] else 0.0
        coeff_c = h_sigma * math.sqrt(cc * (2.0 - cc) * state["mu_eff"])
        for j in range(n):
            state["p_c"][j] = (1.0 - cc) * state["p_c"][j] + coeff_c * y_w[j]

        # diag update
        c1 = state["c1"]
        c_mu = state["c_mu"]
        for j in range(n):
            var = diag[j] * diag[j]
            var *= (1.0 - c1 - c_mu)
            var += c1 * (state["p_c"][j] * state["p_c"][j])
            if c_mu > 0.0:
                s = 0.0
                for i in range(mu):
                    yi = popr[i][2]
                    s += w[i] * (yi[j] * yi[j])
                var += c_mu * s
            if var < 1e-30:
                var = 1e-30
            diag[j] = math.sqrt(var)

        # keep sigma reasonable
        if sigma < 1e-14 * mean_range:
            sigma = 1e-14 * mean_range
        elif sigma > 2.5 * mean_range:
            sigma = 2.5 * mean_range

        state["m"] = m_new
        state["sigma"] = sigma
        state["diag"] = diag
        return state

    while time.time() < deadline:
        gen += 1

        # Occasionally polish current best
        if best_x is not None and random.random() < 0.08:
            t_end = min(deadline, time.time() + max(0.006, 0.016 + 0.0015 * dim))
            xp, fp = local_polish(best_x, best, t_end)
            if fp < best:
                best, best_x = fp, xp

        # Occasionally run a few refinement steps (CMA-like) near best
        if best_x is not None and (refine_state is None or random.random() < 0.10):
            refine_state = init_refine(best_x)

        if refine_state is not None and random.random() < 0.35:
            # small refinement burst, time-safe
            burst = 1 if dim > 30 else 2
            for _ in range(burst):
                if time.time() >= deadline:
                    return best
                refine_state = refine_step(refine_state)

        # Sort population by fitness (for elitism and reinit)
        pop.sort(key=lambda t: t[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]

        # Adaptive DE params (jDE style)
        F = F_min + (F_max - F_min) * random.random()
        CR = CR_min + (CR_max - CR_min) * random.random()
        if dim <= 2:
            CR = max(CR, 0.7)

        # Evolve one generation
        for i in range(NP):
            if time.time() >= deadline:
                return best

            # choose r1,r2,r3 distinct and != i
            idxs = list(range(NP))
            # lightweight selection without building large lists repeatedly:
            # sample until distinct
            r1 = r2 = r3 = i
            while r1 == i:
                r1 = random.randrange(NP)
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(NP)

            x1 = pop[r1][0]
            x2 = pop[r2][0]
            x3 = pop[r3][0]
            xi = pop[i][0]

            # mutation: v = x1 + F*(x2-x3)
            v = [x1[j] + F * (x2[j] - x3[j]) for j in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim) if dim > 0 else 0
            u = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    u[j] = v[j]

            fu, u = safe_eval(u)
            if fu <= pop[i][1]:
                pop[i][0] = u
                pop[i][1] = fu
                if fu < best:
                    best, best_x = fu, u

        # Stagnation and diversity injection
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= 8:
            no_improve_gens = 0
            # Reinitialize a fraction of worst individuals around:
            # - best (exploit) and
            # - Halton/random (explore)
            pop.sort(key=lambda t: t[1])
            k = max(2, NP // 4)
            for t in range(NP - k, NP):
                if time.time() >= deadline:
                    return best
                if best_x is not None and random.random() < 0.60:
                    # jitter around best
                    x = best_x[:]
                    for j in range(dim):
                        r = ranges[j]
                        if r > 0.0:
                            x[j] = x[j] + (0.10 + 0.20 * random.random()) * r * randn()
                    x = repair(x)
                else:
                    if random.random() < 0.55:
                        x = from_unit(halton_point(hal_idx, bases))
                        hal_idx += 1
                    else:
                        x = random_point()
                f, x = safe_eval(x)
                pop[t][0] = x
                pop[t][1] = f
                if f < best:
                    best, best_x = f, x

            # reset refinement near best after restart
            if best_x is not None:
                refine_state = init_refine(best_x)

    return best
