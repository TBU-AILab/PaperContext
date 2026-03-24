import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved (stdlib-only) time-bounded minimizer:
    Hybrid = short L-SHADE/JADE DE (with archive + shrinking pop + success-history)
             + CMA-ES-lite diagonal local search on the incumbent
             + aggressive time-aware restarts/injections
             + optional "mirrored" bound repair.

    Rationale for improving your best (~25.50):
    - Your best code is JADE-like DE + coordinate polish; that polish can be weak on rotated/ill-conditioned basins.
    - This version keeps DE for global search but replaces the end-game with a cheap diagonal-covariance
      evolution strategy (CMA-ES-lite) which is typically much stronger than coordinate search while staying stdlib.
    - Uses weighted success-history memories (L-SHADE idea) + archive to reduce premature convergence.

    Returns:
        best (float): best objective value found within max_time.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- basic utilities ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_repair(x):
        # Mirror reflection into bounds (often better than hard-clip for ES steps).
        for i in range(dim):
            L, U = lo[i], hi[i]
            if x[i] < L or x[i] > U:
                if U <= L:
                    x[i] = L
                    continue
                v = x[i]
                w = U - L
                # map to [0, 2w) then reflect
                y = (v - L) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                x[i] = L + y
        return x

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def safe_eval(x):
        try:
            v = func(x)
            if v is None or isinstance(v, complex):
                return float("inf")
            v = float(v)
            if v != v or v == float("inf") or v == -float("inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------------- scrambled Halton seeding ----------------
    def _primes_upto(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        r = int(n ** 0.5)
        for p in range(2, r + 1):
            if sieve[p]:
                start = p * p
                sieve[start:n + 1:p] = [False] * (((n - start) // p) + 1)
        return [i for i, ok in enumerate(sieve) if ok]

    def _first_n_primes(n):
        if n <= 0:
            return []
        ub = max(50, int(n * (math.log(max(3, n)) + math.log(math.log(max(3, n))) + 3)))
        primes = _primes_upto(ub)
        while len(primes) < n:
            ub = int(ub * 1.7) + 10
            primes = _primes_upto(ub)
        return primes[:n]

    primes = _first_n_primes(dim)
    scramble = [random.random() for _ in range(dim)]

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_value(k, primes[i])
            u = (u + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # DE bounce-back repair (towards base) + clip
    def de_repair(trial, base):
        for j in range(dim):
            if trial[j] < lo[j]:
                r = random.random()
                trial[j] = lo[j] + r * (base[j] - lo[j])
            elif trial[j] > hi[j]:
                r = random.random()
                trial[j] = hi[j] - r * (hi[j] - base[j])
        return clip_inplace(trial)

    # ---------------- initialization ----------------
    # Slightly larger initial pop for better coverage; shrinks over time.
    NP0 = max(18, min(90, 14 + 6 * dim))
    NPmin = max(8, min(36, 10 + 2 * dim))
    NP = NP0

    pop, fit = [], []
    k = 1
    while len(pop) < NP and time.time() < deadline:
        if len(pop) % 4 == 0:
            x = rand_point()
        else:
            x = halton_point(k)
            k += 1

        fx = safe_eval(x)
        # opposition try
        if random.random() < 0.65:
            xo = opposite_point(x)
            fo = safe_eval(xo)
            if fo < fx:
                x, fx = xo, fo

        pop.append(list(x))
        fit.append(fx)

    if not pop:
        return float("inf")

    best_idx = min(range(len(pop)), key=lambda i: fit[i])
    best_x = list(pop[best_idx])
    best = fit[best_idx]

    # ---------------- L-SHADE-style DE state ----------------
    archive = []
    Amax = 2 * NP0

    H = max(6, min(28, 6 + dim // 2))
    M_F = [0.5] * H
    M_CR = [0.5] * H
    mem_idx = 0

    def p_fraction(frac_time, NP_):
        pmin = 2.0 / max(2, NP_)
        # more exploration early, more exploitation late
        p = 0.20 - 0.12 * frac_time
        if p < pmin:
            p = pmin
        if p > 0.25:
            p = 0.25
        return p

    # ---------------- CMA-ES-lite (diagonal) polish ----------------
    # Keeps an internal local model around best and runs bursts late in the budget or when DE stagnates.
    class DiagES:
        __slots__ = ("m", "sig", "w", "mu", "lam", "bestf", "bestx", "stall")
        def __init__(self, x, fx):
            self.m = list(x)
            # initial sigma relative to variable scale
            self.sig = [0.15 * scale[i] for i in range(dim)]
            # population sizes (small, fast)
            self.lam = max(6, min(18, 4 + int(2.5 * math.sqrt(dim))))
            self.mu = max(2, self.lam // 2)
            # log weights
            ws = [math.log((self.mu + 0.5) / (i + 1.0)) for i in range(self.mu)]
            s = sum(ws) if ws else 1.0
            self.w = [wi / s for wi in ws]
            self.bestf = fx
            self.bestx = list(x)
            self.stall = 0

        def ask(self):
            x = [0.0] * dim
            for i in range(dim):
                x[i] = self.m[i] + random.gauss(0.0, self.sig[i])
            reflect_repair(x)
            return x

        def tell(self, samples):
            # samples: list of (f, x) sorted ascending
            # update mean
            oldm = self.m
            newm = [0.0] * dim
            for k_ in range(self.mu):
                wk = self.w[k_]
                xk = samples[k_][1]
                for i in range(dim):
                    newm[i] += wk * xk[i]
            self.m = newm

            # diagonal step-size adaptation using success signal
            # if mean moved and best improved -> mildly increase, else decrease
            moved = 0.0
            for i in range(dim):
                d = (newm[i] - oldm[i]) / (scale[i] if scale[i] != 0 else 1.0)
                moved += d * d
            moved = math.sqrt(moved)

            fbest = samples[0][0]
            if fbest < self.bestf:
                self.bestf = fbest
                self.bestx = list(samples[0][1])
                self.stall = 0
            else:
                self.stall += 1

            # global sigma factor
            if moved > 0.02:
                fac = 1.08
            else:
                fac = 0.92

            if self.stall > 4:
                fac *= 0.85

            for i in range(dim):
                self.sig[i] *= fac
                # clamp to reasonable range
                smin = 1e-12 * scale[i]
                smax = 0.6 * scale[i]
                if self.sig[i] < smin:
                    self.sig[i] = smin
                elif self.sig[i] > smax:
                    self.sig[i] = smax

    es = DiagES(best_x, best)

    # ---------------- main loop ----------------
    it = 0
    no_best_improve = 0
    inj_patience = max(90, 30 * dim)

    # Time partition: mostly DE early, more ES later
    while time.time() < deadline:
        it += 1
        now = time.time()
        frac_time = (now - t0) / max(1e-9, float(max_time))

        # shrink population linearly with time
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac_time))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP < NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            if len(archive) > 2 * NP:
                random.shuffle(archive)
                archive = archive[:2 * NP]

        # late-stage ES bursts (stronger than coordinate polish)
        # also trigger when DE stagnates
        if (frac_time > 0.60 and it % max(8, 2 * dim) == 0) or (no_best_improve > inj_patience and frac_time > 0.35):
            # small burst; keep time-aware
            burst = max(1, min(6, int(0.5 * math.sqrt(dim) + 2)))
            for _ in range(burst):
                if time.time() >= deadline:
                    break
                samples = []
                # sample lambda points
                for _s in range(es.lam):
                    if time.time() >= deadline:
                        break
                    x = es.ask()
                    fx = safe_eval(x)
                    samples.append((fx, x))
                if not samples:
                    break
                samples.sort(key=lambda t: t[0])
                es.tell(samples)
                if es.bestf < best:
                    best = es.bestf
                    best_x = list(es.bestx)
                    no_best_improve = 0

        # injections / partial restart if stagnating
        if no_best_improve > inj_patience and time.time() < deadline:
            nrep = max(1, NP // 4)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:nrep]
            for wi in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.5:
                    x = halton_point(k); k += 1
                else:
                    # around incumbent with wider radius
                    x = [best_x[j] + random.gauss(0.0, 0.45 * scale[j]) for j in range(dim)]
                    reflect_repair(x)
                fx = safe_eval(x)
                pop[wi] = list(x)
                fit[wi] = fx
                if fx < best:
                    best = fx
                    best_x = list(x)
                    no_best_improve = 0
            # reset ES center near new best
            es = DiagES(best_x, best)
            no_best_improve = inj_patience // 2

        # --- DE generation (L-SHADE/JADE style) ---
        order = sorted(range(NP), key=lambda i: fit[i])
        p = p_fraction(frac_time, NP)
        p_count = max(2, int(math.ceil(p * NP)))

        union = pop + archive
        union_n = len(union)

        def pick_r(exclude_pop):
            while True:
                r = random.randrange(union_n)
                if r < NP and r in exclude_pop:
                    continue
                return r

        S_F, S_CR, S_df = [], [], []
        improved_gen = False

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            rH = random.randrange(H)
            muF = M_F[rH]
            muCR = M_CR[rH]

            CR = random.gauss(muCR, 0.1)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            u = random.random()
            u = min(1.0 - 1e-12, max(1e-12, u))
            F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
            tries = 0
            while F <= 0.0 and tries < 8:
                u = random.random()
                u = min(1.0 - 1e-12, max(1e-12, u))
                F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # choose pbest from top p_count
            pbest = order[random.randrange(p_count)]
            xp = pop[pbest]

            excl = {i, pbest}
            r1 = pick_r(excl)
            r2 = pick_r(excl)
            while r2 == r1:
                r2 = pick_r(excl)

            xr1 = union[r1]
            xr2 = union[r2]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])
            de_repair(v, xi)

            # crossover
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    uvec[j] = v[j]
                else:
                    uvec[j] = xi[j]
            de_repair(uvec, xi)

            fu = safe_eval(uvec)

            if fu <= fi:
                # archive store replaced parent
                if len(archive) < Amax:
                    archive.append(list(xi))
                else:
                    archive[random.randrange(Amax)] = list(xi)

                pop[i] = uvec
                fit[i] = fu

                S_F.append(F)
                S_CR.append(CR)
                df = (fi - fu)
                if df < 0.0:
                    df = -df
                S_df.append(df if df > 0.0 else 1e-12)

                if fu < best:
                    best = fu
                    best_x = list(uvec)
                    improved_gen = True

        if improved_gen:
            no_best_improve = 0
            # keep ES centered on latest best (helps late-stage)
            if best < es.bestf:
                es = DiagES(best_x, best)
        else:
            no_best_improve += 1

        # update memories
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = 1.0
            mcr = 0.0
            num = 0.0
            den = 0.0
            for Fv, CRv, w in zip(S_F, S_CR, S_df):
                ww = w / wsum
                mcr += ww * CRv
                num += ww * (Fv * Fv)
                den += ww * Fv
            mF = (num / den) if den > 0.0 else M_F[mem_idx]
            M_F[mem_idx] = mF
            M_CR[mem_idx] = mcr
            mem_idx = (mem_idx + 1) % H

        # archive bound
        Amax = max(2 * NP, 2 * NP0)
        if len(archive) > Amax:
            random.shuffle(archive)
            archive = archive[:Amax]

        # very end: one last tiny ES probe if time remains
        if deadline - time.time() < 0.02:
            break

    # last-chance micro-polish (very cheap)
    if time.time() < deadline:
        # a few greedy mirrored Gaussian probes
        for _ in range(min(10, 2 + dim)):
            if time.time() >= deadline:
                break
            x = [best_x[i] + random.gauss(0.0, 0.02 * scale[i]) for i in range(dim)]
            reflect_repair(x)
            fx = safe_eval(x)
            if fx < best:
                best = fx
                best_x = x

    return best
