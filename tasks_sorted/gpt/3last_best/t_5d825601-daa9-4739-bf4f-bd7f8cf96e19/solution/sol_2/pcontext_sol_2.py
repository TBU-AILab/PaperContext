import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (stdlib-only) with stronger performance than the
    previous JADE-like DE in many black-box settings:

    Core: L-SHADE-style DE
      - adaptive memories for F and CR (success-history)
      - current-to-pbest/1 mutation
      - external archive (JADE/L-SHADE)
      - linear population size reduction (more exploration early, more exploitation late)

    Extras:
      - cheap "polynomial-ish" mutation fallback (rare) to escape traps
      - opportunistic local search (SPSA-like 2-eval step + a few coordinate probes)
      - robust safe_eval (handles exceptions/NaN/inf)

    Returns: best (minimum) fitness found within time.
    """

    t0 = time.time()

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def timed_out():
        return (time.time() - t0) >= max_time

    def clip_reflect_inplace(x):
        # reflect at bounds (better diversity than hard clip)
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            xi = x[i]
            if xi < lo:
                xi = lo + (lo - xi)
                if xi > hi:
                    xi = lo
            elif xi > hi:
                xi = hi - (xi - hi)
                if xi < lo:
                    xi = hi
            x[i] = xi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # --- choose initial population size (L-SHADE typical: ~18*dim, but cap for time) ---
    # keep it moderate; evaluation budget is unknown.
    NP_init = max(20, 10 * dim)
    NP_init = min(NP_init, 120)
    NP_min = max(8, 4 * dim)
    if NP_min > NP_init:
        NP_min = max(8, NP_init // 2)

    # success-history memories (L-SHADE uses H ~ 5..20)
    H = 8
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    # p-best fraction (typical 0.11)
    p_best_rate = 0.11
    p_best_rate = max(0.05, min(0.25, p_best_rate))

    # archive
    archive = []
    # capacity tied to NP, will be updated as NP reduces
    archive_cap = NP_init

    # initialize population
    pop = [rand_vec() for _ in range(NP_init)]
    fit = [float("inf")] * NP_init

    best = float("inf")
    best_x = None

    for i in range(NP_init):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # --- local search around best: very cheap, only when it seems worthwhile ---
    ls_sigma = 0.08  # normalized
    ls_sigma_min = 1e-5
    ls_sigma_max = 0.25

    def local_search(best_x, best_f):
        nonlocal ls_sigma
        if best_x is None:
            return best_x, best_f

        x = best_x[:]
        f = best_f

        # 1) SPSA-like two-evaluation step (works when dim is large)
        if not timed_out():
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            c = ls_sigma
            a = 0.6 * ls_sigma  # step
            x_plus = x[:]
            x_minus = x[:]
            for d in range(dim):
                step = c * spans[d]
                x_plus[d] += delta[d] * step
                x_minus[d] -= delta[d] * step
            clip_reflect_inplace(x_plus)
            clip_reflect_inplace(x_minus)

            f_plus = safe_eval(x_plus)
            if timed_out():
                return x, f
            f_minus = safe_eval(x_minus)

            if f_plus < float("inf") and f_minus < float("inf"):
                # approximate gradient sign and step
                g = (f_plus - f_minus)
                # move opposite if g positive on average
                x_try = x[:]
                for d in range(dim):
                    x_try[d] -= a * spans[d] * (1.0 if g > 0.0 else -1.0) * delta[d]
                clip_reflect_inplace(x_try)
                f_try = safe_eval(x_try)
                if f_try < f:
                    x, f = x_try, f_try
                    ls_sigma = max(ls_sigma_min, ls_sigma * 0.85)
                else:
                    ls_sigma = min(ls_sigma_max, ls_sigma * 1.08)

        # 2) a few coordinate probes (good when close to optimum)
        if timed_out():
            return x, f
        probes = min(6, dim)
        for _ in range(probes):
            if timed_out():
                break
            d = random.randrange(dim)
            step = ls_sigma * spans[d]
            if step <= 0.0:
                continue
            improved = False
            for sgn in (1.0, -1.0):
                if timed_out():
                    break
                xt = x[:]
                xt[d] += sgn * step
                clip_reflect_inplace(xt)
                ft = safe_eval(xt)
                if ft < f:
                    x, f = xt, ft
                    improved = True
                    break
            if improved:
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.9)

        return x, f

    # small helper: draw F from Cauchy-like around mean (standard in SHADE)
    def sample_F(mu):
        for _ in range(8):
            u = random.random()
            # Cauchy via tan(pi*(u-0.5))
            f = mu + 0.1 * math.tan(math.pi * (u - 0.5))
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        # fallback
        return 1.0 if mu > 1.0 else (0.1 if mu <= 0.0 else mu)

    def sample_CR(mu):
        cr = mu + 0.1 * random.gauss(0.0, 1.0)
        if cr < 0.0:
            return 0.0
        if cr > 1.0:
            return 1.0
        return cr

    # progress tracking for LPSR (linear population size reduction)
    start_time = t0
    end_time = t0 + max_time

    # stagnation triggers
    no_improve_gens = 0
    stagnation_limit = max(12, 4 * dim)

    # --- main loop ---
    gen = 0
    while not timed_out():
        gen += 1

        # compute target NP (linear reduction over time)
        now = time.time()
        if now >= end_time:
            break
        frac = (now - start_time) / (end_time - start_time + 1e-12)  # 0..1
        NP_target = int(round(NP_init - frac * (NP_init - NP_min)))
        if NP_target < NP_min:
            NP_target = NP_min

        NP = len(pop)
        if NP_target < NP:
            # remove worst individuals to reduce NP
            # sort indices by fitness ascending, keep best NP_target
            idx = sorted(range(NP), key=lambda i: fit[i])[:NP_target]
            pop = [pop[i] for i in idx]
            fit = [fit[i] for i in idx]
            NP = NP_target
            # shrink archive cap
            archive_cap = max(NP, NP_init // 2)
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

        # p-best selection pool size
        p_count = max(2, int(math.ceil(p_best_rate * NP)))

        # rank indices once per generation
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # generation success data
        S_F = []
        S_CR = []
        S_df = []

        improved_gen = False

        # prebuild combined pool size for archive usage
        for i in range(NP):
            if timed_out():
                return best

            xi = pop[i]
            fi = fit[i]

            # choose memory slot
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            F = sample_F(muF)
            CR = sample_CR(muCR)

            # choose pbest
            pbest_idx = idx_sorted[random.randrange(p_count)]
            xpbest = pop[pbest_idx]

            # choose r1 from population != i
            while True:
                r1 = random.randrange(NP)
                if r1 != i:
                    break

            # choose r2 from pop U archive, != i, != r1
            use_arch = (archive and random.random() < 0.5)
            if use_arch:
                combined_n = NP + len(archive)
                while True:
                    r2c = random.randrange(combined_n)
                    if r2c == i or r2c == r1:
                        continue
                    if r2c < NP:
                        xr2 = pop[r2c]
                    else:
                        xr2 = archive[r2c - NP]
                    break
            else:
                while True:
                    r2 = random.randrange(NP)
                    if r2 != i and r2 != r1:
                        xr2 = pop[r2]
                        break

            xr1 = pop[r1]

            # mutation: current-to-pbest/1
            vi = [0.0] * dim
            for d in range(dim):
                vi[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

            clip_reflect_inplace(vi)

            # crossover (binomial)
            jrand = random.randrange(dim)
            ui = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    ui[d] = vi[d]
                else:
                    ui[d] = xi[d]

            # rare extra mutation to escape (cheap, bound-safe)
            if random.random() < 0.015:
                d = random.randrange(dim)
                # "polynomial-ish": jump toward a random point
                rr = lows[d] + random.random() * spans[d]
                ui[d] = 0.7 * ui[d] + 0.3 * rr
                clip_reflect_inplace(ui)

            fui = safe_eval(ui)

            if fui <= fi:
                # push parent to archive
                archive.append(xi[:])
                if len(archive) > archive_cap:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = ui
                fit[i] = fui

                df = fi - fui
                if df > 0.0:
                    S_F.append(F)
                    S_CR.append(CR)
                    S_df.append(df)

                if fui < best:
                    best = fui
                    best_x = ui[:]
                    improved_gen = True

        # update memories (SHADE)
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = float(len(S_df))
                weights = [1.0 / wsum] * len(S_df)
            else:
                inv = 1.0 / wsum
                weights = [df * inv for df in S_df]

            # weighted arithmetic mean for CR
            meanCR = 0.0
            for w, cr in zip(weights, S_CR):
                meanCR += w * cr

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            meanF = (num / den) if den != 0.0 else M_F[mem_idx]

            # write into memory cyclically
            M_CR[mem_idx] = min(1.0, max(0.0, meanCR))
            M_F[mem_idx] = min(1.0, max(0.05, meanF))
            mem_idx = (mem_idx + 1) % H

        # opportunistic local search:
        # - more likely later in time, or right after improvements
        if best_x is not None and not timed_out():
            if improved_gen or random.random() < (0.05 + 0.25 * frac):
                bx, bf = local_search(best_x, best)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

        # stagnation handling: partial refresh of worst few (keeps within time)
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= stagnation_limit and not timed_out():
            no_improve_gens = 0
            NP = len(pop)
            k = max(2, NP // 6)  # refresh ~16%
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for wi in worst:
                if timed_out():
                    return best
                pop[wi] = rand_vec()
                fit[wi] = safe_eval(pop[wi])
                if fit[wi] < best:
                    best = fit[wi]
                    best_x = pop[wi][:]

            # also thin archive a bit to keep it diverse
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

    return best
