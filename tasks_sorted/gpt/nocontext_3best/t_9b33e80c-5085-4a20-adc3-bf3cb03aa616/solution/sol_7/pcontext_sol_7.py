import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Deadline-safe hybrid optimizer (self-contained; stdlib only).

    Key improvements vs your best (#1):
      - Adds a *true* local optimizer: bounded Nelder–Mead (reflection/expansion/
        contraction/shrink) started repeatedly from the best point found.
      - Keeps global robustness with a compact L-SHADE/JADE-style DE (current-to-pbest/1
        + archive + linear pop-size reduction).
      - Uses time-aware evaluation budgeting and always preserves an end-game slice
        for local search (important for scoring well under tight max_time).
      - Better restart/diversity injection using Halton + opposition + worst replacement.

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    lo = [0.0] * dim
    hi = [0.0] * dim
    span = [0.0] * dim
    for i, (a, b) in enumerate(bounds):
        a = float(a); b = float(b)
        if b < a:
            raise ValueError("Each bound must be (low, high) with high >= low")
        lo[i] = a; hi[i] = b; span[i] = b - a

    # Degenerate case: single point
    if all(s == 0.0 for s in span):
        x0 = [lo[i] for i in range(dim)]
        return float(func(x0))

    # -------------------- utilities --------------------
    random.seed()

    def now():
        return time.time()

    def time_frac():
        return (now() - t0) / max(1e-12, float(max_time))

    def clamp(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    def clamp_vec(x):
        y = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                y[d] = lo[d]
            else:
                y[d] = clamp(x[d], lo[d], hi[d])
        return y

    def repair_midpoint(u, parent):
        # Midpoint repair then clamp (stable for DE)
        for d in range(dim):
            if span[d] == 0.0:
                u[d] = lo[d]
            else:
                if u[d] < lo[d]:
                    u[d] = 0.5 * (lo[d] + parent[d])
                elif u[d] > hi[d]:
                    u[d] = 0.5 * (hi[d] + parent[d])
                if u[d] < lo[d]: u[d] = lo[d]
                elif u[d] > hi[d]: u[d] = hi[d]
        return u

    def l2_dist2(a, b):
        s = 0.0
        for d in range(dim):
            t = a[d] - b[d]
            s += t * t
        return s

    # Halton for injections
    def first_primes(n):
        primes, k = [], 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r: break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton_value(index, base):
        f, r, i = 1.0, 0.0, index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    bases = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                x[d] = lo[d]
            else:
                x[d] = lo[d] + halton_value(k, bases[d]) * span[d]
        return x

    def rand_point():
        x = [0.0] * dim
        for d in range(dim):
            x[d] = lo[d] if span[d] == 0.0 else (lo[d] + random.random() * span[d])
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                xo[d] = lo[d]
            else:
                xo[d] = clamp(lo[d] + hi[d] - x[d], lo[d], hi[d])
        return xo

    # RNG helpers
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_normal(mu, sigma):
        return mu + sigma * randn()

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # time-aware evaluation
    eval_calls = 0
    avg_eval = None  # EMA

    def eval_f(x):
        nonlocal eval_calls, avg_eval
        t1 = now()
        fx = float(func(x))
        t2 = now()
        dt = max(1e-6, t2 - t1)
        eval_calls += 1
        if avg_eval is None:
            avg_eval = dt
        else:
            a = 0.08
            avg_eval = (1.0 - a) * avg_eval + a * dt
        return fx

    def remaining_eval_budget(margin_frac=0.08):
        rem = max(0.0, deadline - now())
        rem2 = max(0.0, rem - margin_frac * float(max_time))
        est = avg_eval if avg_eval is not None else 1e-3
        return int(rem2 / max(est, 1e-6))

    # -------------------- bounded Nelder–Mead (local) --------------------
    def nelder_mead_local(x0, f0, time_limit, frac):
        """
        Bounded Nelder–Mead with restartable small simplex.
        Returns improved (x, f). Uses only function evaluations.
        """
        # If many fixed dims, NM still works but simplex degenerates; handle by tiny jitter.
        # Initial step size shrinks with time
        base = 0.18 * (1.0 - 0.90 * frac) + 0.003
        steps = [base * span[d] for d in range(dim)]
        for d in range(dim):
            if span[d] > 0.0:
                mn = 1e-14 * span[d]
                if steps[d] < mn:
                    steps[d] = mn

        # Build simplex of size dim+1
        simplex = [list(x0)]
        fvals = [f0]

        # Create vertices along coordinate directions (with small random sign)
        for d in range(dim):
            if now() >= time_limit:
                return simplex[0], fvals[0]
            v = list(x0)
            if span[d] == 0.0:
                v[d] = lo[d]
            else:
                sgn = -1.0 if random.random() < 0.5 else 1.0
                v[d] = clamp(v[d] + sgn * steps[d], lo[d], hi[d])
            v = clamp_vec(v)
            fv = eval_f(v)
            simplex.append(v)
            fvals.append(fv)

        # Coefficients
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

        # Iteration budget based on remaining evals
        # Each iter uses ~1-2 evals (sometimes more on shrink).
        max_iters = max(10, min(220, 18 * dim))
        it = 0

        while it < max_iters and now() < time_limit:
            it += 1
            # order
            idx = list(range(len(simplex)))
            idx.sort(key=lambda i: fvals[i])
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            bestx, bestf = simplex[0], fvals[0]
            worstx, worstf = simplex[-1], fvals[-1]
            second_worstf = fvals[-2]

            # centroid of all but worst
            centroid = [0.0] * dim
            inv = 1.0 / float(dim)
            for i in range(dim):
                xi = simplex[i]
                for d in range(dim):
                    centroid[d] += xi[d]
            for d in range(dim):
                centroid[d] *= inv

            # reflect
            xr = [0.0] * dim
            for d in range(dim):
                xr[d] = centroid[d] + alpha * (centroid[d] - worstx[d])
            xr = clamp_vec(xr)
            fr = eval_f(xr)
            if now() >= time_limit:
                # keep best found so far
                idxb = min(range(len(fvals)), key=lambda i: fvals[i])
                return simplex[idxb], fvals[idxb]

            if fr < bestf:
                # expand
                xe = [0.0] * dim
                for d in range(dim):
                    xe[d] = centroid[d] + gamma * (xr[d] - centroid[d])
                xe = clamp_vec(xe)
                fe = eval_f(xe) if now() < time_limit else float("inf")
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
                continue

            if fr < second_worstf:
                simplex[-1], fvals[-1] = xr, fr
                continue

            # contraction
            if fr < worstf:
                # outside contraction
                xc = [0.0] * dim
                for d in range(dim):
                    xc[d] = centroid[d] + rho * (xr[d] - centroid[d])
                xc = clamp_vec(xc)
                fc = eval_f(xc)
                if fc <= fr:
                    simplex[-1], fvals[-1] = xc, fc
                    continue
            else:
                # inside contraction
                xc = [0.0] * dim
                for d in range(dim):
                    xc[d] = centroid[d] - rho * (centroid[d] - worstx[d])
                xc = clamp_vec(xc)
                fc = eval_f(xc)
                if fc < worstf:
                    simplex[-1], fvals[-1] = xc, fc
                    continue

            # shrink towards best
            for i in range(1, len(simplex)):
                if now() >= time_limit:
                    break
                xi = simplex[i]
                xs = [0.0] * dim
                for d in range(dim):
                    xs[d] = bestx[d] + sigma * (xi[d] - bestx[d])
                xs = clamp_vec(xs)
                simplex[i] = xs
                fvals[i] = eval_f(xs)

        idxb = min(range(len(fvals)), key=lambda i: fvals[i])
        return simplex[idxb], fvals[idxb]

    # -------------------- initialize population --------------------
    NP_max = max(18, min(96, 14 + 5 * dim))
    NP_min = max(8,  min(40, 7 + 2 * dim))

    pop, fit = [], []
    best = float("inf")
    bestx = None

    for i in range(NP_max):
        if now() >= deadline:
            return best
        if i < (2 * NP_max) // 3:
            x = halton_point(hal_k); hal_k += 1
        else:
            x = rand_point()

        fx = eval_f(x)
        if now() < deadline:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            if fxo < fx:
                x, fx = xo, fxo

        pop.append(x); fit.append(fx)
        if fx < best:
            best = fx; bestx = x

    # -------------------- DE (L-SHADE-ish) --------------------
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []

    last_best = best
    stagn = 0

    # reserve time for endgame local search
    endgame_frac = 0.22  # last ~22% time prioritize NM + small random polish

    while now() < deadline:
        frac = time_frac()
        if frac >= 1.0:
            break

        # If close to endgame, switch emphasis to local search loops
        if frac >= (1.0 - endgame_frac) and bestx is not None:
            # Spend remaining time on repeated NM restarts + tiny gaussian samples
            while now() < deadline:
                frac2 = time_frac()
                # quick gaussian poke
                if remaining_eval_budget(margin_frac=0.03) <= 5:
                    break
                # local slice
                slice_time = min(deadline, now() + max(0.03 * max_time, 0.18 * (deadline - now())))
                bx, bf = nelder_mead_local(bestx, best, slice_time, frac2)
                if bf < best:
                    best, bestx = bf, bx
                else:
                    # jitter restart around best
                    cand = [0.0] * dim
                    scale = (0.06 * (1.0 - 0.7 * frac2) + 0.0015)
                    for d in range(dim):
                        if span[d] == 0.0:
                            cand[d] = lo[d]
                        else:
                            cand[d] = clamp(bestx[d] + (scale * span[d]) * randn(), lo[d], hi[d])
                    fc = eval_f(cand)
                    if fc < best:
                        best, bestx = fc, cand
            return best

        # linear population reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min
        if len(pop) > target_NP:
            idx = list(range(len(pop)))
            idx.sort(key=lambda i: fit[i])
            keep = idx[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            if len(archive) > target_NP:
                archive = random.sample(archive, target_NP)

        NP = len(pop)
        if NP < 4:
            return best

        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        # pbest fraction
        p = min(0.35, max(2.0 / NP, 0.10 + 0.22 * frac))
        pbest_count = max(2, int(round(p * NP)))
        elites = idx_sorted[:pbest_count]

        union = pop + archive

        # evaluation budgeting
        rem_budget = remaining_eval_budget(margin_frac=0.09)
        if rem_budget < 2 * NP:
            n_trials = 1
            allow_second = False
        else:
            n_trials = 2 if dim <= 24 else 3
            allow_second = (rem_budget > 3 * NP and random.random() < 0.30)

        S_F, S_CR, weights = [], [], []

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            CRi = rand_normal(muCR, 0.1)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            Fi = -1.0
            tries = 0
            while Fi <= 0.0:
                Fi = rand_cauchy(muF, 0.1)
                tries += 1
                if tries > 10:
                    Fi = max(1e-3, muF)
                    break
            if Fi > 1.0:
                Fi = 1.0

            xpbest = pop[elites[random.randrange(len(elites))]]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            xr2 = None
            for _ in range(25):
                cand = union[random.randrange(len(union))]
                if cand is xi or cand is xr1:
                    continue
                xr2 = cand
                break
            if xr2 is None:
                xr2 = union[random.randrange(len(union))]

            trials = []
            for _t in range(n_trials):
                use_ctpbest = (random.random() < (0.80 - 0.20 * (1.0 - frac)))
                v = [0.0] * dim
                if use_ctpbest:
                    for d in range(dim):
                        v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                else:
                    a = random.randrange(NP)
                    b = random.randrange(NP)
                    c = random.randrange(NP)
                    while b == a:
                        b = random.randrange(NP)
                    while c == a or c == b:
                        c = random.randrange(NP)
                    xa, xb, xc = pop[a], pop[b], pop[c]
                    for d in range(dim):
                        v[d] = xa[d] + Fi * (xb[d] - xc[d])

                jrand = random.randrange(dim)
                u = [0.0] * dim
                for d in range(dim):
                    if span[d] == 0.0:
                        u[d] = lo[d]
                    else:
                        u[d] = v[d] if (random.random() < CRi or d == jrand) else xi[d]
                u = repair_midpoint(u, xi)

                db = l2_dist2(u, bestx) if bestx is not None else 0.0
                dpar = l2_dist2(u, xi)
                score = db + (0.02 + 0.10 * (1.0 - frac)) * dpar
                trials.append((score, u))

            trials.sort(key=lambda z: z[0])
            u1 = trials[0][1]
            fu1 = eval_f(u1)

            chosen_u, chosen_fu = u1, fu1
            if allow_second and len(trials) > 1 and now() < deadline:
                u2 = trials[1][1]
                fu2 = eval_f(u2)
                if fu2 < chosen_fu:
                    chosen_u, chosen_fu = u2, fu2

            if chosen_fu <= fi:
                # archive
                if len(archive) < NP:
                    archive.append(xi)
                else:
                    archive[random.randrange(NP)] = xi

                pop[i] = chosen_u
                fit[i] = chosen_fu

                if chosen_fu < best:
                    best = chosen_fu
                    bestx = chosen_u

                df = abs(fi - chosen_fu)
                w = df if df > 0.0 else 1.0
                S_F.append(Fi)
                S_CR.append(CRi)
                weights.append(w)

        # update SHADE memories
        if S_F:
            wsum = sum(weights)
            if wsum <= 0.0:
                wsum = 1.0
            num, den, cr = 0.0, 0.0, 0.0
            for Fi, CRi, w in zip(S_F, S_CR, weights):
                num += w * Fi * Fi
                den += w * Fi
                cr += w * CRi
            newF = (num / den) if den != 0.0 else M_F[h_idx]
            newCR = cr / wsum

            if newF < 0.05: newF = 0.05
            if newF > 0.95: newF = 0.95
            if newCR < 0.0: newCR = 0.0
            if newCR > 1.0: newCR = 1.0

            M_F[h_idx] = newF
            M_CR[h_idx] = newCR
            h_idx = (h_idx + 1) % H

        # opportunistic local NM when improving or stagnating (but before endgame)
        if bestx is not None and now() < deadline:
            if (random.random() < (0.03 + 0.18 * frac)) or (stagn >= 6 and random.random() < 0.55):
                if remaining_eval_budget(margin_frac=0.10) > max(18, 2 * dim):
                    slice_time = min(deadline, now() + min(0.10 * max_time, 0.16 * (deadline - now())))
                    bx, bf = nelder_mead_local(bestx, best, slice_time, frac)
                    if bf < best:
                        best, bestx = bf, bx
                        # inject into population (replace worst)
                        worst_i = max(range(NP), key=lambda j: fit[j])
                        pop[worst_i] = list(bestx)
                        fit[worst_i] = best

        # stagnation/diversity injection
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if (stagn >= 10 or (frac < 0.75 and random.random() < 0.03)) and now() < deadline:
            replace_n = max(2, NP // 5)
            worst = list(range(NP))
            worst.sort(key=lambda i: fit[i], reverse=True)
            for t in range(replace_n):
                if now() >= deadline:
                    return best
                j = worst[t]
                if random.random() < 0.60:
                    xnew = halton_point(hal_k); hal_k += 1
                else:
                    xnew = rand_point()
                fnew = eval_f(xnew)
                if now() < deadline and random.random() < 0.55:
                    xo = opposite_point(xnew)
                    fxo = eval_f(xo)
                    if fxo < fnew:
                        xnew, fnew = xo, fxo
                pop[j] = xnew
                fit[j] = fnew
                if fnew < best:
                    best, bestx = fnew, xnew
            archive = []
            stagn = 0

    return best
