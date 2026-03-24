import random
import time
import math

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def clip(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def eval_f(x):
        return float(func(x))

    # --------- scrambled Halton sequence for robust seeding/diversification ----------
    def _vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, r = divmod(n, base)
            denom *= base
            v += r / denom
        return v

    def _is_prime(p):
        if p < 2:
            return False
        if p % 2 == 0:
            return p == 2
        r = int(math.sqrt(p))
        q = 3
        while q <= r:
            if p % q == 0:
                return False
            q += 2
        return True

    primes = []
    p = 2
    while len(primes) < dim:
        if _is_prime(p):
            primes.append(p)
        p += 1

    scramble = [random.random() for _ in range(dim)]

    def halton_point(k):
        kk = k + 1
        x = [0.0] * dim
        for i in range(dim):
            u = _vdc(kk, primes[i]) + scramble[i]
            u -= int(u)
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # --------- init population ----------
    best = float("inf")
    best_x = None

    # slightly larger pop than previous version; helps multimodal problems
    pop_size = max(24, min(110, 14 * dim))
    init_n = max(pop_size, 18 * dim)

    pop, fit = [], []
    k = 0
    while k < init_n and time.time() < deadline:
        x = halton_point(k) if (k < int(0.85 * init_n)) else rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        pop.append(x)
        fit.append(fx)
        k += 1

    if not pop:
        return best

    # keep best pop_size
    if len(pop) > pop_size:
        idx = list(range(len(pop)))
        idx.sort(key=lambda i: fit[i])
        idx = idx[:pop_size]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    # --------- helpers ----------
    def elite_stats_from_indices(elite_idx):
        m = len(elite_idx)
        mu = [0.0] * dim
        for j in elite_idx:
            x = pop[j]
            for d in range(dim):
                mu[d] += x[d]
        invm = 1.0 / m
        for d in range(dim):
            mu[d] *= invm

        var = [0.0] * dim
        for j in elite_idx:
            x = pop[j]
            for d in range(dim):
                dd = x[d] - mu[d]
                var[d] += dd * dd
        for d in range(dim):
            var[d] *= invm

        sigma = [math.sqrt(v) for v in var]
        # keep sigmas sane
        for d in range(dim):
            s = sigma[d] if sigma[d] > 0.0 else 0.25 * spans[d]
            sigma[d] = max(1e-12 * spans[d], min(0.8 * spans[d], s))
        return mu, sigma

    def l2_dist2(a, b):
        s = 0.0
        for d in range(dim):
            t = a[d] - b[d]
            s += t * t
        return s

    # --------- improved local search: coordinate-wise pattern search + ES bursts ----------
    def local_refine(x0, f0, time_budget, sigma_hint):
        endt = min(deadline, time.time() + time_budget)
        x = x0[:]
        f = f0

        # pattern steps per-coordinate (start from hint, then adapt)
        step = [max(1e-12 * spans[d], min(0.25 * spans[d], 0.6 * sigma_hint[d])) for d in range(dim)]
        step_min = [1e-12 * spans[d] for d in range(dim)]

        # lightweight ES sigmas (coupled to step)
        sig = [max(1e-12 * spans[d], 0.35 * step[d]) for d in range(dim)]

        lam = 6 if dim <= 10 else 8 if dim <= 25 else 10

        # which coordinates to scan each pass
        if dim <= 18:
            coords_all = list(range(dim))
        else:
            coords_all = None

        stall = 0
        while time.time() < endt:
            improved_any = False

            # --- coordinate pattern moves (very effective on many benchmarks) ---
            coords = coords_all
            if coords is None:
                m = max(6, dim // 3)
                coords = random.sample(range(dim), m)

            random.shuffle(coords)
            for d in coords:
                if time.time() >= endt:
                    break
                sd = step[d]
                if sd <= step_min[d] * 2.0:
                    continue

                xd = x[d]

                y = x[:]
                y[d] = clip(xd + sd, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved_any = True
                    continue

                y = x[:]
                y[d] = clip(xd - sd, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved_any = True
                    continue

                # if neither direction helped, shrink this coordinate
                step[d] = max(step_min[d], step[d] * 0.72)
                sig[d] = max(step_min[d], sig[d] * 0.8)

            # --- ES burst around current x (captures interactions between coords) ---
            if time.time() < endt:
                best_y = None
                best_fy = f
                # sparse mutation for high-d
                if dim <= 14:
                    m = dim
                else:
                    m = max(3, dim // 3)

                for _ in range(lam):
                    if time.time() >= endt:
                        break
                    y = x[:]
                    idxs = random.sample(range(dim), m) if m < dim else range(dim)
                    for d in idxs:
                        y[d] = clip(y[d] + random.gauss(0.0, 1.0) * sig[d], lows[d], highs[d])
                    fy = eval_f(y)
                    if fy < best_fy:
                        best_fy, best_y = fy, y

                if best_y is not None and best_fy < f:
                    x, f = best_y, best_fy
                    improved_any = True
                    # slightly expand steps when improving
                    for d in range(dim):
                        step[d] = min(0.35 * spans[d], step[d] * 1.08)
                        sig[d] = min(0.35 * spans[d], sig[d] * 1.08)

            if improved_any:
                stall = 0
            else:
                stall += 1

            # stop if clearly converged or too stalled
            if stall >= 6:
                if max(step) <= max(step_min) * 200.0:
                    break
                # reheat a tiny bit to escape micro-stalls
                if time.time() < endt:
                    for d in range(dim):
                        step[d] = min(0.20 * spans[d], step[d] * 1.18)
                        sig[d] = min(0.20 * spans[d], sig[d] * 1.18)
                stall = 0

        return x, f

    # --------- main optimizer: SHADE/JADE-like DE with p-best + archive + L-SHADE pop reduction ----------
    archive = []
    arch_max = pop_size

    mu_F = 0.62
    mu_CR = 0.70
    pbest_rate = 0.18

    # for occasional surrogate-ish bias: pick farther r1/r2 with some probability
    def pick_far_index(base_x, candidates):
        # pick the farthest among a few random candidates (cheap)
        best_j = None
        best_d = -1.0
        trials = 4
        for _ in range(trials):
            j = random.choice(candidates)
            d2 = l2_dist2(base_x, pop[j])
            if d2 > best_d:
                best_d = d2
                best_j = j
        return best_j if best_j is not None else random.choice(candidates)

    it = 0
    no_improve = 0

    # reduce sorting cost: refresh order periodically
    order = list(range(pop_size))
    order.sort(key=lambda j: fit[j])
    rank_refresh = max(3, pop_size // 3)

    # local refine cadence: more frequent than before, but time-sliced
    refine_every = max(14, 4 * dim)

    # population reduction schedule
    min_pop = max(12, min(40, 6 * dim))
    pop0 = pop_size

    while time.time() < deadline:
        it += 1

        if it % rank_refresh == 0:
            order.sort(key=lambda j: fit[j])

        # L-SHADE style: gradually reduce population size over time
        # (keeps exploration early, speeds convergence late)
        frac_time = (time.time() - t0) / max(1e-9, max_time)
        target_pop = int(round(pop0 - (pop0 - min_pop) * min(1.0, frac_time)))
        if target_pop < pop_size:
            # drop worst individuals
            if it % 5 == 0:  # not every iteration
                order.sort(key=lambda j: fit[j])
                keep = order[:target_pop]
                keep_set = set(keep)
                pop = [pop[j] for j in keep]
                fit = [fit[j] for j in keep]
                pop_size = target_pop
                arch_max = pop_size
                if len(archive) > arch_max:
                    # random trim
                    while len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))
                order = list(range(pop_size))
                order.sort(key=lambda j: fit[j])

        # periodic local refinement of best (and sometimes 2nd best)
        if best_x is not None and (it % refine_every == 0):
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            elite_k = max(5, min(pop_size, 2 * dim))
            elite_idx = order[:elite_k]
            mu, sigma = elite_stats_from_indices(elite_idx)

            # spend a small slice; slightly larger late
            budget = min((0.06 + 0.12 * frac_time) * max_time, 0.22 * remaining)
            bx, bf = local_refine(best_x, best, budget, sigma)
            if bf < best:
                best, best_x = bf, bx[:]
                # inject into population replacing worst
                wi = max(range(pop_size), key=lambda j: fit[j])
                archive.append(pop[wi])
                if len(archive) > arch_max:
                    archive.pop(random.randrange(len(archive)))
                pop[wi] = bx[:]
                fit[wi] = bf
                no_improve = 0

        if time.time() >= deadline:
            break

        # elite stats used for occasional injections
        elite_k = max(5, min(pop_size, 2 * dim))
        elite_idx = order[:elite_k]
        mu, sigma = elite_stats_from_indices(elite_idx)

        # pick target
        i = random.randrange(pop_size)
        xi = pop[i]
        fi = fit[i]

        # choose pbest from top p%
        pcount = max(2, int(pbest_rate * pop_size))
        pbest_i = order[random.randrange(pcount)]
        xpbest = pop[pbest_i]

        # sample Fi (cauchy-like), CRi (normal-like)
        # with tighter tails late for exploitation
        scale = 0.12 * (1.0 - 0.6 * min(1.0, frac_time))
        Fi = 0.5
        for _ in range(12):
            u = random.random()
            cand = mu_F + scale * math.tan(math.pi * (u - 0.5))
            if cand > 0.0:
                Fi = cand
                break
        if Fi > 1.0:
            Fi = 1.0
        if Fi < 1e-6:
            Fi = 1e-6

        CRi = mu_CR + random.gauss(0.0, 0.10 * (1.0 - 0.5 * min(1.0, frac_time)))
        if CRi < 0.0:
            CRi = 0.0
        elif CRi > 1.0:
            CRi = 1.0

        # pick r1, r2 (with diversity bias sometimes)
        idxs = list(range(pop_size))
        idxs.remove(i)

        if random.random() < 0.35 and len(idxs) >= 3:
            r1 = pick_far_index(xi, idxs)
            idxs.remove(r1)
            r2 = pick_far_index(xi, idxs)
        else:
            r1 = random.choice(idxs)
            idxs.remove(r1)
            r2 = random.choice(idxs)

        xr1 = pop[r1]
        xr2 = archive[random.randrange(len(archive))] if (archive and random.random() < 0.40) else pop[r2]

        # mutation: current-to-pbest/1
        v = [0.0] * dim
        for d in range(dim):
            vd = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            # bounce-back bounds (often better than pure clip)
            if vd < lows[d]:
                vd = lows[d] + random.random() * (xi[d] - lows[d])
            elif vd > highs[d]:
                vd = highs[d] - random.random() * (highs[d] - xi[d])
            v[d] = vd

        # binomial crossover
        jrand = random.randrange(dim)
        uvec = xi[:]
        for d in range(dim):
            if d == jrand or random.random() < CRi:
                uvec[d] = v[d]

        # occasional injections:
        r = random.random()
        if r < 0.05:
            # gaussian around elite mean
            for d in range(dim):
                uvec[d] = clip(mu[d] + random.gauss(0.0, 1.0) * (0.85 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif r < 0.08 and best_x is not None:
            # best-centered small gaussian
            for d in range(dim):
                uvec[d] = clip(best_x[d] + random.gauss(0.0, 1.0) * (0.45 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif r < 0.10:
            # rare global re-sample (keeps some global exploration)
            uvec = halton_point(random.randrange(1, 2000000))

        fu = eval_f(uvec)

        if fu <= fi:
            # selection
            archive.append(xi)
            if len(archive) > arch_max:
                archive.pop(random.randrange(len(archive)))
            pop[i] = uvec
            fit[i] = fu

            # update means (JADE/SHADE-ish)
            mu_F = 0.92 * mu_F + 0.08 * Fi
            mu_CR = 0.92 * mu_CR + 0.08 * CRi

            if fu < best:
                best, best_x = fu, uvec[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # stagnation control: diversify + widen pbest selection
        if no_improve > (60 + 7 * dim):
            no_improve = 0
            pbest_rate = min(0.6, pbest_rate * 1.35)

            replace = max(1, pop_size // 4)
            worst = sorted(range(pop_size), key=lambda j: fit[j], reverse=True)[:replace]
            for w in worst:
                if time.time() >= deadline:
                    break
                xnew = halton_point(random.randrange(1, 3000000)) if random.random() < 0.65 else rand_point()
                fnew = eval_f(xnew)
                pop[w] = xnew
                fit[w] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # gentle reheat
            mu_F = min(0.9, mu_F + 0.06)
            mu_CR = min(0.9, mu_CR + 0.06)

        # anneal pbest_rate slowly back
        pbest_rate = max(0.12, pbest_rate * 0.9990)

    return best
