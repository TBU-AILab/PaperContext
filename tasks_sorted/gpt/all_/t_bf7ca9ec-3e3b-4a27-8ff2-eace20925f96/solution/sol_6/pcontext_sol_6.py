import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (standard library only).

    Main improvements vs the provided DE variants:
      - L-SHADE-ish DE/current-to-pbest/1 with *linear population size reduction*.
      - SHADE success-history memories for F and CR (H memories).
      - External archive for diversity.
      - Lightweight "cheap surrogate" in the final phase: repeated small trust-region
        sampling around the incumbent best with adaptive radius.
      - Stagnation-triggered soft restart of worst individuals.

    Returns: best fitness (float).
    """
    t0 = time.time()
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def time_up():
        return (time.time() - t0) >= max_time

    def eval_f(x):
        return float(func(x))

    # Reflection mapping into [lo, hi] (handles far out-of-range values gracefully)
    def reflect(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            w = hi - lo
            v = y[i] - lo
            m = v % (2.0 * w)
            y[i] = lo + (m if m <= w else (2.0 * w - m))
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return reflect([lows[i] + highs[i] - x[i] for i in range(dim)])

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # --- Population schedule (L-SHADE style) ---
    NP_init = max(20, min(90, 14 + 5 * dim))
    NP_min = max(8, min(25, 6 + 2 * dim))

    # --- Initialize population with a touch of opposition sampling ---
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    for _ in range(NP_init):
        x = rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        if time_up():
            return best

    # A few opposition trials (cheap early coverage boost)
    for _ in range(max(2, NP_init // 6)):
        if time_up():
            return best
        i = random.randrange(len(pop))
        xo = opposite_point(pop[i])
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i], fit[i] = xo, fo
            if fo < best:
                best, best_x = fo, xo[:]

    # --- SHADE memories ---
    H = 10
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0
    p = 0.18  # p-best fraction

    # --- Archive ---
    archive = []
    arch_max = len(pop)

    # --- Stagnation / local trust region sampling ---
    last_improve = time.time()
    patience = max(0.10 * max_time, 0.8)

    # trust region parameters (used increasingly near the end)
    tr_sigma = 0.10  # fraction of span (global start)
    tr_sigma_min = 1e-6
    tr_sigma_max = 0.25
    tr_next = time.time() + max(0.20, 0.12 * max_time)
    tr_batch = max(12, 4 * dim)

    def top_p_indices():
        # NP <= 90: full sort is OK
        order = sorted(range(len(pop)), key=lambda i: fit[i])
        pnum = int(math.ceil(p * len(pop)))
        if pnum < 2:
            pnum = 2
        return order, pnum

    def trust_region_explore():
        nonlocal best, best_x, last_improve, tr_sigma
        if best_x is None:
            return

        improved = False
        # mixture: gaussian all-dim + occasional single-dim heavy kick
        for _ in range(tr_batch):
            if time_up():
                break
            if random.random() < 0.70:
                y = best_x[:]
                sig = tr_sigma
                for d in range(dim):
                    y[d] += random.gauss(0.0, sig * spans[d])
            else:
                y = best_x[:]
                j = random.randrange(dim)
                y[j] += (tr_sigma * spans[j]) * math.tan(math.pi * (random.random() - 0.5))
            y = reflect(y)
            fy = eval_f(y)
            if fy < best:
                best, best_x = fy, y[:]
                last_improve = time.time()
                improved = True

        # adapt radius
        if improved:
            tr_sigma = min(tr_sigma_max, tr_sigma * 1.08)
        else:
            tr_sigma = max(tr_sigma_min, tr_sigma * 0.78)

    # --- Main loop (generations) ---
    gen = 0
    while not time_up():
        gen += 1

        # Increase local probing frequency near end
        elapsed = time.time() - t0
        remaining = max(0.0, max_time - elapsed)
        if time.time() >= tr_next and not time_up():
            trust_region_explore()
            # more frequent later
            tr_next = time.time() + max(0.10, 0.05 * remaining)

        # stagnation -> soft restart of worst chunk + archive trim + radius reset
        if (time.time() - last_improve) >= patience and not time_up():
            n = len(pop)
            k = max(2, n // 3)
            worst = sorted(range(n), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                if time_up():
                    break
                if random.random() < 0.45:
                    x = rand_point()
                else:
                    # around best with heavier tails to escape
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += 0.12 * spans[d] * math.tan(math.pi * (random.random() - 0.5))
                    x = reflect(x)
                fx = eval_f(x)
                pop[idx], fit[idx] = x, fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve = time.time()
            if len(archive) > 0:
                archive = archive[len(archive) // 2 :]
            tr_sigma = min(tr_sigma_max, max(tr_sigma, 0.10))
            last_improve = time.time()

        # Linear population size reduction based on time fraction
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        target_NP = int(round(NP_init - (NP_init - NP_min) * min(1.0, max(0.0, frac))))
        if target_NP < NP_min:
            target_NP = NP_min

        # If need to reduce pop: delete worst individuals
        if len(pop) > target_NP:
            order = sorted(range(len(pop)), key=lambda i: fit[i])  # best->worst
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            arch_max = len(pop)
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # Prepare p-best ranking
        order, pnum = top_p_indices()

        # Success lists for SHADE update
        S_F, S_CR, S_w = [], [], []

        NP = len(pop)
        for i in range(NP):
            if time_up():
                break

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            CR = random.gauss(muCR, 0.1)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = cauchy(muF, 0.1)
            tries = 0
            while F <= 0.0 and tries < 8:
                F = cauchy(muF, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            pbest_idx = order[random.randrange(pnum)]
            xpbest = pop[pbest_idx]

            r1 = i
            while r1 == i or r1 == pbest_idx:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            use_arch = (len(archive) > 0 and random.random() < (len(archive) / float(len(archive) + NP)))
            if use_arch:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == pbest_idx or r2 == r1:
                    r2 = random.randrange(NP)
                xr2 = pop[r2]

            # DE/current-to-pbest/1
            v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            u = reflect(u)
            fu = eval_f(u)

            if fu <= fi:
                # archive parent
                archive.append(xi[:])
                if len(archive) > arch_max:
                    j = random.randrange(len(archive))
                    archive[j] = archive[-1]
                    archive.pop()

                pop[i], fit[i] = u, fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = time.time()

                w = fi - fu
                if w <= 0.0:
                    w = 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

        # Update SHADE memories
        if S_F:
            wsum = sum(S_w)
            if wsum <= 0.0:
                wsum = float(len(S_w))

            meanCR = 0.0
            for cr, w in zip(S_CR, S_w):
                meanCR += cr * (w / wsum)

            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_w):
                wf = w / wsum
                num += wf * f * f
                den += wf * f
            meanF = (num / den) if den > 0.0 else M_F[k_mem]

            M_CR[k_mem] = meanCR
            M_F[k_mem] = meanF
            k_mem = (k_mem + 1) % H

    return best
