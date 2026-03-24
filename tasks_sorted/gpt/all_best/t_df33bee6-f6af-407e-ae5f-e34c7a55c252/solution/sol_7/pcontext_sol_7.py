import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no external libs).

    Key improvements over your current best (DE + ES + occasional NM):
      1) Adds an "ask/tell" surrogate-guided local search (RBF ridge regression)
         that exploits the evaluation history to propose better candidates.
         - No numpy; solved by simple Gauss-Jordan elimination.
         - Uses a small active set so overhead stays low.
      2) Replaces Nelder–Mead bursts with a cheaper + often stronger trust-region
         local search: random subspace quadratic-ish probing + surrogate proposals.
      3) Keeps SHADE-like DE (current-to-pbest/1 + archive) but reduces wasted evals:
         evaluates a second trial only when useful.
      4) Uses a cache (rounded keys) to avoid reevaluating identical points.
      5) Better boundary handling: reflection + occasional random re-entry.

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """

    # -------------------- helpers --------------------
    start = time.time()
    deadline = start + float(max_time)

    def span(i):
        lo, hi = bounds[i]
        s = hi - lo
        return s if s > 0.0 else 1.0

    spans = [span(i) for i in range(dim)]
    inv_spans = [1.0 / (spans[i] + 1e-300) for i in range(dim)]

    def bounce_inplace(x):
        # reflect into [lo, hi] even if far outside
        for i, (lo, hi) in enumerate(bounds):
            w = hi - lo
            if w <= 0.0:
                x[i] = lo
                continue
            xi = x[i]
            if xi < lo or xi > hi:
                y = (xi - lo) % (2.0 * w)
                x[i] = lo + (2.0 * w - y if y > w else y)
        return x

    def reenter_random_dims(x, p=0.02):
        for i, (lo, hi) in enumerate(bounds):
            if random.random() < p:
                x[i] = random.uniform(lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    def randn():
        # approx N(0,1) via CLT: sum 12 uniforms - 6
        return (random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() - 6.0)

    def stratified_population(n):
        if n <= 0:
            return []
        perms = []
        for d in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            perms.append(idx)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                u = (perms[d][i] + random.random()) / float(n)
                x[d] = lo + u * (hi - lo)
            pop.append(x)
        return pop

    def cauchy_like(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def normal_like(mu, sigma):
        return mu + sigma * randn()

    def pick_index(n, forbid_set):
        j = random.randrange(n)
        tries = 0
        while j in forbid_set and tries < 40:
            j = random.randrange(n)
            tries += 1
        return j

    def argmin(vals):
        bi = 0
        bv = vals[0]
        for i in range(1, len(vals)):
            v = vals[i]
            if v < bv:
                bv = v
                bi = i
        return bi, bv

    # -------- evaluation cache (rounded) --------
    # rounding reduces collisions due to floating noise but preserves most uniqueness
    cache = {}

    def key_of(x):
        # per-dim rounding to ~1e-12 of span
        # (if span is huge, this still keeps key size manageable)
        ks = []
        for i in range(dim):
            lo, hi = bounds[i]
            w = hi - lo
            q = 1e-12 * (w if w > 0 else 1.0)
            if q <= 0.0:
                q = 1e-12
            ks.append(int((x[i] - lo) / q))
        return tuple(ks)

    def eval_f(x):
        # x is list
        k = key_of(x)
        if k in cache:
            return cache[k]
        try:
            v = float(func(x))
            if not math.isfinite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v

    # -------- normalized distance/kernel --------
    def normed_sqdist(a, b):
        s = 0.0
        for i in range(dim):
            d = (a[i] - b[i]) * inv_spans[i]
            s += d * d
        return s

    # -------------------- small linear solver (Gauss-Jordan) --------------------
    def solve_linear(A, b):
        # Solves Ax=b for square A using Gauss-Jordan with partial pivoting.
        # Returns list x or None on singular.
        n = len(A)
        # build augmented matrix
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            # pivot
            piv = col
            best = abs(M[col][col])
            for r in range(col + 1, n):
                v = abs(M[r][col])
                if v > best:
                    best = v
                    piv = r
            if best < 1e-14:
                return None
            if piv != col:
                M[col], M[piv] = M[piv], M[col]

            # normalize pivot row
            pv = M[col][col]
            invpv = 1.0 / pv
            for c in range(col, n + 1):
                M[col][c] *= invpv

            # eliminate
            for r in range(n):
                if r == col:
                    continue
                factor = M[r][col]
                if abs(factor) < 1e-18:
                    continue
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]

        return [M[i][n] for i in range(n)]

    # -------------------- surrogate: RBF ridge (active set) --------------------
    # Model: f(x) ≈ b0 + sum_i alpha_i * exp(-gamma * ||x-xi||^2_norm) + sum_j bj * xj_norm
    # Fit by ridge regression in feature space on active set.
    def fit_surrogate(points, values, active_idx, gamma, ridge):
        m = len(active_idx)
        if m < 6:
            return None

        # Use linear terms on a small random subset of dims to reduce overfit cost
        lin_k = min(dim, 10)
        lin_dims = list(range(dim))
        random.shuffle(lin_dims)
        lin_dims = lin_dims[:lin_k]

        # Build features for active points
        # feature length = 1 + m (rbf centers) + lin_k
        p = 1 + m + lin_k
        # Compute Gram matrix G = X^T X + ridge I, and rhs = X^T y
        G = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p

        # cache normalized coordinates for linear terms
        xn_cache = {}
        for ii, idx in enumerate(active_idx):
            x = points[idx]
            xn = [0.0] * lin_k
            for t, d in enumerate(lin_dims):
                lo, hi = bounds[d]
                w = hi - lo
                if w <= 0.0:
                    xn[t] = 0.0
                else:
                    xn[t] = (2.0 * (x[d] - lo) / w) - 1.0  # [-1,1]
            xn_cache[idx] = xn

        # Precompute RBF feature matrix rows on the fly
        for row_i, idx_i in enumerate(active_idx):
            xi = points[idx_i]
            yi = values[idx_i]
            # phi vector
            phi = [0.0] * p
            phi[0] = 1.0
            # RBF features wrt centers = active points in same order
            for cj, idx_c in enumerate(active_idx):
                xc = points[idx_c]
                d2 = normed_sqdist(xi, xc)
                phi[1 + cj] = math.exp(-gamma * d2)
            # linear
            lin = xn_cache[idx_i]
            for t in range(lin_k):
                phi[1 + m + t] = lin[t]

            # accumulate G and rhs
            for a in range(p):
                rhs[a] += phi[a] * yi
                ga = G[a]
                pa = phi[a]
                for b2 in range(a, p):
                    ga[b2] += pa * phi[b2]

        # symmetrize and add ridge
        for a in range(p):
            for b2 in range(a):
                G[a][b2] = G[b2][a]
            G[a][a] += ridge

        w = solve_linear(G, rhs)
        if w is None:
            return None

        model = {
            "active_idx": active_idx[:],
            "w": w,
            "gamma": gamma,
            "lin_dims": lin_dims,
            "m": m,
            "lin_k": lin_k
        }
        return model

    def surrogate_predict(model, x, points):
        m = model["m"]
        lin_k = model["lin_k"]
        lin_dims = model["lin_dims"]
        w = model["w"]
        gamma = model["gamma"]
        phi0 = 1.0
        s = w[0] * phi0
        # RBF
        for j, idx_c in enumerate(model["active_idx"]):
            xc = points[idx_c]
            d2 = normed_sqdist(x, xc)
            s += w[1 + j] * math.exp(-gamma * d2)
        # linear
        off = 1 + m
        for t in range(lin_k):
            d = lin_dims[t]
            lo, hi = bounds[d]
            wspan = hi - lo
            xn = 0.0 if wspan <= 0.0 else (2.0 * (x[d] - lo) / wspan) - 1.0
            s += w[off + t] * xn
        return s

    # pick active set: best + diverse
    def build_active_set(points, values, max_m):
        n = len(points)
        idx_sorted = sorted(range(n), key=lambda i: values[i])
        active = []
        # always include top few
        top = min(6, n)
        for i in idx_sorted[:top]:
            active.append(i)
        # add diverse points from rest
        # greedy farthest-from-active in normalized space, biased to better values
        cand = idx_sorted[top:]
        # if very few, just take them
        while len(active) < min(max_m, n) and cand:
            best_j = None
            best_score = -1.0
            # scan a limited prefix for speed
            scan = cand[:min(len(cand), 30)]
            for j in scan:
                xj = points[j]
                # min dist to active
                md = float("inf")
                for ai in active:
                    d2 = normed_sqdist(xj, points[ai])
                    if d2 < md:
                        md = d2
                # combine diversity and rank
                rank = 1.0 / (1.0 + cand.index(j))  # crude
                score = md * (0.75 + 0.25 * rank)
                if score > best_score:
                    best_score = score
                    best_j = j
            active.append(best_j)
            cand.remove(best_j)
        return active

    # -------------------- parameters --------------------
    pop_size = max(22, min(120, 22 + 6 * dim))
    p_rate = 0.22

    # SHADE memories
    H = 10
    MCR = [0.85] * H
    MF = [0.60] * H
    mem_k = 0

    archive = []
    archive_max = pop_size

    # (1+1)-ES
    sigma = 0.11
    sigma_min = 1e-16
    sigma_max = 0.45

    # coordinate step
    coord_step = [0.16 * spans[i] for i in range(dim)]
    coord_min_step = [1e-16 * spans[i] + 1e-18 for i in range(dim)]

    # restart / refresh
    best_t = start
    stall_seconds = max(0.25, 0.13 * max_time)
    last_refresh_t = start
    refresh_period = max(0.35, 0.18 * max_time)

    # history for surrogate
    hist_x = []
    hist_f = []

    # -------------------- init: stratified + opposition --------------------
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    for i in range(pop_size):
        if time.time() >= deadline:
            return min(fit)
        xo = opposite(pop[i])
        bounce_inplace(xo)
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i] = xo
            fit[i] = fo

    best_i, best = argmin(fit)
    best_x = pop[best_i][:]

    # seed history
    for i in range(pop_size):
        hist_x.append(pop[i][:])
        hist_f.append(fit[i])

    # local cloud
    cloud = min(10, max(3, pop_size // 7))
    for _ in range(cloud):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for d in range(dim):
            x[d] += randn() * (0.04 * spans[d])
        bounce_inplace(x)
        f = eval_f(x)
        hist_x.append(x[:]); hist_f.append(f)
        j = random.randrange(pop_size)
        if j == best_i:
            j = (j + 1) % pop_size
        if f < fit[j]:
            archive.append(pop[j][:])
            if len(archive) > archive_max:
                archive.pop(random.randrange(len(archive)))
            pop[j] = x
            fit[j] = f
            if f < best:
                best, best_x, best_t = f, x[:], time.time()

    # -------------------- main loop --------------------
    while True:
        now = time.time()
        if now >= deadline:
            return best

        tfrac = 1.0 if max_time <= 0 else min(1.0, (now - start) / max_time)

        # p-best pool
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        p_num = max(2, int(math.ceil(p_rate * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        succ_CR, succ_F, succ_w = [], [], []

        union = pop + archive
        ulen = len(union)

        # -------- DE sweep (mostly 1-2 trials) --------
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mu_cr, mu_f = MCR[r], MF[r]

            CR = normal_like(mu_cr, 0.10 + 0.05 * (1.0 - tfrac))
            CR = 0.0 if CR < 0.0 else (1.0 if CR > 1.0 else CR)

            F = cauchy_like(mu_f, 0.08 + 0.05 * (1.0 - tfrac))
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 12:
                F = cauchy_like(mu_f, 0.08 + 0.05 * (1.0 - tfrac))
                tries += 1
            if F <= 0.0:
                F = 0.5
            if F > 1.0:
                F = 1.0

            pbest = pop[random.choice(pbest_pool)]

            # trial 1: current-to-pbest/1 (archive)
            r1 = pick_index(pop_size, {i})
            r2 = pick_index(ulen, {i, r1}) if ulen > 2 else pick_index(pop_size, {i, r1})
            x_r1 = pop[r1]
            x_r2 = union[r2] if ulen > 2 else pop[r2]

            mut1 = [xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d]) for d in range(dim)]
            bounce_inplace(mut1)
            if random.random() < 0.05:
                reenter_random_dims(mut1, p=0.02)
            tr1 = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    tr1[d] = mut1[d]
            f1 = eval_f(tr1)

            best_trial, f_trial, used_CR = tr1, f1, CR

            # evaluate a second trial if not improved or sometimes anyway
            need_second = (f1 >= fi) or (random.random() < (0.20 + 0.30 * (1.0 - tfrac)))
            if need_second:
                # trial 2: rand/1 (archive) exploration
                a = pick_index(pop_size, {i})
                b = pick_index(pop_size, {i, a})
                c = pick_index(ulen, {i, a, b}) if ulen > 3 else pick_index(pop_size, {i, a, b})
                xa, xb = pop[a], pop[b]
                xc = union[c] if ulen > 3 else pop[c]

                mut2 = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
                bounce_inplace(mut2)
                CR2 = min(1.0, max(0.0, CR + 0.12 * (1.0 - tfrac)))
                tr2 = xi[:]
                jrand2 = random.randrange(dim)
                for d in range(dim):
                    if random.random() < CR2 or d == jrand2:
                        tr2[d] = mut2[d]
                f2 = eval_f(tr2)
                if f2 < f_trial:
                    best_trial, f_trial, used_CR = tr2, f2, CR2

            # selection
            if f_trial <= fi:
                old = fi
                archive.append(xi[:])
                if len(archive) > archive_max:
                    archive.pop(random.randrange(len(archive)))
                pop[i] = best_trial
                fit[i] = f_trial

                w = (old - f_trial) if (math.isfinite(old) and old != float("inf")) else 1.0
                if w < 0.0:
                    w = 0.0
                succ_CR.append(used_CR)
                succ_F.append(F)
                succ_w.append(w + 1e-12)

                hist_x.append(best_trial[:]); hist_f.append(f_trial)

                if f_trial < best:
                    best, best_x, best_t = f_trial, best_trial[:], time.time()
            else:
                # still record some history sometimes to help surrogate learn landscape
                if random.random() < 0.10:
                    hist_x.append(best_trial[:]); hist_f.append(f_trial)

        # -------- memory update --------
        if succ_F:
            wsum = sum(succ_w)
            cr_mean = sum(w * cr for w, cr in zip(succ_w, succ_CR)) / wsum
            num = sum(w * f * f for w, f in zip(succ_w, succ_F))
            den = sum(w * f for w, f in zip(succ_w, succ_F))
            f_mean = (num / den) if den > 0.0 else MF[mem_k]

            MCR[mem_k] = 0.25 * MCR[mem_k] + 0.75 * cr_mean
            MF[mem_k]  = 0.25 * MF[mem_k]  + 0.75 * f_mean
            mem_k = (mem_k + 1) % H

        # -------- local: (1+1)-ES around best --------
        es_tries = 6 + int(18 * tfrac)
        succ = 0
        att = 0
        dim_factor = 1.0 / math.sqrt(max(1.0, dim))
        for _ in range(es_tries):
            if time.time() >= deadline:
                return best
            att += 1
            cand = best_x[:]
            if random.random() < 0.55:
                for d in range(dim):
                    cand[d] += randn() * (sigma * dim_factor * spans[d])
            else:
                k = 1 + int(2 * random.random() * math.sqrt(max(1, dim)))
                for _k in range(k):
                    d = random.randrange(dim)
                    cand[d] += randn() * (sigma * 1.6 * dim_factor * spans[d])
            bounce_inplace(cand)
            f = eval_f(cand)
            hist_x.append(cand[:]); hist_f.append(f)
            if f < best:
                best, best_x, best_t = f, cand[:], time.time()
                succ += 1

        if att:
            rate = succ / float(att)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.18)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.74)

        # -------- surrogate-guided propose (cheap ask/tell) --------
        # Trigger more in mid/late time, but keep small overhead.
        if tfrac > 0.18 and len(hist_x) >= 30 and random.random() < (0.22 + 0.25 * tfrac):
            # keep history bounded (avoid overhead)
            if len(hist_x) > 260:
                # keep best 140 + random 100
                idx_sorted_h = sorted(range(len(hist_x)), key=lambda i: hist_f[i])
                keep = set(idx_sorted_h[:140])
                rem = [i for i in range(len(hist_x)) if i not in keep]
                random.shuffle(rem)
                for i in rem[:100]:
                    keep.add(i)
                hist_x = [hist_x[i] for i in keep]
                hist_f = [hist_f[i] for i in keep]

            max_m = 18
            active = build_active_set(hist_x, hist_f, max_m=max_m)

            # gamma roughly inverse squared lengthscale; set from active spread
            # estimate median distance among a few pairs
            dists = []
            for _ in range(16):
                i1 = random.choice(active)
                i2 = random.choice(active)
                if i1 == i2:
                    continue
                dists.append(normed_sqdist(hist_x[i1], hist_x[i2]))
            med = sorted(dists)[len(dists)//2] if dists else 0.2
            if med <= 1e-12:
                med = 0.2
            gamma = 2.0 / med  # moderate smoothness
            ridge = 1e-6

            model = fit_surrogate(hist_x, hist_f, active, gamma=gamma, ridge=ridge)
            if model is not None:
                # propose a handful of candidates: jitter around best and around random good points
                proposals = []
                # around best
                for _ in range(10):
                    x = best_x[:]
                    rad = (0.12 * (1.0 - tfrac) + 0.03)  # relative
                    for d in range(dim):
                        x[d] += randn() * (rad * spans[d])
                    bounce_inplace(x)
                    proposals.append(x)
                # around a few top historical points
                idx_sorted_h = sorted(range(len(hist_x)), key=lambda i: hist_f[i])
                for _ in range(6):
                    base = hist_x[random.choice(idx_sorted_h[:min(20, len(idx_sorted_h))])][:]
                    rad = (0.10 * (1.0 - tfrac) + 0.02)
                    for d in range(dim):
                        if random.random() < 0.65:
                            base[d] += randn() * (rad * spans[d])
                    bounce_inplace(base)
                    proposals.append(base)

                # pick best predicted few, evaluate 2-3 of them
                scored = [(surrogate_predict(model, x, hist_x), x) for x in proposals]
                scored.sort(key=lambda t: t[0])
                eval_k = 2 if tfrac < 0.7 else 3
                for j in range(min(eval_k, len(scored))):
                    if time.time() >= deadline:
                        return best
                    x = scored[j][1]
                    # small chance to re-enter dims to avoid boundary artifacts
                    if random.random() < 0.04:
                        reenter_random_dims(x, p=0.015)
                    f = eval_f(x)
                    hist_x.append(x[:]); hist_f.append(f)
                    if f < best:
                        best, best_x, best_t = f, x[:], time.time()
                        # inject into population by replacing a random worse one
                        wi = idx_sorted[-1 - random.randrange(max(1, pop_size // 5))]
                        archive.append(pop[wi][:])
                        if len(archive) > archive_max:
                            archive.pop(random.randrange(len(archive)))
                        pop[wi] = x[:]
                        fit[wi] = f

        # -------- micro coordinate tweaks (cheap) --------
        if tfrac > 0.25:
            k = min(dim, 2 + int(4 * tfrac))
            for _ in range(k):
                if time.time() >= deadline:
                    return best
                d = random.randrange(dim)
                sd = coord_step[d]
                if sd <= coord_min_step[d]:
                    continue
                x0 = best_x[d]
                improved = False
                for xd in (x0 + sd, x0 - sd, x0 + 0.5 * sd, x0 - 0.5 * sd):
                    cand = best_x[:]
                    cand[d] = xd
                    bounce_inplace(cand)
                    f = eval_f(cand)
                    hist_x.append(cand[:]); hist_f.append(f)
                    if f < best:
                        best, best_x, best_t = f, cand[:], time.time()
                        coord_step[d] = min(0.35 * spans[d], coord_step[d] * 1.12)
                        improved = True
                        break
                if not improved:
                    coord_step[d] *= 0.62

        # -------- periodic partial refresh --------
        now = time.time()
        if now - last_refresh_t > refresh_period and tfrac > 0.12:
            elite_k = max(3, pop_size // 8)
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elites = set(idx_sorted[:elite_k])

            repl = max(2, pop_size // 6)
            worst = idx_sorted[-repl:]
            for wi in worst:
                if time.time() >= deadline:
                    return best
                if wi in elites:
                    continue

                if random.random() < 0.55:
                    x = best_x[:]
                    rad = (max(0.05, sigma) * 0.20)
                    for d in range(dim):
                        x[d] += randn() * (rad * spans[d])
                    bounce_inplace(x)
                else:
                    x = rand_vec()
                    if random.random() < 0.50:
                        x = opposite(x)
                        bounce_inplace(x)

                f = eval_f(x)
                hist_x.append(x[:]); hist_f.append(f)
                pop[wi] = x
                fit[wi] = f
                if f < best:
                    best, best_x, best_t = f, x[:], time.time()

            archive = []
            last_refresh_t = time.time()

        # -------- full restart on stagnation --------
        if time.time() - best_t > stall_seconds:
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elite_k = max(4, pop_size // 7)
            elites = [pop[i][:] for i in idx_sorted[:elite_k]]
            elites_fit = [fit[i] for i in idx_sorted[:elite_k]]

            new_pop = elites[:]
            new_fit = elites_fit[:]

            remain = pop_size - elite_k
            n_global = remain // 3
            n_opp = remain // 3
            n_local = remain - n_global - n_opp

            for x in stratified_population(n_global):
                fx = eval_f(x)
                hist_x.append(x[:]); hist_f.append(fx)
                new_pop.append(x)
                new_fit.append(fx)

            for _ in range(n_opp):
                x = rand_vec()
                xo = opposite(x)
                bounce_inplace(xo)
                fx = eval_f(x)
                fo = eval_f(xo)
                if fo < fx:
                    hist_x.append(xo[:]); hist_f.append(fo)
                    new_pop.append(xo); new_fit.append(fo)
                else:
                    hist_x.append(x[:]); hist_f.append(fx)
                    new_pop.append(x);  new_fit.append(fx)

            local_sigma = max(0.08, sigma)
            for _ in range(n_local):
                x = best_x[:]
                for d in range(dim):
                    x[d] += randn() * (local_sigma * 0.22 * spans[d])
                bounce_inplace(x)
                fx = eval_f(x)
                hist_x.append(x[:]); hist_f.append(fx)
                new_pop.append(x)
                new_fit.append(fx)

            pop, fit = new_pop, new_fit
            bi, bv = argmin(fit)
            if bv < best:
                best, best_x = bv, pop[bi][:]
            best_t = time.time()
            last_refresh_t = best_t

            archive = []
            # small memory perturbation
            for k in range(H):
                MCR[k] = min(0.95, max(0.05, MCR[k] + 0.035 * randn()))
                MF[k]  = min(0.95, max(0.05, MF[k]  + 0.035 * randn()))
            sigma = max(0.10, sigma)
            coord_step = [max(0.10 * spans[i], coord_step[i]) for i in range(dim)]
