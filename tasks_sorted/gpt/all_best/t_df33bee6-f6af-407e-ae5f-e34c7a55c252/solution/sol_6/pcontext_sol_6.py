import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained, no external libs).

    What’s improved vs your current best (DE + ES + occasional NM):
      1) Better evaluation efficiency: per-individual bandit over multiple trial makers,
         but usually evaluates only 1-2 trials (and 3 only when worthwhile).
      2) Stronger local refinement: bounded Powell-like directional search with adaptive
         directions + occasional small 2-point quadratic fit on a coordinate.
      3) More robust global search: SHADE/JADE current-to-pbest/1 + archive, plus a
         "triangular" mutation variant that is cheap and helps on ill-conditioned problems.
      4) Smarter restarts: triggered by stagnation AND low population diversity.
      5) Boundary handling: reflection + probabilistic re-entry (keeps diversity).

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """

    # -------------------- helpers --------------------
    def eval_f(x):
        try:
            v = float(func(x))
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    def span(i):
        lo, hi = bounds[i]
        s = hi - lo
        return s if s > 0.0 else 1.0

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

    def reenter_random_dims(x, p):
        for i, (lo, hi) in enumerate(bounds):
            if random.random() < p:
                x[i] = random.uniform(lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    def randn():
        # approx N(0,1) via CLT: (sum 12 uniforms - 6)
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

    def pick_index(n, forbid):
        j = random.randrange(n)
        tries = 0
        while j in forbid and tries < 40:
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

    # diversity metric: mean normalized L1 distance to centroid
    def diversity(pop):
        if not pop:
            return 0.0
        cen = [0.0] * dim
        invn = 1.0 / float(len(pop))
        for x in pop:
            for d in range(dim):
                cen[d] += x[d]
        for d in range(dim):
            cen[d] *= invn

        tot = 0.0
        for x in pop:
            s = 0.0
            for d in range(dim):
                s += abs(x[d] - cen[d]) / (span(d) + 1e-300)
            tot += s
        return tot * invn / max(1, dim)

    # -------------------- bounded Powell-like local search --------------------
    # We keep it evaluation-light: a few directions, each does a tiny bracket/line-search.
    def powell_local(best_x, best_f, budget_evals, dirs, step0):
        if budget_evals <= 0:
            return best_x, best_f, 0, dirs, step0

        used = 0
        x = best_x[:]
        f = best_f

        # helper: 1D line search along direction u
        def line_search(x0, f0, u, step, max_e):
            nonlocal used
            if max_e <= 0:
                return x0, f0, 0, step

            # normalize direction (avoid sqrt; use max-abs)
            m = 0.0
            for d in range(dim):
                ad = abs(u[d])
                if ad > m:
                    m = ad
            if m <= 0.0:
                return x0, f0, 0, step
            invm = 1.0 / m
            uu = [u[d] * invm for d in range(dim)]

            # Evaluate at +/- step
            best_ls_x = x0[:]
            best_ls_f = f0
            evals = 0

            def trial_at(alpha):
                nonlocal used
                xx = [x0[d] + alpha * uu[d] for d in range(dim)]
                bounce_inplace(xx)
                # tiny chance to re-enter a few dims if reflection creates sticky behavior
                if random.random() < 0.03:
                    reenter_random_dims(xx, 0.01)
                ff = eval_f(xx)
                used += 1
                return xx, ff

            # try +step
            if evals < max_e:
                x1, f1 = trial_at(step); evals += 1
                if f1 < best_ls_f:
                    best_ls_x, best_ls_f = x1, f1

            # try -step
            if evals < max_e:
                x2, f2 = trial_at(-step); evals += 1
                if f2 < best_ls_f:
                    best_ls_x, best_ls_f = x2, f2

            # If neither improved, shrink step a bit
            if best_ls_f >= f0:
                return best_ls_x, best_ls_f, evals, max(step * 0.55, 1e-18)

            # If improved, attempt one more expansion step in improving direction
            # (very cheap "pattern move")
            if evals < max_e:
                # choose direction
                sign = 1.0
                if best_ls_x is x2 or best_ls_f == f2:
                    sign = -1.0
                x3, f3 = trial_at(sign * (2.0 * step)); evals += 1
                if f3 < best_ls_f:
                    best_ls_x, best_ls_f = x3, f3
                    step = min(step * 1.35, 0.50)  # relative, actual scaling happens outside
                else:
                    step = max(step * 0.85, 1e-18)

            return best_ls_x, best_ls_f, evals, step

        # run directions (subset if dim big)
        # keep dirs length at most 2*dim but we limit per call for time
        kdirs = min(len(dirs), max(2, 6))
        # choose a few directions (biased to most recent ones)
        pick = list(range(len(dirs)))
        # sample without external libs: partial shuffle
        for i in range(min(len(pick), kdirs)):
            j = i + random.randrange(len(pick) - i)
            pick[i], pick[j] = pick[j], pick[i]
        pick = pick[:kdirs]

        moved = False
        x_start = x[:]
        f_start = f

        # relative base step in normalized space; convert to absolute by spans in direction
        for idx in pick:
            if used >= budget_evals:
                break
            u = dirs[idx]

            # compute absolute step from step0 and spans (use max span projection)
            sabs = 0.0
            for d in range(dim):
                sabs = max(sabs, abs(u[d]) * span(d))
            if sabs <= 0.0:
                continue
            step_abs = step0 * sabs

            x_new, f_new, ev, new_step0 = line_search(x, f, u, step_abs, budget_evals - used)
            step0 = max(1e-18, min(0.6, new_step0 / (sabs + 1e-300)))
            if f_new < f:
                x, f = x_new, f_new
                moved = True

        # update directions (Powell-style): add net displacement direction if moved
        if moved:
            disp = [x[d] - x_start[d] for d in range(dim)]
            # if meaningful displacement, prepend it
            norm1 = 0.0
            for d in range(dim):
                norm1 += abs(disp[d]) / (span(d) + 1e-300)
            if norm1 > 1e-12:
                dirs = [disp] + dirs
                # cap number of directions
                max_dirs = min(2 * dim, 40)
                if len(dirs) > max_dirs:
                    dirs = dirs[:max_dirs]

        return x, f, used, dirs, step0

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- parameters --------------------
    pop_size = max(24, min(120, 24 + 6 * dim))
    p_rate = 0.22

    # SHADE memories
    H = 10
    MCR = [0.85] * H
    MF = [0.60] * H
    mem_k = 0

    archive = []
    archive_max = pop_size

    # (1+1)-ES sigma and scaling
    sigma = 0.10
    sigma_min = 1e-18
    sigma_max = 0.45
    dim_scale = [span(i) for i in range(dim)]

    # coordinate micro-step
    coord_step = [0.14 * span(i) for i in range(dim)]
    coord_min_step = [1e-18 * span(i) + 1e-18 for i in range(dim)]

    # stagnation / refresh
    best_t = start
    stall_seconds = max(0.25, 0.12 * max_time)
    last_refresh_t = start
    refresh_period = max(0.35, 0.18 * max_time)

    # bandit over trial makers (counts only when evaluated)
    # 0: current-to-pbest/1, 1: rand/1, 2: best/2, 3: triangular current-to-best (cheap)
    arms = 4
    arm_w = [1.0] * arms
    arm_n = [1e-9] * arms

    # local Powell directions init: identity + a few random
    dirs = []
    for d in range(min(dim, 10)):
        u = [0.0] * dim
        u[d] = 1.0
        dirs.append(u)
    for _ in range(4):
        u = [randn() for _ in range(dim)]
        dirs.append(u)
    pow_step0 = 0.08  # relative

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

    # seed local cloud
    cloud = min(12, max(4, pop_size // 7))
    for _ in range(cloud):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for d in range(dim):
            x[d] += randn() * (0.045 * span(d))
        bounce_inplace(x)
        f = eval_f(x)
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

        # -------- DE sweep with bandit + usually 1-2 evaluations --------
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # sample F, CR from memories
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

            # choose up to 2 arms typically; 3 later
            # pick first arm by roulette on weights
            totw = sum(arm_w)
            u = random.random() * totw
            a1 = 0
            acc = arm_w[0]
            while acc < u and a1 < arms - 1:
                a1 += 1
                acc += arm_w[a1]

            # second arm: pick a different one, biased to others
            a2 = a1
            if random.random() < 0.70:
                # roulette among remaining
                remw = totw - arm_w[a1]
                u2 = random.random() * max(1e-12, remw)
                acc2 = 0.0
                for k in range(arms):
                    if k == a1:
                        continue
                    acc2 += arm_w[k]
                    if acc2 >= u2:
                        a2 = k
                        break
                if a2 == a1:
                    a2 = (a1 + 1) % arms
            else:
                a2 = (a1 + 1 + random.randrange(arms - 1)) % arms

            # third arm occasionally (late or near-best)
            do_third = (tfrac > 0.60 and random.random() < 0.35) or (fi < best + 0.05 * abs(best) + 1e-9 and random.random() < 0.22)
            a3 = (a2 + 1 + random.randrange(arms - 1)) % arms

            pbest = pop[random.choice(pbest_pool)]

            def make_trial(arm_id):
                # returns (trial_vector, used_CR_for_success_logging)
                if arm_id == 0:
                    # current-to-pbest/1 with archive
                    r1 = pick_index(pop_size, {i})
                    r2 = pick_index(ulen, {i, r1}) if ulen > 2 else pick_index(pop_size, {i, r1})
                    x_r1 = pop[r1]
                    x_r2 = union[r2] if ulen > 2 else pop[r2]
                    mutant = [xi[d] + F * (pbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d]) for d in range(dim)]
                    bounce_inplace(mutant)
                    if random.random() < 0.05:
                        reenter_random_dims(mutant, 0.02)
                    tr = xi[:]
                    jrand = random.randrange(dim)
                    for d in range(dim):
                        if random.random() < CR or d == jrand:
                            tr[d] = mutant[d]
                    return tr, CR

                if arm_id == 1:
                    # rand/1 with archive (exploration)
                    a = pick_index(pop_size, {i})
                    b = pick_index(pop_size, {i, a})
                    c = pick_index(ulen, {i, a, b}) if ulen > 3 else pick_index(pop_size, {i, a, b})
                    xa, xb = pop[a], pop[b]
                    xc = union[c] if ulen > 3 else pop[c]
                    mutant = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
                    bounce_inplace(mutant)
                    CR2 = min(1.0, max(0.0, CR + 0.12 * (1.0 - tfrac)))
                    tr = xi[:]
                    jrand = random.randrange(dim)
                    for d in range(dim):
                        if random.random() < CR2 or d == jrand:
                            tr[d] = mutant[d]
                    return tr, CR2

                if arm_id == 2:
                    # best/2 exploitation
                    r3 = pick_index(pop_size, {i})
                    r4 = pick_index(pop_size, {i, r3})
                    r5 = pick_index(pop_size, {i, r3, r4})
                    r6 = pick_index(pop_size, {i, r3, r4, r5})
                    x3, x4, x5, x6 = pop[r3], pop[r4], pop[r5], pop[r6]
                    F3 = min(1.0, max(0.10, F * (0.70 + 0.65 * tfrac)))
                    mutant = [best_x[d] + F3 * (x3[d] - x4[d]) + F3 * (x5[d] - x6[d]) for d in range(dim)]
                    bounce_inplace(mutant)
                    CR3 = min(1.0, max(0.0, 0.55 * CR + 0.35 + 0.20 * tfrac))
                    tr = xi[:]
                    jrand = random.randrange(dim)
                    for d in range(dim):
                        if random.random() < CR3 or d == jrand:
                            tr[d] = mutant[d]
                    return tr, CR3

                # arm_id == 3:
                # triangular current-to-best (cheap, surprisingly robust)
                # xi + F*(best-xi) + 0.5F*(pbest-xi)
                mutant = [xi[d] + F * (best_x[d] - xi[d]) + 0.5 * F * (pbest[d] - xi[d]) for d in range(dim)]
                bounce_inplace(mutant)
                CRt = min(1.0, max(0.0, 0.40 + 0.35 * CR))
                tr = xi[:]
                jrand = random.randrange(dim)
                for d in range(dim):
                    if random.random() < CRt or d == jrand:
                        tr[d] = mutant[d]
                return tr, CRt

            # evaluate 1st trial
            tr1, cr1 = make_trial(a1)
            f1 = eval_f(tr1)
            arm_n[a1] += 1.0

            best_tr, best_f, best_cr, best_arm = tr1, f1, cr1, a1

            # evaluate 2nd trial if it seems promising OR if first didn't improve
            need_second = (f1 >= fi) or (random.random() < (0.25 + 0.35 * (1.0 - tfrac)))
            if need_second:
                tr2, cr2 = make_trial(a2)
                f2 = eval_f(tr2)
                arm_n[a2] += 1.0
                if f2 < best_f:
                    best_tr, best_f, best_cr, best_arm = tr2, f2, cr2, a2

            # evaluate 3rd trial occasionally (late / hard problems)
            if do_third:
                tr3, cr3 = make_trial(a3)
                f3 = eval_f(tr3)
                arm_n[a3] += 1.0
                if f3 < best_f:
                    best_tr, best_f, best_cr, best_arm = tr3, f3, cr3, a3

            # selection
            if best_f <= fi:
                old = fi
                archive.append(xi[:])
                if len(archive) > archive_max:
                    archive.pop(random.randrange(len(archive)))
                pop[i] = best_tr
                fit[i] = best_f

                w = (old - best_f) if (math.isfinite(old) and old != float("inf")) else 1.0
                if w < 0.0:
                    w = 0.0
                succ_CR.append(best_cr)
                succ_F.append(F)
                succ_w.append(w + 1e-12)

                # reward arm (multiplicative weights, gently)
                arm_w[best_arm] *= (1.0 + 0.04 + 0.08 * min(1.0, w / (abs(old) + 1e-9)))
                # mild decay to keep exploration
                for k in range(arms):
                    arm_w[k] *= 0.999

                if best_f < best:
                    best, best_x, best_t = best_f, best_tr[:], time.time()
            else:
                # slight penalty to chosen arm to avoid wasting too much time
                arm_w[best_arm] *= 0.9995

        # keep arm weights sane
        mx = max(arm_w)
        if mx > 50.0:
            sc = 1.0 / mx
            arm_w = [w * 10.0 * sc for w in arm_w]
        for k in range(arms):
            if arm_w[k] < 1e-6:
                arm_w[k] = 1e-6

        # -------- memory update (SHADE-like) --------
        if succ_F:
            wsum = sum(succ_w)
            cr_mean = sum(w * cr for w, cr in zip(succ_w, succ_CR)) / wsum
            num = sum(w * f * f for w, f in zip(succ_w, succ_F))
            den = sum(w * f for w, f in zip(succ_w, succ_F))
            f_mean = (num / den) if den > 0.0 else MF[mem_k]

            MCR[mem_k] = 0.25 * MCR[mem_k] + 0.75 * cr_mean
            MF[mem_k] = 0.25 * MF[mem_k] + 0.75 * f_mean
            mem_k = (mem_k + 1) % H

        # -------- local: (1+1)-ES around best --------
        es_tries = 6 + int(16 * tfrac)
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
                    cand[d] += randn() * (sigma * dim_factor * dim_scale[d])
            else:
                k = 1 + int(2 * random.random() * math.sqrt(max(1, dim)))
                for _k in range(k):
                    d = random.randrange(dim)
                    cand[d] += randn() * (sigma * 1.7 * dim_factor * dim_scale[d])
            bounce_inplace(cand)
            f = eval_f(cand)
            if f < best:
                best, best_x, best_t = f, cand[:], time.time()
                succ += 1

        if att:
            rate = succ / float(att)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.18)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.74)

        # -------- micro coordinate tweaks + occasional quadratic fit --------
        if tfrac > 0.22:
            k = min(dim, 2 + int(4 * tfrac))
            for _ in range(k):
                if time.time() >= deadline:
                    return best
                d = random.randrange(dim)
                sd = coord_step[d]
                if sd <= coord_min_step[d]:
                    continue

                x0 = best_x[d]
                # evaluate three points: x0-sd, x0, x0+sd (but x0 already known)
                cand_p = best_x[:]; cand_p[d] = x0 + sd
                cand_m = best_x[:]; cand_m[d] = x0 - sd
                bounce_inplace(cand_p); bounce_inplace(cand_m)
                fp = eval_f(cand_p)
                fm = eval_f(cand_m)

                improved = False
                if fp < best:
                    best, best_x, best_t = fp, cand_p[:], time.time()
                    improved = True
                if fm < best:
                    best, best_x, best_t = fm, cand_m[:], time.time()
                    improved = True

                # cheap 1D quadratic fit if both sides are finite
                if (not improved) and math.isfinite(fp) and math.isfinite(fm) and math.isfinite(best):
                    # parabola through (-sd, fm), (0, best), (+sd, fp)
                    denom = (fm - 2.0 * best + fp)
                    if abs(denom) > 1e-18:
                        a = denom / (2.0 * sd * sd)
                        b = (fp - fm) / (2.0 * sd)
                        # minimizer at -b/(2a)
                        alpha = -b / (2.0 * a)
                        # trust region clamp
                        alpha = max(-1.5 * sd, min(1.5 * sd, alpha))
                        cand_q = best_x[:]
                        cand_q[d] = x0 + alpha
                        bounce_inplace(cand_q)
                        fq = eval_f(cand_q)
                        if fq < best:
                            best, best_x, best_t = fq, cand_q[:], time.time()
                            improved = True

                if improved:
                    coord_step[d] = min(0.40 * span(d), coord_step[d] * 1.12)
                else:
                    coord_step[d] *= 0.62

        # -------- occasional Powell-like directional burst (strong local) --------
        # Run when we have a decent incumbent and either late in time or recently improved.
        if tfrac > 0.10 and (time.time() - best_t) < 0.50 * stall_seconds:
            if random.random() < (0.06 + 0.12 * tfrac):
                budget = 6 + int(10 * tfrac)
                best_x, best, _, dirs, pow_step0 = powell_local(best_x, best, budget, dirs, pow_step0)
                best_t = time.time()

        # -------- periodic partial refresh --------
        now = time.time()
        if now - last_refresh_t > refresh_period and tfrac > 0.10:
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
                    for d in range(dim):
                        x[d] += randn() * (max(0.05, sigma) * 0.20 * span(d))
                    bounce_inplace(x)
                else:
                    x = rand_vec()
                    if random.random() < 0.50:
                        x = opposite(x)
                        bounce_inplace(x)

                f = eval_f(x)
                pop[wi] = x
                fit[wi] = f
                if f < best:
                    best, best_x, best_t = f, x[:], time.time()

            archive = []
            last_refresh_t = time.time()

        # -------- full restart on stagnation OR low diversity --------
        div = diversity(pop)
        if (time.time() - best_t > stall_seconds) or (tfrac > 0.20 and div < 0.015 and time.time() - best_t > 0.35 * stall_seconds):
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
                new_pop.append(x)
                new_fit.append(eval_f(x))

            for _ in range(n_opp):
                x = rand_vec()
                xo = opposite(x)
                bounce_inplace(xo)
                fx = eval_f(x)
                fo = eval_f(xo)
                if fo < fx:
                    new_pop.append(xo); new_fit.append(fo)
                else:
                    new_pop.append(x);  new_fit.append(fx)

            local_sigma = max(0.08, sigma)
            for _ in range(n_local):
                x = best_x[:]
                for d in range(dim):
                    x[d] += randn() * (local_sigma * 0.22 * span(d))
                bounce_inplace(x)
                new_pop.append(x)
                new_fit.append(eval_f(x))

            pop, fit = new_pop, new_fit
            bi, bv = argmin(fit)
            if bv < best:
                best, best_x = bv, pop[bi][:]
            best_t = time.time()
            last_refresh_t = best_t

            archive = []

            # perturb memories a bit
            for k in range(H):
                MCR[k] = min(0.95, max(0.05, MCR[k] + 0.035 * randn()))
                MF[k] = min(0.95, max(0.05, MF[k] + 0.035 * randn()))

            # reset some local radii
            sigma = max(0.10, sigma)
            coord_step = [max(0.12 * span(i), coord_step[i]) for i in range(dim)]
            pow_step0 = min(0.12, max(0.04, pow_step0 * 1.10))
