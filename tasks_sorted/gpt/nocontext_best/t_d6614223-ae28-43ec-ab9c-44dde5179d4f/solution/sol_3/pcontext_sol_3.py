import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Hybrid algorithm (robust, fast in limited time):
      1) Multi-start opposition+LHS-ish seeding
      2) Differential Evolution, "current-to-pbest/1" (JADE-like) with adaptive F/CR
         - uses an archive to improve exploration
         - pbest fraction shrinks over time
      3) Occasional low-cost local refinement (coordinate/pattern) on improvements
      4) Stagnation-triggered partial restart of worst individuals

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(bounds[i][0]) for i in range(dim)]
    highs = [float(bounds[i][1]) for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    def now():
        return time.time()

    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opp_vec(x):
        # opposition point around center of bounds
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # simple normal sampler (Box-Muller) to sample F/CR
    spare = [None]
    def randn():
        if spare[0] is not None:
            z = spare[0]
            spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        spare[0] = z1
        return z0

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    # LHS-ish initializer
    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = []
            for d in range(dim):
                u = (perms[d][i] + random.random()) / n
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    # low-cost local refinement (coordinate / pattern search)
    def local_refine(x, fx, budget):
        step = [0.04 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]
        grow = 1.25
        shrink = 0.5

        xb = x[:]
        fb = fx
        evals = 0

        while evals < budget and now() < deadline:
            improved = False
            for d in range(dim):
                if evals >= budget or now() >= deadline:
                    break
                sd = step[d]
                if sd <= min_step[d]:
                    continue

                # + step
                xp = xb[:]
                xp[d] += sd
                clip(xp)
                fp = eval_f(xp); evals += 1
                if fp < fb:
                    xb, fb = xp, fp
                    improved = True
                    continue

                if evals >= budget or now() >= deadline:
                    break

                # - step
                xm = xb[:]
                xm[d] -= sd
                clip(xm)
                fm = eval_f(xm); evals += 1
                if fm < fb:
                    xb, fb = xm, fm
                    improved = True
                    continue

            if improved:
                for d in range(dim):
                    step[d] = min(0.25 * spans[d], step[d] * grow)
            else:
                tiny = True
                for d in range(dim):
                    step[d] = max(min_step[d], step[d] * shrink)
                    if step[d] > 10.0 * min_step[d]:
                        tiny = False
                if tiny:
                    break

        return xb, fb

    # --- population sizing ---
    pop_size = max(18, min(90, 10 * dim))
    if max_time <= 0.2:
        pop_size = max(12, min(pop_size, 20))

    # Initialize population with LHS + opposition points + random fill
    pop = []
    fit = []

    n_lhs = min(pop_size, max(6, int(math.sqrt(pop_size) + 5)))
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return float("inf") if not fit else min(fit)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

        if len(pop) < pop_size and now() < deadline:
            xo = opp_vec(x)
            clip(xo)
            fxo = eval_f(xo)
            pop.append(xo); fit.append(fxo)

    while len(pop) < pop_size and now() < deadline:
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    # Best
    best_i = min(range(len(fit)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]
    if now() >= deadline:
        return best

    # --- JADE-like control ---
    mu_F = 0.65
    mu_CR = 0.55
    F_min, F_max = 0.05, 1.0

    # external archive (stores replaced solutions)
    archive = []
    archive_max = pop_size

    # stagnation / restart controls
    last_best = best
    no_improve_gens = 0
    refine_cooldown = 0

    def pick_indices_excluding(n, excl):
        # returns a random index in [0,n) not in set excl (small set)
        j = random.randrange(n)
        while j in excl:
            j = random.randrange(n)
        return j

    gen = 0
    while now() < deadline:
        gen += 1

        # sort indices by fitness (ascending)
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])

        # pbest fraction: start larger (explore), shrink (exploit)
        remaining = max(0.0, deadline - now())
        frac = remaining / max(1e-12, max_time)  # 1 -> early, 0 -> late
        pfrac = 0.35 * frac + 0.08  # in [0.08, 0.43]
        pcount = max(2, int(pfrac * pop_size))

        # memory of successful parameters
        SF = []
        SCR = []
        dF = []  # fitness improvements for weighting

        improved_this_gen = False

        for i in range(pop_size):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick pbest among top pcount
            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            # sample F ~ N(mu_F, 0.15) but ensure >0; retry few times
            F = None
            for _ in range(6):
                cand = mu_F + 0.15 * randn()
                if cand > 0:
                    F = cand
                    break
            if F is None:
                F = mu_F
            F = clamp(F, F_min, F_max)

            # sample CR ~ N(mu_CR, 0.1) and clamp
            CR = clamp(mu_CR + 0.10 * randn(), 0.0, 1.0)

            # Choose r1 from population, r2 from union(pop, archive)
            r1 = pick_indices_excluding(pop_size, {i, pbest_idx})
            x_r1 = pop[r1]

            union = pop + archive
            union_n = len(union)
            # ensure r2 not equal to xi or x_r1 (by identity of index in pop part)
            # easiest: pick an index from union and allow archive choices freely.
            # if it hits i or r1 in pop-part, resample a few times.
            r2u = random.randrange(union_n)
            for _ in range(10):
                if r2u < pop_size and (r2u == i or r2u == r1):
                    r2u = random.randrange(union_n)
                else:
                    break
            x_r2 = union[r2u]

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d])
            clip(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            fu = eval_f(u)

            if fu <= fi:
                # archive old
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = u
                fit[i] = fu

                # record success
                SF.append(F)
                SCR.append(CR)
                dF.append(max(0.0, fi - fu))

                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_this_gen = True

                    # small local refine on new best (throttled)
                    if refine_cooldown <= 0 and now() < deadline:
                        budget = max(8, min(140, 6 * dim))
                        if frac < 0.35:
                            budget = max(budget, min(240, 10 * dim))
                        xb2, fb2 = local_refine(best_x, best, budget)
                        if fb2 < best:
                            best = fb2
                            best_x = xb2[:]
                        refine_cooldown = 2

        refine_cooldown -= 1

        # update mu_F, mu_CR using successful parameters
        if SF:
            # weighted by improvement
            wsum = sum(dF) + 1e-12
            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for w, F in zip(dF, SF):
                num += w * F * F
                den += w * F
            lehmerF = num / (den + 1e-12)
            meanCR = 0.0
            for w, CR in zip(dF, SCR):
                meanCR += w * CR
            meanCR /= wsum

            c = 0.12  # smoothing
            mu_F = (1.0 - c) * mu_F + c * clamp(lehmerF, F_min, F_max)
            mu_CR = (1.0 - c) * mu_CR + c * clamp(meanCR, 0.0, 1.0)

        # stagnation detection + partial restart of worst
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # if stagnating, re-seed some worst points (but keep best)
        if no_improve_gens >= max(4, int(1.5 * math.sqrt(pop_size))) and now() < deadline:
            no_improve_gens = 0
            k = max(2, pop_size // 6)  # replace ~16%
            worst = sorted(range(pop_size), key=lambda j: fit[j], reverse=True)
            replaced = 0
            for idx in worst:
                if replaced >= k or now() >= deadline:
                    break
                if pop[idx] is best_x:
                    continue
                # mixture of random and opposition of best (diversify)
                if random.random() < 0.5:
                    x = rand_vec()
                else:
                    x = opp_vec(best_x)
                    # add jitter
                    for d in range(dim):
                        x[d] += (random.random() - 0.5) * 0.15 * spans[d]
                    clip(x)
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                replaced += 1
                if fx < best:
                    best = fx
                    best_x = x[:]

            # also trim archive (keep it bounded)
            if len(archive) > archive_max:
                archive = archive[-archive_max:]

    return best
