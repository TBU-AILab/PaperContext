import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libs).

    Hybrid strategy tuned for robustness under unknown objective landscapes:
      1) Latin-hypercube-like init + opposites (better initial coverage than pure random)
      2) DE/current-to-pbest/1 (JADE-style) with:
         - per-individual CR from N(muCR, 0.1), F from Cauchy(muF, 0.1)
         - success-history adaptation of muCR, muF
         - p-best selection + "archive" for difference vectors (diversity)
      3) Lightweight local search around incumbent best (coordinate + random direction),
         with decaying step size
      4) Stagnation actions: inject new samples (some near best, some global)

    Returns:
        best (float): best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]

    def now():
        return time.time()

    def clip_inplace(x):
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
            else:
                if x[i] < lo[i]:
                    x[i] = lo[i]
                elif x[i] > hi[i]:
                    x[i] = hi[i]

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                y[i] = lo[i]
            else:
                y[i] = lo[i] + (hi[i] - x[i])
        return y

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
            else:
                x[i] = lo[i] + random.random() * span[i]
        return x

    def rand_unit_vec():
        v = [random.gauss(0.0, 1.0) for _ in range(dim)]
        s = math.sqrt(sum(a*a for a in v))
        if s <= 0.0:
            j = random.randrange(dim)
            v = [0.0] * dim
            v[j] = 1.0
            return v
        return [a / s for a in v]

    # ---- distributions for JADE ----
    def clamp(x, a, b):
        return a if x < a else b if x > b else x

    def rand_cauchy(loc, scale):
        # Inverse CDF: loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def sample_F(muF):
        # re-sample until positive; then clamp
        for _ in range(30):
            f = rand_cauchy(muF, 0.1)
            if f > 0.0:
                return clamp(f, 0.05, 1.0)
        return clamp(muF, 0.05, 1.0)

    def sample_CR(muCR):
        cr = random.gauss(muCR, 0.1)
        return clamp(cr, 0.0, 1.0)

    # ---- population size ----
    # Slightly larger than your #1 sometimes helps, but keep bounded.
    pop_size = max(18, min(90, 14 * dim))

    # ---- LHS-like initialization ----
    # For each dimension, create permuted strata; then build vectors.
    strata = []
    for d in range(dim):
        if fixed[d]:
            strata.append([lo[d]] * pop_size)
        else:
            perm = list(range(pop_size))
            random.shuffle(perm)
            vals = []
            for k in range(pop_size):
                # sample uniformly inside stratum
                u = (perm[k] + random.random()) / pop_size
                vals.append(lo[d] + u * span[d])
            strata.append(vals)

    pop = []
    for i in range(pop_size):
        x = [strata[d][i] for d in range(dim)]
        pop.append(x)

    # Add opposites and keep best pop_size
    candidates = []
    for x in pop:
        candidates.append(x)
        candidates.append(opposite(x))

    scored = []
    for x in candidates:
        if now() >= deadline:
            break
        clip_inplace(x)
        scored.append((safe_eval(x), x[:]))
    if not scored:
        return float("inf")
    scored.sort(key=lambda t: t[0])
    pop = [x for _, x in scored[:pop_size]]
    fit = [f for f, _ in scored[:pop_size]]

    best_i = min(range(len(fit)), key=lambda i: fit[i])
    best = fit[best_i]
    best_x = pop[best_i][:]

    # ---- JADE parameters ----
    muF = 0.6
    muCR = 0.5
    c = 0.1  # learning rate for mu updates
    p = 0.2  # p-best fraction
    archive = []  # stores replaced solutions for diversity
    arch_max = pop_size

    # ---- local search step sizes ----
    step = [0.20 * s for s in span]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # ---- stagnation ----
    no_improve_gens = 0
    stagnate_after = max(18, 5 * dim)
    local_every = 2  # attempt local search frequently, but very small budget each time

    gen = 0
    while now() < deadline:
        gen += 1

        # sort indices by fitness (for p-best selection)
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p * pop_size)))

        SF = []
        SCR = []
        dF_sum = 0.0  # sum of improvements for weighting

        improved_gen = False

        for i in range(pop_size):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick p-best
            pbest_idx = order[random.randrange(pcount)]
            xp = pop[pbest_idx]

            # choose r1 from population excluding i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            xr1 = pop[r1]

            # choose r2 from pop U archive excluding i and r1 and pbest ideally
            pool = pop + archive
            # try a few times to avoid collisions; if not possible, accept
            r2 = None
            for _ in range(20):
                cand = random.randrange(len(pool))
                x2 = pool[cand]
                if x2 is xi or x2 is xr1 or x2 is xp:
                    continue
                r2 = cand
                break
            if r2 is None:
                r2 = random.randrange(len(pool))
            xr2 = pool[r2]

            CR = sample_CR(muCR)
            F = sample_F(muF)

            # mutation: current-to-pbest/1 with archive
            v = [0.0] * dim
            for d in range(dim):
                if fixed[d]:
                    v[d] = lo[d]
                else:
                    v[d] = xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
            clip_inplace(u)

            fu = safe_eval(u)

            if fu <= fi:
                # success
                pop[i] = u
                fit[i] = fu

                # archive keeps old xi
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_gen = True

                # record for parameter adaptation (weighted by improvement)
                df = max(0.0, fi - fu)
                if df > 0.0:
                    SF.append(F)
                    SCR.append(CR)
                    dF_sum += df
                else:
                    # still a "success" but no improvement; low weight
                    SF.append(F)
                    SCR.append(CR)
                    dF_sum += 1e-12

        # adapt muF, muCR
        if SF:
            # weights proportional to improvements (JADE)
            # if all df are tiny, this still behaves well.
            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            # arithmetic mean for CR: sum(w*CR)/sum(w)
            # build weights from improvements implicitly by using df; if not tracked per item,
            # approximate equal weights (still works). We'll approximate equal weights for speed.
            # (Keeping strict df weights would require storing df per success.)
            # Use robust updates:
            meanCR = sum(SCR) / len(SCR)
            num = sum(f*f for f in SF)
            den = sum(SF)
            meanF = (num / den) if den > 0 else muF

            muCR = (1 - c) * muCR + c * meanCR
            muF = (1 - c) * muF + c * meanF
            muCR = clamp(muCR, 0.0, 1.0)
            muF = clamp(muF, 0.05, 1.0)

        # occasional opposition injection for a few worst
        if (gen % max(8, 2 * dim)) == 0 and now() + 0.002 < deadline:
            worst = order[-max(1, pop_size // 10):]
            for idx in worst:
                if now() >= deadline:
                    return best
                ox = opposite(pop[idx])
                clip_inplace(ox)
                fox = safe_eval(ox)
                if fox < fit[idx]:
                    pop[idx] = ox
                    fit[idx] = fox
                    if fox < best:
                        best = fox
                        best_x = ox[:]
                        improved_gen = True

        # local refinement around best (tiny budget, frequent)
        if (gen % local_every) == 0 and now() + 0.002 < deadline:
            trials = 0
            max_trials = min(6 + dim, 28)
            while trials < max_trials and now() < deadline:
                trials += 1
                x = best_x[:]

                if random.random() < 0.7:
                    d = random.randrange(dim)
                    if fixed[d] or step[d] <= min_step[d]:
                        continue
                    x[d] += step[d] * (1.0 if random.random() < 0.5 else -1.0)
                else:
                    dirv = rand_unit_vec()
                    # average step across non-fixed dims
                    ssum = 0.0
                    cnt = 0
                    for d in range(dim):
                        if not fixed[d]:
                            ssum += step[d]
                            cnt += 1
                    if cnt == 0:
                        break
                    scale = (ssum / cnt) * (0.35 + 0.65 * random.random())
                    for d in range(dim):
                        if not fixed[d]:
                            x[d] += scale * dirv[d]

                clip_inplace(x)
                fx = safe_eval(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    improved_gen = True
                else:
                    # gentle decay
                    for d in range(dim):
                        if not fixed[d] and step[d] > min_step[d]:
                            step[d] *= 0.997

        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # stagnation: inject new samples (mix of near-best and global)
        if no_improve_gens >= stagnate_after and now() + 0.01 < deadline:
            no_improve_gens = 0
            # shrink steps to focus
            for d in range(dim):
                if not fixed[d] and step[d] > min_step[d]:
                    step[d] = max(min_step[d], step[d] * 0.7)

            # replace some worst
            order = sorted(range(pop_size), key=lambda i: fit[i])
            replace = order[-max(2, pop_size // 5):]
            for idx in replace:
                if now() >= deadline:
                    return best
                if random.random() < 0.65:
                    x = best_x[:]
                    for d in range(dim):
                        if not fixed[d]:
                            sd = max(step[d], 1e-15)
                            x[d] += random.gauss(0.0, sd)
                else:
                    x = rand_vec()
                clip_inplace(x)
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

            # keep archive bounded and fresh
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]

    return best
