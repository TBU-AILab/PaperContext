import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (no external libs).

    Main changes vs previous:
      - Proper population-based DE (jDE: self-adaptive F/CR per individual)
      - Opposition-based initialization + Latin-hypercube-ish stratification
      - Periodic local refinement on current best using (1+1)-ES with 1/5th rule
      - Cheap diversity maintenance (replace worst sometimes; occasional re-seeding)
      - Careful time checks; returns best fitness found

    Works well as a robust general-purpose black-box bounded optimizer.
    """

    t_end = time.time() + float(max_time)

    # ---- bounds / scaling helpers ----
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    for i in range(dim):
        if hi[i] < lo[i]:
            lo[i], hi[i] = hi[i], lo[i]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand01():
        return random.random()

    def random_point():
        return [lo[i] + rand01() * span[i] for i in range(dim)]

    def opposite_point(x):
        # Opposite in box: x' = lo + hi - x
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def eval_f(x):
        return float(func(list(x)))

    def lhs_like_points(n):
        # Per-dimension stratified sampling with independent permutations
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = [0.0] * dim
            for j in range(dim):
                u = (perms[j][i] + rand01()) / n
                x[j] = lo[j] + u * span[j]
            pts.append(x)
        return pts

    # ---- quick gaussian generator (Box-Muller) ----
    def gauss():
        u1 = max(1e-12, rand01())
        u2 = rand01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---- choose population size (kept modest for speed) ----
    # DE typically wants >= 5*dim but that's often too expensive; we compromise.
    NP = max(12, min(80, 10 + 3 * dim))

    # ---- initialize population: LHS-like + opposition; keep best NP ----
    pop = []
    fits = []

    # Create 2*NP candidates then select best NP to start (opposition helps a lot)
    init_pts = lhs_like_points(NP)
    cand = []
    for x in init_pts:
        cand.append(x)
        cand.append(opposite_point(x))

    # If dim is small, add a few pure random injections for robustness
    extra = max(0, min(2 * NP, 10))
    for _ in range(extra):
        cand.append(random_point())

    # Evaluate until time runs out or candidates exhausted
    scored = []
    for x in cand:
        if time.time() >= t_end:
            break
        clip_inplace(x)
        fx = eval_f(x)
        scored.append((fx, x))

    if not scored:
        return float("inf")

    scored.sort(key=lambda t: t[0])
    scored = scored[:NP]

    for fx, x in scored:
        pop.append(list(x))
        fits.append(float(fx))

    best = fits[0]
    best_x = list(pop[0])

    # ---- jDE parameters per individual ----
    # Each individual has its own F and CR that self-adapt.
    F = [0.5] * NP
    CR = [0.9] * NP
    tau1 = 0.1  # probability to change F
    tau2 = 0.1  # probability to change CR

    # ---- local refinement state (1+1-ES around best) ----
    # step sizes in absolute coordinates
    es_step = [0.2 * s if s > 0 else 0.0 for s in span]
    es_min = [max(1e-12, 1e-10 * s) if s > 0 else 0.0 for s in span]
    es_max = [0.5 * s if s > 0 else 0.0 for s in span]
    es_trials = 0
    es_succ = 0

    def local_refine_burst(burst_evals):
        nonlocal best, best_x, es_trials, es_succ, es_step
        # (1+1)-ES with 1/5 success rule
        for _ in range(burst_evals):
            if time.time() >= t_end:
                return
            # propose
            y = list(best_x)
            for j in range(dim):
                if span[j] <= 0:
                    continue
                y[j] += gauss() * es_step[j]
            clip_inplace(y)
            fy = eval_f(y)

            es_trials += 1
            if fy < best:
                best = fy
                best_x = y
                es_succ += 1

            # adapt every 20 trials
            if es_trials >= 20:
                rate = es_succ / float(es_trials)
                # 1/5 rule: if success > 0.2 increase step else decrease
                if rate > 0.2:
                    mult = 1.25
                else:
                    mult = 0.82
                for j in range(dim):
                    if span[j] <= 0:
                        continue
                    es_step[j] = max(es_min[j], min(es_max[j], es_step[j] * mult))
                es_trials = 0
                es_succ = 0

    # ---- DE operators ----
    def pick3distinct(exclude):
        # return indices a,b,c all distinct and != exclude
        a = exclude
        while a == exclude:
            a = random.randrange(NP)
        b = a
        while b == a or b == exclude:
            b = random.randrange(NP)
        c = b
        while c == a or c == b or c == exclude:
            c = random.randrange(NP)
        return a, b, c

    # ---- main loop ----
    gen = 0
    while time.time() < t_end:
        gen += 1

        # Occasionally spend a few evals on local refinement (cheap and effective)
        # Frequency increases as time passes (more exploitation near end).
        if gen % 7 == 0:
            # keep this small to not starve DE
            local_refine_burst(burst_evals=2 + (dim // 10))

        # One DE "generation" (iterating individuals; time-bounded)
        for i in range(NP):
            if time.time() >= t_end:
                return best

            # jDE self-adaptation
            Fi = F[i]
            CRi = CR[i]
            if rand01() < tau1:
                # Fi in (0.1, 0.9)
                Fi = 0.1 + 0.8 * rand01()
            if rand01() < tau2:
                CRi = rand01()

            # Mutation: current-to-best/1 + rand/1 blend (robust)
            a, b, c = pick3distinct(i)
            xi = pop[i]
            xa = pop[a]
            xb = pop[b]
            xc = pop[c]

            # choose a "p-best" from top fraction for stronger guidance
            p = max(2, int(0.2 * NP))
            pbest_idx = random.randrange(p)
            # ensure sorted order for p-best selection cheaply by scanning
            # (NP is small; O(NP) scan is fine)
            # find p-best index by partial selection:
            # create list of indices sorted by fits once per generation? too costly.
            # We'll do a quick selection: sample a few and take best.
            # But for stability, use true p-best by a small partial approach.
            # Here: do a tiny tournament among 6 randoms + current best.
            best_idx = 0
            cand_idx = [0]
            for _ in range(min(6, NP-1)):
                cand_idx.append(random.randrange(NP))
            for idx in cand_idx:
                if fits[idx] < fits[best_idx]:
                    best_idx = idx
            xbest = pop[best_idx]

            # Build mutant
            v = [0.0] * dim
            for j in range(dim):
                # current-to-best plus differential
                v[j] = xi[j] + Fi * (xbest[j] - xi[j]) + Fi * (xa[j] - xb[j]) + 0.2 * Fi * (xb[j] - xc[j])

            # Binomial crossover
            u = list(xi)
            jrand = random.randrange(dim)
            for j in range(dim):
                if rand01() < CRi or j == jrand:
                    u[j] = v[j]

            clip_inplace(u)

            fu = eval_f(u)
            if fu <= fits[i]:
                pop[i] = u
                fits[i] = fu
                F[i] = Fi
                CR[i] = CRi

                if fu < best:
                    best = fu
                    best_x = list(u)

        # Diversity maintenance / reseed:
        # If population collapses, inject one random point replacing current worst.
        if gen % 11 == 0 and time.time() < t_end:
            # compute spread quickly
            fmin = best
            fmax = max(fits)
            if fmax - fmin < 1e-12:
                # full collapse
                worst = max(range(NP), key=lambda k: fits[k])
                x = random_point()
                if time.time() >= t_end:
                    break
                fx = eval_f(x)
                pop[worst] = x
                fits[worst] = fx
                # reset its parameters
                F[worst] = 0.5
                CR[worst] = 0.9
                if fx < best:
                    best = fx
                    best_x = list(x)
            else:
                # mild: sometimes replace worst if clearly bad
                if rand01() < 0.25:
                    worst = max(range(NP), key=lambda k: fits[k])
                    x = random_point()
                    if time.time() >= t_end:
                        break
                    fx = eval_f(x)
                    if fx < fits[worst]:
                        pop[worst] = x
                        fits[worst] = fx
                        if fx < best:
                            best = fx
                            best_x = list(x)

    return best
