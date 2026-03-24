import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs).
    Hybrid strategy:
      1) Low-discrepancy (Halton) + random exploration
      2) (1+λ)-ES local search with SUCCESS-BASED step-size adaptation (1/5 rule style)
      3) Coordinate/pattern search refinement (cheap, robust)
      4) Occasional "heavy-tail" jumps + restarts on stagnation
    Returns:
      best (float): best (minimum) fitness found within max_time seconds
    """

    # -------------------- basic helpers --------------------
    t0 = time.time()
    deadline = t0 + max_time

    def now():
        return time.time()

    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def in_bounds(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # ranges and eps
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    scale = [r if r > 0 else 1.0 for r in ranges]

    # -------------------- Halton sequence (fast, deterministic) --------------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(idx, base):
        vdc = 0.0
        denom = 1.0
        while idx > 0:
            idx, rem = divmod(idx, base)
            denom *= base
            vdc += rem / denom
        return vdc

    bases = first_primes(dim)
    halton_k = 1

    def halton_point(k):
        u = [van_der_corput(k, bases[i]) for i in range(dim)]
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            x.append(lo + u[i] * (hi - lo))
        return x

    # -------------------- initialization: spend small budget on coverage --------------------
    best = float("inf")
    best_x = None

    # a bit more init than before: often pays off greatly
    init_n = max(30, 12 * dim)
    for _ in range(init_n):
        if now() >= deadline:
            return best
        if random.random() < 0.75:
            x = halton_point(halton_k); halton_k += 1
        else:
            x = rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        return best

    # -------------------- local search state --------------------
    # global step-size (normalized to each dimension by range)
    # start slightly larger than before; will adapt quickly
    sigma_global = 0.20  # in units of "fraction of range"
    sigma_min = 1e-12
    sigma_max = 0.75

    # ES parameters
    lam = max(8, 4 * dim)  # offspring per generation
    # success-based adaptation target ~ 0.2
    success_target = 0.20
    adapt_up = 1.25
    adapt_down = 0.85

    # stagnation / restart
    no_improve = 0
    stagnate_limit = 60 + 15 * dim

    # pattern search step per coordinate (absolute in variable units)
    coord_step = [0.05 * scale[i] for i in range(dim)]
    coord_step_min = [1e-12 * scale[i] for i in range(dim)]

    # -------------------- sampling utilities --------------------
    def cauchy():
        # heavy-tail step: tan(pi*(u-0.5))
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    def make_es_candidate(center):
        # mixture: mostly Gaussian (good local), sometimes Cauchy (escapes)
        x = []
        use_cauchy = (random.random() < 0.10)
        for i in range(dim):
            if use_cauchy:
                step = 0.15 * sigma_global * scale[i] * cauchy()
            else:
                step = random.gauss(0.0, sigma_global * scale[i])
            x.append(center[i] + step)
        return in_bounds(x)

    def try_pattern_refine(x0, f0):
        # simple coordinate search (one pass) with current coord_step
        x = x0[:]
        f = f0
        improved = False
        for i in range(dim):
            if now() >= deadline:
                break
            step = coord_step[i]
            if step <= coord_step_min[i]:
                continue

            # try +step
            xp = x[:]
            xp[i] = clip(xp[i] + step, bounds[i][0], bounds[i][1])
            fp = eval_f(xp)
            if fp < f:
                x, f = xp, fp
                improved = True
                continue

            if now() >= deadline:
                break

            # try -step
            xm = x[:]
            xm[i] = clip(xm[i] - step, bounds[i][0], bounds[i][1])
            fm = eval_f(xm)
            if fm < f:
                x, f = xm, fm
                improved = True

        return x, f, improved

    # -------------------- main loop --------------------
    # Alternate: ES generations + periodic pattern search.
    gen = 0
    while True:
        if now() >= deadline:
            return best

        gen += 1

        # --- (1+λ)-ES around best_x ---
        best_gen_x = None
        best_gen_f = float("inf")
        success = 0

        for _ in range(lam):
            if now() >= deadline:
                return best
            # occasional global injection
            if random.random() < 0.08:
                # mix halton and random
                if random.random() < 0.7:
                    cand = halton_point(halton_k); halton_k += 1
                else:
                    cand = rand_point()
            else:
                cand = make_es_candidate(best_x)

            fc = eval_f(cand)
            if fc < best_gen_f:
                best_gen_f = fc
                best_gen_x = cand
            if fc < best:
                success += 1

        # Adapt sigma based on success rate
        succ_rate = success / float(lam)
        if succ_rate > success_target:
            sigma_global = min(sigma_max, sigma_global * adapt_up)
        else:
            sigma_global = max(sigma_min, sigma_global * adapt_down)

        # Accept the best offspring if it improves
        if best_gen_f < best:
            best = best_gen_f
            best_x = best_gen_x
            no_improve = 0
            # when we improve, also allow slightly larger coordinate steps
            for i in range(dim):
                coord_step[i] = min(0.25 * scale[i], coord_step[i] * 1.10)
        else:
            no_improve += 1
            # shrink coordinate steps slowly when not improving
            for i in range(dim):
                coord_step[i] = max(coord_step_min[i], coord_step[i] * 0.95)

        # --- periodic deterministic refinement ---
        # Run a coordinate pass every few generations (cheap and robust).
        if (gen % 4 == 0) and (now() < deadline):
            x2, f2, imp = try_pattern_refine(best_x, best)
            if f2 < best:
                best, best_x = f2, x2
                no_improve = 0
                # after a refine improvement, tighten sigma to exploit
                sigma_global = max(sigma_min, sigma_global * 0.90)

        # --- stagnation handling: restart/diversify ---
        if no_improve >= stagnate_limit:
            no_improve = 0

            # keep best, but jump the center near good regions via multi-try global samples
            trials = max(10, 4 * dim)
            candidate_centers = []
            for _ in range(trials):
                if now() >= deadline:
                    return best
                if random.random() < 0.70:
                    x = halton_point(halton_k); halton_k += 1
                else:
                    x = rand_point()
                fx = eval_f(x)
                candidate_centers.append((fx, x))
                if fx < best:
                    best, best_x = fx, x

            candidate_centers.sort(key=lambda t: t[0])
            # pick one of the top few (adds diversity)
            pick_pool = min(len(candidate_centers), max(3, dim // 2))
            best_x = candidate_centers[random.randrange(pick_pool)][1]

            # reset step sizes to explore again
            sigma_global = min(0.35, max(0.08, sigma_global * 1.5))
            coord_step = [0.08 * scale[i] for i in range(dim)]
