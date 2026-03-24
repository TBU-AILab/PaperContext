import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Key improvements vs provided version:
      - Adds a robust trust-region local optimizer (BOBYQA-lite / Powell-style) in normalized space.
        This typically gives large gains on smooth-ish problems and still helps on noisy ones.
      - Keeps a simpler, stronger global phase: DE/current-to-best/1 with success-based parameter adaption.
      - Better restart logic: diversified restarts + best-centered sampling with shrinking radius.
      - Removes heavy cache quantization (can hurt); replaces with tiny LRU-ish exact cache on rounded u.
      - More disciplined late exploitation: trust-region dominates near the end.

    Returns:
      best fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-9

    # ---------- helpers ----------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if span[i] <= 0.0:
            span[i] = 0.0

    def clamp01(u):
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def u_to_x(u):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] == 0.0:
                x[i] = lo[i]
            else:
                ui = u[i]
                if ui < 0.0:
                    ui = 0.0
                elif ui > 1.0:
                    ui = 1.0
                x[i] = lo[i] + ui * span[i]
        return x

    def rand_u():
        return [random.random() for _ in range(dim)]

    # Box-Muller gaussian
    _has_spare = False
    _spare = 0.0
    def gauss():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare = z1
        _has_spare = True
        return z0

    # Small rounded-key cache (helps when local search revisits points)
    cache = {}
    cache_order = []
    CACHE_MAX = 8000 if dim <= 20 else (5000 if dim <= 60 else 3000)

    def cache_key(u):
        # rounding rather than bucketing: keeps locality without collapsing too much
        # fewer decimals in high dim to keep key size manageable
        dec = 6 if dim <= 12 else (5 if dim <= 30 else 4)
        return tuple(round(ui, dec) for ui in u)

    evals = 0
    def evaluate_u(u):
        nonlocal evals
        u = clamp01(u[:])
        k = cache_key(u)
        if k in cache:
            return cache[k], u
        x = u_to_x(u)
        try:
            v = float(func(x))
            if not is_finite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        cache_order.append(k)
        if len(cache_order) > CACHE_MAX:
            old = cache_order.pop(0)
            # guard in case duplicates
            if old in cache:
                del cache[old]
        evals += 1
        return v, u

    # ---------- Global phase: DE with success-based adaption ----------
    pop = int(14 + 4 * math.sqrt(max(1, dim)))
    pop = max(18, min(90, pop))

    U = []
    F = []
    best_u = None
    best_f = float("inf")

    # diversified init: random + opposite + a few axis-biased points
    init = pop
    for i in range(init):
        if time.time() >= deadline:
            return best_f
        u = rand_u()
        fu, u = evaluate_u(u)
        U.append(u); F.append(fu)
        if fu < best_f:
            best_f, best_u = fu, u[:]

        if time.time() >= deadline:
            return best_f
        uo = [1.0 - ui for ui in u]
        fo, uo = evaluate_u(uo)
        U.append(uo); F.append(fo)
        if fo < best_f:
            best_f, best_u = fo, uo[:]

    # add a handful of "almost-corners" (helps on boundary optima)
    corner_trials = min(2 * dim, 24)
    for _ in range(corner_trials):
        if time.time() >= deadline:
            return best_f
        u = [0.03 if random.random() < 0.5 else 0.97 for _ in range(dim)]
        # randomize a few dims away from corners
        for _k in range(max(1, dim // 6)):
            j = random.randrange(dim)
            u[j] = random.random()
        fu, u = evaluate_u(u)
        U.append(u); F.append(fu)
        if fu < best_f:
            best_f, best_u = fu, u[:]

    # trim to pop
    order = list(range(len(F)))
    order.sort(key=lambda i: F[i])
    order = order[:pop]
    U = [U[i] for i in order]
    F = [F[i] for i in order]

    # DE adaptive memories
    Fm = 0.6
    Crm = 0.85
    succ_F = []
    succ_Cr = []

    last_best = best_f
    last_improve_t = time.time()
    restarts = 0

    # ---------- Trust-region local search (Powell / BOBYQA-lite) ----------
    def trust_region_local(u0, f0, time_limit_frac=0.18):
        """
        Derivative-free trust-region with rotating coordinate directions + quadratic-ish step acceptance.
        Works in u-space. Very evaluation-efficient late in the run.
        """
        if u0 is None:
            return f0, u0

        t_start = time.time()
        t_limit = t_start + max(0.0, time_limit_frac * max_time)

        u = u0[:]
        fu = f0

        # initial radius based on remaining time and dimension
        rem = max(0.0, deadline - time.time())
        # start modest; shrink/expand based on success
        rad = 0.25 if rem > 0.6 * max_time else 0.18
        rad = min(rad, 0.35)
        rad_min = 1e-7
        rad_max = 0.5

        # directions: start with basis, later add a few random directions
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = 1.0
            dirs.append(d)

        # try to keep evaluation count low in high dim
        extra_dirs = 0 if dim <= 10 else (4 if dim <= 30 else 6)
        for _ in range(extra_dirs):
            v = [gauss() for _ in range(dim)]
            n = math.sqrt(sum(x*x for x in v)) or 1.0
            v = [x / n for x in v]
            dirs.append(v)

        no_improve_rounds = 0
        while time.time() < min(deadline, t_limit):
            improved_any = False

            # explore along directions (both signs), accept best
            best_cand_u = None
            best_cand_f = fu

            # shuffle to avoid bias
            random.shuffle(dirs)

            for dvec in dirs:
                if time.time() >= min(deadline, t_limit):
                    break

                # propose +/- step
                for sgn in (1.0, -1.0):
                    uc = u[:]
                    step = rad * sgn
                    for j in range(dim):
                        uc[j] += step * dvec[j]
                    uc = clamp01(uc)
                    fc, uc = evaluate_u(uc)
                    if fc < best_cand_f:
                        best_cand_f = fc
                        best_cand_u = uc[:]

            if best_cand_u is not None:
                # successful step
                u = best_cand_u
                fu = best_cand_f
                improved_any = True
                rad = min(rad_max, rad * 1.35)
                no_improve_rounds = 0
            else:
                # no improvement: shrink radius
                rad *= 0.55
                no_improve_rounds += 1

            if rad < rad_min:
                break
            if no_improve_rounds >= 3 and rad < 1e-4:
                break

        return fu, u

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = (time.time() - t0) / max(eps_time, max_time)
        rem = deadline - time.time()

        # stagnation detection + restart
        if best_f < last_best - 1e-12:
            last_best = best_f
            last_improve_t = time.time()

        stagn = time.time() - last_improve_t
        if stagn > (0.18 + 0.06 * min(4, restarts)) * max_time and rem > 0.08 * max_time:
            restarts += 1
            last_improve_t = time.time()

            # keep elite; refill with mix of:
            #  - best-centered gaussian with decreasing sigma
            #  - random/opposition
            ord2 = list(range(pop))
            ord2.sort(key=lambda i: F[i])
            keep = max(6, pop // 4)
            U = [U[i] for i in ord2[:keep]]
            F = [F[i] for i in ord2[:keep]]

            sigma = max(0.04, 0.22 / (1.0 + 0.5 * restarts))
            while len(U) < pop and time.time() < deadline:
                r = random.random()
                if best_u is not None and r < 0.65:
                    u = best_u[:]
                    for j in range(dim):
                        u[j] += sigma * gauss()
                    u = clamp01(u)
                else:
                    u = rand_u()
                fu, u = evaluate_u(u)
                U.append(u); F.append(fu)
                if fu < best_f:
                    best_f, best_u = fu, u[:]

                if len(U) < pop and time.time() < deadline and random.random() < 0.25:
                    uo = [1.0 - ui for ui in u]
                    fo, uo = evaluate_u(uo)
                    U.append(uo); F.append(fo)
                    if fo < best_f:
                        best_f, best_u = fo, uo[:]

        # decide phase: DE early/mid, local trust-region late
        if elapsed > 0.72 or rem < 0.25 * max_time:
            # spend most remaining time in local, but interleave tiny global kicks
            if best_u is not None:
                f_loc, u_loc = trust_region_local(best_u, best_f, time_limit_frac=0.12 if elapsed < 0.9 else 0.20)
                if f_loc < best_f:
                    best_f, best_u = f_loc, u_loc[:]
                    # inject improved best into population replacing worst
                    worst = max(range(pop), key=lambda i: F[i])
                    U[worst] = best_u[:]
                    F[worst] = best_f

            # small "kick" to avoid local traps
            if time.time() < deadline and random.random() < 0.35:
                u = best_u[:] if best_u is not None else rand_u()
                kick = 0.08 if elapsed < 0.9 else 0.04
                for j in range(dim):
                    u[j] += kick * gauss()
                u = clamp01(u)
                fu, u = evaluate_u(u)
                if fu < best_f:
                    best_f, best_u = fu, u[:]
                # replace worst if good
                w = max(range(pop), key=lambda i: F[i])
                if fu < F[w]:
                    U[w] = u[:]
                    F[w] = fu
            continue

        # --- DE/current-to-best/1/bin with success-adaptation ---
        # sample parameters around memories
        def sample_F():
            # jDE-style
            v = Fm + 0.25 * gauss()
            if v < 0.1: v = 0.1
            if v > 1.0: v = 1.0
            return v

        def sample_Cr():
            v = Crm + 0.20 * gauss()
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            return v

        # ensure best index known
        best_i = min(range(pop), key=lambda i: F[i])
        if F[best_i] < best_f:
            best_f = F[best_i]
            best_u = U[best_i][:]

        indices = list(range(pop))
        random.shuffle(indices)

        gen_succ_F = []
        gen_succ_Cr = []

        for i in indices:
            if time.time() >= deadline:
                break

            Fi = sample_F()
            Cri = sample_Cr()

            # pick r1,r2 distinct and not i
            r1 = r2 = i
            while r1 == i:
                r1 = random.randrange(pop)
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop)

            xi = U[i]
            xb = U[best_i]
            x1 = U[r1]
            x2 = U[r2]

            # current-to-best/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xb[j] - xi[j]) + Fi * (x1[j] - x2[j])

            # bin crossover
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if random.random() < Cri or j == jrand:
                    trial[j] = v[j]
            trial = clamp01(trial)

            ft, trial = evaluate_u(trial)
            if ft <= F[i]:
                U[i] = trial
                F[i] = ft
                gen_succ_F.append(Fi)
                gen_succ_Cr.append(Cri)
                if ft < best_f:
                    best_f = ft
                    best_u = trial[:]
                    best_i = i

        # update memories using successful params
        if gen_succ_F:
            # Lehmer mean for F, arithmetic for Cr
            num = sum(f*f for f in gen_succ_F)
            den = sum(gen_succ_F) + 1e-12
            Fm = max(0.15, min(0.95, num / den))
            Crm = max(0.05, min(0.98, sum(gen_succ_Cr) / len(gen_succ_Cr)))

        # occasional micro-local improvement on best (cheap)
        if best_u is not None and (gen % 3 == 0) and (time.time() + 0.002 < deadline):
            f_loc, u_loc = trust_region_local(best_u, best_f, time_limit_frac=0.03)
            if f_loc < best_f:
                best_f, best_u = f_loc, u_loc[:]
                worst = max(range(pop), key=lambda k: F[k])
                U[worst] = best_u[:]
                F[worst] = best_f

    return best_f
