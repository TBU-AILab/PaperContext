import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free optimizer (no external libs):
      - Scaled search in [0,1]^dim for numerical stability
      - Low-discrepancy initialization (Halton) + a few random points
      - Maintain an elite set of best points
      - Multiple local runs using (1+1)-ES with 1/5 success rule (fast, robust)
      - Occasional coordinate/pattern refinement around best
      - Time-aware, exception-safe objective calls

    Returns:
      best (float): best objective value found
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # -------- utilities in scaled space --------
    def clip01(u):
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def u_to_x(u):
        return [lows[i] + u[i] * span_safe[i] for i in range(dim)]

    def safe_eval_u(u):
        try:
            return float(func(u_to_x(u)))
        except Exception:
            return float("inf")

    # Box-Muller normal
    _bm_has = False
    _bm_next = 0.0
    def randn():
        nonlocal _bm_has, _bm_next
        if _bm_has:
            _bm_has = False
            return _bm_next
        u1 = random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(max(1e-300, u1)))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_next = z1
        _bm_has = True
        return z0

    def rand_u():
        return [random.random() for _ in range(dim)]

    # -------- Halton sequence for init (better coverage than LHS-lite) --------
    def _first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def _halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = _first_primes(max(1, dim))

    def halton_point(k):  # k>=1
        return [_halton_value(k, primes[d]) for d in range(dim)]

    # -------- time-aware evaluation budget estimate --------
    # Probe a couple evaluations to estimate dt
    probe_u = rand_u()
    t_probe = time.time()
    _ = safe_eval_u(probe_u)
    dt1 = time.time() - t_probe
    t_probe = time.time()
    _ = safe_eval_u(rand_u())
    dt2 = time.time() - t_probe
    eval_dt = max(1e-6, 0.5 * (dt1 + dt2))

    # Conservative max evaluations
    time_left = max(0.0, deadline - time.time())
    max_evals = int(max(20, (time_left / eval_dt) * 0.90))

    evals = 0
    best_val = float("inf")
    best_u = rand_u()

    # -------- keep an elite set --------
    elite_k = max(5, min(25, 2 * dim + 5))
    elite = []  # list of (val, u)

    def elite_add(v, u):
        nonlocal best_val, best_u, elite
        if v < best_val:
            best_val, best_u = v, u[:]
        elite.append((v, u[:]))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_k:
            elite = elite[:elite_k]

    # -------- initialization: Halton + random --------
    init_n = max(12, min(80, 6 * dim + 20))
    init_n = min(init_n, max(12, max_evals // 3))
    # Halton indices start at 1
    for k in range(1, init_n + 1):
        if time.time() >= deadline or evals >= max_evals:
            return best_val
        u = halton_point(k)
        v = safe_eval_u(u)
        evals += 1
        elite_add(v, u)

    extra_r = min(max(0, init_n // 3), max(0, max_evals // 10))
    for _ in range(extra_r):
        if time.time() >= deadline or evals >= max_evals:
            return best_val
        u = rand_u()
        v = safe_eval_u(u)
        evals += 1
        elite_add(v, u)

    # -------- coordinate/pattern refine in scaled space --------
    def coord_refine(u, step0, passes):
        nonlocal evals
        cur = u[:]
        curv = safe_eval_u(cur)
        evals += 1
        elite_add(curv, cur)
        step = step0
        for _ in range(passes):
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for i in order:
                if time.time() >= deadline or evals >= max_evals:
                    return
                up = cur[:]; up[i] += step
                um = cur[:]; um[i] -= step
                clip01(up); clip01(um)
                vp = safe_eval_u(up); evals += 1
                vm = safe_eval_u(um); evals += 1
                if vp < curv or vm < curv:
                    if vp <= vm:
                        cur, curv = up, vp
                    else:
                        cur, curv = um, vm
                    elite_add(curv, cur)
                    improved = True
                    if time.time() >= deadline or evals >= max_evals:
                        return
            if not improved:
                step *= 0.5

    # -------- (1+1)-ES local search with 1/5 success rule --------
    def one_plus_one_es(start_u, start_sigma, max_steps):
        nonlocal evals
        u = start_u[:]
        fu = safe_eval_u(u)
        evals += 1
        elite_add(fu, u)

        sigma = start_sigma
        sigma_min = 1e-6
        sigma_max = 0.5

        # 1/5 success rule window
        succ = 0
        win = 0
        win_size = 12

        # occasional Cauchy-like big jump probability
        for _ in range(max_steps):
            if time.time() >= deadline or evals >= max_evals:
                return

            # build candidate
            v = u[:]
            # mix: mostly Gaussian, sometimes heavier tail
            heavy = (random.random() < 0.12)
            for i in range(dim):
                z = randn()
                if heavy:
                    # approximate heavier tail by scaling with 1/|N| (bounded)
                    z = z / max(0.35, abs(randn()))
                v[i] += sigma * z
            clip01(v)

            fv = safe_eval_u(v)
            evals += 1

            win += 1
            if fv < fu:
                u, fu = v, fv
                elite_add(fu, u)
                succ += 1

            # adapt sigma every window
            if win >= win_size:
                rate = succ / float(win)
                # if too successful -> increase; else decrease
                if rate > 0.20:
                    sigma *= 1.35
                else:
                    sigma *= 0.72
                if sigma < sigma_min:
                    sigma = sigma_min
                elif sigma > sigma_max:
                    sigma = sigma_max
                succ = 0
                win = 0

    # -------- main loop: alternate ES on elites + refinements + occasional global samples --------
    # Base sigma scales with dimension (scaled space)
    base_sigma = 0.20 / math.sqrt(max(1, dim))

    rounds = 0
    while time.time() < deadline and evals < max_evals:
        rounds += 1

        # choose a start: mostly from elite, sometimes random
        if elite and random.random() < 0.85:
            idx = int((random.random() ** 2.2) * len(elite))  # bias to better
            start = elite[idx][1]
        else:
            start = rand_u()

        # jitter start slightly to diversify
        u0 = start[:]
        jit = base_sigma * (1.5 if rounds < 4 else 0.8)
        for i in range(dim):
            u0[i] += randn() * jit
        clip01(u0)

        # ES local steps allocation (time-aware-ish)
        steps = 40 + 10 * dim
        steps = min(steps, 250)

        # sigma schedule: larger earlier, smaller later
        sigma0 = base_sigma * (1.8 if rounds < 4 else 1.0)
        if elite:
            # tighten if we already have many elites (refine)
            sigma0 *= max(0.6, 1.2 - 0.03 * len(elite))

        one_plus_one_es(u0, sigma0, steps)

        # small coordinate refinement around current best
        if time.time() < deadline and evals < max_evals:
            coord_refine(best_u, step0=max(0.01, base_sigma * 0.8), passes=2)

        # occasional pure global sample (insurance)
        if time.time() < deadline and evals < max_evals and random.random() < 0.25:
            ug = halton_point(1 + (rounds * 7) % 9973) if random.random() < 0.6 else rand_u()
            vg = safe_eval_u(ug)
            evals += 1
            elite_add(vg, ug)

    return best_val
