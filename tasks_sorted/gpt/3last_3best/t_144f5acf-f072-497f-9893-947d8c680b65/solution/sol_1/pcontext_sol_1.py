import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded optimizer (no external libs):
      - LHS-like stratified initialization + a few random points
      - Multi-start Nelder–Mead (simplex) in scaled [0,1]^dim
      - Occasional local Gaussian perturbation + coordinate micro-search
      - Budget/time aware, robust to exceptions

    Returns:
      best (float): best objective value found
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ----- bounds prep -----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]
    # handle degenerate spans
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    def clip01(u):
        # in-place
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def u_to_x(u):
        # map [0,1] -> bounds
        return [lows[i] + u[i] * span_safe[i] for i in range(dim)]

    def safe_eval_u(u):
        # evaluate at u (scaled). robust to failures.
        try:
            x = u_to_x(u)
            v = func(x)
            return float(v)
        except Exception:
            return float("inf")

    def rand_u():
        return [random.random() for _ in range(dim)]

    def lhs_u(n):
        # simple LHS in [0,1]^dim
        coords = []
        for d in range(dim):
            arr = [(k + random.random()) / n for k in range(n)]
            random.shuffle(arr)
            coords.append(arr)
        pts = []
        for k in range(n):
            pts.append([coords[d][k] for d in range(dim)])
        return pts

    # Gaussian using Box-Muller (no numpy)
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

    # ----- choose evaluation budget style via time sampling -----
    # estimate evaluation time quickly (very light)
    probe_u = rand_u()
    probe_t0 = time.time()
    _ = safe_eval_u(probe_u)
    eval_dt = max(1e-6, time.time() - probe_t0)

    # target number of evaluations we can afford
    time_left = max(0.0, deadline - time.time())
    # be conservative: use only part of the theoretical budget
    max_evals = int(max(10, (time_left / eval_dt) * 0.85))

    # ----- initialization: LHS + random -----
    best_val = float("inf")
    best_u = None

    init_n = max(12, min(60, 6 * dim + 12))
    init_n = min(init_n, max(12, max_evals // 6))  # keep it proportional to budget

    # LHS points
    for u in lhs_u(init_n):
        if time.time() >= deadline:
            return best_val
        v = safe_eval_u(u)
        if v < best_val:
            best_val, best_u = v, u[:]

    # extra random points (cheap exploration)
    extra_n = min(max(0, init_n // 2), max(0, max_evals // 10))
    for _ in range(extra_n):
        if time.time() >= deadline:
            return best_val
        u = rand_u()
        v = safe_eval_u(u)
        if v < best_val:
            best_val, best_u = v, u[:]

    if best_u is None:
        best_u = rand_u()
        best_val = safe_eval_u(best_u)

    evals_used = init_n + extra_n + 1

    # ----- Nelder–Mead on [0,1]^dim -----
    # Standard coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho   = 0.5   # contraction
    sigma = 0.5   # shrink

    def nm_optimize(start_u, start_scale, max_iters, max_local_evals):
        nonlocal best_val, best_u, evals_used

        # build simplex around start_u
        n = dim
        simplex = [start_u[:]]
        for i in range(n):
            v = start_u[:]
            v[i] += start_scale
            clip01(v)
            simplex.append(v)

        fvals = []
        for u in simplex:
            if time.time() >= deadline:
                return
            fv = safe_eval_u(u)
            evals_used += 1
            fvals.append(fv)
            if fv < best_val:
                best_val, best_u = fv, u[:]
            if evals_used >= max_evals:
                return

        it = 0
        local_evals = len(simplex)
        while it < max_iters and local_evals < max_local_evals:
            if time.time() >= deadline or evals_used >= max_evals:
                return
            it += 1

            # sort simplex by f
            order = sorted(range(n + 1), key=lambda k: fvals[k])
            simplex = [simplex[k] for k in order]
            fvals   = [fvals[k] for k in order]

            # centroid of best n points
            centroid = [0.0] * n
            for j in range(n):
                s = 0.0
                for i in range(n):  # exclude worst (index n)
                    s += simplex[i][j]
                centroid[j] = s / n

            worst = simplex[-1]
            f_worst = fvals[-1]
            f_best = fvals[0]
            f_second_worst = fvals[-2]

            # reflect
            xr = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
            clip01(xr)
            fr = safe_eval_u(xr)
            evals_used += 1
            local_evals += 1
            if fr < best_val:
                best_val, best_u = fr, xr[:]
            if evals_used >= max_evals:
                return

            if fr < f_best:
                # expand
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
                clip01(xe)
                fe = safe_eval_u(xe)
                evals_used += 1
                local_evals += 1
                if fe < best_val:
                    best_val, best_u = fe, xe[:]
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
                continue

            if fr < f_second_worst:
                # accept reflection
                simplex[-1], fvals[-1] = xr, fr
                continue

            # contraction
            if fr < f_worst:
                # outside contraction
                xc = [centroid[j] + rho * (xr[j] - centroid[j]) for j in range(n)]
            else:
                # inside contraction
                xc = [centroid[j] - rho * (centroid[j] - worst[j]) for j in range(n)]
            clip01(xc)
            fc = safe_eval_u(xc)
            evals_used += 1
            local_evals += 1
            if fc < best_val:
                best_val, best_u = fc, xc[:]
            if evals_used >= max_evals:
                return

            if fc < f_worst:
                simplex[-1], fvals[-1] = xc, fc
                continue

            # shrink toward best
            bestp = simplex[0]
            for i in range(1, n + 1):
                if time.time() >= deadline or evals_used >= max_evals:
                    return
                simplex[i] = [bestp[j] + sigma * (simplex[i][j] - bestp[j]) for j in range(n)]
                clip01(simplex[i])
                fvals[i] = safe_eval_u(simplex[i])
                evals_used += 1
                local_evals += 1
                if fvals[i] < best_val:
                    best_val, best_u = fvals[i], simplex[i][:]
                if evals_used >= max_evals:
                    return

    # ----- multi-start schedule -----
    # start scales in [0,1] coordinates
    # larger first to move quickly, then smaller for refinement
    scales = [0.18, 0.10, 0.06, 0.03, 0.015]

    # build a small pool of promising starts from init LHS + best + random
    # (we don't store all init points; just generate a few around best)
    starts = [best_u[:]]

    # add perturbed variants of best
    for s in (0.08, 0.05, 0.03):
        if len(starts) >= 6:
            break
        u = best_u[:]
        for i in range(dim):
            u[i] += randn() * s
        clip01(u)
        starts.append(u)

    # add some random starts
    while len(starts) < 10:
        starts.append(rand_u())

    # local coordinate micro-search (cheap refinement) around current best
    def coord_refine(u, step0, passes=1):
        nonlocal best_val, best_u, evals_used
        ucur = u[:]
        vcur = safe_eval_u(ucur)
        evals_used += 1
        if vcur < best_val:
            best_val, best_u = vcur, ucur[:]
        step = step0
        for _ in range(passes):
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for i in order:
                if time.time() >= deadline or evals_used >= max_evals:
                    return
                up = ucur[:]
                um = ucur[:]
                up[i] += step
                um[i] -= step
                clip01(up); clip01(um)
                vp = safe_eval_u(up); evals_used += 1
                vm = safe_eval_u(um); evals_used += 1
                if vp < vcur or vm < vcur:
                    if vp <= vm:
                        ucur, vcur = up, vp
                    else:
                        ucur, vcur = um, vm
                    improved = True
                    if vcur < best_val:
                        best_val, best_u = vcur, ucur[:]
                if time.time() >= deadline or evals_used >= max_evals:
                    return
            if not improved:
                step *= 0.5
        if vcur < best_val:
            best_val, best_u = vcur, ucur[:]

    # main loop over starts/scales until time/budget ends
    si = 0
    while time.time() < deadline and evals_used < max_evals:
        start = starts[si % len(starts)]
        scale = scales[min(len(scales) - 1, si // len(starts))]

        # a little "annealed" noise on the start to diversify
        u0 = start[:]
        jitter = scale * 0.5
        for i in range(dim):
            u0[i] += randn() * jitter
        clip01(u0)

        # allocate modest local budget per start
        # (more if dim is small)
        max_iters = 30 + 8 * dim
        max_local_evals = 80 + 20 * dim

        nm_optimize(u0, start_scale=scale, max_iters=max_iters, max_local_evals=max_local_evals)

        # quick coordinate refine at the end of each local run
        if time.time() < deadline and evals_used < max_evals:
            coord_refine(best_u, step0=max(0.002, scale * 0.25), passes=2)

        # occasional global jump if we stagnate (very cheap)
        if time.time() < deadline and evals_used < max_evals and random.random() < 0.35:
            ug = rand_u()
            vg = safe_eval_u(ug)
            evals_used += 1
            if vg < best_val:
                best_val, best_u = vg, ug[:]

        si += 1

    return best_val
