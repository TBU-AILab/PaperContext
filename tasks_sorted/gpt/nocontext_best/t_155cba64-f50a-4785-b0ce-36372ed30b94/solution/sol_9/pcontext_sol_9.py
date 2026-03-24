import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Key upgrades vs the provided algorithm:
      1) Better global exploration: Latin-hypercube style init + occasional heavy-tailed jumps.
      2) Stronger local convergence: embedded Nelder–Mead (simplex) around incumbent.
      3) More reliable step control: success-based adaptation + per-dimension scaling in normalized space.
      4) Less wasted evals: quantized cache (near-duplicate suppression).
    Returns: best (minimum) fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        try:
            v = float(func([]))
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    # --- bounds prep ---
    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if not (span[i] > 0.0):
            span[i] = 1.0

    def repair_u(u):
        # clamp to [0,1]
        out = u[:]
        for i in range(dim):
            if out[i] < 0.0:
                out[i] = 0.0
            elif out[i] > 1.0:
                out[i] = 1.0
        return out

    def to_x(u):
        return [lo[i] + span[i] * u[i] for i in range(dim)]

    # --- caching on a quantized grid in normalized space ---
    grid = {}
    # a little coarser cache in higher dim to keep memory in check
    q = 8e-5 if dim <= 8 else (2.5e-4 if dim <= 20 else 7.5e-4)

    def key_u(u):
        # tuple of ints
        return tuple(int(u[i] / q + 0.5) for i in range(dim))

    def eval_u(u):
        uu = repair_u(u)
        k = key_u(uu)
        v = grid.get(k)
        if v is not None:
            return v, uu
        x = to_x(uu)
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")
        grid[k] = v
        return v, uu

    def rand_u():
        return [random.random() for _ in range(dim)]

    # --- init: Latin-hypercube-ish batches (good space-filling quickly) ---
    best = float("inf")
    best_u = rand_u()

    def lhs_batch(n):
        # returns n points in [0,1]^dim with per-dim stratification
        pts = [[0.0] * dim for _ in range(n)]
        for j in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            invn = 1.0 / n
            for i in range(n):
                # jitter inside stratum
                pts[i][j] = (perm[i] + random.random()) * invn
        return pts

    n_init = max(40, min(360, 26 * dim + 60))
    batch = 40 if n_init >= 40 else n_init
    done = 0
    while done < n_init and time.time() < deadline:
        n = min(batch, n_init - done)
        for u in lhs_batch(n):
            if time.time() >= deadline:
                return best
            v, uu = eval_u(u)
            if v < best:
                best, best_u = v, uu
        done += n

    # add a few random probes
    for _ in range(max(12, dim)):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(rand_u())
        if v < best:
            best, best_u = v, uu

    # --- Nelder-Mead in normalized space (bounded via repair) ---
    def nelder_mead(u0, v0, iters, step0):
        """
        Lightweight Nelder–Mead on [0,1]^dim. Very effective for local refinement.
        """
        # build simplex: u0 plus axis steps
        simplex = [(v0, u0[:])]
        for j in range(dim):
            u = u0[:]
            u[j] += step0
            v, uu = eval_u(u)
            simplex.append((v, uu))
        simplex.sort(key=lambda t: t[0])

        # coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        for _ in range(iters):
            if time.time() >= deadline:
                break

            simplex.sort(key=lambda t: t[0])
            fbest, ubest = simplex[0]
            fworst, uworst = simplex[-1]
            f2, u2 = simplex[-2]

            # centroid of all but worst
            centroid = [0.0] * dim
            inv = 1.0 / dim
            for i in range(dim):  # over vertices 0..dim-1 (excluding worst)
                ui = simplex[i][1]
                for j in range(dim):
                    centroid[j] += ui[j]
            for j in range(dim):
                centroid[j] *= inv

            # reflect
            ur = [centroid[j] + alpha * (centroid[j] - uworst[j]) for j in range(dim)]
            fr, ur = eval_u(ur)

            if fr < fbest:
                # expand
                ue = [centroid[j] + gamma * (ur[j] - centroid[j]) for j in range(dim)]
                fe, ue = eval_u(ue)
                simplex[-1] = (fe, ue) if fe < fr else (fr, ur)
            elif fr < f2:
                simplex[-1] = (fr, ur)
            else:
                # contract
                if fr < fworst:
                    # outside contraction
                    uc = [centroid[j] + rho * (ur[j] - centroid[j]) for j in range(dim)]
                else:
                    # inside contraction
                    uc = [centroid[j] - rho * (centroid[j] - uworst[j]) for j in range(dim)]
                fc, uc = eval_u(uc)
                if fc < fworst:
                    simplex[-1] = (fc, uc)
                else:
                    # shrink towards best
                    new_simplex = [simplex[0]]
                    for i in range(1, dim + 1):
                        ui = simplex[i][1]
                        us = [ubest[j] + sigma * (ui[j] - ubest[j]) for j in range(dim)]
                        fs, us = eval_u(us)
                        new_simplex.append((fs, us))
                    simplex = new_simplex

        simplex.sort(key=lambda t: t[0])
        return simplex[0]

    # --- main search: evolution-style sampling + occasional NM + restarts ---
    mean = best_u[:]

    lam = max(22, 10 * dim)
    if lam % 2 == 1:
        lam += 1
    mu = max(6, lam // 4)

    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    sig = [0.22] * dim
    sig_min = 1e-7
    sig_max = 0.55
    gsig = 1.0

    # step adaptation
    cs = 0.18

    # restart tracking
    last_improve = time.time()
    stall_seconds = max(0.25, 0.10 * max_time)

    # NM scheduling
    nm_every = 6 if dim <= 12 else 8
    nm_step = 0.08 if dim <= 10 else 0.06  # in normalized space

    gen = 0
    while time.time() < deadline:
        gen += 1
        best_before = best

        off = []
        # generate antithetic pairs; sometimes use heavy-tail (Cauchy-ish) for escape
        heavy = (gen % 9 == 0)

        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            # gaussian direction
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            if heavy:
                # occasional heavy-tailed scale (approx Cauchy) for global jumps
                # tan(pi*(u-0.5)) is Cauchy; clamp extreme
                u = random.random()
                c = math.tan(math.pi * (u - 0.5))
                if c > 25.0: c = 25.0
                if c < -25.0: c = -25.0
                scale = 1.0 + 0.35 * abs(c)
            else:
                scale = 1.0

            step = gsig * scale

            u1 = [mean[j] + step * sig[j] * z[j] for j in range(dim)]
            v1, uu1 = eval_u(u1)
            off.append((v1, uu1))

            u2 = [mean[j] - step * sig[j] * z[j] for j in range(dim)]
            v2, uu2 = eval_u(u2)
            off.append((v2, uu2))

            if v1 < best:
                best, best_u = v1, uu1
            if v2 < best:
                best, best_u = v2, uu2

        off.sort(key=lambda t: t[0])

        # recombine mean
        new_mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            ui = off[i][1]
            for j in range(dim):
                new_mean[j] += w * ui[j]
        mean = repair_u(new_mean)

        # adapt per-dim sig from elite dispersion
        c_diag = 0.12 + 1.2 / (dim + 10.0)
        if c_diag > 0.28:
            c_diag = 0.28

        for j in range(dim):
            mj = mean[j]
            s2 = 0.0
            for i in range(mu):
                w = weights[i]
                d = off[i][1][j] - mj
                s2 += w * d * d
            target = math.sqrt(max(1e-32, s2))
            sj = (1.0 - c_diag) * sig[j] + c_diag * (target + 1e-12)
            if sj < sig_min:
                sj = sig_min
            elif sj > sig_max:
                sj = sig_max
            sig[j] = sj

        # global success-based gsig (aim ~0.2 success)
        success = 1.0 if best < best_before else 0.0
        gsig *= math.exp(cs * (success - 0.2))
        if gsig < 1e-7:
            gsig = 1e-7
        elif gsig > 8.0:
            gsig = 8.0

        if best < best_before:
            last_improve = time.time()

        # periodic Nelder–Mead intensification near current best
        if (gen % nm_every == 0) and time.time() < deadline:
            # run a small, time-safe NM
            iters = 18 + 3 * dim
            f0, u0 = best, best_u
            f1, u1 = nelder_mead(u0, f0, iters=iters, step0=nm_step)
            if f1 < best:
                best, best_u = f1, u1
                mean = best_u[:]
                last_improve = time.time()
                # shrink sig a bit after a good local improvement
                for j in range(dim):
                    sig[j] = max(sig_min, sig[j] * 0.85)

        # restart if stalled or steps collapsed
        stalled = (time.time() - last_improve) > stall_seconds
        tiny = all((gsig * sig[j]) < (2.0e-6) for j in range(dim))
        if stalled or tiny:
            # keep incumbent, diversify around it + random injections
            mean = best_u[:]
            gsig = min(2.5, max(0.9, gsig * 1.6))
            # inflate sig to re-explore
            for j in range(dim):
                sig[j] = min(sig_max, max(sig[j], 0.20))

            # inject a few random candidates
            inj = max(10, dim)
            for _ in range(inj):
                if time.time() >= deadline:
                    return best
                v, uu = eval_u(rand_u())
                if v < best:
                    best, best_u = v, uu
                    mean = best_u[:]
                    last_improve = time.time()

    return best
