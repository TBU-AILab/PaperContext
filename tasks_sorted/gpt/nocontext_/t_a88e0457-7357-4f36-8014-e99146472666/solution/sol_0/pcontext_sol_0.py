import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - Sobol-like-ish random initialization (simple stratified / LHS)
      - (mu + lambda) evolution strategy with self-adaptive step sizes
      - occasional coordinate/local refinement
      - restarts if stagnating

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """
    t0 = time.time()

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        # clip to bounds
        y = [0.0] * dim
        for i in range(dim):
            xi = x[i]
            if xi < lows[i]:
                xi = lows[i]
            elif xi > highs[i]:
                xi = highs[i]
            y[i] = xi
        return y

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_fit(x):
        # func expects an "array-like"; list is fine for typical call signatures
        return float(func(x))

    def lhs_samples(n):
        # very small Latin-hypercube sampler (no numpy)
        # returns list of vectors
        # For each dimension, create n strata and permute them
        perms = []
        for _ in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        samples = []
        for k in range(n):
            x = []
            for i in range(dim):
                # stratum index
                s = perms[i][k]
                # jitter inside stratum
                u = (s + random.random()) / n
                x.append(lows[i] + u * spans[i])
            samples.append(x)
        return samples

    # --- initial sampling ---
    best = float("inf")
    best_x = None

    # choose population sizes based on dimension
    mu = max(5, min(20, 4 + dim))          # parents
    lam = max(10, min(60, 8 + 4 * dim))    # offspring
    init_n = max(mu, min(5 * mu, 50))      # initial LHS samples

    # initial sigma per coordinate (relative to span)
    base_sigma = [0.15 * s if s > 0 else 1.0 for s in spans]

    # Evaluate initial points (LHS)
    for x in lhs_samples(init_n):
        if time.time() - t0 >= max_time:
            return best
        f = eval_fit(x)
        if f < best:
            best, best_x = f, x

    # Create initial parents around best + randoms
    parents = []
    # parent structure: (fitness, x, sigmas)
    for _ in range(mu):
        x = best_x[:] if best_x is not None else rand_vec()
        # add some randomization
        x = [x[i] + random.gauss(0.0, base_sigma[i]) for i in range(dim)]
        x = clip(x)
        sig = base_sigma[:]
        f = eval_fit(x)
        parents.append((f, x, sig))
        if f < best:
            best, best_x = f, x

    parents.sort(key=lambda t: t[0])
    parents = parents[:mu]

    # --- strategy parameters ---
    # self-adaptation parameters (log-normal)
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(dim))
    tau0 = 1.0 / math.sqrt(2.0 * dim)

    # Local search frequency and stagnation control
    last_improve_time = time.time()
    no_improve_iters = 0
    iter_count = 0

    def local_refine(x0, sig0):
        """Small coordinate-wise refinement around x0 using current sigma."""
        x = x0[:]
        fx = eval_fit(x)
        improved = False
        # try +/- along a few random coordinates
        coords = list(range(dim))
        random.shuffle(coords)
        m = min(dim, 8)  # limit attempts
        for j in coords[:m]:
            step = sig0[j]
            if step <= 0:
                continue
            for direction in (-1.0, 1.0):
                xn = x[:]
                xn[j] = xn[j] + direction * step
                xn = clip(xn)
                fn = eval_fit(xn)
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
        return x, fx, improved

    # --- main loop ---
    while True:
        if time.time() - t0 >= max_time:
            return best

        iter_count += 1

        # If stagnating, restart partially
        if no_improve_iters > 30 and (time.time() - last_improve_time) > 0.25 * max_time:
            # restart: keep best, resample others
            new_parents = [(best, best_x[:], base_sigma[:])]
            for _ in range(mu - 1):
                x = rand_vec()
                sig = base_sigma[:]
                f = eval_fit(x)
                new_parents.append((f, x, sig))
                if f < best:
                    best, best_x = f, x
                    last_improve_time = time.time()
                    no_improve_iters = 0
            new_parents.sort(key=lambda t: t[0])
            parents = new_parents[:mu]
            no_improve_iters = 0

        # generate offspring
        offspring = []
        # use global best as occasional parent to intensify
        for k in range(lam):
            if time.time() - t0 >= max_time:
                return best

            # select parent (tournament)
            a = parents[random.randrange(mu)]
            b = parents[random.randrange(mu)]
            parent = a if a[0] < b[0] else b
            _, px, psig = parent

            # self-adapt sigmas
            global_noise = random.gauss(0.0, 1.0)
            child_sig = [0.0] * dim
            for i in range(dim):
                # log-normal update, clamp to reasonable range
                s = psig[i] * math.exp(tau0 * global_noise + tau * random.gauss(0.0, 1.0))
                # avoid collapse or explosion
                min_s = 1e-12 * (spans[i] if spans[i] > 0 else 1.0)
                max_s = 0.5 * (spans[i] if spans[i] > 0 else 1.0)
                if s < min_s:
                    s = min_s
                elif s > max_s:
                    s = max_s
                child_sig[i] = s

            # mutate
            cx = [px[i] + random.gauss(0.0, child_sig[i]) for i in range(dim)]
            cx = clip(cx)

            cf = eval_fit(cx)
            offspring.append((cf, cx, child_sig))

            if cf < best:
                best, best_x = cf, cx
                last_improve_time = time.time()
                no_improve_iters = 0

        # (mu + lambda) selection
        pool = parents + offspring
        pool.sort(key=lambda t: t[0])
        parents = pool[:mu]

        # occasional local refinement on current best parent
        if iter_count % 10 == 0:
            if time.time() - t0 >= max_time:
                return best
            topf, topx, tops = parents[0]
            rx, rf, imp = local_refine(topx, tops)
            if rf < topf:
                parents[0] = (rf, rx, tops[:])
                parents.sort(key=lambda t: t[0])
            if rf < best:
                best, best_x = rf, rx
                last_improve_time = time.time()
                no_improve_iters = 0

        # stagnation tracking
        if parents[0][0] >= best:
            no_improve_iters += 1
        else:
            no_improve_iters = 0
