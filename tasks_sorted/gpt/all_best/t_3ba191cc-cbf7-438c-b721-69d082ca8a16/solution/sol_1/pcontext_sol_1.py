import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Hybrid approach:
      1) Differential Evolution with best/1 and current-to-best/1 (more exploitative than rand/1)
      2) jDE-style self-adaptation of F and CR per individual
      3) Opposition-based initialization (often gives a better starting population)
      4) Lightweight local search around incumbent best (adaptive step size)
      5) Stagnation-triggered partial restart (keeps elites)

    Returns
    -------
    best : float
        Best objective value found within time budget.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def time_up():
        return time.time() >= deadline

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Opposition point: x_op = low + high - x
    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # --- Population size: keep moderate for speed; scale gently with dim
    pop_size = max(16, min(80, 6 * dim + 10))

    # jDE adaptation settings
    tau1 = 0.1   # prob to reset F
    tau2 = 0.1   # prob to reset CR
    F_low, F_high = 0.15, 0.95

    # Initialize population with opposition-based initialization
    pop = [rand_vec() for _ in range(pop_size)]
    opp = [clip_inplace(opposite(x[:])) for x in pop]

    fits = []
    all_candidates = pop + opp
    # Evaluate and pick best pop_size among the 2*pop_size candidates
    for x in all_candidates:
        if time_up():
            # if we cannot evaluate, return best of what we have
            return min(fits) if fits else float("inf")
        fits.append(func(x))

    # Select top pop_size
    idx_sorted = sorted(range(len(all_candidates)), key=lambda i: fits[i])
    pop = [all_candidates[i][:] for i in idx_sorted[:pop_size]]
    fit_pop = [fits[i] for i in idx_sorted[:pop_size]]

    # Per-individual parameters
    F_i = [0.6 for _ in range(pop_size)]
    CR_i = [0.9 for _ in range(pop_size)]

    best_idx = min(range(pop_size), key=lambda i: fit_pop[i])
    best = fit_pop[best_idx]
    best_vec = pop[best_idx][:]

    # Stagnation / restart parameters
    no_improve_gens = 0
    restart_patience = max(20, 5 * dim)
    elite_count = max(2, pop_size // 8)
    restart_frac = 0.35

    # Local search controls
    # Step shrinks as time passes and also when we stagnate
    base_local_trials = 10

    gen = 0
    while not time_up():
        gen += 1
        improved = False

        # refresh best index (cheap)
        best_idx = min(range(pop_size), key=lambda i: fit_pop[i])
        if fit_pop[best_idx] < best:
            best = fit_pop[best_idx]
            best_vec = pop[best_idx][:]
            improved = True

        # Evolve population
        for i in range(pop_size):
            if time_up():
                return best

            # Self-adapt F and CR (jDE)
            Fi = F_i[i]
            CR = CR_i[i]
            if random.random() < tau1:
                Fi = F_low + random.random() * (F_high - F_low)
            if random.random() < tau2:
                CR = random.random()

            # Choose strategy: mix exploitative strategies for faster convergence
            # 0: DE/best/1/bin
            # 1: DE/current-to-best/1/bin
            strategy = 0 if random.random() < 0.55 else 1

            # Pick r1, r2 distinct and != i
            # (and also r1 != r2)
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = random.sample(idxs, 2)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            if strategy == 0:
                # v = best + F*(xr1 - xr2)
                v = [best_vec[j] + Fi * (xr1[j] - xr2[j]) for j in range(dim)]
            else:
                # v = xi + F*(best - xi) + F*(xr1 - xr2)
                v = [xi[j] + Fi * (best_vec[j] - xi[j]) + Fi * (xr1[j] - xr2[j]) for j in range(dim)]

            clip_inplace(v)

            # Binomial crossover
            j_rand = random.randrange(dim)
            trial = xi[:]  # start from target
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]

            f_trial = func(trial)
            if f_trial <= fit_pop[i]:
                pop[i] = trial
                fit_pop[i] = f_trial
                F_i[i] = Fi
                CR_i[i] = CR

                if f_trial < best:
                    best = f_trial
                    best_vec = trial[:]
                    improved = True

        # Stagnation bookkeeping
        if improved:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # Local refinement around best (adaptive coordinate + random direction steps)
        # Do it more when close to the end of time OR when stagnating.
        if time_up():
            return best

        elapsed = time.time() - t0
        remain = max(0.0, deadline - time.time())
        progress = min(0.999999, elapsed / max(1e-12, float(max_time)))

        # step factor decays with progress; slightly increases if stagnating
        decay = (1.0 - progress)
        stagn_boost = 1.0 + min(2.0, no_improve_gens / max(1.0, restart_patience))
        step_scale = 0.15 * decay * stagn_boost  # overall local step magnitude

        # Run local search occasionally; also near the end to exploit.
        do_local = (gen % 8 == 0) or (remain < 0.25 * float(max_time)) or (no_improve_gens >= 6)
        if do_local and step_scale > 1e-12:
            trials = base_local_trials + (dim // 3)
            x0 = best_vec[:]
            f0 = best

            # Coordinate-wise small moves + a few random-direction moves
            for _ in range(trials):
                if time_up():
                    return best

                cand = x0[:]

                if random.random() < 0.6:
                    # coordinate perturbation (1-3 coords)
                    k = 1 if dim == 1 else (1 + (1 if random.random() < 0.7 else min(2, dim - 1)))
                    for __ in range(k):
                        j = random.randrange(dim)
                        # symmetric noise ~ [-1,1]
                        u = (random.random() + random.random()) - 1.0
                        cand[j] += u * step_scale * spans[j]
                else:
                    # random direction perturbation
                    # create small direction with sum of uniforms to mimic gaussian-ish
                    for j in range(dim):
                        u = (random.random() + random.random() + random.random() - 1.5)  # ~[-1.5,1.5]
                        cand[j] += u * (0.5 * step_scale) * spans[j]

                clip_inplace(cand)
                fc = func(cand)
                if fc < f0:
                    x0, f0 = cand, fc
                    if fc < best:
                        best, best_vec = fc, cand[:]
                        improved = True
                        no_improve_gens = 0

        # Partial restart on stagnation: keep elites, reinit worst
        if no_improve_gens >= restart_patience:
            # identify elites
            elite_idxs = sorted(range(pop_size), key=lambda k: fit_pop[k])[:elite_count]
            elite_set = set(elite_idxs)

            # restart some of the worst non-elites
            k_restart = max(1, int(pop_size * restart_frac))
            worst = [k for k in sorted(range(pop_size), key=lambda k: fit_pop[k], reverse=True) if k not in elite_set]
            worst = worst[:k_restart]

            for k in worst:
                if time_up():
                    return best
                x = rand_vec()
                fx = func(x)
                pop[k] = x
                fit_pop[k] = fx
                F_i[k] = 0.6
                CR_i[k] = 0.9
                if fx < best:
                    best = fx
                    best_vec = x[:]
                    improved = True

            no_improve_gens = 0

    return best
