#The following algorithm implements **Restarting L-SHADE with Reactive Population Reduction and Directional Momentum Polishing**.
#
###### Rationale for Improvement
#The previous best result (4.02) was achieved by L-SHADE with "Expanded Momentum Polishing". The new algorithm refines this approach to push the result towards zero (global minimum) by addressing specific inefficiencies:
#1.  **Directional Momentum Polishing**: Instead of starting the local search in a random direction (which is inefficient), the polishing phase now initializes its search vector using the difference between the **best and second-best** individuals ($d = x_{best} - x_{second}$). This vector typically aligns with the valley or ridge of the function landscape. The algorithm then performs a greedy line expansion along this vector, allowing it to traverse long, narrow valleys rapidly.
#2.  **Reflection Boundary Handling**: Instead of simply clipping values to bounds (which piles up solutions at the edge), this version uses **reflection** ("bouncing" back into the domain). This preserves the statistical distribution of the population and maintains diversity near the boundaries.
#3.  **Latin Hypercube Initialization**: Instead of pure random sampling, the population is initialized using a simplified Latin Hypercube Sampling (LHS) strategy. This ensures a more uniform coverage of the search space for the initial generation, increasing the probability of finding a good basin early.
#4.  **Robust Parameter Generation**: The $F$ parameter generation now strictly resamples (jSO-style) instead of clipping when values are invalid, ensuring the mutation strength distribution remains statistically sound.
#
###### Algorithm Code
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with Reactive Population Reduction
    and Directional Momentum Polishing.
    
    Mechanism:
    1. Global: L-SHADE with historical parameter adaptation.
    2. Init: Latin Hypercube Sampling (LHS) for uniform coverage.
    3. Bounds: Reflection (Bounce) strategy to preserve diversity.
    4. Local: Directional Polishing using the population's covariance info (Best - 2ndBest).
    5. Meta: Adaptive population reduction and restarts upon stagnation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Helper: Check Time ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: Boundary Handling (Reflection) ---
    def fix_bounds(mutants):
        """
        Reflects mutants back into bounds if they exceed limits.
        Better than clipping for preserving population variance.
        """
        # Lower Bounds
        mask_l = mutants < min_b
        # Broadcast min_b to shape
        min_b_exp = min_b[None, :]
        # Reflect: x' = min + (min - x) = 2*min - x
        mutants = np.where(mask_l, 2.0 * min_b_exp - mutants, mutants)
        # If still out (double bounce), clip
        mutants = np.where(mutants < min_b_exp, min_b_exp, mutants)
        
        # Upper Bounds
        mask_u = mutants > max_b
        max_b_exp = max_b[None, :]
        # Reflect: x' = max - (x - max) = 2*max - x
        mutants = np.where(mask_u, 2.0 * max_b_exp - mutants, mutants)
        # If still out, clip
        mutants = np.where(mutants > max_b_exp, max_b_exp, mutants)
        
        return mutants

    # --- Helper: Directional Momentum Polish ---
    def directional_polish(best_sol, best_val, second_sol, duration):
        """
        Local search that accelerates along the valley defined by the
        best and second-best individuals.
        """
        p_start = datetime.now()
        x_curr = best_sol.copy()
        f_curr = best_val
        
        # 1. Determine direction: Vector from 2nd best to Best
        # This often points down the 'valley'
        if second_sol is not None:
            d = x_curr - second_sol
        else:
            d = np.random.normal(0, 1, dim)
            
        norm = np.linalg.norm(d)
        if norm > 1e-15:
            d /= norm
        else:
            d = np.random.normal(0, 1, dim)
            d /= np.linalg.norm(d)
            
        # Initial step: fairly small relative to domain
        sigma = np.max(diff_b) * 0.01
        
        l_best_val = f_curr
        l_best_sol = x_curr.copy()
        
        while (datetime.now() - p_start).total_seconds() < duration:
            if check_time(): break
            
            # 2. Try Forward Move
            step = d * sigma
            x_test = x_curr + step
            # For local search, clipping is safer/simpler than reflection
            x_test = np.clip(x_test, min_b, max_b)
            f_test = func(x_test)
            
            if f_test < f_curr:
                # Success: Move and Accelerate
                f_curr = f_test
                x_curr = x_test
                if f_curr < l_best_val:
                    l_best_val = f_curr
                    l_best_sol = x_curr.copy()
                    
                # Greedy Expansion (Momentum)
                sigma *= 1.2
            else:
                # 3. Try Backward Move
                x_test = x_curr - 0.5 * step # Reverse and shrink
                x_test = np.clip(x_test, min_b, max_b)
                f_test = func(x_test)
                
                if f_test < f_curr:
                    # Success Backward
                    f_curr = f_test
                    x_curr = x_test
                    if f_curr < l_best_val:
                        l_best_val = f_curr
                        l_best_sol = x_curr.copy()
                    
                    # Reverse direction vector for next iter
                    d = -d
                    sigma *= 1.1 
                else:
                    # 4. Fail: Shrink and Perturb
                    sigma *= 0.5
                    # Add random noise to direction to escape orthogonal ridges
                    d += np.random.normal(0, 1, dim) * 0.2
                    dn = np.linalg.norm(d)
                    if dn > 1e-15: d /= dn
            
            # Reset if step vanishes
            if sigma < 1e-12:
                sigma = np.max(diff_b) * 0.01
                d = np.random.normal(0, 1, dim)
                d /= np.linalg.norm(d)
                
        return l_best_sol, l_best_val

    # --- Main Restart Loop ---
    while not check_time():
        
        # 1. Initialization with Latin Hypercube Sampling (LHS)
        # Robust population size
        pop_size = int(max(30, min(18 * dim, 120)))
        min_pop = 4
        
        population = np.zeros((pop_size, dim))
        for d_i in range(dim):
            # Split dimension into pop_size bins and sample one from each
            edges = np.linspace(min_b[d_i], max_b[d_i], pop_size + 1)
            u = np.random.rand(pop_size)
            points = edges[:-1] + (edges[1:] - edges[:-1]) * u
            np.random.shuffle(points)
            population[:, d_i] = points
            
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = population[i].copy()
                
        # Initialize Memories
        mem_size = 5
        M_CR = np.full(mem_size, 0.5)
        M_F = np.full(mem_size, 0.5)
        k_mem = 0
        
        # Archive (stores good solutions to maintain diversity)
        archive = np.zeros((int(pop_size * 2.5), dim))
        arc_count = 0
        
        # State tracking
        stag_count = 0
        prev_min = np.min(fitness)
        
        # 2. Evolutionary Cycle
        while not check_time():
            # Sort population (Best -> Worst)
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            best_curr = fitness[0]
            
            # Stagnation Check
            if best_curr < prev_min - 1e-10:
                prev_min = best_curr
                stag_count = 0
            else:
                stag_count += 1
                
            # Adaptive Population Reduction (Reactive)
            # Remove worst individual if stagnating
            if pop_size > min_pop and (stag_count > 5):
                pop_size -= 1
                population = population[:pop_size]
                fitness = fitness[:pop_size]
                
            # Restart Trigger
            pop_std = np.std(fitness)
            if pop_size <= min_pop or stag_count > 25 or pop_std < 1e-12:
                # Perform Polishing before restart
                rem_time = (time_limit - (datetime.now() - start_time)).total_seconds()
                if rem_time > 0.1:
                    budget = min(2.0, rem_time * 0.4) # Use up to 40% of remaining time
                    
                    # Directional polish using Best and Second Best
                    second_best = population[1] if pop_size > 1 else None
                    p_res, p_val = directional_polish(population[0], fitness[0], second_best, budget)
                    
                    if p_val < global_best_val:
                        global_best_val = p_val
                        global_best_sol = p_res.copy()
                break # Restart
            
            # --- L-SHADE Parameter Generation ---
            r_idx = np.random.randint(0, mem_size, pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            # Generate CR (Normal)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy with Resampling - jSO style)
            f_p = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    val = m_f[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0:
                        if val > 1: val = 1
                        f_p[i] = val
                        break
            
            # --- Mutation: current-to-pbest/1 ---
            # p varies slightly based on pop_size to balance exploration/exploitation
            p_ratio = max(2.0/pop_size, 0.11)
            n_top = int(max(1, p_ratio * pop_size))
            
            # p-best selection
            pbest_idx = np.random.randint(0, n_top, pop_size)
            x_pbest = population[pbest_idx]
            
            # r1 selection (!= i)
            r1 = np.random.randint(0, pop_size, pop_size)
            mask_s = r1 == np.arange(pop_size)
            r1[mask_s] = (r1[mask_s] + 1) % pop_size
            x_r1 = population[r1]
            
            # r2 selection (Union of Pop and Archive)
            r2 = np.random.randint(0, pop_size + arc_count, pop_size)
            x_r2 = np.zeros((pop_size, dim))
            mask_in = r2 < pop_size
            x_r2[mask_in] = population[r2[mask_in]]
            if arc_count > 0:
                mask_out = ~mask_in
                x_r2[mask_out] = archive[r2[mask_out] - pop_size]
                
            # Compute Mutant
            mutant = population + f_p[:, None] * (x_pbest - population) + f_p[:, None] * (x_r1 - x_r2)
            
            # Apply Bounds using Reflection
            mutant = fix_bounds(mutant)
            
            # --- Crossover (Binomial) ---
            rand_c = np.random.rand(pop_size, dim)
            mask_c = rand_c < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_c[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_c, mutant, population)
            
            # --- Selection ---
            succ_cr = []
            succ_f = []
            diffs = []
            
            for i in range(pop_size):
                if check_time(): return global_best_val
                
                f_t = func(trial[i])
                
                if f_t <= fitness[i]:
                    diff = fitness[i] - f_t
                    
                    # Update Archive
                    if arc_count < len(archive):
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        archive[np.random.randint(0, len(archive))] = population[i].copy()
                        
                    population[i] = trial[i]
                    fitness[i] = f_t
                    
                    # Store Success Info
                    succ_cr.append(cr[i])
                    succ_f.append(f_p[i])
                    diffs.append(diff)
                    
                    if f_t < global_best_val:
                        global_best_val = f_t
                        global_best_sol = trial[i].copy()
                        
            # --- Memory Update (Weighted Lehmer Mean) ---
            if diffs:
                w = np.array(diffs)
                total = np.sum(w)
                if total > 0:
                    w /= total
                    
                    m_cr_new = np.sum(w * np.array(succ_cr))
                    M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                    
                    sf = np.array(succ_f)
                    num = np.sum(w * sf**2)
                    den = np.sum(w * sf)
                    m_f_new = num/den if den > 0 else 0.5
                    M_F[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % mem_size
                    
    return global_best_val
