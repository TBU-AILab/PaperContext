#To improve upon the previous SHADE algorithm, I propose implementing **L-SHADE (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)** with a **Restart Mechanism**.
#
#**Why this is better:**
#1.  **Linear Population Reduction (L-SHADE):** Standard SHADE uses a fixed population size. L-SHADE starts with a large population to maximize exploration (finding the general area of the global minimum) and linearly reduces the population size over time. This forces the algorithm to shift from exploration to exploitation (refining the solution) as time runs out, mimicking an annealing process.
#2.  **Restart Mechanism:** The previous result (34.62) suggests the algorithm might have converged prematurely to a local optimum. By detecting stagnation (low population variance), we can trigger a "soft restart" — keeping the best solution found so far but scattering the rest of the population to new random positions to hunt for better basins of attraction within the remaining time.
#
#Here is the fully operational, self-contained Python code:
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution) with Restart capability.
    """
    # --- Timer Setup ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def get_time_elapsed_ratio():
        """Returns 0.0 to 1.0 representing time usage."""
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Preprocessing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- L-SHADE Configuration ---
    # Initial Population: Start large to explore. 
    # Reducing from ~18*dim down to 4 is standard for L-SHADE.
    # We clip max initial size to avoid excessive evaluation costs in short timeframes.
    r_N_init = 18
    N_init = int(round(r_N_init * dim))
    N_init = np.clip(N_init, 50, 500) 
    
    N_min = 4 # Minimum population size
    
    # SHADE Memory parameters
    H = 6 
    
    # --- State Variables ---
    # These are wrapped in a mutable structure or locals to allow "Restarts"
    # to reset them easily.
    
    state = {
        'population': None,
        'fitness': None,
        'M_CR': np.full(H, 0.5),
        'M_F': np.full(H, 0.5),
        'k_mem': 0,
        'archive': [],
        'N_current': N_init,
        'best_val': float('inf'),
        'best_vec': None
    }

    # --- Helper: Initialization / Restart ---
    def initialize_population(current_best_vec=None, current_best_val=float('inf')):
        """
        Initializes population. If a best vector exists (Restart), 
        include it to preserve learned knowledge.
        """
        pop = min_b + np.random.rand(state['N_current'], dim) * diff_b
        fit = np.full(state['N_current'], float('inf'))
        
        # Evaluate initial population
        for i in range(state['N_current']):
            if check_timeout(): break
            
            # If restarting, keep the best one at index 0
            if i == 0 and current_best_vec is not None:
                pop[i] = current_best_vec
                fit[i] = current_best_val
                continue

            try:
                val = func(pop[i])
            except:
                val = float('inf')
            
            fit[i] = val
            
            if val < state['best_val']:
                state['best_val'] = val
                state['best_vec'] = pop[i].copy()
                
        state['population'] = pop
        state['fitness'] = fit
        state['M_CR'] = np.full(H, 0.5)
        state['M_F'] = np.full(H, 0.5)
        state['k_mem'] = 0
        state['archive'] = []

    # Initial boot
    initialize_population()

    # --- Main Optimization Loop ---
    while not check_timeout():
        
        # 1. Calculate Linear Population Reduction
        # N_next = Round [ (N_min - N_init) * (time_elapsed / max_time) + N_init ]
        time_ratio = get_time_elapsed_ratio()
        N_target = int(round((N_min - N_init) * time_ratio + N_init))
        N_target = max(N_min, N_target)

        # 2. Reduce Population if needed
        if state['N_current'] > N_target:
            # Sort by fitness (ascending)
            sort_idx = np.argsort(state['fitness'])
            state['population'] = state['population'][sort_idx]
            state['fitness'] = state['fitness'][sort_idx]
            
            # Truncate to new size
            state['N_current'] = N_target
            state['population'] = state['population'][:N_target]
            state['fitness'] = state['fitness'][:N_target]
            
            # Resize Archive (Archive size should not exceed Population size)
            if len(state['archive']) > N_target:
                # Randomly remove excess
                random.shuffle(state['archive'])
                state['archive'] = state['archive'][:N_target]

        # Local references for speed
        pop = state['population']
        fit = state['fitness']
        N = state['N_current']
        
        # 3. Parameter Generation
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, N)
        m_cr = state['M_CR'][r_idx]
        m_f = state['M_F'][r_idx]
        
        # CR ~ Normal(M_CR, 0.1), clipped [0, 1]
        # If M_CR is -1 (terminal), CR=0 (not used here, but standard logic)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(N)
        f = np.minimum(f, 1.0)
        
        # Repair F <= 0
        bad_f = f <= 0
        while np.any(bad_f):
            count_bad = np.sum(bad_f)
            f[bad_f] = m_f[bad_f] + 0.1 * np.random.standard_cauchy(count_bad)
            f = np.minimum(f, 1.0)
            bad_f = f <= 0

        # 4. Mutation: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # p-best selection: top p% individuals
        # p is random in [2/N, 0.2]
        p_val = np.random.uniform(2.0/N, 0.2)
        p_val = max(p_val, 2.0/N) # Safety
        top_p_cnt = int(p_val * N)
        top_p_cnt = max(top_p_cnt, 1)
        
        # Sort to find pbest
        sorted_indices = np.argsort(fit)
        pbest_indices = sorted_indices[np.random.randint(0, top_p_cnt, N)]
        x_pbest = pop[pbest_indices]
        
        # r1: random from pop, r1 != i
        # We generate random indices and just hope r1 != i for vectorization speed,
        # or accept slight bias. For exactness:
        r1_indices = np.random.randint(0, N, N)
        # Fix collisions r1 == i
        collisions = (r1_indices == np.arange(N))
        r1_indices[collisions] = (r1_indices[collisions] + 1) % N
        x_r1 = pop[r1_indices]
        
        # r2: random from Union(Pop, Archive), r2 != i, r2 != r1
        # Construct Union
        archive_np = np.array(state['archive']) if len(state['archive']) > 0 else np.empty((0, dim))
        union_pop = pop 
        if len(archive_np) > 0:
            union_pop = np.vstack((pop, archive_np))
        
        curr_union_size = len(union_pop)
        r2_indices = np.random.randint(0, curr_union_size, N)
        
        # Logic to ensure r2 != r1 and r2 != i is complex vectorized. 
        # SHADE implementations often relax this strictly or retry. 
        # Simple relaxation: Re-roll collisions once.
        collision_r2 = (r2_indices == np.arange(N)) | (r2_indices == r1_indices)
        if np.any(collision_r2):
            r2_indices[collision_r2] = np.random.randint(0, curr_union_size, np.sum(collision_r2))
            
        x_r2 = union_pop[r2_indices]
        
        # Generate Mutant
        mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, N)
        rand_vals = np.random.rand(N, dim)
        mask = (rand_vals < cr[:, None])
        mask[np.arange(N), j_rand] = True
        
        trial = np.where(mask, mutant, pop)
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Selection & Updates
        succ_f = []
        succ_cr = []
        diff_f = []
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(N):
            if check_timeout(): return state['best_val']
            
            try:
                t_val = func(trial[i])
            except:
                t_val = float('inf')
                
            if t_val <= fit[i]:
                if t_val < fit[i]:
                    state['archive'].append(pop[i].copy())
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_f.append(fit[i] - t_val)
                
                new_pop[i] = trial[i]
                new_fit[i] = t_val
                
                if t_val < state['best_val']:
                    state['best_val'] = t_val
                    state['best_vec'] = trial[i].copy()
                    
        state['population'] = new_pop
        state['fitness'] = new_fit
        
        # Maintain Archive Limit (Size <= N)
        if len(state['archive']) > N:
            random.shuffle(state['archive'])
            state['archive'] = state['archive'][:N]
            
        # Update Memory (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            diff_f = np.array(diff_f)
            
            total_diff = np.sum(diff_f)
            weights = diff_f / total_diff if total_diff > 0 else np.ones(len(diff_f))/len(diff_f)
            
            # Update M_CR
            if np.max(succ_cr) == 0:
                m_cr_new = 0
            else:
                m_cr_new = np.sum(weights * succ_cr)
            
            # Update M_F (Lehmer)
            num = np.sum(weights * (succ_f**2))
            den = np.sum(weights * succ_f)
            m_f_new = num / den if den > 0 else 0.5
            
            k = state['k_mem']
            state['M_CR'][k] = np.clip(m_cr_new, 0, 1)
            state['M_F'][k] = np.clip(m_f_new, 0, 1)
            state['k_mem'] = (k + 1) % H
            
        # --- Restart Logic ---
        # If population variance is extremely low, we are likely stuck in a local optimum.
        # If we have time left, restart the population around the bounds, keeping the best.
        # Condition: Std Dev < Threshold AND Time < 85% (save time for final convergence)
        if N >= 4:
            std_dev = np.std(state['fitness'])
            if std_dev < 1e-9 and get_time_elapsed_ratio() < 0.85:
                # Trigger Restart
                # Reset population size to a portion of initial to avoid being too slow
                state['N_current'] = int(N_init * 0.5) 
                state['N_current'] = np.clip(state['N_current'], 10, 100)
                initialize_population(state['best_vec'], state['best_val'])

    return state['best_val']
