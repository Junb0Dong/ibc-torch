import numpy as np

# Simple 2D energy function: E(a) = (a[0] - 0.5)^2 + (a[1] - 0.5)^2, minimum at [0.5, 0.5]
def energy_function(a):
    return (a[:, 0] - 0.5)**2 + (a[:, 1] - 0.5)**2

# Derivative-Free Optimizer
def derivative_free_optimizer(N_samples=100, N_iters=1000, sigma_init=0.33, K=0.5, a_min=-1, a_max=1):
    # Step 1: Initialize samples from uniform
    samples = np.random.uniform(a_min, a_max, (N_samples, 2))
    
    sigma = sigma_init
    for t in range(N_iters):
        # Step 2: Compute energies for all samples
        energies = energy_function(samples)
        
        # Step 3: Softmax probabilities (low energy -> high prob)
        exp_energies = np.exp(-energies)
        probs = exp_energies / np.sum(exp_energies) # softmax
        # print("probs max:", probs.max(), "probs min:",probs.min())
        
        if t < N_iters - 1:
            # Step 4a: Resample elites with replacement based on probs
            indices = np.random.choice(range(N_samples), size=N_samples, p=probs)
            new_samples = samples[indices]
            
            # Step 4b: Add Gaussian noise for exploration
            noise = np.random.normal(0, sigma, (N_samples, 2))
            new_samples += noise
            
            # Step 4c: Clip to bounds
            new_samples = np.clip(new_samples, a_min, a_max)
            
            # Update samples and shrink sigma
            samples = new_samples
            sigma *= K
            # print(f"Iteration {t+1}, sigma: {sigma}")
        else:
            # Final step: Select argmax prob sample as â
            best_idx = np.argmax(probs)
            print("the max probs", probs[best_idx])
            print("prob of best sample:", probs[best_idx])
            a_hat = samples[best_idx]

    return a_hat, energies[best_idx]  # Return final action and its energy

# Run the optimizer
result = derivative_free_optimizer()
print("Final action (â):", result[0])
print("Final energy:", result[1])