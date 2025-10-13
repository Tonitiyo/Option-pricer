# Main

from pricing import result_mc, result_bs
from greeks import greeks_mc, greeks_bs
from plot import plot_gbm, plot_option_convergence
from test import convergence_test

# Parameters
S_0, r, vol, T, K = 40, 0.032, 0.2, 1/3, 40

# Example usage
print("Monte Carlo:", result_mc(S_0, r, vol, T, K))
print("Black–Scholes:", result_bs(S_0, r, vol, T, K))

print("\nMonte Carlo Greeks:\n", greeks_mc(S_0, r, vol, T, K))
print("\nBlack–Scholes Greeks:\n", greeks_bs(S_0, r, vol, T, K))

# Plotting examples
plot_gbm(S_0, r, vol, T, n_simulations=20)
plot_option_convergence(S_0, r, vol, T, K, max_simulations=100000, step=10000)

# Convergence test
print("\nConvergence Table:\n", convergence_test(S_0, r, vol, T, K))

