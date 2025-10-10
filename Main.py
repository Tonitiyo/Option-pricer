# Main

from pricing import monte_carlo_call_put, BS_call_put
from greeks import greeks_MC, greeks_BS
from plots import plot_gbm, plot_option_convergence
from tests import convergence_test

# Parameters
S_0, r, vol, T, K = 40, 0.03, 0.2, 1/3, 40

# Example usage
print("Monte Carlo:", monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=100000))
print("Black–Scholes:", BS_call_put(S_0, r, vol, T, K))

print("\nMonte Carlo Greeks:\n", greeks_MC(S_0, r, vol, T, K))
print("\nBlack–Scholes Greeks:\n", greeks_BS(S_0, r, vol, T, K))

# Plotting examples
plot_gbm(S_0, r, vol, T, n_simulations=20)
plot_option_convergence(S_0, r, vol, T, K, max_simulations=50000, step=500)

# Convergence test
print("\nConvergence Table:\n", convergence_test(S_0, r, vol, T, K))
