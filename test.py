#Test 

import pandas as pd
from pricing import monte_carlo_call_put

def convergence_test(S_0, r, vol, T, K):
    results = []
    for n in [10, 100, 1000, 10000, 10000000]:
        call, put = monte_carlo_call_put(S_0, r, vol, T, K, n)
        results.append({'Simulations': n, 'Call': call, 'Put': put})
    return pd.DataFrame(results)

print(convergence_test(40, 0.03, 0.2, 1/3, 40))

# plot conergence
import matplotlib.pyplot as plt
def plot_convergence(S_0, r, vol, T, K, max_simulations=100000, step=1000):
    simulations = list(range(10, max_simulations + 1, step))
    call_prices = []
    put_prices = []
    
    for n in simulations:
        call, put = monte_carlo_call_put(S_0, r, vol, T, K, n)
        call_prices.append(call)
        put_prices.append(put)
    
    plt.figure(figsize=(12, 6))
    plt.plot(simulations, call_prices, label='Call Price', color='blue')
    plt.plot(simulations, put_prices, label='Put Price', color='orange')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price')
    plt.title('Convergence of Option Prices with Increasing Simulations')
    plt.legend()
    plt.grid(True)
    plt.show()

print(plot_convergence(40, 0.03, 0.2, 1/3, 40, max_simulations=10000000, step=10000))