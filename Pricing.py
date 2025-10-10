### Pricing 

import numpy as np
import scipy.stats as stats
import pandas as pd

# Parameters
S_0 = 40
r = .03
vol = .2
T = 1/3

# Calculate a call and put option using Monte Carlo simulation
def monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=10000):
    # Simulate end-of-period stock prices
    Z = np.random.normal(0, 1, n_simulations)
    S_T = S_0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
    
    # Calculate call and put payoffs
    call_payoffs = np.maximum(S_T - K, 0)
    put_payoffs = np.maximum(K - S_T, 0)
    
    # Discount payoffs back to present value
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_price = np.exp(-r * T) * np.mean(put_payoffs)
    
    return call_price, put_price

def result_mc(S_0, r, vol, T, K):
    mc_10_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=10)
    mc_100_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=100)
    mc_1000_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=1000)
    mc_10000_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=10000)
    mc_100000_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=100000)
    results_mc = pd.DataFrame({
        'Methods': ['MC 10 scenarios', 'MC 100 scenarios', 'MC 1000 scenarios', 'MC 10000 scenarios', 'MC 100000 scenarios'],
        'Call Price': [mc_10_scenarios[0], mc_100_scenarios[0], mc_1000_scenarios[0], mc_10000_scenarios[0], mc_100000_scenarios[0]],
        'Put Price': [mc_10_scenarios[1], mc_100_scenarios[1],  mc_1000_scenarios[1], mc_10000_scenarios[1], mc_100000_scenarios[1]]
    })
    return results_mc


#Pricing options using B&S formula
def BS_call_put(S_0, r, vol, T, K):
    #Calculate d1 and d2
    d1 = (np.log(S_0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    call_price = S_0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S_0 * stats.norm.cdf(-d1)
    
    return call_price, put_price

print(BS_call_put(S_0, r, vol, T, K = 40))

def result_bs(S_0, r, vol, T, K):
    bs_price = BS_call_put(S_0, r, vol, T, K)
    bs_results = pd.DataFrame({
        'Methods' : ["Black and Scholes"],
        'Call Price' : [bs_price[0]],
        'Put Price' : [bs_price[1]]
    })
    return bs_results


def result(S_0, r, vol, T, K):
    mc_results = result_mc(S_0, r, vol, T, K)
    bs_results = result_bs(S_0, r, vol, T, K)
    combined_results = pd.concat([mc_results, bs_results], ignore_index=True)
    return combined_results

print(result(S_0, r, vol, T, K = 40))