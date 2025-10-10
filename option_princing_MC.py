import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display

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

print(monte_carlo_call_put(S_0, r, vol, T, K = 40, n_simulations=100000))

#Result function to number of scenarios
def result_MC(S_0, r, vol, T, K):
    mc_10_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=10)
    mc_100_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=100)
    mc_1000_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=1000)
    mc_10000_scenarios = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=10000)
    mc_scenarios = pd.DataFrame({
        'Methods': ['MC 10 scenarios', 'MC 100 scenarios', 'MC 1000 scenarios', 'MC 10000 scenarios'],
        'Call Price': [mc_10_scenarios[0], mc_100_scenarios[0], mc_1000_scenarios[0], mc_10000_scenarios[0]],
        'Put Price': [mc_10_scenarios[1], mc_100_scenarios[1],  mc_1000_scenarios[1], mc_10000_scenarios[1]]
    })
    return mc_scenarios


#Calcul du delta, gamma et vega 
def greeks(S_0, r, vol, T, K, n_simulations=10000):
    # Base prices
    call_price, put_price = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations)
    
    # Price with S_0 + 1
    call_price_up, put_price_up = monte_carlo_call_put(S_0 + 1, r, vol, T, K, n_simulations)
    
    # Price with S_0 - 1
    call_price_down, put_price_down = monte_carlo_call_put(S_0 - 1, r, vol, T, K, n_simulations)
    
    # Delta calculation
    delta_call = (call_price_up - call_price_down) / 2
    delta_put = (put_price_up - put_price_down) / 2
    
    # Price with vol + 0.01
    call_price_vol_up, put_price_vol_up = monte_carlo_call_put(S_0, r, vol + 0.01, T, K, n_simulations)
    
    # Price with vol - 0.01
    call_price_vol_down, put_price_vol_down = monte_carlo_call_put(S_0, r, vol - 0.01, T, K, n_simulations)
    
    # Vega calculation
    vega_call = (call_price_vol_up - call_price_vol_down) / 2
    vega_put = (put_price_vol_up - put_price_vol_down) / 2

    # Gamma calculation 
    gamma_call = (call_price_up - 2 * call_price + call_price_down)
    gamma_put = (put_price_up - 2 * put_price + put_price_down)
    # Store results in a DataFrame
         
    greeks_df = pd.DataFrame({
        'Greeks': ['Delta Call', 'Delta Put', 'gamma_call', 'gamma_put', 'Vega Call', 'Vega Put'],
        'Values': [delta_call, delta_put, gamma_call, gamma_put, vega_call, vega_put]
    })
    
    return greeks_df

#print(greeks(S_0, r, vol, T, K = 40))

#Plot gbm simulations
def plot_gbm(S_0, r, vol, T, n_simulations=1000, n_steps=100):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps)
    
    plt.figure(figsize=(10, 6))
    
    for _ in range(n_simulations):
        Z = np.random.normal(0, 1, n_steps)
        S_t = S_0 * np.exp(np.cumsum((r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z))
        plt.plot(t, S_t)
    
    plt.title('Geometric Brownian Motion Simulations')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid()
    plt.show()

#Show underlying asset simulations
#print(plot_gbm(S_0, r, vol, T, n_simulations=1000, n_steps=100))

#Plot option price convergence
def plot_option_convergence(S_0, r, vol, T, K, max_simulations=100000, step=100):
    simulations = list(range(step, max_simulations + 1, step))
    call_prices = []
    put_prices = []
    
    for n in simulations:
        call_price, put_price = monte_carlo_call_put(S_0, r, vol, T, K, n)
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(simulations, call_prices, label='Call Price', color='blue')
    plt.plot(simulations, put_prices, label='Put Price', color='orange')
    plt.title('Option Price Convergence')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid()
    plt.show()

#print(plot_option_convergence(S_0, r, vol, T, K = 40, max_simulations=100000, step=100))



#Pricing options using B&S formula
def BS_call_put(S_0, r, vol, T, K):
    #Calculate d1 and d2
    d1 = (np.log(S_0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    call_price = S_0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S_0 * stats.norm.cdf(-d1)
    
    return call_price, put_price

def delta_vega_BS(S_0, r, vol, T, K):
    d1 = (np.log(S_0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    
    delta_call = stats.norm.cdf(d1)
    delta_put = stats.norm.cdf(d1) - 1
    vega = S_0 * stats.norm.pdf(d1) * np.sqrt(T)
    
    greeks_bs_df = pd.DataFrame({
        'Greeks': ['Delta Call', 'Delta Put', 'Vega'],
        'Values': [delta_call, delta_put, vega]
    })
    
    return greeks_bs_df

def results(S_0, r, vol, T, K):
    mc_results = result_MC(S_0, r, vol, T, K)
    bs_call, bs_put = BS_call_put(S_0, r, vol, T, K)
    greeks_mc = greeks(S_0, r, vol, T, K)
    greeks_bs = delta_vega_BS(S_0, r, vol, T, K)
    
    bs_df = pd.DataFrame({
        'Methods': ['Black-Scholes'],
        'Call Price': [bs_call],
        'Put Price': [bs_put]
    })
    
    final_results = pd.concat([mc_results, bs_df], ignore_index=True)
    
    print("Option Prices:")
    print(final_results)
    print("\nGreeks from Monte Carlo:")
    print(greeks_mc)
    print("\nGreeks from Black-Scholes:")
    print(greeks_bs)
    
    return final_results

#print(results(S_0, r, vol, T, K = 40))


# export results to excel file
"""
results_df = results(S_0, r, vol, T, K = 40)
results_df.to_csv('option_pricing_results.csv', index=False)
"""
