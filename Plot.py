### Plot 

import numpy as np
import matplotlib.pyplot as plt
from pricing import monte_carlo_call_put

def plot_gbm(S_0, r, vol, T, n_simulations=10, n_steps=100):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps)
    
    plt.figure(figsize=(10, 6))
    for _ in range(n_simulations):
        Z = np.random.normal(0, 1, n_steps)
        S_t = S_0 * np.exp(np.cumsum((r - 0.5 * vol**2)*dt + vol*np.sqrt(dt)*Z))
        plt.plot(t, S_t)
    
    plt.title("Geometric Brownian Motion Simulations")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.grid()
    plt.show()


def plot_option_convergence(S_0, r, vol, T, K, max_simulations=100000, step=1000):
    simulations = range(step, max_simulations + 1, step)
    call_prices, put_prices = [], []
    
    for n in simulations:
        call_price, put_price = monte_carlo_call_put(S_0, r, vol, T, K, n)
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(simulations, call_prices, label="Call Price")
    plt.plot(simulations, put_prices, label="Put Price")
    plt.title("Monte Carlo Price Convergence")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid()
    plt.show()
