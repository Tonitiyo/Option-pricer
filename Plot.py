### Plot 

import numpy as np
import matplotlib.pyplot as plt
from pricing import monte_carlo_call_put, BS_call_put
from greeks import greeks_mc, greeks_bs

# Parameters
S_0 = 40
r = .032
vol = .2
T = 1/3
k = 40


def plot_gbm(S_0, r, vol, T, n_simulations=10000, n_steps=100):
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


def plot_convergence(S_0, r, vol, T, K, max_simulations=100000, step=2000):
    
    #Show Monte Carlo convergence toward Black–Scholes analytical prices.
    
    simulations = np.arange(step, max_simulations + 1, step)
    call_prices, put_prices = [], []

    # Simulation Monte Carlo pour différents N
    for n in simulations:
        call, put = monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=n)
        call_prices.append(call)
        put_prices.append(put)

    # Référence Black–Scholes
    bs_call, bs_put = BS_call_put(S_0, r, vol, T, K)

    # --- PLOT ---
    plt.figure(figsize=(12, 6))
    
    # Courbes Monte Carlo
    plt.plot(simulations, call_prices, color='royalblue', lw=1.4, label="Monte Carlo Call Price")
    plt.plot(simulations, put_prices, color='darkorange', lw=1.4, label="Monte Carlo Put Price")

    # Lignes horizontales BS
    plt.axhline(bs_call, color='red', linestyle='--', lw=2, label=f"BS Call = {bs_call:.4f}")
    plt.axhline(bs_put, color='green', linestyle='--', lw=2, label=f"BS Put = {bs_put:.4f}")

    # Texte d’annotation
    plt.text(max_simulations * 0.6, bs_call + 0.01, "→ BS Call target", color='red', fontsize=10)
    plt.text(max_simulations * 0.6, bs_put - 0.02, "→ BS Put target", color='green', fontsize=10)

    # Mise en forme
    plt.title("Monte Carlo Convergence toward Black–Scholes Prices", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Simulations", fontsize=12)
    plt.ylabel("Option Price", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


print(plot_convergence(S_0, r, vol, T, k, max_simulations=100000, step=2000))
