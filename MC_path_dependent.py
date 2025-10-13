import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S_0 = 40
r = .03
vol = .2
T = 1/3

# Pricing Monte Carlo euro option call and put in path dependent 
def monte_carlo_euro_option(S0, K, T, r, sigma, M, I):
    # S0: initial stock price
    # K: strike price
    # T: maturity time (in year)
    # r: risk-free interest rate
    # sigma: volatility
    # M: number of time steps
    # I: number of simulations
    
   # Show that when number of time steps increases for a Monte Carlo, the price converges to the Black-Scholes price *
 
    dt = T / M  # time step
    S = np.zeros((M + 1, I))
    S[0] = S0   
    for t in range(1, M + 1):
        Z = np.random.standard_normal(I)  # pseudorandom numbers
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    # European Call and Put Payoffs
    C = np.exp(-r * T) * np.maximum(S[-1] - K, 0)  # discounted call payoff
    P = np.exp(-r * T) * np.maximum(K - S[-1], 0)  # discounted put payoff
    C0 = np.mean(C)  # Monte Carlo estimator
    P0 = np.mean(P)
    return C0, P0

print(monte_carlo_euro_option(S_0, K = 40, T = T, r = r, sigma = vol, M = 100000, I = 10000))
