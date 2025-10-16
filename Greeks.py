### Greeks  

import numpy as np
import pandas as pd
import scipy.stats as stats
from pricing import monte_carlo_call_put

# Parameters
S_0 = 40
r = .032
vol = .2
T = 1/3


#Calcul du delta, gamma et vega 
def greeks_mc(S_0, r, vol, T, K, n_simulations=10000):
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

def greeks_bs(S_0, r, vol, T, K):
    d1 = (np.log(S_0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    
    delta_call = stats.norm.cdf(d1)
    delta_put = stats.norm.cdf(d1) - 1
    vega = S_0 * stats.norm.pdf(d1) * np.sqrt(T)
    gamma = stats.norm.pdf(d1) / (S_0 * vol * np.sqrt(T))

# Store results in a DataFrame
    
    
    greeks_bs_df = pd.DataFrame({
        'Greeks': ['Delta Call', 'Delta Put', 'Vega', 'gamma'],
        'Values': [delta_call, delta_put, vega, gamma]
    })
    
    return greeks_bs_df

print([greeks_bs(S_0, r, vol, T, K = 40)], [greeks_mc(S_0, r, vol, T, K = 40)])
